"""
Inference script for ML Pipeline Debugger environment.
Follows OpenEnv hackathon submission requirements strictly.

Environment variables:
    API_BASE_URL   - LLM API endpoint (default: OpenAI)
    MODEL_NAME     - Model to use (default: gpt-4o)
    HF_TOKEN       - HuggingFace token (no default - required)
    LOCAL_IMAGE_NAME - Optional: local Docker image name
"""

import os
import sys
import json

from openai import OpenAI

# ── Environment variables (as required by checklist) ──────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o")
HF_TOKEN     = os.getenv("HF_TOKEN")           # No default — required
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Optional

# ── OpenAI client configured via environment variables ────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert ML engineer who debugs broken PyTorch training scripts.

You receive a broken script, a symptom description, and the bug category.
Return ONLY the complete fixed Python script — no markdown, no explanation, no code fences.
Every epoch MUST print exactly: loss:X.XXXXXX
"""

# ── Fix function using OpenAI client ──────────────────────────────────────────
def get_fix(broken_script: str, task_description: str, bug_type: str) -> str:
    """Call LLM to fix the broken script."""
    user_msg = f"""BUG TYPE: {bug_type}
SYMPTOM: {task_description}

BROKEN SCRIPT:
{broken_script}

Return the complete fixed Python script only. No markdown. No explanation."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=2000,
    )

    fixed = response.choices[0].message.content.strip()

    # Strip markdown fences if model added them
    if fixed.startswith("```"):
        fixed = "\n".join(
            line for line in fixed.split("\n")
            if not line.startswith("```")
        )

    return fixed


# ── Main inference loop ────────────────────────────────────────────────────────
def main():
    # Get Space URL from args or environment
    if len(sys.argv) > 1:
        space_url = sys.argv[1]
    else:
        space_url = os.getenv(
            "SPACE_URL",
            "https://shivamtiwari84-ml-pipeline-debugger.hf.space"
        )

    print(f"START", flush=True)
    print(json.dumps({
        "event": "START",
        "space_url": space_url,
        "model": MODEL_NAME,
    }), flush=True)

    # Import client
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from client import MLPipelineDebuggerEnv
        from models import MLDebugAction
    except ImportError:
        print(json.dumps({"event": "ERROR", "msg": "client/models import failed"}), flush=True)
        sys.exit(1)

    results = []

    with MLPipelineDebuggerEnv(base_url=space_url).sync() as env:

        for episode in range(15):

            # Reset — get broken script
            result = env.reset()
            obs    = result.observation

            print(json.dumps({
                "event":      "STEP",
                "episode":    episode + 1,
                "task_id":    obs.task_id,
                "difficulty": obs.difficulty,
                "bug_type":   obs.bug_type,
            }), flush=True)

            # Get fix from LLM
            try:
                fixed_code = get_fix(
                    obs.broken_script,
                    obs.task_description,
                    obs.bug_type,
                )
            except Exception as e:
                print(json.dumps({
                    "event": "STEP",
                    "episode": episode + 1,
                    "error": str(e),
                    "score": 0.0,
                }), flush=True)
                results.append(0.0)
                continue

            # Submit fix
            step_result = env.step(MLDebugAction(
                fixed_code=fixed_code,
                explanation="LLM inference fix",
                task_id=obs.task_id,
            ))

            score = float(step_result.reward or 0.0)
            results.append(score)

            print(json.dumps({
                "event":      "STEP",
                "episode":    episode + 1,
                "task_id":    obs.task_id,
                "difficulty": obs.difficulty,
                "bug_type":   obs.bug_type,
                "score":      score,
            }), flush=True)

    # Final summary
    avg = sum(results) / len(results) if results else 0.0

    by_diff = {}
    for i, r in enumerate(results):
        # approximate difficulty from position
        diff = "easy" if i < 5 else "medium" if i < 10 else "hard"
        by_diff.setdefault(diff, []).append(r)

    print(json.dumps({
        "event":          "END",
        "total_episodes": len(results),
        "average_score":  round(avg, 3),
        "all_scores":     results,
    }), flush=True)

    print(f"END", flush=True)


if __name__ == "__main__":
    main()