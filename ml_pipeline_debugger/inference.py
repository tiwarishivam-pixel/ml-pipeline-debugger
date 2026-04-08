"""
Inference script for ML Pipeline Debugger environment.
Follows OpenEnv hackathon submission requirements strictly.

Environment variables:
    API_BASE_URL     - LLM API endpoint (default: OpenAI)
    MODEL_NAME       - Model to use (default: gpt-4o)
    HF_TOKEN         - HuggingFace/API token (no default - required)
    LOCAL_IMAGE_NAME - Optional: local Docker image name
"""

import os
import sys

from openai import OpenAI

# ── Environment variables ──────────────────────────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "gpt-4o")
HF_TOKEN         = os.getenv("HF_TOKEN")           # No default — required
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")   # Optional

# ── OpenAI client configured via environment variables ────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

SYSTEM_PROMPT = """You are an expert ML engineer who debugs broken PyTorch training scripts.

You receive a broken script, a symptom description, and the bug category.
Return ONLY the complete fixed Python script — no markdown, no explanation, no code fences.
Every epoch MUST print exactly: loss:X.XXXXXX
"""


def get_fix(broken_script, task_description, bug_type):
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
    if fixed.startswith("```"):
        fixed = "\n".join(l for l in fixed.split("\n") if not l.startswith("```"))
    return fixed


def main():
    space_url = os.getenv(
        "SPACE_URL",
        "https://shivamtiwari84-ml-pipeline-debugger.hf.space"
    )

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        from client import MLPipelineDebuggerEnv
        from models import MLDebugAction
    except ImportError:
        print("[START] task=ml_pipeline_debugger", flush=True)
        print("[STEP] step=1 reward=0.0", flush=True)
        print("[END] task=ml_pipeline_debugger score=0.0 steps=1", flush=True)
        sys.exit(1)

    total_score = 0.0
    total_steps = 0

    with MLPipelineDebuggerEnv(base_url=space_url).sync() as env:

        for episode in range(15):

            result = env.reset()
            obs    = result.observation
            task   = obs.task_id

            print(f"[START] task={task}", flush=True)

            try:
                fixed_code = get_fix(
                    obs.broken_script,
                    obs.task_description,
                    obs.bug_type,
                )
            except Exception:
                fixed_code = obs.broken_script

            step_result = env.step(MLDebugAction(
                fixed_code=fixed_code,
                explanation="LLM inference fix",
                task_id=task,
            ))

            score = float(step_result.reward or 0.0)
            total_score += score
            total_steps += 1

            print(f"[STEP] step={total_steps} reward={score}", flush=True)
            print(f"[END] task={task} score={score} steps={total_steps}", flush=True)

    avg = round(total_score / total_steps, 3) if total_steps > 0 else 0.0
    print(f"[END] task=ml_pipeline_debugger score={avg} steps={total_steps}", flush=True)


if __name__ == "__main__":
    main()
