"""
Baseline inference script for the ML Pipeline Debugger environment.

Uses the OpenAI API to run a baseline agent against all three tasks.
Reads OPENAI_API_KEY from environment variables.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline.py --url http://localhost:8000
    python baseline.py --url https://<your-hf-space>.hf.space

Produces a reproducible baseline score on all 3 task difficulties.
"""

import argparse
import os
import statistics
import sys
import time

from openai import OpenAI

# ── try importing the client (works both installed and local) ──────────────────
try:
    from ml_pipeline_debugger.client import MLPipelineDebuggerEnv
    from ml_pipeline_debugger.models import MLDebugAction
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from client import MLPipelineDebuggerEnv
    from models import MLDebugAction

# ── system prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert ML engineer specialised in debugging PyTorch training failures.

You will receive a broken Python training script along with:
- The symptom (what's going wrong: exploding loss, NaN, flatline, etc.)
- The observed loss curve from running the broken script
- The bug category (what type of error was injected)

Your job: identify the exact bug and return a COMPLETE, FIXED Python script
that can be run as-is.

CRITICAL RULES:
1. Return ONLY the complete fixed Python script — nothing else, no markdown fences
2. Every epoch must print: loss:X.XXXXXX  (the grader parses this exact format)
3. Do not change the model architecture or dataset unless the bug is there
4. The fix should be minimal — change only what is broken

Bug categories you may encounter:
- learning_rate_too_high / lr_zero: fix the optimizer lr value
- weight_decay_too_high: fix the weight_decay parameter  
- momentum_negative: fix the momentum sign
- epochs_zero: fix the range() argument
- div_by_zero_std: add a guard before dividing by std
- log_of_zero: clip values before log transform
- sqrt_negative: use abs() before sqrt
- inf_from_scale: remove or fix the extreme scaling factor
- missing_fillna: fill NaN values before tensor conversion
- zero_weight_init: remove constant_ init or use proper init
- frozen_layer: set requires_grad=True on all parameters
- missing_optimizer_step: add optimizer.step() call
- wrong_activation: replace with torch.relu or nn.ReLU()
"""


def build_user_prompt(obs) -> str:
    """Format the observation into a clear prompt for the model."""
    curve_str = str(obs.loss_curve[:15]) if obs.loss_curve else "[]"
    return f"""TASK: {obs.task_id} | DIFFICULTY: {obs.difficulty} | BUG TYPE: {obs.bug_type}

SYMPTOM:
{obs.task_description}

LOSS CURVE (first steps from running broken script):
{curve_str}

BROKEN SCRIPT:
{obs.broken_script}

Return the complete fixed Python script only. No explanation, no markdown."""


def run_baseline_episode(env, client: OpenAI, model: str = "gpt-4o") -> dict:
    """Run one episode: reset → query LLM → step → return result."""
    # Reset and get broken script
    result = env.reset()
    obs = result.observation

    task_id = obs.task_id
    difficulty = obs.difficulty
    bug_type = obs.bug_type

    # Query the LLM
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(obs)},
            ],
            temperature=0.0,   # deterministic
            max_tokens=2000,
        )
        fixed_code = response.choices[0].message.content.strip()

        # Strip markdown fences if the model added them anyway
        if fixed_code.startswith("```"):
            lines = fixed_code.split("\n")
            fixed_code = "\n".join(
                line for line in lines
                if not line.startswith("```")
            )

    except Exception as e:
        fixed_code = obs.broken_script  # fallback — submit broken script
        print(f"  [WARN] LLM call failed: {e}. Submitting broken script.")

    # Submit the fix and get graded
    step_result = env.step(MLDebugAction(
        fixed_code=fixed_code,
        explanation="Baseline LLM fix",
        task_id=task_id,
    ))

    return {
        "task_id":   task_id,
        "difficulty": difficulty,
        "bug_type":  bug_type,
        "score":     step_result.reward or 0.0,
        "feedback":  step_result.observation.task_description,
    }


def main():
    parser = argparse.ArgumentParser(description="ML Pipeline Debugger baseline agent")
    parser.add_argument("--url",     default="http://localhost:8000", help="Environment server URL")
    parser.add_argument("--model",   default="gpt-4o",               help="OpenAI model to use")
    parser.add_argument("--episodes",type=int, default=15,           help="Total episodes to run")
    parser.add_argument("--seed",    type=int, default=42,           help="Not used — for documentation")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print(f"ML Pipeline Debugger — Baseline Evaluation")
    print(f"Model:    {args.model}")
    print(f"Server:   {args.url}")
    print(f"Episodes: {args.episodes}")
    print("=" * 55)

    results = []

    with MLPipelineDebuggerEnv(base_url=args.url).sync() as env:
        for ep in range(1, args.episodes + 1):
            print(f"\nEpisode {ep:2d}/{args.episodes}", end=" ")
            t0 = time.time()

            try:
                r = run_baseline_episode(env, client, args.model)
                results.append(r)
                elapsed = time.time() - t0
                print(f"| {r['task_id']} ({r['difficulty']:6s}) | bug: {r['bug_type']:30s} | score: {r['score']:.2f} | {elapsed:.1f}s")
            except Exception as e:
                print(f"| ERROR: {e}")

    # ── summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("RESULTS SUMMARY")
    print("=" * 55)

    by_difficulty = {"easy": [], "medium": [], "hard": []}
    for r in results:
        by_difficulty[r["difficulty"]].append(r["score"])

    for diff, scores in by_difficulty.items():
        if scores:
            print(f"  {diff:6s}  n={len(scores):2d}  mean={statistics.mean(scores):.3f}  "
                  f"min={min(scores):.2f}  max={max(scores):.2f}")

    all_scores = [r["score"] for r in results]
    if all_scores:
        print(f"\n  OVERALL  n={len(all_scores):2d}  mean={statistics.mean(all_scores):.3f}")

    print("\nNote: Baseline scores are recorded here for the hackathon submission.")
    print("These scores are reproducible by re-running this script.")


if __name__ == "__main__":
    main()
