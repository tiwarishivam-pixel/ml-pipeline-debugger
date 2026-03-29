# Module 5: Training a Debugging Agent with GRPO

## What This Module Covers

Fine-tune a small LLM (Qwen3-1.7B) using GRPO to automatically fix broken
ML training scripts. The model learns by interacting with the ML Pipeline
Debugger environment — fixing scripts, getting scored, and improving.

## Why GRPO Works Here

GRPO generates a group of fixes for the same broken script, scores each one
using the environment's deterministic grader, then updates the model based
on which fixes scored highest within the group. No value model needed.

## Reward Functions

| Reward | What it measures | Range |
|--------|-----------------|-------|
| `reward_runs` | Does the fixed script run without crashing? | 0.0 or 0.3 |
| `reward_decreasing` | Is the loss curve going down? | 0.0 or 0.4 |
| `reward_threshold` | Does the final loss reach the target? | 0.0 or 0.3 |
| `reward_total` | Full grader score (sum of above) | 0.0 – 1.0 |

## Hardware Requirements

- **GPU:** A100 40GB (Colab Pro or similar)
- **Training time:** ~2 hours for 500 episodes
- **Peak memory:** ~35GB

## Setup

```bash
pip install trl>=0.17.0 openenv-core transformers torch numpy vllm
export OPENAI_API_KEY=sk-...  # only needed for baseline comparison
```

## The Training Loop

```
for each batch:
  1. Sample broken scripts from environment (reset())
  2. Generate N candidate fixes using the model
  3. Submit each fix to the grader (step())
  4. Get scores 0.0-1.0 per fix
  5. GRPO: update model toward higher-scoring fixes
```

## Expected Learning Curve

| Training step | Task 1 score | Task 2 score | Task 3 score |
|---------------|-------------|-------------|-------------|
| 0 (base) | 0.3 | 0.3 | 0.2 |
| 100 | 0.6 | 0.45 | 0.25 |
| 500 | 0.85 | 0.65 | 0.40 |
| 1000 | 0.95 | 0.75 | 0.55 |

Task 3 (silent underfitting) remains challenging because it requires
multi-file reasoning with no error message — the core research challenge.
