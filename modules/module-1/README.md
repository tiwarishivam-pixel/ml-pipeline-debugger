# Module 1: Connecting to the ML Pipeline Debugger

## What This Module Covers

Connect to the ML Pipeline Debugger environment and use the standard OpenEnv
3-method interface: `reset()`, `step()`, `state()`. No building required —
just use the running environment.

## The Environment in One Sentence

The agent receives a broken PyTorch training script, fixes it, and gets a
score 0.0–1.0 based on whether the fix actually trains correctly.

## The 3-Method Interface

| Method | What it does | Returns |
|--------|-------------|---------|
| `reset()` | Samples a random bug variant | Broken script + loss curve + symptom |
| `step(action)` | Submits fixed code | Reward 0.0–1.0 + grader feedback |
| `state()` | Episode metadata | task_id, difficulty, step_count |

## What the Agent Sees (Observation)

```
broken_script     — the complete broken Python training script
error_log         — stderr if the script crashed (None if silent failure)
loss_curve        — first N loss values from running the broken script
task_description  — plain-English symptom description
task_id           — task1 | task2 | task3
difficulty        — easy | medium | hard
bug_type          — e.g. lr_too_high | div_by_zero_std | zero_weight_init
```

## What the Agent Returns (Action)

```
fixed_code    — the complete corrected Python script (must print loss:X.XXXXXX)
explanation   — why it was broken and what was changed
task_id       — which task this fix is for
```

## Setup

```bash
pip install openenv-core torch numpy
cd ml_pipeline_debugger
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

## Key Takeaway

Once you know `reset()` → `step()` → check `reward`, you know every OpenEnv
environment. The observation fields change, but the interface never does.
