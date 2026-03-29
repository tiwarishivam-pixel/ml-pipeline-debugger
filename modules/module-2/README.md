# Module 2: Writing Agents Against the ML Pipeline Debugger

## What This Module Covers

Build four debugging agents of increasing sophistication, run them against
all three task difficulties, and compare their scores. You will see exactly
where LLMs succeed and where they still fail on hard ML debugging.

## The Four Agents

| Agent | Strategy | Expected score |
|-------|----------|---------------|
| Naive | Returns the broken script unchanged | 0.2–0.3 |
| Keyword | Patches based on bug_type string matching | 0.3–0.5 |
| LLM (GPT-4o) | Reads script, reasons, fixes | 0.5–0.85 |
| Chain-of-Thought LLM | Thinks step by step before fixing | 0.6–0.9 |

## What Makes a Good Debugging Agent

A good agent must:
1. Read the `loss_curve` and recognise the failure pattern (exploding, NaN, flat)
2. Read the `bug_type` hint and map it to a class of fixes
3. Read the `broken_script` and find the exact broken line
4. Return a *complete*, runnable script that prints `loss:X.XXXXXX` every epoch

## Why Task 3 Is Hard for LLMs

Task 3 (silent underfitting) has no error message and a loss curve that looks
almost healthy (drops slightly then flatlines). GPT-4o frequently patches the
wrong thing — changing learning rate when the real issue is weight initialisation.
This is the environment's core research value.

## Setup

```bash
pip install openenv-core torch numpy openai
export OPENAI_API_KEY=sk-...
uvicorn server.app:app --host 0.0.0.0 --port 8000
```
