---
title: ML Pipeline Debugger
emoji: 🦀
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
---

# ML Pipeline Debugger

An **OpenEnv** reinforcement learning environment where agents learn to diagnose and fix broken PyTorch training pipelines. 

Unlike standard code-generation tasks, this environment uses a **deterministic grader** that executes the agent's fix in an isolated subprocess. Scoring is based on real training behavior (loss convergence)—no LLM judges, no fuzzy matching.

---

## 🚀 Motivation

Every ML engineer has faced the "silent failure": a training run where the loss explodes to `NaN` or flatlines without an error message. Finding the cause manually takes hours. 

This environment trains agents to automate that debugging—making them useful for:
* **Automated Code Review** at ML-first companies.
* **CI/CD Pipelines** that catch regressions in model logic.
* **Training Monitors** that suggest fixes in real-time.

---

## 📊 Environment Overview

| Property | Value |
| :--- | :--- |
| **Action Space** | `MLDebugAction` (Fixed script + Explanation) |
| **Observation Space** | `MLDebugObservation` (Broken script, loss curve, logs) |
| **Reward** | Float `0.0` – `1.0` (Based on training metrics) |
| **Episode Length** | 1 Step (One fix attempt per episode) |
| **Tasks** | 3 Difficulty Levels (Easy / Medium / Hard) |
| **Bug Variants** | 14 Unique logical/numerical bugs |

---

## 🛠️ Task breakdown

### 1. Hyperparameter Bomb (Easy)
The script runs but won't learn due to a catastrophic hyperparameter setting (e.g., `LR=10.0`).
* **Goal:** Identify the outlier value and bring it to a sane range.

### 2. Data Pipeline NaN Bomb (Medium)
A subtle mathematical error in `preprocess()` (e.g., `sqrt(-x)`) causes NaNs to propagate.
* **Goal:** Trace the numerical instability back to the data transformation layer.

### 3. Silent Underfitting (Hard)
The "Senior Engineer" test. The script runs perfectly, but a structural bug (e.g., a frozen layer or missing `optimizer.step()`) prevents learning.
* **Goal:** Deep architectural inspection of the PyTorch logic.

---
## Setup and Usage

### Install

```bash
git clone https://huggingface.co/spaces/<your-username>/ml-pipeline-debugger
cd ml-pipeline-debugger
pip install -e .
```

### Run locally

```bash
# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Or via the entry point
python -m server.app
```

### Run with Docker

```bash
docker build -t ml-pipeline-debugger:latest -f server/Dockerfile .
docker run -d -p 8000:8000 ml-pipeline-debugger:latest
```

### Connect with the Python client

```python
import asyncio
from ml_pipeline_debugger.client import MLPipelineDebuggerEnv
from ml_pipeline_debugger.models import MLDebugAction

async def main():
    async with MLPipelineDebuggerEnv(base_url="http://localhost:8000") as env:
        result = await env.reset()
        obs = result.observation

        print(obs.task_id)          # task1 / task2 / task3
        print(obs.difficulty)       # easy / medium / hard
        print(obs.bug_type)         # e.g. lr_too_high
        print(obs.broken_script)    # the code to fix
        print(obs.loss_curve[:5])   # first 5 loss values

        result = await env.step(MLDebugAction(
            fixed_code="...your fixed script...",
            explanation="Changed lr from 10.0 to 0.01",
            task_id=obs.task_id,
        ))
        print(result.reward)   # 0.0 – 1.0

asyncio.run(main())

# Sync usage
with MLPipelineDebuggerEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    result = env.step(MLDebugAction(fixed_code="..."))
    print(result.reward)
```

---

## Grader Design

All graders follow the same deterministic pattern:

1. Run `fixed_code` in an isolated subprocess with a 45–90 second timeout
2. Parse `loss:X.XXXXXX` lines from stdout
3. Score mathematically — no LLM, no fuzzy matching

This guarantees identical scores across repeated runs. The grader cannot be fooled by a well-written explanation — only a fix that actually trains correctly earns a high score.

---

## Baseline Scores

Evaluated using GPT-4o (temperature=0), 15 episodes (5 per difficulty):

| Difficulty | Mean score | Notes |
|---|---|---|
| Easy | TBD | Run `python baseline.py` to reproduce |
| Medium | TBD | |
| Hard | TBD | |
| **Overall** | **TBD** | |

To reproduce:

```bash
export OPENAI_API_KEY=sk-...
python baseline.py --url http://localhost:8000 --episodes 15
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `WORKERS` | 4 | Uvicorn worker processes |
| `PORT` | 8000 | Server port |
| `MAX_CONCURRENT_ENVS` | 50 | Max WebSocket sessions |
| `OPENAI_API_KEY` | — | Required for baseline script only |
