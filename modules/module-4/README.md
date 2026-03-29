# Module 4: Building the ML Pipeline Debugger — Selected Problem Statement

## Why This Is the Selected PS

Our project is a **custom OpenEnv environment built from scratch** following
the exact Module 4 pattern. Every component maps directly:

| Module 4 pattern | Our implementation |
|-----------------|-------------------|
| `models.py` types | `MLDebugAction`, `MLDebugObservation` |
| `server/environment.py` | `MlPipelineDebuggerEnvironment` |
| `server/tasks/` | `task1_hyperparams.py`, `task2_nan_pipeline.py`, `task3_silent_underfit.py` |
| `client.py` | `MLPipelineDebuggerEnv` (WebSocket client) |
| `server/app.py` | FastAPI wiring via `create_app()` |
| `server/Dockerfile` | Multi-stage build using openenv-base |

## The 3-Component Pattern

```
ml_pipeline_debugger/
├── models.py                    ← MLDebugAction + MLDebugObservation
├── client.py                    ← WebSocket client (what users import)
├── server/
│   ├── app.py                   ← FastAPI via create_app()
│   ├── ml_pipeline_debugger_environment.py  ← reset() + step()
│   ├── Dockerfile
│   └── tasks/
│       ├── grader_utils.py      ← subprocess runner + loss parser
│       ├── task1_hyperparams.py ← 5 bug variants + grader
│       ├── task2_nan_pipeline.py← 5 bug variants + grader
│       └── task3_silent_underfit.py ← 4 bug variants + grader
├── openenv.yaml
└── pyproject.toml
```

## The Grader Design (Key Innovation)

Every grader follows the same deterministic pattern:

```python
def grade_taskN(fixed_code: str) -> Tuple[float, str]:
    # 1. Run in isolated subprocess
    success, stdout, stderr = run_script(fixed_code, timeout=45)
    if not success:
        return 0.0, "Script crashed"

    # 2. Parse loss:X.XXXXXX from stdout
    losses = parse_losses(stdout)

    # 3. Score mathematically — no LLM, no fuzzy matching
    score = 0.0
    if runs_clean:       score += 0.3
    if loss_decreasing:  score += 0.4
    if threshold_reached:score += 0.3
    return score, feedback
```

**No LLM judge. No regex on explanation text. Pure execution-based scoring.**

## Build Steps (Scaffold → Implement → Validate → Deploy)

```bash
# Step 1: Scaffold
openenv init ml_pipeline_debugger
cd ml_pipeline_debugger

# Step 2: Implement (edit models.py, server/environment.py, tasks/)
# Step 3: Validate
openenv validate .          # Must print: [OK] Ready for multi-mode deployment

# Step 4: Test locally
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Step 5: Deploy
openenv push --repo-id YOUR_USERNAME/ml-pipeline-debugger
```

## Hackathon Scoring Alignment

| Criterion (weight) | How we meet it |
|-------------------|----------------|
| Real-world utility (30%) | ML debugging is a daily pain at Meta/HF |
| Task & grader quality (25%) | 3 tasks, 14 bug variants, deterministic graders |
| Environment design (20%) | Partial credit rewards, clean state management |
| Code quality (15%) | OpenEnv spec compliant, typed models, Dockerfile |
| Creativity & novelty (10%) | No existing OpenEnv environment does ML debugging |
