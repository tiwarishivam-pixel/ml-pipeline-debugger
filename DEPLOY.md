# Deployment & Submission Guide

Complete step-by-step instructions to deploy and submit before **8 April 2026, 11:59 PM IST**.

---

## Prerequisites

```bash
pip install openenv-core torch numpy openai huggingface_hub
huggingface-cli login   # enter your HF token
```

---

## Step 1 — Validate locally

```bash
cd ml_pipeline_debugger
openenv validate .
# Expected: [OK] : Ready for multi-mode deployment
```

---

## Step 2 — Run locally and smoke-test

```bash
# Terminal 1 — start server
cd ml_pipeline_debugger
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — smoke test
curl http://localhost:8000/health
# {"status":"healthy"}

python3 -c "
import sys; sys.path.insert(0, 'ml_pipeline_debugger')
from client import MLPipelineDebuggerEnv
from models import MLDebugAction

with MLPipelineDebuggerEnv(base_url='http://localhost:8000').sync() as env:
    obs = env.reset().observation
    print('task:', obs.task_id, '| difficulty:', obs.difficulty, '| bug:', obs.bug_type)
    result = env.step(MLDebugAction(
        fixed_code=obs.broken_script,
        task_id=obs.task_id,
    ))
    print('reward:', result.reward, '| done:', result.done)
"
```

---

## Step 3 — Run the baseline script

```bash
export OPENAI_API_KEY=sk-...
python3 ml_pipeline_debugger/baseline.py \
    --url http://localhost:8000 \
    --model gpt-4o \
    --episodes 15
```

Copy the printed scores table into `ml_pipeline_debugger/README.md`
under the **Baseline Scores** section.

---

## Step 4 — Deploy to HuggingFace Spaces

```bash
cd ml_pipeline_debugger
openenv push --repo-id YOUR_HF_USERNAME/ml-pipeline-debugger
```

Wait ~2 minutes for the Space to build. Then verify:

```bash
curl https://YOUR_HF_USERNAME-ml-pipeline-debugger.hf.space/health
# {"status":"healthy"}
```

Your environment is now live at:
- **API:**    https://YOUR_HF_USERNAME-ml-pipeline-debugger.hf.space
- **Web UI:** https://YOUR_HF_USERNAME-ml-pipeline-debugger.hf.space/web
- **Docs:**   https://YOUR_HF_USERNAME-ml-pipeline-debugger.hf.space/docs

---

## Step 5 — Validate the live Space

```bash
openenv validate https://YOUR_HF_USERNAME-ml-pipeline-debugger.hf.space
# Expected: [OK]
```

---

## Step 6 — Submit

Paste this URL on the Scaler dashboard before the deadline:

```
https://YOUR_HF_USERNAME-ml-pipeline-debugger.hf.space
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ImportError: attempted relative import` | Run from inside the `ml_pipeline_debugger/` directory |
| `openenv push` auth error | Run `huggingface-cli login` first |
| Space fails to build | Check Space logs — likely missing `torch` in requirements.txt |
| `/reset` times out | Normal on free HF tier under load — upgrade to CPU Upgrade |
| Grader returns 0.0 | Script does not print `loss:X.XXXXXX` format |

---

## Deadline

**8 April 2026, 11:59 PM IST**

Do not wait until the last day. Deploy by April 6 and use April 7-8 as buffer.
