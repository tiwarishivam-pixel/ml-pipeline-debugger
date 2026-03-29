# Module 3: Deploying the ML Pipeline Debugger

## What This Module Covers

Three deployment methods for the ML Pipeline Debugger, from fastest to
most production-ready: Uvicorn locally, Docker locally, and HuggingFace Spaces.

## Three Ways to Run

### 1. Uvicorn (fastest iteration)
```bash
cd ml_pipeline_debugger
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Docker (isolated, production-like)
```bash
cd ml_pipeline_debugger
docker build -t ml-pipeline-debugger:latest -f server/Dockerfile .
docker run -d -p 8000:8000 \
    -e WORKERS=4 \
    -e MAX_CONCURRENT_ENVS=50 \
    ml-pipeline-debugger:latest
```

### 3. HuggingFace Spaces (public, shareable)
```bash
cd ml_pipeline_debugger
openenv push --repo-id YOUR_USERNAME/ml-pipeline-debugger
```

After deployment:
- **API:** `https://YOUR_USERNAME-ml-pipeline-debugger.hf.space`
- **Web UI:** `https://YOUR_USERNAME-ml-pipeline-debugger.hf.space/web`
- **Docs:** `https://YOUR_USERNAME-ml-pipeline-debugger.hf.space/docs`
- **Health:** `https://YOUR_USERNAME-ml-pipeline-debugger.hf.space/health`

## Verify Deployment

```bash
curl https://YOUR_USERNAME-ml-pipeline-debugger.hf.space/health
# {"status": "healthy"}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKERS` | 4 | Uvicorn worker processes |
| `PORT` | 8000 | Server port |
| `MAX_CONCURRENT_ENVS` | 50 | Max WebSocket sessions |

## Connecting After Deployment

```python
from ml_pipeline_debugger.client import MLPipelineDebuggerEnv
from ml_pipeline_debugger.models import MLDebugAction

url = "https://YOUR_USERNAME-ml-pipeline-debugger.hf.space"

with MLPipelineDebuggerEnv(base_url=url).sync() as env:
    result = env.reset()
    print(result.observation.task_id)
```

## Hardware Recommendations

| Tier | Use case | Max concurrent sessions |
|------|----------|------------------------|
| CPU Basic (Free) | Development, demos | ~128 |
| CPU Upgrade | Production training | ~512 |
