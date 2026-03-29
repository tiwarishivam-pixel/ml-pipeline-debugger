"""FastAPI application for the ML Pipeline Debugger Environment."""

import sys
import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Run: pip install openenv-core") from e

# Support three import modes:
# 1. Installed as package  →  from ml_pipeline_debugger.models import ...
# 2. Run from inside dir   →  from models import ...
# 3. Relative (in-package) →  from ..models import ...
try:
    from ..models import MLDebugAction, MLDebugObservation
    from .ml_pipeline_debugger_environment import MlPipelineDebuggerEnvironment
except ImportError:
    try:
        from models import MLDebugAction, MLDebugObservation
        from server.ml_pipeline_debugger_environment import MlPipelineDebuggerEnvironment
    except ImportError:
        # Add parent dir to path so bare `import models` works
        _here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        from models import MLDebugAction, MLDebugObservation
        from server.ml_pipeline_debugger_environment import MlPipelineDebuggerEnvironment

app = create_app(
    MlPipelineDebuggerEnvironment,
    MLDebugAction,
    MLDebugObservation,
    env_name="ml_pipeline_debugger",
    max_concurrent_envs=50,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
