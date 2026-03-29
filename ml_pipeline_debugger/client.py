"""ML Pipeline Debugger Environment Client."""

from typing import Dict
import sys, os

try:
    from openenv.core import EnvClient
    from openenv.core.client_types import StepResult
    from openenv.core.env_server.types import State
except ImportError as e:
    raise ImportError("openenv-core is required: pip install openenv-core") from e

# Support both installed-package and local-path imports
try:
    from .models import MLDebugAction, MLDebugObservation
except ImportError:
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from models import MLDebugAction, MLDebugObservation


class MLPipelineDebuggerEnv(
    EnvClient[MLDebugAction, MLDebugObservation, State]
):
    """
    Client for the ML Pipeline Debugger environment.

    Connects via WebSocket to a running server instance.
    Each client gets its own isolated environment session.

    Example (async):
        async with MLPipelineDebuggerEnv(base_url="http://localhost:8000") as env:
            result = await env.reset()
            obs = result.observation
            print(obs.task_id, obs.difficulty, obs.bug_type)
            result = await env.step(MLDebugAction(
                fixed_code="...your fixed script...",
                explanation="Fixed the learning rate",
                task_id=obs.task_id,
            ))
            print(result.reward)

    Example (sync):
        with MLPipelineDebuggerEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset()
            result = env.step(MLDebugAction(fixed_code="..."))
            print(result.reward)
    """

    def _step_payload(self, action: MLDebugAction) -> Dict:
        return {
            "fixed_code":  action.fixed_code,
            "explanation": action.explanation,
            "task_id":     action.task_id,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MLDebugObservation]:
        obs_data = payload.get("observation", {})
        observation = MLDebugObservation(
            broken_script=obs_data.get("broken_script", ""),
            error_log=obs_data.get("error_log"),
            loss_curve=obs_data.get("loss_curve", []),
            task_description=obs_data.get("task_description", ""),
            task_id=obs_data.get("task_id", ""),
            difficulty=obs_data.get("difficulty", ""),
            bug_type=obs_data.get("bug_type", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
