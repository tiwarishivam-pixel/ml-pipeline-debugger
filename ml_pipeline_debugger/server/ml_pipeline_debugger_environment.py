"""
ML Pipeline Debugger Environment.

The agent receives a broken PyTorch training script and must fix it.
Three tasks of increasing difficulty are randomly assigned each episode.

Episode flow:
  reset() → agent sees broken script + loss curve + symptom description
  step()  → agent submits fixed_code → grader runs it → reward returned
  done=True after the first step() (one fix attempt per episode)
"""

import math
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MLDebugAction, MLDebugObservation
    from .tasks.task1_hyperparams import grade_task1, sample_task1
    from .tasks.task2_nan_pipeline import grade_task2, sample_task2
    from .tasks.task3_silent_underfit import grade_task3, sample_task3
except ImportError:
    from models import MLDebugAction, MLDebugObservation
    from server.tasks.task1_hyperparams import grade_task1, sample_task1
    from server.tasks.task2_nan_pipeline import grade_task2, sample_task2
    from server.tasks.task3_silent_underfit import grade_task3, sample_task3

import random

_TASKS = {
    "task1": {"difficulty": "easy",   "sample": sample_task1, "grade": grade_task1},
    "task2": {"difficulty": "medium", "sample": sample_task2, "grade": grade_task2},
    "task3": {"difficulty": "hard",   "sample": sample_task3, "grade": grade_task3},
}


class MlPipelineDebuggerEnvironment(Environment):
    """
    RL environment for ML pipeline debugging.

    Each episode:
      1. A task is randomly selected (task1 / task2 / task3)
      2. A bug variant within that task is randomly selected
      3. The broken script + symptom are shown to the agent
      4. Agent submits a fixed script
      5. The grader runs the fix and returns a score 0.0–1.0
      6. Episode ends (done=True)

    Agents improve by learning to diagnose and fix progressively
    harder categories of training failures.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task_id: str = "task1"
        self._current_bug_type: str = ""
        self._current_difficulty: str = "easy"
        self._episode_done: bool = False

    # ── reset ──────────────────────────────────────────────────────────────────

    def reset(self) -> MLDebugObservation:
        """
        Start a new episode by sampling a random task and bug variant.

        Returns the broken script, observed loss curve, and symptom description.
        The agent must call step() with its fixed code to receive a reward.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode_done = False

        # Randomly pick one of the three tasks
        task_id = random.choice(list(_TASKS.keys()))
        task = _TASKS[task_id]
        self._current_task_id = task_id
        self._current_difficulty = task["difficulty"]

        bug_key, broken_script, loss_curve, description = task["sample"]()
        self._current_bug_type = bug_key

        # Sanitise loss curve — replace NaN/Inf with sentinel for JSON serialisation
        safe_curve = [
            l if (not math.isnan(l) and not math.isinf(l)) else -1.0
            for l in loss_curve
        ]

        return MLDebugObservation(
            broken_script=broken_script,
            error_log=None,
            loss_curve=safe_curve,
            task_description=description,
            task_id=task_id,
            difficulty=self._current_difficulty,
            bug_type=self._current_bug_type,
            done=False,
            reward=0.0,
        )

    # ── step ───────────────────────────────────────────────────────────────────

    def step(self, action: MLDebugAction) -> MLDebugObservation:
        """
        Grade the agent's fix and return a reward.

        The agent submits fixed_code. The grader runs it in a subprocess,
        parses the loss curve, and returns a score 0.0–1.0.

        Episode ends immediately (done=True) — one attempt per episode.
        """
        self._state.step_count += 1

        if self._episode_done:
            # Agent called step() after episode already ended
            return MLDebugObservation(
                broken_script="Episode already finished. Call reset() to start a new one.",
                error_log="Episode done — call reset()",
                loss_curve=[],
                task_description="Episode finished.",
                task_id=self._current_task_id,
                difficulty=self._current_difficulty,
                bug_type=self._current_bug_type,
                done=True,
                reward=0.0,
            )

        # Run the grader for the current task
        grade_fn = _TASKS[self._current_task_id]["grade"]
        score, feedback = grade_fn(action.fixed_code)

        self._episode_done = True

        return MLDebugObservation(
            broken_script=feedback,   # feedback doubles as the post-episode observation
            error_log=None,
            loss_curve=[],
            task_description=(
                f"Episode complete.\n"
                f"Task: {self._current_task_id} ({self._current_difficulty})\n"
                f"Bug: {self._current_bug_type}\n"
                f"Score: {score}/1.0\n\n"
                f"{feedback}"
            ),
            task_id=self._current_task_id,
            difficulty=self._current_difficulty,
            bug_type=self._current_bug_type,
            done=True,
            reward=score,
            metadata={
                "task_id": self._current_task_id,
                "difficulty": self._current_difficulty,
                "bug_type": self._current_bug_type,
                "score": score,
                "feedback": feedback,
            },
        )

    # ── state ──────────────────────────────────────────────────────────────────

    @property
    def state(self) -> State:
        return self._state
