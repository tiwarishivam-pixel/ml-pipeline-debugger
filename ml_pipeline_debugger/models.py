"""
Data models for the ML Pipeline Debugger environment.

The agent receives a broken training script and must return a fixed version.
Three task difficulties: hyperparameter bug, NaN data pipeline, silent underfitting.
"""

from typing import List, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class MLDebugAction(Action):
    """
    What the agent submits after analysing the broken script.

    The agent must return the complete fixed Python script.
    Partial fixes are accepted — the grader scores partial credit.
    """

    fixed_code: str = Field(
        ...,
        description=(
            "The complete corrected Python training script. "
            "Must be runnable as-is via subprocess. "
            "Print loss as 'loss:X.XXXXXX' every epoch for the grader to parse."
        ),
    )
    explanation: str = Field(
        default="",
        description=(
            "Brief explanation of what was wrong and what was fixed. "
            "Not scored directly but useful for debugging and human review."
        ),
    )
    task_id: str = Field(
        default="",
        description="Which task this submission is for: task1 | task2 | task3",
    )


class MLDebugObservation(Observation):
    """
    What the agent sees at the start of each episode.

    Contains the broken script, any error output, the observed loss curve
    (first N steps of running the broken script), and a plain-English description
    of the symptom.
    """

    broken_script: str = Field(
        ...,
        description="The complete broken Python training script the agent must fix.",
    )
    error_log: Optional[str] = Field(
        default=None,
        description=(
            "Stderr output from running the broken script, if it crashed. "
            "None if the script ran but produced bad training behaviour."
        ),
    )
    loss_curve: List[float] = Field(
        default_factory=list,
        description=(
            "Loss values from the first N training steps of the broken script. "
            "May contain NaN or Inf. Exploding, flat, or NaN curves "
            "are all signals the agent should use to diagnose the bug."
        ),
    )
    task_description: str = Field(
        ...,
        description="Plain-English description of the observed symptom.",
    )
    task_id: str = Field(
        ...,
        description="Unique identifier: task1 | task2 | task3",
    )
    difficulty: str = Field(
        ...,
        description="Difficulty level: easy | medium | hard",
    )
    bug_type: str = Field(
        ...,
        description=(
            "Category of the injected bug. "
            "E.g. 'learning_rate_too_high' | 'division_by_zero_normalisation' | 'zero_weight_init'"
        ),
    )
