"""
WEGH OpenEnv Grader Hooks.
Evaluates model actions by handling both Pydantic objects and serialized JSON dicts.
Phase 2 receives observations over HTTP as plain dictionaries, not instantiated models.
"""

from typing import Any
from models import CPUObservation
from server.wegh_env import sanitize_grade


def extract_score(*args: Any, **kwargs: Any) -> float:
    obs = (
        kwargs.get("observation")
        or kwargs.get("obs")
        or kwargs.get("final_observation")
        or kwargs.get("last_obs")
        or kwargs.get("result")
    )
    if not obs and args:
        obs = args[-1]

    if isinstance(obs, CPUObservation):
        return sanitize_grade(obs.final_score)

    if isinstance(obs, dict) and "final_score" in obs:
        return sanitize_grade(obs["final_score"])

    if isinstance(obs, (int, float)):
        return sanitize_grade(obs)

    return 0.001


def grade_iot_8bit(*args: Any, **kwargs: Any) -> float:
    return extract_score(*args, **kwargs)


def grade_rv32im(*args: Any, **kwargs: Any) -> float:
    return extract_score(*args, **kwargs)


def grade_mseries_superscalar(*args: Any, **kwargs: Any) -> float:
    return extract_score(*args, **kwargs)
