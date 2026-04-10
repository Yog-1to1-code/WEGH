"""
WEGH OpenEnv Grader Hooks.
Evaluates model actions by querying the Go engine's /grade endpoint.
Phase 2 receives observations over HTTP as plain dictionaries.
Grading is handled server-side in Go for deterministic, verifiable scoring.
"""

import os
import logging
from typing import Any

import httpx

logger = logging.getLogger("wegh.graders")

# Go engine URL — matches the self-contained server
GO_ENGINE_URL = os.getenv("WEGH_ENGINE_URL", "http://127.0.0.1:7860")


def _query_grade() -> float:
    """Query the Go engine's /grade endpoint for the current episode score."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{GO_ENGINE_URL}/grade")
            if resp.status_code == 200:
                data = resp.json()
                score = data.get("score", 0.001)
                return _sanitize(score)
    except Exception as e:
        logger.warning("Failed to query /grade: %s", e)
    return 0.001


def _sanitize(raw_score: Any) -> float:
    """Sanitize to (0, 1) exclusive — never return exactly 0 or 1."""
    try:
        if raw_score is None:
            return 0.001
        score = float(raw_score)
    except (ValueError, TypeError):
        return 0.001

    if score >= 1.0:
        return 0.999
    elif score <= 0.0:
        return 0.001
    return score


def extract_score(*args: Any, **kwargs: Any) -> float:
    """Extract score from observation dict or query /grade endpoint.
    
    Priority: /grade endpoint → observation final_score → default.
    """
    # First try the Go engine's deterministic grading
    grade_score = _query_grade()
    if grade_score > 0.01:
        return grade_score

    # Fallback: extract from observation
    obs = (
        kwargs.get("observation")
        or kwargs.get("obs")
        or kwargs.get("final_observation")
        or kwargs.get("last_obs")
        or kwargs.get("result")
    )
    if not obs and args:
        obs = args[-1]

    if isinstance(obs, dict) and "final_score" in obs:
        return _sanitize(obs["final_score"])
    elif isinstance(obs, (int, float)):
        return _sanitize(obs)

    return 0.001


def grade_iot_8bit(*args: Any, **kwargs: Any) -> float:
    """Grade Task 0: 8-Bit IoT Microcontroller."""
    return extract_score(*args, **kwargs)


def grade_rv32im(*args: Any, **kwargs: Any) -> float:
    """Grade Task 1: RV32IM 5-Stage Pipelined Core."""
    return extract_score(*args, **kwargs)


def grade_mseries_superscalar(*args: Any, **kwargs: Any) -> float:
    """Grade Task 2: M-Series Heterogeneous Superscalar."""
    return extract_score(*args, **kwargs)
