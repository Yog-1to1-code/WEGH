# WEGH — Environment Client
# Standalone HTTP client — NO openenv-core dependency.
# Communicates directly with the Go engine via REST.

from __future__ import annotations
from typing import Any, Dict, Optional
import requests

try:
    from .models import CPUAction, CPUObservation, CPUState
except (ModuleNotFoundError, ImportError):
    from models import CPUAction, CPUObservation, CPUState


class WEGHEnv:
    """Client for connecting to a running WEGH environment.

    Usage:
        env = WEGHEnv(base_url="http://localhost:7860")
        reset_data = env.reset(task="rv32im", seed=42)
        step_data = env.step({"action": {"type": "resize", ...}})
        grade_data = env.grade()
        env.close()
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def health(self) -> bool:
        """Check if the environment server is healthy."""
        try:
            r = self._session.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def reset(self, task: str = "rv32im", seed: int = 42) -> Optional[Dict[str, Any]]:
        """Start a new episode with the given task and seed."""
        try:
            payload = {"task": task, "seed": seed}
            r = self._session.post(
                f"{self.base_url}/reset", json=payload, timeout=self.timeout
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[ERROR] Failed to reset environment: {e}", flush=True)
            return None

    def step(self, action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Take an action and receive the next observation and reward."""
        try:
            r = self._session.post(
                f"{self.base_url}/step", json=action, timeout=self.timeout
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[ERROR] Failed to step environment: {e}", flush=True)
            return None

    def grade(self) -> Dict[str, Any]:
        """Get the episode grade/score after completion."""
        try:
            r = self._session.get(
                f"{self.base_url}/grade", timeout=self.timeout
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[ERROR] Failed to grade: {e}", flush=True)
            return {"score": 0.01, "sub_scores": {}, "exploit_detected": False}

    def state(self) -> Optional[Dict[str, Any]]:
        """Get the current environment state."""
        try:
            r = self._session.get(
                f"{self.base_url}/state", timeout=self.timeout
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[ERROR] Failed to get state: {e}", flush=True)
            return None

    def tasks(self) -> list:
        """Get available tasks."""
        try:
            r = self._session.get(
                f"{self.base_url}/tasks", timeout=self.timeout
            )
            r.raise_for_status()
            return r.json()
        except Exception:
            return []

    def close(self) -> None:
        """Close the client session."""
        try:
            self._session.close()
        except Exception:
            pass

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # Docker image support (compatibility with old interface)
    @classmethod
    def from_docker_image(cls, image_name: str, task: str = "rv32im", **kwargs):
        """Create client that connects to a Docker container."""
        return cls(base_url="http://localhost:7860", **kwargs)

    # .sync() compatibility shim
    def sync(self):
        """Return self — already synchronous."""
        return self
