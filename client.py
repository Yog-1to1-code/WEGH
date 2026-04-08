# WEGH — OpenEnv Client
# Provides the client-side API for remote connections to WEGH.

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import CPUAction, CPUObservation, CPUState
except (ModuleNotFoundError, ImportError):
    from models import CPUAction, CPUObservation, CPUState


class WEGHEnv(EnvClient[CPUAction, CPUObservation, CPUState]):
    """Client for connecting to a running WEGH environment.

    Usage (sync):
        with WEGHEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset(task="iot_8bit")
            obs = result.observation
            result = env.step(CPUAction(action_type="resize", ...))
    """

    def _step_payload(self, action: CPUAction) -> Dict[str, Any]:
        """Convert CPUAction to JSON payload for step message."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CPUObservation]:
        """Parse server response into StepResult[CPUObservation]."""
        obs_data = payload.get("observation", {})

        # OpenEnv puts reward and done at the root level
        reward = payload.get("reward", obs_data.get("reward", 0.0))
        done = payload.get("done", obs_data.get("done", False))

        # Inject into obs_data for CPUObservation fields
        obs_data["reward"] = reward
        obs_data["done"] = done

        valid_keys = CPUObservation.model_fields.keys()
        clean_msg = {k: v for k, v in obs_data.items() if k in valid_keys}
        observation = CPUObservation(**clean_msg)

        return StepResult(
            observation=observation,
            reward=float(reward) if reward is not None else 0.0,
            done=bool(done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CPUState:
        """Parse server response into CPUState."""
        valid_keys = CPUState.model_fields.keys()
        clean_msg = {k: v for k, v in payload.items() if k in valid_keys}
        return CPUState(**clean_msg)
