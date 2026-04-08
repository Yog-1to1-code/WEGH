# WEGH — OpenEnv Client
# Provides the client-side API for remote connections to WEGH.

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.env_client import StepResult
from models import CPUAction, CPUObservation, CPUState


class WEGHEnv(EnvClient[CPUAction, CPUObservation, CPUState]):
    """Client for connecting to a running WEGH environment.
    
    Usage (async):
        async with WEGHEnv(base_url="https://your-space.hf.space") as client:
            result = await client.reset()
            obs = result.observation
            result = await client.step(CPUAction(action_type="resize", ...))
    
    Usage (sync):
        with WEGHEnv(base_url="...").sync() as client:
            result = client.reset()
            obs = result.observation
            result = client.step(CPUAction(action_type="resize", ...))
    """
    
    def _parse_result(self, message: Dict[str, Any]) -> StepResult[CPUObservation]:
        """Convert JSON response from env server to StepResult[CPUObservation]."""
        obs_dict = message.get('observation', message.copy())
        
        # OpenEnv puts reward and done in root, not inside 'observation' block
        reward = message.get('reward', obs_dict.get('reward', 0.0))
        done = message.get('done', obs_dict.get('done', False))
        
        # Also inject reward/done into observation for backward compat
        obs_dict['reward'] = reward
        obs_dict['done'] = done
            
        valid_keys = CPUObservation.model_fields.keys()
        clean_msg = {k: v for k, v in obs_dict.items() if k in valid_keys}
        observation = CPUObservation(**clean_msg)
        
        return StepResult(
            observation=observation,
            reward=float(reward) if reward is not None else 0.0,
            done=bool(done),
        )
        
    def _step_payload(self, action: CPUAction) -> dict:
        return action.model_dump()
