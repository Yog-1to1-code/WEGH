# WEGH — OpenEnv Client
# Provides the client-side API for remote connections to WEGH.

from typing import Any

from openenv.core import EnvClient
from models import CPUAction, CPUObservation, CPUState


class WEGHEnv(EnvClient[CPUAction, CPUObservation, CPUState]):
    """Client for connecting to a running WEGH environment.
    
    Usage (async):
        async with WEGHEnv(base_url="https://your-space.hf.space") as client:
            result = await client.reset()
            result = await client.step(CPUAction(action_type="resize", ...))
    
    Usage (sync):
        with WEGHEnv(base_url="...").sync() as client:
            result = client.reset()
            result = client.step(CPUAction(action_type="resize", ...))
    """
    
    def _parse_state(self, message: dict) -> CPUObservation:
        if 'observation' in message:
            message = message['observation']
        valid_keys = CPUObservation.model_fields.keys()
        clean_msg = {k: v for k, v in message.items() if k in valid_keys}
        return CPUObservation(**clean_msg)
        
    def _parse_result(self, message: dict) -> CPUObservation:
        obs_dict = message.get('observation', message.copy())
        # OpenEnv puts reward and done in root, not inside 'observation' block
        if 'reward' in message:
            obs_dict['reward'] = message['reward']
        if 'done' in message:
            obs_dict['done'] = message['done']
            
        valid_keys = CPUObservation.model_fields.keys()
        clean_msg = {k: v for k, v in obs_dict.items() if k in valid_keys}
        return CPUObservation(**clean_msg)
        
    def _step_payload(self, action: CPUAction) -> dict:
        return action.model_dump()
