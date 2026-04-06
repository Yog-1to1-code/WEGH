# WEGH — Go Engine Client
# Handles all HTTP communication between Python and the Go simulation daemon.

import json
import logging
import time
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("wegh.go_client")


class GoEngineClient:
    """HTTP client for the Go simulation engine daemon."""
    
    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url
        self.client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            # Keep-alive connection pooling for fast localhost calls
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )
        self._healthy = False
    
    def wait_for_ready(self, max_wait: float = 15.0) -> bool:
        """Wait for Go engine to be ready. Returns True if ready."""
        start = time.time()
        while time.time() - start < max_wait:
            try:
                resp = self.client.get("/health")
                if resp.status_code == 200:
                    self._healthy = True
                    logger.info("Go engine ready at %s", self.base_url)
                    return True
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass
            time.sleep(0.5)
        logger.error("Go engine not ready after %.1fs", max_wait)
        return False
    
    def reset(self, episode_id: str, task_id: int, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize a new episode graph in the Go engine."""
        payload = {
            "episode_id": episode_id,
            "task_id": task_id,
            "task_config": task_config,
        }
        try:
            resp = self.client.post("/reset", json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("Go engine /reset failed: %s", e)
            return {"status": "error", "error": str(e)}
    
    def step(self, episode_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply an action to the episode graph and get updated metrics."""
        payload = {
            "episode_id": episode_id,
            "action": action,
        }
        try:
            resp = self.client.post("/step", json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("Go engine /step failed: %s", e)
            return {
                "status": "error",
                "error": str(e),
                "valid": False,
                "metrics": {
                    "ipc": 0, "throughput_gips": 0, "total_power_mw": 0,
                    "total_area_mm2": 0, "max_power_density": 0,
                    "thermal_celsius": 0, "hotspot_count": 0,
                    "throttled_factor": 1.0, "perf_per_watt": 0,
                    "effective_clock_ghz": 0,
                },
                "components": [],
                "connections": [],
                "validation_errors": [str(e)],
                "engineering_notes": f"ENGINE ERROR: {e}",
            }
    
    def health(self) -> Dict[str, Any]:
        """Check engine health."""
        try:
            resp = self.client.get("/health")
            return resp.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()
