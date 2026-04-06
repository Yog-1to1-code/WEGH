# WEGH — FastAPI Application
# Creates the OpenEnv-compliant FastAPI app using create_fastapi_app helper.

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_fastapi_app
from models import CPUAction, CPUObservation
from server.wegh_env import WEGHEnvironment


def create_env():
    """Factory function for creating the environment instance."""
    return WEGHEnvironment()


# Create the FastAPI app using OpenEnv's helper
# This sets up WebSocket endpoints, health checks, and proper routing
app = create_fastapi_app(create_env, CPUAction, CPUObservation)

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)

if __name__ == "__main__":
    main()

