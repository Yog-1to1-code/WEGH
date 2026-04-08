# WEGH — FastAPI Application
# Creates the OpenEnv-compliant FastAPI app using create_app helper.

import argparse

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    from openenv.core.env_server import create_fastapi_app as create_app

try:
    from ..models import CPUAction, CPUObservation
    from .wegh_env import WEGHEnvironment
except (ModuleNotFoundError, ImportError):
    from models import CPUAction, CPUObservation
    from server.wegh_env import WEGHEnvironment


app = create_app(
    WEGHEnvironment,
    CPUAction,
    CPUObservation,
    env_name="wegh",
    max_concurrent_envs=1,
)


def main():
    """Entry point for: uv run --project . server"""
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
