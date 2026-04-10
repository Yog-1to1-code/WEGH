# WEGH — Python Server Entry Point
# The Go engine serves all OpenEnv endpoints directly.
# This module provides the Python entry point required by pyproject.toml [project.scripts]
# and handles binary discovery + fallback.

import os
import subprocess
import sys


def main():
    """Start the WEGH Go environment server."""
    port = os.getenv("PORT", "7860")

    # Binary search paths (Docker → local)
    binary_paths = [
        "/app/go-engine",              # Docker path
        os.path.join(os.getcwd(), "go-engine"),   # Local
        os.path.join(os.getcwd(), "engine", "go-engine"),  # Dev layout
        "./go-engine",
    ]

    binary = None
    for path in binary_paths:
        if os.path.exists(path):
            binary = path
            break

    if binary:
        print(f"[WEGH] Starting Go server on port {port}", flush=True)
        os.execv(binary, [binary, f"--bind=0.0.0.0:{port}"])
    else:
        # Fallback: try go run for development
        print(f"[WEGH] Binary not found, trying go run ...", flush=True)
        go_src = os.path.join(os.getcwd(), "engine", "cmd", "server")
        if not os.path.exists(go_src):
            go_src = os.path.join(os.getcwd(), "cmd", "server")

        try:
            proc = subprocess.run(
                ["go", "run", "."],
                cwd=go_src,
                env={**os.environ, "PORT": port}
            )
            sys.exit(proc.returncode)
        except FileNotFoundError:
            print(
                "[WEGH] ERROR: Neither compiled binary nor 'go' found.\n"
                "Build with: cd engine && go build -o go-engine ./cmd/server\n"
                "Or run in Docker: docker build -t wegh . && docker run -p 7860:7860 wegh",
                file=sys.stderr
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
