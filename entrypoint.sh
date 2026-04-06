#!/bin/bash
set -e

echo "=== WEGH Engine Starting ==="
echo "Starting Python/OpenEnv server on port 7860..."
echo "(The Go simulator daemon will be spawned automatically internally)"

# Start Python Uvicorn server (foreground)
exec uvicorn server.app:app --host 0.0.0.0 --port 7860
