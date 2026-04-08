#!/bin/bash
set -e

echo "=== WEGH Engine Starting ==="
echo "Starting Python/OpenEnv server on port ${PORT:-8000}..."
echo "(The Go simulator daemon will be spawned automatically internally)"

# Start Python Uvicorn server (foreground)
# OpenEnv framework expects container-internal port 8000
exec uvicorn server.app:app --host 0.0.0.0 --port "${PORT:-8000}"
