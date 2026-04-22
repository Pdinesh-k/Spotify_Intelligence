#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate
pkill -f "uvicorn main:app" 2>/dev/null || true
sleep 1
echo "Starting Spotify Intelligence Agent → http://localhost:8000"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
