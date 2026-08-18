#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Starting Tool Calling from Scratch..."

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Error: Ollama is not running. Start it with: ollama serve"
    exit 1
fi

echo "Ollama is running."

# Kill any leftover process on port 8000
if lsof -ti:8000 > /dev/null 2>&1; then
    echo "Port 8000 in use — killing old process..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    sleep 1
fi

echo "Open http://localhost:8000 in your browser"
echo "Press Ctrl+C to stop."
echo ""

uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
