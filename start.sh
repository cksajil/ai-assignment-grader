#!/bin/bash
set -e

# MODEL="${MODEL_NAME:-qwen2.5-coder:7b}"
MODEL="${MODEL_NAME:-qwen2.5-coder:3b}

echo "================================================"
echo "  AI Assignment Grader — Starting up"
echo "================================================"

export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_ORIGINS=*
export OLLAMA_KEEP_ALIVE=-1

export OLLAMA_NUM_THREADS=2          # match free tier vCPU count exactly
export OLLAMA_MAX_LOADED_MODELS=1    # don't waste RAM on model slots
export OLLAMA_FLASH_ATTENTION=1      # reduces memory usage per token

echo "[1/3] Starting Ollama server..."
ollama serve &

echo "[2/3] Waiting for Ollama..."
MAX_WAIT=60; COUNT=0
until curl -s http://localhost:11434 > /dev/null 2>&1; do
    [ $COUNT -ge $MAX_WAIT ] && echo "ERROR: Ollama did not start" && exit 1
    sleep 1; COUNT=$((COUNT+1))
    echo "  ...waiting ($COUNT/${MAX_WAIT}s)"
done
echo "  Ollama ready!"

echo "[3/3] Pulling model: $MODEL"
ollama pull "$MODEL"

echo "Warming up model..."
curl -s http://localhost:11434/api/generate \
  -d "{\"model\": \"$MODEL\", \"prompt\": \"\", \"keep_alive\": -1}" > /dev/null
echo "Model warm."

echo ""
echo "================================================"
echo "  Launching Gradio on port 7860"
echo "================================================"

exec python app.py
