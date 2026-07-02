#!/bin/bash
# Quickstart: Full Longest6 evaluation pipeline
# Usage: ./quickstart_longest6.sh [model_path]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="${1:-/app/models/transfuser_official/model.pth}"

echo "=== Longest6 Evaluation Quickstart ==="
echo "Model: $MODEL_PATH"

echo "[1/4] Starting CARLA server..."
bash "$SCRIPT_DIR/run_carla_docker.sh" up

echo "[2/4] Waiting for CARLA..."
sleep 10

echo "[3/4] Running evaluation..."
bash "$SCRIPT_DIR/run_client_docker.sh" "$MODEL_PATH"

echo "[4/4] Generating demo video..."
bash "$SCRIPT_DIR/generate_video_from_frames.sh" \
    "/app/outputs/frames/longest6" \
    "/app/outputs/videos/longest6/demo_$(date +%Y%m%d_%H%M%S).mp4" \
    20 || echo "Note: No frames found, skipping video."

echo "=== Done! ==="
echo "Check outputs/metrics/longest6/ for results."
echo "Check outputs/videos/longest6/ for demo video."
