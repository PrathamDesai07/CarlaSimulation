#!/bin/bash
set -e

echo "=== CARLA Longest6 Evaluation Client ==="
echo "CARLA_HOST: ${CARLA_HOST:-carla-server}"
echo "CARLA_PORT: ${CARLA_PORT:-2000}"
echo "MODEL_PATH: ${MODEL_PATH:-/app/models/transfuser_official/model.pth}"
echo "FRAME_PATH: ${FRAME_PATH:-/app/outputs/frames/longest6}"

# Step 1: Wait for CARLA server to be ready
echo ""
echo "--- Waiting for CARLA server ---"
bash /app/docker/client/scripts/wait_for_carla.sh \
    "${CARLA_HOST:-carla-server}" \
    "${CARLA_PORT:-2000}" \
    120

# Step 2: Run environment verification
echo ""
echo "--- Environment Verification ---"
python3 /app/docker/client/scripts/verify_environment.py || {
    echo "WARNING: Some environment checks failed. Proceeding anyway."
}

# Step 3: Run evaluation with FRAME_PATH for video capture
echo ""
echo "--- Starting Evaluation ---"
FRAME_PATH="${FRAME_PATH:-/app/outputs/frames/longest6}" \
python3 /app/scripts/run_longest6_eval.py \
    --carla-host "${CARLA_HOST:-carla-server}" \
    --carla-port "${CARLA_PORT:-2000}" \
    --model-path "${MODEL_PATH:-/app/models/transfuser_official/model.pth}" \
    --config /app/config/longest6_config.yaml \
    --frames-dir "${FRAME_PATH:-/app/outputs/frames/longest6}" \
    --video-output "${VIDEO_OUTPUT:-/app/outputs/videos/longest6/demo_$(date +%Y%m%d_%H%M%S).mp4}" \
    "$@"
