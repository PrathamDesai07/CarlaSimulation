#!/bin/bash
set -e
export PYTHONUNBUFFERED=1

echo "=== CARLA Longest6 Evaluation Client ==="
echo "CARLA_HOST: ${CARLA_HOST:-carla-server}"
echo "CARLA_PORT: ${CARLA_PORT:-2000}"
echo "MODEL_PATH: ${MODEL_PATH:-/app/models/transfuser_official/model.pth}"
echo "FRAME_PATH: ${FRAME_PATH:-/app/outputs/frames/longest6}"
echo "ROUTE_IDS: ${ROUTE_IDS:-}"
echo "OUTPUT_DIR: ${OUTPUT_DIR:-/app/outputs/metrics/longest6}"
echo "TRAFFIC_MANAGER_PORT: ${TRAFFIC_MANAGER_PORT:-8000}"

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

# Build optional args from env vars
ROUTE_IDS_ARG=""
if [ -n "$ROUTE_IDS" ]; then
    ROUTE_IDS_ARG="--route-ids $ROUTE_IDS"
fi

OUTPUT_DIR_ARG=""
if [ -n "$OUTPUT_DIR" ]; then
    OUTPUT_DIR_ARG="--output-dir $OUTPUT_DIR"
fi

TM_PORT_ARG=""
if [ -n "$TRAFFIC_MANAGER_PORT" ]; then
    TM_PORT_ARG="--traffic-manager-port $TRAFFIC_MANAGER_PORT"
fi

FRAME_PATH="${FRAME_PATH:-/app/outputs/frames/longest6}" \
python3 /app/scripts/run_longest6_eval.py \
    --carla-host "${CARLA_HOST:-carla-server}" \
    --carla-port "${CARLA_PORT:-2000}" \
    --model-path "${MODEL_PATH:-/app/models/transfuser_official/model.pth}" \
    --config /app/config/longest6_config.yaml \
    --frames-dir "${FRAME_PATH:-/app/outputs/frames/longest6}" \
    --video-output "${VIDEO_OUTPUT:-/app/outputs/videos/longest6/demo_$(date +%Y%m%d_%H%M%S).mp4}" \
    $ROUTE_IDS_ARG \
    $OUTPUT_DIR_ARG \
    $TM_PORT_ARG \
    "$@"
