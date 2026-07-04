#!/bin/bash
# Run evaluation client Docker container
# Usage: ./run_client_docker.sh <model_path>
#   model_path: path to .pth file (host path or container path)
#               If a host path under the project root is given, it's
#               converted to the in-container /app/... path automatically.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_DIR="$(cd "$SCRIPT_DIR/../docker/client" && pwd)"
NETWORK_NAME="carla-longest6-net"

MODEL_INPUT="${1:-${PROJECT_DIR}/models/transfuser_official/model.pth}"

# Convert host path to container path if it starts with PROJECT_DIR
if [[ "$MODEL_INPUT" == "$PROJECT_DIR"* ]]; then
    CONTAINER_MODEL_PATH="/app${MODEL_INPUT#$PROJECT_DIR}"
else
    CONTAINER_MODEL_PATH="$MODEL_INPUT"
fi

# Ensure shared network exists
docker network inspect "$NETWORK_NAME" &>/dev/null || \
    docker network create "$NETWORK_NAME" --driver bridge

echo "Starting evaluation client..."
echo "  Host model path:    $MODEL_INPUT"
echo "  Container model:    $CONTAINER_MODEL_PATH"
echo "  Network:            $NETWORK_NAME"

cd "$COMPOSE_DIR"
docker compose run --rm \
    -e MODEL_PATH="$CONTAINER_MODEL_PATH" \
    -e FRAME_PATH="${FRAME_PATH:-/app/outputs/frames/longest6}" \
    -e VIDEO_OUTPUT="${VIDEO_OUTPUT:-/app/outputs/videos/longest6/demo.mp4}" \
    client

echo "Evaluation complete."
