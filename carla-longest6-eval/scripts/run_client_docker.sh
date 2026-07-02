#!/bin/bash
# Run evaluation client Docker container
# Usage: ./run_client_docker.sh <model_path>

set -e

MODEL_PATH="${1:-/app/models/transfuser_official/model.pth}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_DIR="$(cd "$SCRIPT_DIR/../docker/client" && pwd)"

echo "Starting evaluation client..."
echo "  Model: $MODEL_PATH"

cd "$COMPOSE_DIR"
docker-compose run --rm \
    -e MODEL_PATH="$MODEL_PATH" \
    client

echo "Evaluation complete."
