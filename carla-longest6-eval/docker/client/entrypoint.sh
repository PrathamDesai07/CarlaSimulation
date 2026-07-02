#!/bin/bash
set -e

echo "=== CARLA Longest6 Evaluation Client ==="
echo "CARLA_HOST: ${CARLA_HOST:-carla-server}"
echo "CARLA_PORT: ${CARLA_PORT:-2000}"
echo "MODEL_PATH: ${MODEL_PATH:-/app/models/transfuser_official/model.pth}"

/opt/conda/bin/python3 /app/scripts/run_longest6_eval.py \
    --carla-host "${CARLA_HOST:-carla-server}" \
    --carla-port "${CARLA_PORT:-2000}" \
    --model-path "${MODEL_PATH:-/app/models/transfuser_official/model.pth}" \
    --config /app/config/longest6_config.yaml \
    "$@"
