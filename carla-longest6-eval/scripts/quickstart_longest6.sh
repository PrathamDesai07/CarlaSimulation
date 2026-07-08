#!/bin/bash
# Quickstart: Full Longest6 evaluation pipeline
# Usage: ./quickstart_longest6.sh [model_path]
#   model_path: path to model file (default: models/transfuser_official/model.pth)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_PATH="${1:-${PROJECT_DIR}/models/transfuser_official/model.pth}"
NETWORK_NAME="carla-longest6-net"

wait_for_carla() {
    local host="$1" port="$2" timeout="$3"
    echo "Waiting for CARLA at $host:$port..."
    for i in $(seq 1 "$timeout"); do
        if timeout 2 bash -c "echo > /dev/tcp/$host/$port" 2>/dev/null; then
            echo "CARLA is ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: CARLA did not become ready after ${timeout}s"
    return 1
}

echo "=== Longest6 Evaluation Quickstart ==="
echo "Model: $MODEL_PATH"
echo ""

# Step 1: Start CARLA server
echo "[1/4] Starting CARLA server..."
bash "$SCRIPT_DIR/run_carla_docker.sh" up

# Step 2: Wait for CARLA
echo "[2/4] Waiting for CARLA..."
wait_for_carla "localhost" 2000 120

# Step 3: Run evaluation
echo "[3/4] Running evaluation..."

# Set frame/video output paths (container paths — outputs/ is mounted)
export FRAME_PATH="${PROJECT_DIR}/outputs/frames/longest6"
export VIDEO_OUTPUT="${PROJECT_DIR}/outputs/videos/longest6/demo_$(date +%Y%m%d_%H%M%S).mp4"

bash "$SCRIPT_DIR/run_client_docker.sh" "$MODEL_PATH"

# Step 4: Generate demo video (if frames exist)
echo "[4/4] Generating demo video..."
FRAMES_DIR="${PROJECT_DIR}/outputs/frames/longest6"
VIDEOS_DIR="${PROJECT_DIR}/outputs/videos/longest6"
mkdir -p "$VIDEOS_DIR"

if [ -d "$FRAMES_DIR" ] && [ "$(ls -A "$FRAMES_DIR" 2>/dev/null)" ]; then
    bash "$SCRIPT_DIR/generate_video_from_frames.sh" \
        "$FRAMES_DIR" \
        "$VIDEOS_DIR/demo_$(date +%Y%m%d_%H%M%S).mp4" \
        20
else
    echo "  No frames found at $FRAMES_DIR — skipping video generation."
fi

echo ""
echo "=== Done! ==="
echo "  Metrics:  ${PROJECT_DIR}/outputs/metrics/longest6/"
echo "  Videos:   ${PROJECT_DIR}/outputs/videos/longest6/"
