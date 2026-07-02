#!/bin/bash
# Generate video from captured frames using ffmpeg
# Usage: ./generate_video_from_frames.sh <frames_dir> <output_path> [fps]

set -e

FRAMES_DIR="${1:-/app/outputs/frames/longest6}"
OUTPUT_PATH="${2:-/app/outputs/videos/longest6/demo.mp4}"
FPS="${3:-20}"

if [ ! -d "$FRAMES_DIR" ]; then
    echo "ERROR: Frames directory not found: $FRAMES_DIR"
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "Generating video from frames..."
echo "  Frames: $FRAMES_DIR"
echo "  Output: $OUTPUT_PATH"
echo "  FPS: $FPS"

ffmpeg -y \
    -framerate "$FPS" \
    -pattern_type glob \
    -i "$FRAMES_DIR/*.png" \
    -c:v libx264 \
    -preset medium \
    -crf 23 \
    -pix_fmt yuv420p \
    "$OUTPUT_PATH"

echo "Video saved to: $OUTPUT_PATH"
