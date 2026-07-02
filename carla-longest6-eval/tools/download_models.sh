#!/bin/bash
# Download model files from Google Drive
# Usage: bash tools/download_models.sh

set -e

MODEL_DIR="$(cd "$(dirname "$0")/../models" && pwd)"
TRANSFUSER_DIR="$(cd "$(dirname "$0")/../transfuser" && pwd)"
DRIVE_FOLDER_ID="1yG9LbVtSLaneHKlB5GL5Vhzz5Miue5Me"

echo "=== Model Download Script ==="
echo ""
echo "This script downloads model files from the Google Drive folder:"
echo "  https://drive.google.com/drive/folders/$DRIVE_FOLDER_ID"
echo ""
echo "Since Google Drive doesn't support direct CLI downloads for large files easily,"
echo "please download manually from the link above and place files as follows:"
echo ""
echo "1. models/transfuser_official/model.pth"
echo "2. models/crossvit_50/model50.pth"
echo "3. Extract transfuser-2022.zip into transfuser/transfuser-2022/"
echo "4. Place bi-attenfusion/ folder into transfuser/bi-attenfusion/"
echo ""
echo "Alternatively, if you have gdown installed:"
echo "  pip install gdown"
echo "  gdown --folder https://drive.google.com/drive/folders/$DRIVE_FOLDER_ID"
echo ""

# Check if gdown is available
if command -v gdown &> /dev/null; then
    echo "gdown found. Attempting automatic download..."
    echo ""
    
    # Create directories
    mkdir -p "$MODEL_DIR/transfuser_official" "$MODEL_DIR/crossvit_50"
    mkdir -p "$TRANSFUSER_DIR/transfuser-2022" "$TRANSFUSER_DIR/bi-attenfusion"
    
    # Download the folder
    gdown --folder "https://drive.google.com/drive/folders/$DRIVE_FOLDER_ID" -O /tmp/carla_models --remaining-ok || true
    
    echo ""
    echo "Download attempted. Check /tmp/carla_models/ for downloaded files."
    echo "Manually organize them as described above."
else
    echo "gdown not installed. To enable automatic downloads:"
    echo "  pip install gdown"
    echo ""
    echo "Manual download link:"
    echo "  https://drive.google.com/drive/folders/$DRIVE_FOLDER_ID"
fi
