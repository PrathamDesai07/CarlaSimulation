#!/bin/bash
# Environment verification script for Phase 1.1
# Runs all checks and outputs a summary.
# Usage: bash infra/check_environment.sh

set -e

echo "============================================"
echo "  CARLA Longest6 - Environment Verification"
echo "============================================"
echo ""

# 1. OS
echo "[1/8] OS Version"
cat /etc/os-release 2>/dev/null | grep -E "PRETTY_NAME|VERSION_ID" | head -2
echo ""

# 2. NVIDIA Driver
echo "[2/8] NVIDIA Driver"
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "NOT FOUND"
echo ""

# 3. GPU Info
echo "[3/8] GPU Info"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv 2>/dev/null || echo "NOT FOUND"
echo ""

# 4. CUDA
echo "[4/8] CUDA Version"
nvcc --version 2>/dev/null | grep "release" || echo "NOT FOUND"
echo ""

# 5. Docker
echo "[5/8] Docker"
docker --version 2>/dev/null || echo "NOT FOUND"
docker compose version 2>/dev/null || echo "NOT FOUND"
echo ""

# 6. Docker GPU Access
echo "[6/8] Docker GPU Access"
docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu24.04 nvidia-smi \
    --query-gpu=name --format=csv,noheader 2>/dev/null || echo "FAILED"
echo ""

# 7. Python & Packages
echo "[7/8] Python Packages"
python3 -c "
import torch, numpy, cv2, PIL, yaml, carla, tqdm, scipy
print(f'torch:       {torch.__version__}')
print(f'numpy:       {numpy.__version__}')
print(f'opencv:      {cv2.__version__}')
print(f'pillow:      {PIL.__version__}')
print(f'pyyaml:      {yaml.__version__}')
print(f'carla:       {carla.__version__ if hasattr(carla, \"__version__\") else \"0.9.16\"}')
print(f'tqdm:        {tqdm.__version__}')
print(f'scipy:       {scipy.__version__}')
" 2>/dev/null
echo ""

# 8. PyTorch CUDA
echo "[8/8] PyTorch CUDA"
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU device:     {torch.cuda.get_device_name(0)}')
print(f'GPU count:      {torch.cuda.device_count()}')
print(f'Torch CUDA:     {torch.version.cuda}')
" 2>/dev/null
echo ""

echo "============================================"
echo "  Verification Complete"
echo "============================================"
