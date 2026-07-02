# User Guide — CARLA Longest6 Evaluation

## Prerequisites

- Ubuntu 22.04+
- NVIDIA GPU (L40S or RTX 5090)
- NVIDIA Driver 545+, CUDA 12.1+
- Docker 24+ with NVIDIA Container Toolkit
- Docker Compose 2.21+

## 1. Environment Setup

```bash
# Verify GPU
nvidia-smi

# Verify Docker GPU access
docker run --gpus all nvidia/cuda:12.1-base nvidia-smi
```

### Build mmcv from source

```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.7.0
export MMCV_WITH_OPS=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
pip install -e .
```

## 2. Project Setup

```bash
git clone <repo-url> carla-longest6-eval
cd carla-longest6-eval
```

Download from: https://drive.google.com/drive/folders/1yG9LbVtSLaneHKlB5GL5Vhzz5Miue5Me?usp=sharing

1. Extract `transfuser-2022.zip` → `transfuser/transfuser-2022/`
2. Place `bi-attenfusion` → `transfuser/bi-attenfusion/`
3. Place `model.pth` → `models/transfuser_official/`
4. Place `model50.pth` → `models/crossvit_50/`

```bash
# Build Docker images
cd docker/carla && docker-compose build
cd ../client && docker-compose build
```

## 3. Run Evaluation

```bash
# Start CARLA
cd docker/carla && docker-compose up -d

# Run official TransFuser
cd ../client && docker-compose run --rm \
    -e MODEL_PATH=/app/models/transfuser_official/model.pth \
    client

# Or CrossViT-Fusion
cd docker/client && docker-compose run --rm \
    -e MODEL_PATH=/app/models/crossvit_50/model50.pth \
    client
```

## 4. Outputs

- Metrics: `outputs/metrics/longest6/longest6_<model>.json`
- Summary: `outputs/metrics/longest6/longest6_<model>.txt`
- Video: `outputs/videos/longest6/demo.mp4`

## 5. Troubleshooting

| Issue | Solution |
|-------|----------|
| mmcv import error | Rebuild mmcv from source with correct CUDA arch flags |
| Can't connect to CARLA | `docker ps` to check CARLA is running |
| Empty sensor data | Ensure CARLA has `-RenderOffScreen` flag |
| GPU OOM | Lower camera resolution or quality level |
| Container networking | Check `docker network ls` for `carla-net` |
