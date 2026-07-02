# Environment Baseline

## Prerequisites

| Component | Required Version | Verification Command |
|-----------|-----------------|---------------------|
| OS | Ubuntu 22.04+ | `cat /etc/os-release` |
| NVIDIA Driver | 545+ | `nvidia-smi` |
| CUDA | 12.1+ | `nvcc --version` |
| Docker | 24+ | `docker --version` |
| Docker Compose | 2.21+ | `docker compose version` |
| NVIDIA Container Toolkit | Latest | `docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi` |

## GPU Verification

```bash
nvidia-smi
docker run --gpus all nvidia/cuda:12.1-base nvidia-smi
```

## Notes

- L40S uses Ada Lovelace architecture. RTX 5090 uses Blackwell. Both require drivers 545+ and CUDA 12.1+.
- mmcv precompiled wheels do not support Blackwell. Must rebuild from source (see `docs/mmcv_compatibility_notes.md`).
