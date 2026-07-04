# Environment Baseline — Verified Server State

Real verification results from the target machine (run: July 2, 2026).

## System

| Component | Version | Source |
|-----------|---------|--------|
| OS | Ubuntu 24.04.4 LTS (Noble Numbat) | `cat /etc/os-release` |
| GPU | NVIDIA L4 (24 GB VRAM, Ada Lovelace) | `nvidia-smi` |
| NVIDIA Driver | 580.159.03 | `nvidia-smi` |
| CUDA (driver API) | 13.0 | `nvidia-smi` |
| CUDA (compiler) | 13.0.88 | `nvcc --version` |
| Docker | 28.0.1 | `docker --version` |
| Docker Compose | v2.27.0 | `docker compose version` |
| NVIDIA Container Toolkit | 1.19.1 | `dpkg -l \| grep nvidia-container-toolkit` |

## GPU Verification

```bash
$ nvidia-smi
Thu Jul  2 20:28:55 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.159.03          Driver Version: 580.159.03     CUDA Version: 13.0       |
+-----------------------------------------+------------------------+----------------------+
|   0  NVIDIA L4                      Off |   00000000:00:04.0 Off |                    0 |
| N/A   36C    P8             16W /   72W |       0MiB /  23034MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

## Docker GPU Access Verification

```bash
$ docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu24.04 nvidia-smi
# Output confirms GPU visible inside container:
#   0  NVIDIA L4                    -       23034MiB
```

Docker daemon is configured with NVIDIA runtime:

```bash
$ cat /etc/docker/daemon.json
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
```

## Python Environment

| Package | Version | Status |
|---------|---------|--------|
| Python | 3.12.11 | ✅ |
| torch | 2.8.0+cu128 | ✅ |
| torchvision | 0.23.0+cu128 | ✅ |
| numpy | 1.26.4 | ✅ |
| pillow | 12.2.0 | ✅ |
| PyYAML | 6.0.3 | ✅ |
| scipy | 1.11.4 | ✅ |
| tqdm | 4.68.3 | ✅ |
| opencv-python | **not installed** | ❌ Needed |
| carla-client | **not installed** | ❌ Needed (comes from CARLA pip) |
| mmcv | **not installed** | ❌ Must build from source |

## mmcv Compatibility Analysis

**GPU Architecture:** NVIDIA L4 uses Ada Lovelace architecture (compute capability ~8.9).
The TransFuser training code (`model.py`) uses mmcv with custom CUDA ops, but the
**evaluation code (`model_eval.py`) has zero mmcv dependencies** — it uses pure-PyTorch
and torchvision fallbacks for all operations.

mmcv is built from source in the Dockerfile for safety (in case any code path
inadvertently imports it), but the actual evaluation forward pass does not require it.

```bash
# mmcv source build (included in Dockerfile, or run manually):
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.7.0
export MMCV_WITH_OPS=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
pip install -e .
```

## Verification Script

Run the full verification at any time with:

```bash
bash infra/check_environment.sh
```

## Action Items — Resolved

| # | Item | Status |
|---|------|--------|
| 1 | NVIDIA driver 580.159.03 installed | ✅ Done |
| 2 | CUDA 13.0 tools available | ✅ Done |
| 3 | Docker 28.0.1 with NVIDIA runtime | ✅ Done |
| 4 | Docker GPU access working | ✅ Done (nvidia/cuda:12.8.0-runtime-ubuntu24.04 verified) |
| 5 | Docker Compose v2.27.0 | ✅ Done |
| 6 | Install opencv-python | ✅ Done (4.8.1.78, pinned for numpy compat) |
| 7 | Install carla-client (0.9.16) | ✅ Done (`pip install carla==0.9.16`) |
| 8 | Build mmcv from source | ❌ Pending — see docs/mmcv_compatibility_notes.md |
| 9 | Verify PyTorch CUDA | ✅ True — NVIDIA L4 detected via PyTorch 2.8.0+cu128 |
