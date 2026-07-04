# mmcv Compatibility Notes

## Problem

The official mmcv used by TransFuser (`mmcv-full==1.6.0` / `mmcv==1.7.0`) does not
support RTX 50-series (Blackwell) or Ada Lovelace (L40S) GPUs out of the box.
Precompiled wheels on PyPI only ship CUDA kernels for compute capability up to 8.0
(Turing/Ampere), missing 8.9 (Ada) and 9.0 (Blackwell).

## Why This Project Does Not Need mmcv at Evaluation Time

**Critical finding:** The evaluation code (`model_eval.py`) has **zero mmcv dependencies**.
It was explicitly refactored to use pure-PyTorch and torchvision fallbacks:

| mmcv operation | model_eval.py replacement | Status |
|---|---|---|
| `mmcv.ops.batched_nms` | `torchvision.ops.nms` | ✅ Swapped |
| `mmdet.get_local_maximum` | Custom pure-PyTorch `_get_local_maximum` | ✅ In-file |
| `mmdet.get_topk_from_heatmap` | Custom pure-PyTorch `_get_topk_from_heatmap` | ✅ In-file |
| `mmdet.transpose_and_gather_feat` | Custom pure-PyTorch `_transpose_and_gather_feat` | ✅ In-file |
| `mmcv.cnn.bias_init_with_prob` | Manual bias init | ✅ In-file |
| `mmcv.cnn.normal_init` | Manual normal init | ✅ In-file |
| `mmdet.LidarCenterNetHead` torch.compile | `force_fp32` decorator (avoids autocast issues) | ✅ In-file |

The 18 backbone files (`transfuser.py`, `attenchange.py`, `geometric_fusion.py`, etc.)
also have **zero mmcv imports** — they only depend on `timm`, `torch`, and `torchvision`.

mmcv is only required for **training** (`model.py`), which is outside the scope of
this evaluation project.

## What We Do in the Dockerfile

The Dockerfile at `docker/client/Dockerfile` builds mmcv from source with correct
architecture flags. This is kept for:

1. Training (`model.py`) compatibility if ever needed
2. Any script that inadvertently imports mmcv
3. The verification script which tests mmcv ops explicitly

```dockerfile
RUN git clone https://github.com/open-mmlab/mmcv.git /tmp/mmcv \
    && cd /tmp/mmcv \
    && git checkout v1.7.0 \
    && MMCV_WITH_OPS=1 TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" pip install -e . \
    && rm -rf /tmp/mmcv
```

## Manual Build Instructions (if needed outside Docker)

```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.7.0
export MMCV_WITH_OPS=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"  # 8.9=Ada (L40S), 9.0=Blackwell (RTX 5090)
pip install -e .
```

## Verified Combinations

| PyTorch | CUDA | mmcv | Notes |
|---------|------|------|-------|
| 2.1.0 | 12.1 | 1.7.0 (source build) | Used in this project |
| 2.8.0+cu128 | 13.0 (host driver) | 1.7.0 (source build) | Verified on L40S host |

## Verification

```bash
# Test mmcv ops (requires GPU)
python3 -c "
import torch
from mmcv.ops import nms
boxes = torch.rand(100, 4).cuda()
scores = torch.rand(100).cuda()
keep = nms(boxes, scores, 0.5)
print(f'mmcv nms works: {len(keep)} boxes kept')
"

# Test full model forward pass without CARLA
python3 scripts/test_model_forward.py
```
