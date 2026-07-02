# mmcv Compatibility Notes

## Problem

Official mmcv used by TransFuser doesn't support RTX 50-series (Blackwell architecture) GPUs. Precompiled wheels lack CUDA kernels for Blackwell SM architecture.

## Solution: Rebuild mmcv from source

```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.7.0
export MMCV_WITH_OPS=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"  # Blackwell = 9.0
pip install -e .
```

## Alternative: Replace with torch-native ops

- `mmcv.ops.nms` → `torchvision.ops.nms`
- `mmcv.ops.roi_align` → `torchvision.ops.roi_align`
- Remove sync_bn dependency (single-GPU eval doesn't need it)

## Verified Combinations

| PyTorch | CUDA | mmcv | Notes |
|---------|------|------|-------|
| 2.1.0 | 12.1 | 1.7.0 (source build) | Recommended for L40S |
| 2.2.0 | 12.1 | 1.7.0 (source build) | Tested on Ada |

## Verification

```python
import torch
from mmcv.ops import nms
boxes = torch.rand(100, 4).cuda()
scores = torch.rand(100).cuda()
keep = nms(boxes, scores, 0.5)
print(f"mmcv nms works: {len(keep)} boxes kept")
```
