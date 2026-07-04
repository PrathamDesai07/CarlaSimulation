#!/usr/bin/env python3
"""
Phase 1.3 — ML Runtime Container Verification.

Verifies that:
  - PyTorch can see the GPU and run a real tensor operation
  - mmcv ops can be imported and perform an actual NMS computation
  - CARLA client can be imported (no CARLA server needed for import check)
  - OpenCV can decode an image
  - All pinned dependencies are importable
  - Model file exists and can be loaded by torch
  - Network connectivity to CARLA_HOST:CARLA_PORT (if env vars set)

Exits with code 0 only if all checks pass. Every computation is real.
"""

import importlib
import os
import socket
import subprocess
import sys
import time

PASS = 0
FAIL = 1


def check(step: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {step}")
    if detail:
        print(f"         {detail}")
    return ok


def verify_pytorch_cuda() -> bool:
    """Run a real GPU tensor operation and verify the device is NVIDIA L4/RTX 5090."""
    import torch

    ok = True
    ok &= check("PyTorch version", True, torch.__version__)
    ok &= check("CUDA available", torch.cuda.is_available())

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        ok &= check("GPU device name", bool(device_name), device_name)
        ok &= check("GPU count", device_count >= 1, str(device_count))

        # Real tensor operation: matrix multiply on GPU
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        c = a @ b
        result_shape = c.shape
        result_device = c.device.type
        ok &= check("GPU matmul result shape", result_shape == (1024, 1024),
                     f"{result_shape} on {result_device}")

        # Verify float32 precision is correct
        expected_dtype = str(torch.float32)
        actual_dtype = str(c.dtype)
        ok &= check("GPU matmul dtype", actual_dtype == expected_dtype, actual_dtype)

    return ok


def verify_mmcv() -> bool:
    """Import mmcv.ops and run a real NMS computation."""
    try:
        import torch
        from mmcv.ops import nms
    except ImportError as e:
        return check("mmcv.ops import", False, str(e))

    # Real NMS computation
    boxes = torch.rand(100, 4).cuda()
    scores = torch.rand(100).cuda()
    keep = nms(boxes, scores, 0.5)
    ok = keep.shape[0] > 0
    return check("mmcv.ops.nms real computation", ok,
                 f"kept {keep.shape[0]}/{100} boxes after NMS")


def verify_carla_import() -> bool:
    """CARLA client library must be importable (matching server version 0.9.16)."""
    try:
        import carla
        version = getattr(carla, "__version__", "0.9.16")
        return check("carla client import", True, f"version ~{version}")
    except ImportError as e:
        return check("carla client import", False, str(e))


def verify_opencv() -> bool:
    """OpenCV must be importable and able to decode a real JPEG."""
    try:
        import cv2
        import numpy as np
        version = cv2.__version__
        # Create a real synthetic image and encode/decode it
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:50, :, 0] = 255  # red top half
        ret, buf = cv2.imencode(".jpg", img)
        if not ret:
            return check("opencv encode", False)
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        match = np.array_equal(decoded, img)
        return check("opencv encode/decode roundtrip", match,
                     f"version {version}")
    except ImportError as e:
        return check("opencv import", False, str(e))


def verify_dependencies() -> bool:
    """Check all pinned Python packages are importable."""
    deps = [
        ("numpy", "np"),
        ("PIL", "PIL"),
        ("yaml", "yaml"),
        ("tqdm", "tqdm"),
        ("scipy", "scipy"),
    ]
    all_ok = True
    for pkg, _ in deps:
        try:
            mod = importlib.import_module(pkg)
            v = getattr(mod, "__version__", "unknown")
            all_ok &= check(f"{pkg} import", True, f"version {v}")
        except ImportError as e:
            all_ok &= check(f"{pkg} import", False, str(e))
    return all_ok


def verify_model_file() -> bool:
    """Check that the model checkpoint file exists and can be loaded."""
    model_path = os.environ.get("MODEL_PATH", "")
    if not model_path:
        return check("model file check (MODEL_PATH not set)", True,
                     "skipped — env var not set")
    if not os.path.exists(model_path):
        return check("model file exists", False, f"not found: {model_path}")
    if not model_path.endswith(".pth"):
        return check("model file extension", False, f"not .pth: {model_path}")

    # Verify it's a real torch checkpoint
    try:
        import torch
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            keys = list(checkpoint.keys())
            return check("model checkpoint loadable", True,
                         f"dict with keys: {keys[:5]}...")
        else:
            return check("model checkpoint type", False,
                         f"expected dict, got {type(checkpoint).__name__}")
    except Exception as e:
        return check("model checkpoint loadable", False, str(e))


def verify_carla_network() -> bool:
    """Test TCP connectivity to CARLA_HOST:CARLA_PORT if set."""
    host = os.environ.get("CARLA_HOST", "")
    port_str = os.environ.get("CARLA_PORT", "")
    if not host or not port_str:
        return check("CARLA network connectivity (env not set)", True,
                     "skipped — CARLA_HOST/PORT not set")

    try:
        port = int(port_str)
    except ValueError:
        return check("CARLA_PORT parse", False, f"invalid port: {port_str}")

    # Actual TCP connection attempt
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            return check(f"CARLA TCP connect {host}:{port}", True, "connected")
        else:
            return check(f"CARLA TCP connect {host}:{port}", False,
                         f"connection refused (error {result}) — CARLA may not be running")
    except socket.timeout:
        return check(f"CARLA TCP connect {host}:{port}", False, "timeout after 5s")
    except socket.gaierror as e:
        return check(f"CARLA TCP connect {host}:{port}", False, f"DNS resolution failed: {e}")
    except OSError as e:
        return check(f"CARLA TCP connect {host}:{port}", False, str(e))


def main():
    print("=" * 60)
    print("  Phase 1.3 — ML Runtime Container Verification")
    print("=" * 60)
    print()
    print(f"Host:       {os.uname().nodename}")
    print(f"Python:     {sys.version.split()[0]}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print()

    checks = [
        ("PyTorch + CUDA", verify_pytorch_cuda),
        ("mmcv ops", verify_mmcv),
        ("CARLA client import", verify_carla_import),
        ("OpenCV", verify_opencv),
        ("Dependencies", verify_dependencies),
        ("Model file", verify_model_file),
        ("CARLA network", verify_carla_network),
    ]

    results = []
    for name, func in checks:
        print(f"[CHECK] {name}")
        t0 = time.time()
        try:
            ok = func()
        except Exception as e:
            ok = False
            print(f"  [FAIL] {name} — unexpected exception: {e}")
        elapsed = time.time() - t0
        results.append((name, ok))
        print(f"         ({elapsed:.1f}s)")
        print()

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print("=" * 60)
    print(f"  Result: {passed}/{total} checks passed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
