#!/usr/bin/env python3
"""
Phase 2.2 Verification — Real model forward pass (no CARLA, no mmcv).

Tests:
  1. Import model_eval.py (which has zero mmcv/mmdet dependencies)
  2. Instantiate LidarCenterNet with both backbones (transFuser, crossvit_fusion)
  3. Run a real GPU forward pass with synthetic tensors
  4. Verify output shapes and device placement
  5. Load model weights from disk (if checkpoint exists)
  6. Check model_eval.py has no mmcv imports at all

Exits 0 on success, non-zero on failure.
"""

import sys, os, importlib

PASS = 0
FAIL = 1
results = []


def check(name: str, ok: bool, detail: str = ""):
    results.append((name, ok))
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")
    return ok


def verify_import_chain():
    """Verify the entire import chain resolves without mmcv."""
    sys.path.insert(0, "transfuser/transfuser-2022/team_code_transfuser")

    # 1. First verify model_eval.py has zero mmcv imports
    print("\n[CHECK] model_eval.py has zero mmcv imports")
    import ast
    with open("transfuser/transfuser-2022/team_code_transfuser/model_eval.py") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                names = [alias.name for alias in node.names]
            for name in names:
                if "mmcv" in name or "mmdet" in name or "mmengine" in name:
                    return check("model_eval.py mmcv-free", False,
                                 f"Found mmcv import: {name}")
    check("model_eval.py mmcv-free", True)

    # 2. Import config and model
    from config import GlobalConfig
    from model_eval import LidarCenterNet
    check("GlobalConfig importable", True)
    check("LidarCenterNet importable", True)
    return GlobalConfig, LidarCenterNet


def verify_cuda_available():
    import torch
    if not torch.cuda.is_available():
        return check("CUDA available", False,
                     "No GPU found — forward pass requires CUDA")
    device_name = torch.cuda.get_device_name(0)
    check("CUDA available", True, f"device: {device_name}")
    return True


def verify_forward_pass(GlobalConfig, LidarCenterNet):
    import torch
    import numpy as np

    device = torch.device("cuda")

    for backbone_name, model_tag in [
        ("transFuser", "official TransFuser"),
        ("crossvit_fusion", "CrossViT-Fusion"),
    ]:
        print(f"\n[CHECK] Forward pass: {model_tag} (backbone={backbone_name})")

        # Configure for eval
        config = GlobalConfig(setting="eval")
        config.backbone = backbone_name
        config.multitask = False
        config.use_point_pillars = False
        config.use_target_point_image = True
        config.sync_batch_norm = False

        # Set architectures based on backbone
        if backbone_name == "transFuser":
            img_arch = "regnety_032"
            lidar_arch = "regnety_032"
            use_vel = False
        else:
            img_arch = "resnet34"
            lidar_arch = "resnet18"
            use_vel = True

        try:
            model = LidarCenterNet(
                config, device, backbone_name,
                image_architecture=img_arch,
                lidar_architecture=lidar_arch,
                use_velocity=use_vel,
            )
            model.eval()
            check(f"{model_tag} instantiation", True)
        except Exception as e:
            check(f"{model_tag} instantiation", False, str(e))
            continue

        # Create synthetic inputs matching expected shapes.
        # use_target_point_image=True → backbone expects lidar_bev with 2 channels,
        # then forward_ego concats target_point_image (1ch) to make 3ch before backbone.
        batch_size = 1
        rgb = torch.randn(batch_size, 3, config.img_resolution[0],
                          config.img_resolution[1], device=device)
        lidar_bev = torch.randn(batch_size, 2, 256, 256, device=device)
        target_point = torch.randn(batch_size, 2, device=device)
        target_point_image = torch.randn(batch_size, 1, 256, 256, device=device)
        ego_vel = torch.randn(batch_size, 1, device=device)

        # Run forward
        try:
            with torch.no_grad():
                pred_wp, bboxes = model.forward_ego(
                    rgb, lidar_bev, target_point, target_point_image,
                    ego_vel, debug=False
                )

            wp_ok = isinstance(pred_wp, torch.Tensor) and pred_wp.shape == (1, 4, 2)
            bb_ok = isinstance(bboxes, list)
            wp_on_cuda = pred_wp.device.type == "cuda" if wp_ok else False

            check(f"{model_tag} forward pass", True,
                  f"pred_wp: {pred_wp.shape}, bboxes: {len(bboxes)} detections")
            check(f"{model_tag} output on GPU", wp_on_cuda)
        except Exception as e:
            check(f"{model_tag} forward pass", False, str(e))

        # Clean up
        del model
        torch.cuda.empty_cache()

    return True


def verify_model_loading(GlobalConfig, LidarCenterNet):
    """Load both checkpoints and verify they match their backbones."""
    import torch

    models_to_test = [
        ("models/transfuser_official/model.pth", "transFuser",
         "regnety_032", "regnety_032"),
        ("models/crossvit_50/model50.pth", "crossvit_fusion",
         "resnet34", "resnet18"),
    ]

    for model_path, backbone, img_arch, lidar_arch in models_to_test:
        print(f"\n[CHECK] Loading checkpoint: {model_path}")
        if not os.path.exists(model_path):
            check(f"Checkpoint exists: {model_path}", False, "file not found")
            continue

        try:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
            keys = list(ckpt.keys())
            check(f"Checkpoint loadable: {model_path}", True,
                  f"dict with {len(keys)} keys")

            # Verify backbone compatibility
            if backbone == "crossvit_fusion":
                has_crossvit = any("crossvit" in k for k in keys)
                check(f"CrossViT keys present in checkpoint",
                      has_crossvit, f"crossvit keys: {sum(1 for k in keys if 'crossvit' in k)}")
            else:
                has_transformer = any("transformer" in k for k in keys)
                check(f"Transformer keys present in checkpoint",
                      has_transformer, f"transformer keys: {sum(1 for k in keys if 'transformer' in k)}")
        except Exception as e:
            check(f"Checkpoint loadable: {model_path}", False, str(e))

    return True


def verify_no_mmcv_in_backbones():
    """Backbone files should not import mmcv either."""
    backbone_files = [
        "transfuser.py", "geometric_fusion.py", "late_fusion.py",
        "latentTF.py", "attenchange.py", "Bi-Attenfusion.py",
        "no_bi_attn.py", "no_multiscale.py", "no_vel.py",
        "no_bi_ms.py", "std_crossvit.py", "no_attn.py",
        "no_attn_no_ms.py", "single_modal_backbone.py",
    ]
    base = "transfuser/transfuser-2022/team_code_transfuser"

    all_clean = True
    for fname in backbone_files:
        fpath = os.path.join(base, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            content = f.read()
        if "mmcv" in content or "mmdet" in content or "mmengine" in content:
            all_clean = False
            check(f"Backbone {fname} has mmcv import", False)
        else:
            check(f"Backbone {fname} is mmcv-free", True)

    return all_clean


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    project_root = os.getcwd()
    print(f"  Project root: {project_root}")
    print(f"  Python:       {sys.version.split()[0]}")
    print()

    GlobalConfig, LidarCenterNet = verify_import_chain()

    print("\n" + "-" * 60)
    print("  Forward Pass Tests (requires GPU)")
    print("-" * 60)
    if verify_cuda_available():
        verify_forward_pass(GlobalConfig, LidarCenterNet)
    else:
        print("  Skipping forward pass tests — no CUDA device.")

    print("\n" + "-" * 60)
    print("  Checkpoint Loading Tests")
    print("-" * 60)
    verify_model_loading(GlobalConfig, LidarCenterNet)

    print("\n" + "-" * 60)
    print("  Backbone mmcv-free Verification")
    print("-" * 60)
    verify_no_mmcv_in_backbones()

    # Summary
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"  Result: {passed}/{total} checks passed")
    print("=" * 60)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
