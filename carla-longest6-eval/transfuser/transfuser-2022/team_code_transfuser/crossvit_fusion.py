"""
CrossViT Fusion Backbone — wrapper for attenchange.py

The CrossViTFusionBackbone class is defined in attenchange.py.
This file re-exports all public names so that model.py and model_eval.py
imports resolve correctly.
"""
from attenchange import (
    CrossViTFusionBackbone,
    CrossAttention,
    CrossViTBlock,
    MultiScaleCrossViT,
    ImageCNN,
    LidarEncoder,
)

def normalize_imagenet(x):
    """ImageNet normalization transform for CARLA camera input."""
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x / 255.0 - mean) / std
