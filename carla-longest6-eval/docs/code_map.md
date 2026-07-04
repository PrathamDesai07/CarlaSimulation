# Code Map — TransFuser + CrossViT-Fusion (bi-attenfusion)

## Source Code Layout

All evaluation code lives under `transfuser/transfuser-2022/team_code_transfuser/`.
There is no separate `bi-attenfusion/` directory — the crossvit_fusion code is
integrated directly into `team_code_transfuser/` alongside the standard TransFuser backbones.

```
transfuser/transfuser-2022/
├── team_code_transfuser/               # All model/agent/eval code
│   ├── config.py                       # GlobalConfig — all hyperparameters/settings
│   ├── model.py                        # Training model (with mmcv/mmdet for loss computation)
│   ├── model_eval.py                   # Evaluation model (no mmcv loss deps, pure PyTorch)
│   ├── submission_agent.py             # CARLA leaderboard agent (HybridAgent)
│   ├── submission_agent_v2.py          # Alternative agent version
│   ├── train.py                        # Training loop entrypoint
│   ├── data.py                         # Dataset loader (CARLA_Data) + sensor preprocessing
│   ├── utils.py                        # Coordinate transforms (LiDAR↔vehicle↔BEV image)
│   │
│   │  # Backbone implementations (model architecture)
│   ├── transfuser.py                   # TransfuserBackbone — GPT-based fusion (official TransFuser)
│   ├── geometric_fusion.py             # GeometricFusionBackbone
│   ├── late_fusion.py                  # LateFusionBackbone
│   ├── latentTF.py                     # latentTFBackbone
│   ├── Bi-Attenfusion.py               # CrossViT cross-attention block definitions
│   ├── attenchange.py                  # CrossViTFusionBackbone — full crossvit_fusion backbone
│   ├── crossvit_fusion.py              # Alias/wrapper → re-exports from attenchange.py
│   │
│   │  # Ablation study backbones
│   ├── no_bi_attn.py                   # NoBiAttnCrossViTBackbone
│   ├── no_multiscale.py                # NoMultiScaleCrossViTBackbone
│   ├── no_vel.py                       # NoVelCrossViTBackbone
│   ├── no_bi_ms.py                     # NoBiMsCrossViTBackbone
│   ├── std_crossvit.py                 # StdCrossViTBackbone (nn.MultiheadAttention)
│   ├── no_attn.py                      # NoAttnCrossViTBackbone (MLP-only)
│   ├── no_attn_no_ms.py                # NoAttnNoMsCrossViTBackbone
│   │
│   │  # Supporting modules
│   ├── point_pillar.py                 # PointPillarNet (LiDAR voxelization alternative)
│   ├── single_modal_backbone.py        # ImageOnlyBackbone, LidarOnlyBackbone
│   │
│   │  # Import wrappers (so model.py/model_eval.py imports resolve)
│   ├── no_bi_attn_crossvit.py          → re-exports NoBiAttnCrossViTBackbone
│   ├── no_multiscale_crossvit.py       → re-exports NoMultiScaleCrossViTBackbone
│   ├── no_vel_crossvit.py              → re-exports NoVelCrossViTBackbone
│   ├── no_bi_ms_crossvit.py            → re-exports NoBiMsCrossViTBackbone
│   ├── no_attn_crossvit.py             → re-exports NoAttnCrossViTBackbone
│   ├── no_attn_no_ms_crossvit.py       → re-exports NoAttnNoMsCrossViTBackbone
│   │
│   └── requirements.txt                # Python deps for this project
│
├── team_code_autopilot/                # Privileged autopilot agent (data collection)
│   ├── autopilot.py                    # AutopilotAgent — privileged expert for data gen
│   ├── data_agent.py                   # Data collection agent
│   ├── nav_planner.py                  # Global route planner
│   └── utils/
│       ├── lts_rendering.py
│       └── map_utils.py
│
├── leaderboard/                        # CARLA Leaderboard evaluation framework
│   ├── leaderboard/
│   │   ├── leaderboard_evaluator.py    # Leaderboard evaluation orchestrator
│   │   ├── autoagents/
│   │   │   ├── autonomous_agent.py     # Base class for agent plugins
│   │   │   └── agent_wrapper.py        # Sensor setup bridge
│   │   ├── scenarios/
│   │   └── utils/
│   └── scripts/
│
├── scenario_runner/                    # CARLA scenario runner
├── tools/                              # Dataset generation tools, route generation
└── models_2022/                        # (from S3) Pretrained checkpoints
```

## Key Classes and Their Locations

| Class / Function | File | Purpose |
|---|---|---|
| `GlobalConfig` | `config.py` | All hyperparameters: sensor config, model architecture, training, PID controller |
| `LidarCenterNet` | `model_eval.py` | Full evaluation model: backbone → features → GRU waypoints → PID control |
| `TransfuserBackbone` | `transfuser.py` | Official TransFuser: ResNet encoders + GPT multi-scale fusion transformers |
| `CrossViTFusionBackbone` | `attenchange.py` | CrossViT-Fusion: cross-modal attention (image↔LiDAR at 4 scales) |
| `LidarCenterNetHead` | `model_eval.py` | CenterNet-based 3D object detection head (heatmap + wh + offset + yaw) |
| `PIDController` | `model_eval.py` | PID controller for steer/throttle/brake from predicted waypoints |
| `HybridAgent` | `submission_agent.py` | CARLA agent: loads model, sets up sensors, runs eval loop |
| `ImageCNN` | `transfuser.py` (and in all backbone files) | ResNet/RegNet/ConvNext image encoder |
| `LidarEncoder` | `transfuser.py` (and in all backbone files) | ResNet/RegNet/ConvNext LiDAR encoder |

## Data Flow (Evaluation)

```
CARLA Server (carla-server:2000)
    │
    ▼  ← sensors: RGB camera (3×), LiDAR, GNSS, IMU
HybridAgent (submission_agent.py)
    │
    ├── Sensor data preprocessing (data.py):
    │   ├── rgb_crop_resize → (3, 160, 704)
    │   ├── lidar_to_histogram_features → (2, 256, 256) BEV
    │   ├── lidar_bev_cam_correspondences → geometric mapping
    │   └── target_point → global route guidance (2,)
    │
    ▼
LidarCenterNet.forward_ego() (model_eval.py)
    │
    ├── Backbone forward:
    │   ├── ImageCNN(rgb) → image_features_grid
    │   ├── LidarEncoder(lidar_bev) → lidar_features
    │   └── Fusion (transFuser / crossvit_fusion / etc.) → fused_features
    │
    ├── Waypoint prediction:
    │   ├── Join(fused_features) → 64-dim embedding
    │   └── GRU decoder (autoregressive, pred_len=4) → pred_wp
    │
    ├── PIDController(pred_wp, ego_vel) → steer, throttle, brake
    │
    └── Object detection (optional):
        └── LidarCenterNetHead(features) → 3D bboxes + yaw + velocity

    ▼
CARLA vehicle.apply_control() → simulator steps
    │
    ▼ (per tick)
Metrics tracked:
    - Route Completion (%)  — distance along route / total route distance
    - Driving Score         — Route Completion × infraction penalty
    - Collision count       — collisions with vehicles/pedestrians/static
    - Lane invasion count   — crossing lane markings
    - Red light infraction  — running red lights
    - Agent blocked count   — stuck/stopped abnormally
    - Off-road count        — driving outside drivable area
```

## Model Weights Architecture Mapping

| Checkpoint | backbone config | Model Class | Encoder |
|---|---|---|---|
| `models/transfuser_official/model.pth` | `transFuser` | `TransfuserBackbone` | ResNet34 image + ResNet18 LiDAR |
| `models/crossvit_50/model50.pth` | `crossvit_fusion` | `CrossViTFusionBackbone` | ResNet34 image + ResNet18 LiDAR |

## Import Chain (critical for Phase 2.2 mmcv fix)

```
model_eval.py
    ├── utils.py                     — pure NumPy, no mmcv
    ├── transfuser.py                — timm, torchvision (no mmcv)
    ├── geometric_fusion.py          — timm (no mmcv)
    ├── late_fusion.py               — timm (no mmcv)
    ├── latentTF.py                  — timm (no mmcv)
    ├── attenchange.py / crossvit_fusion.py  — timm (no mmcv)
    ├── no_*.py / std_crossvit.py     — timm (no mmcv)
    ├── point_pillar.py              — pure torch (no mmcv)
    ├── single_modal_backbone.py     — timm (no mmcv)
    └── data.py                      — skimage, cv2, ujson (no mmcv)

model.py (training only — NOT used at eval time)
    └── Depends on mmcv.ops.batched_nms, mmcv.cnn, mmdet.*
        → This is why model_eval.py exists as a separate eval-only file
```

The `model_eval.py` file is explicitly designed to have **zero mmcv dependencies**,
which means our mmcv-from-source build is only needed for the CenterNet head's
`batched_nms` and related ops — but `model_eval.py` uses a pure-PyTorch fallback.
