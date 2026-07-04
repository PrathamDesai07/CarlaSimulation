# Args Reference

There are three args.txt files in `config_internal/raw_args/`:

| File | Model | Backbone | Used By |
|------|-------|----------|---------|
| `args-1.txt` | CrossViT-Fusion (model50.pth) | `crossvit_fusion` | `model50.pth` training |
| `args-crossvit.txt` | CrossViT-Fusion (variant) | `transFuser` | Variant from Google Drive |
| `args-transfuser-official.txt` | Official TransFuser (S3) | `transFuser` | `model.pth` from AWS |

## args-1.txt — CrossViT-Fusion (`crossvit_fusion`, ResNet34+ResNet18)

| Field | Value | Description |
|-------|-------|-------------|
| `id` | `crossvit_2x5090` | Experiment identifier |
| `epochs` | `50` | Number of training epochs |
| `lr` | `0.0001` | Learning rate |
| `batch_size` | `16` | Training batch size |
| `logdir` | `/home/xhh/cb/transfuser/logs/crossvit_2x5090` | Log directory |
| `load_file` | `null` | Checkpoint to resume from |
| `start_epoch` | `0` | Starting epoch |
| `setting` | `02_05_withheld` | Dataset split: Town02+Town05 withheld for validation |
| `root_dir` | `/home/xhh/cb/transfuser/transfuser-2022/data` | Dataset root |
| `schedule` | `1` | Enable LR scheduling |
| `schedule_reduce_epoch_01` | `35` | First LR reduction at epoch 35 |
| `schedule_reduce_epoch_02` | `45` | Second LR reduction at epoch 45 |
| `backbone` | `crossvit_fusion` | **CrossViT-Fusion**: cross-modal attention fusion |
| `image_architecture` | `resnet34` | Image encoder: ResNet-34 |
| `lidar_architecture` | `resnet18` | LiDAR encoder: ResNet-18 |
| `use_velocity` | `1` | Use velocity as transformer input |
| `n_layer` | `4` | Number of transformer layers (per scale) |
| `wp_only` | `0` | Predict waypoints only (0 = full model) |
| `use_target_point_image` | `1` | Render target point into BEV input |
| `use_point_pillars` | `0` | Disable PointPillars (use voxelization) |
| `parallel_training` | `1` | Multi-GPU training (DDP) |
| `val_every` | `5` | Validate every 5 epochs |
| `no_bev_loss` | `0` | Include BEV loss |
| `sync_batch_norm` | `1` | Synchronized batch norm for multi-GPU |
| `zero_redundancy_optimizer` | `0` | Disable ZeRO optimizer |
| `use_disk_cache` | `0` | No disk caching |

## args-transfuser-official.txt — Official TransFuser (`transFuser`, RegNetY_032+RegNetY_032)

| Field | Value | Description |
|-------|-------|-------------|
| `id` | `TransFuserAllNLayer4NoVelocityTPReg32Reg32Seed1` | Experiment identifier |
| `epochs` | `41` | Number of training epochs |
| `lr` | `0.0001` | Learning rate |
| `batch_size` | `12` | Training batch size |
| `backbone` | `transFuser` | **TransFuser**: GPT-based multi-scale fusion |
| `image_architecture` | `regnety_032` | **RegNetY-3.2GF** image encoder (not ResNet) |
| `lidar_architecture` | `regnety_032` | **RegNetY-3.2GF** LiDAR encoder (not ResNet) |
| `use_velocity` | `0` | **No velocity input** (different from crossvit variant) |
| `n_layer` | `4` | Number of GPT layers |
| `num_transformer` | `4` | Number of transformer blocks per scale |
| `use_target_point_image` | `1` | Render target point into BEV |
| `wp_only` | `0` | Full model (not waypoints-only) |
| `schedule` | `1` | Enable LR scheduling |

## Key Differences Between Configs

| Parameter | crossvit_fusion (model50) | transFuser official (model.pth) |
|---|---|---|
| `backbone` | `crossvit_fusion` | `transFuser` |
| `image_architecture` | `resnet34` | `regnety_032` |
| `lidar_architecture` | `resnet18` | `regnety_032` |
| `use_velocity` | `1` (yes) | `0` (no) |
| `batch_size` | 16 | 12 |
| `epochs` | 50 | 41 |
| `schedule_reduce_epoch_01` | 35 | (default) |
| `schedule_reduce_epoch_02` | 45 | (default) |

## Important

- For `model50.pth`, set `backbone: crossvit_fusion` in `config/longest6_config.yaml`
- For official TransFuser `model.pth`, set `backbone: transFuser`
- The official model uses **RegNetY** encoders, not ResNet — this matters for `config.py`
- Paths (`logdir`, `root_dir`) are training-only and overridden for evaluation
