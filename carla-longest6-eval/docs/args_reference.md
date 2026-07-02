# Args Reference — args-1.txt

## Configuration Fields

| Field | Value | Description |
|-------|-------|-------------|
| `id` | `crossvit_2x5090` | Experiment identifier |
| `epochs` | `50` | Number of training epochs |
| `lr` | `0.0001` | Learning rate |
| `batch_size` | `16` | Training batch size |
| `logdir` | `/home/xhh/cb/transfuser/logs/crossvit_2x5090` | Log directory |
| `load_file` | `null` | Checkpoint to resume from |
| `start_epoch` | `0` | Starting epoch |
| `setting` | `02_05_withheld` | Dataset split setting |
| `root_dir` | `/home/xhh/cb/transfuser/transfuser-2022/data` | Dataset root |
| `schedule` | `1` | Enable LR scheduling |
| `schedule_reduce_epoch_01` | `35` | First LR reduction epoch |
| `schedule_reduce_epoch_02` | `45` | Second LR reduction epoch |
| `backbone` | `crossvit_fusion` | Model backbone architecture |
| `image_architecture` | `resnet34` | Image encoder backbone |
| `lidar_architecture` | `resnet18` | LiDAR encoder backbone |
| `use_velocity` | `1` | Use velocity input |
| `n_layer` | `4` | Number of transformer layers |
| `wp_only` | `0` | Predict waypoints only |
| `use_target_point_image` | `1` | Use target point image input |
| `use_point_pillars` | `0` | Use PointPillars LiDAR |
| `parallel_training` | `1` | Multi-GPU training |
| `val_every` | `5` | Validate every N epochs |
| `no_bev_loss` | `0` | Disable BEV loss |
| `sync_batch_norm` | `1` | Synchronized batch norm |
| `zero_redundancy_optimizer` | `0` | Use ZeRO optimizer |
| `use_disk_cache` | `0` | Cache dataset on disk |

## Important

- For `model50.pth`, `backbone` must be `crossvit_fusion`.
- For official TransFuser model, `backbone` is `transfuser`.
- Paths (`logdir`, `root_dir`) are overridden for Docker container paths.
