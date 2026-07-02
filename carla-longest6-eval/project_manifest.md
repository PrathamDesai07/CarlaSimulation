# Project Manifest

## Overview

**Project:** CARLA Longest6 Evaluation — autonomous driving evaluation on Longest6 route using CARLA 0.9.16 with TransFuser and CrossViT-Fusion models. Targets Ubuntu + RTX 50-series GPU (L40S / RTX 5090) + Docker.

## Key Scripts

| Script | Path | Purpose |
|--------|------|---------|
| Evaluation | `scripts/run_longest6_eval.py` | Main evaluation entrypoint |
| Video gen | `scripts/generate_video_from_frames.sh` | FFmpeg video from frames |
| CARLA start | `scripts/run_carla_docker.sh` | Start/stop CARLA Docker |
| Client start | `scripts/run_client_docker.sh` | Run evaluation client |
| Quickstart | `scripts/quickstart_longest6.sh` | Full pipeline one-shot |

## Configs

| Config | Path | Purpose |
|--------|------|---------|
| Eval config | `config/longest6_config.yaml` | Route, weather, sensors, metrics |
| Camera config | `config/camera.yaml` | Camera intrinsics and position |
| Model registry | `config/models.yaml` | Named model paths and backbones |

## Docker

| Component | Path |
|-----------|------|
| CARLA Dockerfile | `docker/carla/Dockerfile` |
| CARLA compose | `docker/carla/docker-compose.yml` |
| Client Dockerfile | `docker/client/Dockerfile` |
| Client compose | `docker/client/docker-compose.yml` |
| Requirements | `docker/client/requirements.txt` |
| Pinned deps | `docker/client/requirements_fixed.txt` |

## Documentation

| Doc | Path |
|-----|------|
| User guide | `docs/user_guide.md` |
| Code map | `docs/code_map.md` |
| Args reference | `docs/args_reference.md` |
| mmcv notes | `docs/mmcv_compatibility_notes.md` |
| Model setup | `docs/model_setup.md` |

## Outputs

| Output | Path | Format |
|--------|------|--------|
| Metrics JSON | `outputs/metrics/longest6/` | `.json` |
| Metrics summary | `outputs/metrics/longest6/` | `.txt` |
| Eval logs | `outputs/eval_logs/longest6/` | `.log` |
| Frames | `outputs/frames/longest6/` | `.png` |
| Demo video | `outputs/videos/longest6/` | `.mp4` |

## Schema

| Schema | Path |
|--------|------|
| Metrics schema | `schemas/metrics_schema.json` |
