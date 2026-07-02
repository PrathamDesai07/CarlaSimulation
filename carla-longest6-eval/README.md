# Carla Longest6 Evaluation

**Autonomous driving evaluation on the Longest6 test route using CARLA 0.9.16 with TransFuser and CrossViT-Fusion models, targeting Ubuntu + RTX 50-series GPU (L40S / RTX 5090).**

## Overview

This project evaluates autonomous driving performance on the **Longest6** test route in **CARLA 0.9.16** using pre-trained models (TransFuser official and CrossViT-Fusion). It outputs standard TransFuser metrics — Route Completion, Driving Score, and violation metrics — and optionally generates a demo video of the run.

**Source repository reference:** [autonomousvision/transfuser](https://github.com/autonomousvision/transfuser)

---

## Table of Contents

- [Phase 1 – Environment & Infrastructure](#phase-1--environment--infrastructure)
  - [1.1 – Server & GPU Baseline](#11--server--gpu-baseline)
  - [1.2 – CARLA via Docker (Headless)](#12--carla-via-docker-headless)
  - [1.3 – ML Runtime Container](#13--ml-runtime-container)
- [Phase 2 – Model Integration & Evaluation Logic](#phase-2--model-integration--evaluation-logic)
  - [2.1 – Codebase Ingestion & Structure Mapping](#21--codebase-ingestion--structure-mapping)
  - [2.2 – mmcv & GPU Compatibility Fix](#22--mmcv--gpu-compatibility-fix)
  - [2.3 – Longest6 Route Evaluation Script](#23--longest6-route-evaluation-script)
  - [2.4 – Model Selection Interface](#24--model-selection-interface)
- [Phase 3 – Video Generation, Packaging & Documentation](#phase-3--video-generation-packaging--documentation)
  - [3.1 – Video Generation from Headless CARLA](#31--video-generation-from-headless-carla)
  - [3.2 – Metrics Export & Reporting](#32--metrics-export--reporting)
  - [3.3 – User-Facing Documentation](#33--user-facing-documentation)
  - [3.4 – Final Packaging & Submission Bundle](#34--final-packaging--submission-bundle)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Deliverables](#deliverables)

---

## Phase 1 – Environment & Infrastructure

### 1.1 – Server & GPU Baseline

**Goal:** Verify the L40S/RTX 5090 headless server is ready to run CARLA + CUDA + Docker.

**Actions:**
1. Verify NVIDIA driver and CUDA versions with `nvidia-smi`.
2. Verify Docker can access the GPU: `docker run --gpus all nvidia/cuda:12.1-base nvidia-smi`.
3. Determine exact CUDA/PyTorch version matching TransFuser dependency constraints (mmcv compatibility with Ada/Blackwell architecture GPUs).
4. Install Docker and NVIDIA Container Toolkit if not already present.

**Deliverables:**
- `infra/README_environment.md` — checklist of OS, driver, CUDA, Docker, and NVIDIA Container Toolkit versions with verification commands.

---

### 1.2 – CARLA via Docker (Headless)

**Goal:** Run CARLA 0.9.16 in a Docker container in headless/off-screen mode on the L40S.

**Actions:**
1. Use the official CARLA 0.9.16 Docker image from [CARLA Docker Hub](https://hub.docker.com/r/carlasimulator/carla).
2. Configure GPU access (`--gpus all`).
3. Enable off-screen rendering (`-RenderOffScreen` or EGL config).
4. Verify with a simple Python script that synchronous sensor data can be fetched in headless mode.
5. Set up docker-compose for reproducible startup.

**Files to create:**
- `docker/carla/Dockerfile` — extends base CARLA 0.9.16 image with extras (ffmpeg, Python deps).
- `docker/carla/docker-compose.yml` — CARLA service with GPU access, ports (2000-2002), off-screen rendering.
- `docker/carla/entrypoint.sh` — entrypoint script for CARLA container.
- `docker/carla/scripts/wait_for_carla.sh` — helper to wait until CARLA server is ready.

**Deliverable:**
- Ability to start CARLA in headless mode on L40S with a single `docker-compose up` command.
- `infra/carla_run_notes.md` — notes on CARLA flags, port usage, troubleshooting.

---

### 1.3 – ML Runtime Container

**Goal:** Build a separate Docker container with all Python dependencies for TransFuser/bi-attenfusion evaluation, able to connect to the CARLA server container.

**Actions:**
1. Create a Python environment with PyTorch, mmcv (or patched replacement), and all TransFuser dependencies compatible with L40S/RTX 5090.
2. Mount or copy `transfuser-2022` and `bi-attenfusion` (CrossViT-Fusion) code into the container.
3. Ensure the container can network with the CARLA container via Docker compose networking.

**Files to create:**
- `docker/client/Dockerfile` — evaluation client image with all deps.
- `docker/client/requirements.txt` — Python dependencies (torch, torchvision, carla client, mmcv, numpy, opencv-python, etc.).
- `docker/client/requirements_fixed.txt` — pinned versions tested on L40S/RTX 5090.
- `docker/client/entrypoint.sh` — entrypoint for the client container.
- `docker/client/scripts/wait_for_carla.sh` — waits for CARLA TCP port before starting eval.
- `infra/networking.md` — explains how client container talks to CARLA container.

**Deliverable:**
- Working client container that can import TransFuser modules and open a CARLA client connection.

---

## Phase 2 – Model Integration & Evaluation Logic

### 2.1 – Codebase Ingestion & Structure Mapping

**Goal:** Understand and map the TransFuser 2022 + CrossViT-Fusion (bi-attenfusion) codebase layout and entrypoints.

**Actions:**
1. Download and extract `transfuser-2022.zip` from the [Google Drive link](https://drive.google.com/drive/folders/1yG9LbVtSLaneHKlB5GL5Vhzz5Miue5Me?usp=sharing).
2. Inspect the `bi-attenfusion` folder (CrossViT-Fusion code corresponding to `model50.pth`).
3. Identify:
   - Training and evaluation entrypoint scripts.
   - Model definition files (backbone, fusion layers, CrossViT fusion).
   - The argument parser to understand `args-1.txt` configuration.
   - CARLA client connection and agent logic.
   - Metrics computation code.

**Files to create:**
- `docs/code_map.md` — architecture map: which script starts evaluation, where models are defined, CARLA client logic, metrics computation.
- `docs/args_reference.md` — explanation of all settings in `args-1.txt` (epochs, lr, batch_size, backbone, use_velocity, n_layer, wp_only, etc.).

**Deliverable:**
- Clear documentation of the evaluation pipeline and where to hook Longest6 + metrics.

---

### 2.2 – mmcv & GPU Compatibility Fix

**Goal:** Resolve the core blocker — mmcv is incompatible with RTX 50-series GPUs (Blackwell architecture). The official mmcv used by TransFuser has not been updated for a long time and doesn't support newer GPU architectures.

**Actions:**
1. Determine the exact mmcv version used in TransFuser and the nature of the incompatibility (likely missing CUDA SM support for Blackwell/Ada architecture in precompiled wheels).
2. Evaluate options:
   - **Option A:** Rebuild mmcv from source on the target CUDA version with correct architecture flags (`TORCH_CUDA_ARCH_LIST`).
   - **Option B:** Swap out mmcv-specific operations with torch-native equivalents where possible.
   - **Option C:** Pin PyTorch/CUDA/mmcv to a known working trio for Ada architecture GPUs.
3. Test minimal model forward pass (dummy input, no CARLA) to confirm the fix works.
4. Document the exact solution.

**Files to create:**
- `docker/client/requirements_fixed.txt` — final pinned dependency list that works on L40S/RTX 5090.
- `docs/mmcv_compatibility_notes.md` — what was changed, exact versions, any patches applied, build instructions.

**Deliverable:**
- A container where all TransFuser modules can be imported and a test forward pass runs without error.

---

### 2.3 – Longest6 Route Evaluation Script

**Goal:** Implement the main evaluation script that runs the Longest6 route in CARLA 0.9.16 using a chosen model checkpoint and outputs TransFuser-standard metrics.

**Actions:**
1. Lock the evaluation to the **Longest6** route (no route switching required).
2. Implement the script to:
   - Connect to CARLA server (via Docker service hostname/port).
   - Load the specified `.pth` model file from a configurable path.
   - Drive the agent through the Longest6 route.
   - Collect all required metrics: Route Completion, Driving Score, violation metrics (collisions, lane invasions, red light infractions, etc.).
3. Save per-tick logs, per-episode summaries, and final aggregated metrics.

**Files to create:**
- `scripts/run_longest6_eval.py` — main evaluation entrypoint.
- `config/longest6_config.yaml` — route definition, weather settings, traffic settings, number of episodes, evaluation parameters.
- `config/camera.yaml` — camera intrinsics, resolution, FPS, FOV, mounting position for video capture.

**Deliverable:**
- One-command evaluation for Longest6 that produces numeric metrics for any given `.pth` model file.

---

### 2.4 – Model Selection Interface

**Goal:** Provide a simple mechanism to switch between different model checkpoints.

**Actions:**
1. Add CLI argument `--model-path` or `--model` to the evaluation script.
2. Support both:
   - Official TransFuser `model.pth`.
   - CrossViT-Fusion `model50.pth` (from the `bi-attenfusion` code).
3. Ensure the model loader handles different architectures correctly (the `args-1.txt` config defines `backbone: crossvit_fusion` for model50).

**Files to create:**
- `config/models.yaml` — named model entries (e.g., `transfuser_official`, `crossvit_50`) with file paths, backbone types, and architecture configs.
- `docs/model_setup.md` — explains where to place model files, naming conventions, and how to switch models.

**Deliverable:**
- Verified evaluation runs for both models:
  - Official TransFuser `model.pth`
  - CrossViT-Fusion `model50.pth`

---

## Phase 3 – Video Generation, Packaging & Documentation

### 3.1 – Video Generation from Headless CARLA

**Goal:** Produce a video file of a Longest6 evaluation run on the L40S in headless mode.

**Actions:**
1. Attach a front-facing RGB camera sensor in the evaluation script.
2. Save individual frames (PNG/JPEG) during the run to a designated directory.
3. After the run completes, use ffmpeg to encode frames into an H.264 MP4 video.
4. Ensure frame ordering and timestamps are consistent.

**Files to create:**
- `config/camera.yaml` — camera intrinsics, resolution (e.g., 1920x1080), FPS, FOV, mounting position.
- `scripts/generate_video_from_frames.sh` — shell script wrapping ffmpeg to produce MP4 from frames.

**Deliverable:**
- Reproducible process to generate a demo video from any evaluation run, fully headless.

---

### 3.2 – Metrics Export & Reporting

**Goal:** Output all required metrics in both machine-readable (JSON) and human-readable (text summary) formats.

**Required metrics:**
- Route Completion (%) — percentage of the Longest6 route completed.
- Driving Score — aggregate score combining route completion and infractions.
- Violation metrics:
  - Collision count and collision score.
  - Lane invasion count and score.
  - Red light infraction count and score.
  - Other agent-related violation scores.
- Additional TransFuser-standard metrics.

**Files to create:**
- `schemas/metrics_schema.json` — JSON schema defining all metric fields, types, and validation rules.
- `outputs/metrics/` — target directory for per-run metric files.
  - `longest6_<model_id>.json` — machine-readable metrics.
  - `longest6_<model_id>.txt` — human-readable summary.

**Deliverable:**
- Verified metrics files for at least both models on Longest6.

---

### 3.3 – User-Facing Documentation

**Goal:** Produce complete Markdown documentation for installation, usage, and output interpretation.

**Actions:**
1. Document:
   - Prerequisites (Ubuntu, NVIDIA driver, Docker, GPU).
   - How to build and run both CARLA and client containers.
   - Where to place model files and how to configure them.
   - How to run Longest6 evaluation for a given model.
   - Where metrics and videos are stored and how to interpret them.
   - Troubleshooting common issues (mmcv compatibility, networking, etc.).

**Files to create:**
- `README.md` (this file) — high-level intro, quickstart commands, structure overview.
- `docs/user_guide.md` — full step-by-step guide with commands, expected outputs, and troubleshooting.

**Deliverable:**
- Self-contained Markdown guide that someone can follow on an Ubuntu + L40S/RTX 5090 server with Docker.

---

### 3.4 – Final Packaging & Submission Bundle

**Goal:** Assemble everything into a clean, runnable package with all deliverables.

**Actions:**
1. Ensure the project structure is clean and well-organized.
2. Run a fresh end-to-end test on the target server from a clean clone to validate all instructions in the user guide.
3. Generate the demo video for at least the official TransFuser model.
4. Create a project manifest for quick reference.

**Files to create:**
- `project_manifest.md` — one-page reference listing all major scripts, configs, and their purposes.
- `outputs/videos/demo_longest6_official.mp4` — demonstration video of a successful run.

**Deliverable:**
- Complete runnable source code.
- Demo video.
- Markdown user guide and project manifest.
- Verified end-to-end on Ubuntu + L40S/RTX 5090 with Docker.

---

## Project Structure

```
carla-longest6-eval/
├── README.md                        # This file
├── project_manifest.md              # Quick reference of all components
├── docs/
│   ├── user_guide.md                # Full step-by-step user guide
│   ├── code_map.md                  # Architecture and code mapping
│   ├── args_reference.md            # Explanation of args-1.txt settings
│   ├── mmcv_compatibility_notes.md  # mmcv fix documentation
│   └── model_setup.md               # Model placement and switching guide
├── infra/
│   ├── README_environment.md        # Environment baseline checklist
│   ├── carla_run_notes.md           # CARLA runtime notes
│   └── networking.md                # Container networking explanation
├── docker/
│   ├── carla/
│   │   ├── Dockerfile               # CARLA image (extends official)
│   │   ├── docker-compose.yml       # CARLA service definition
│   │   ├── entrypoint.sh            # CARLA container entrypoint
│   │   └── scripts/
│   │       └── wait_for_carla.sh    # Wait for CARLA readiness
│   └── client/
│       ├── Dockerfile               # Evaluation client image
│       ├── requirements.txt         # Python dependencies
│       ├── requirements_fixed.txt   # Pinned working versions
│       ├── entrypoint.sh            # Client container entrypoint
│       └── scripts/
│           └── wait_for_carla.sh    # Wait for CARLA server
├── config/
│   ├── longest6_config.yaml         # Longest6 evaluation configuration
│   ├── camera.yaml                  # Camera sensor configuration
│   └── models.yaml                  # Model registry with paths
├── schemas/
│   └── metrics_schema.json          # Metrics output JSON schema
├── scripts/
│   ├── run_longest6_eval.py         # Main evaluation entrypoint
│   ├── generate_video_from_frames.sh# FFmpeg video generation
│   ├── run_carla_docker.sh          # Helper to start CARLA
│   ├── run_client_docker.sh         # Helper to start client
│   └── quickstart_longest6.sh       # Full pipeline quickstart
├── transfuser/
│   ├── transfuser-2022/             # Official TransFuser code (extracted)
│   └── bi-attenfusion/              # CrossViT-Fusion code for model50.pth
├── models/
│   ├── transfuser_official/
│   │   └── model.pth                # Official TransFuser checkpoint
│   └── crossvit_50/
│       └── model50.pth              # CrossViT-Fusion checkpoint
├── config_internal/
│   └── raw_args/
│       └── args-1.txt               # Original training args reference
├── outputs/
│   ├── logs/
│   │   ├── client/                  # Client runtime logs
│   │   └── carla/                   # CARLA server logs
│   ├── eval_logs/
│   │   └── longest6/                # Per-tick/per-episode logs
│   ├── metrics/
│   │   └── longest6/                # Final metrics (JSON + TXT)
│   ├── frames/
│   │   └── longest6/                # Captured video frames
│   └── videos/
│       └── longest6/                # Generated demo videos
└── tools/
    └── helper_scripts/              # Small utilities and validators
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url> carla-longest6-eval
cd carla-longest6-eval

# 2. Download and place model files
# Download from: https://drive.google.com/drive/folders/1yG9LbVtSLaneHKlB5GL5Vhzz5Miue5Me?usp=sharing
# Place transfuser-2022.zip in transfuser/ and extract
# Place model.pth in models/transfuser_official/
# Place model50.pth in models/crossvit_50/

# 3. Start CARLA server
cd docker/carla && docker-compose up -d

# 4. Build and run evaluation client
cd ../client && docker-compose run --rm client \
  --model-path /app/models/transfuser_official/model.pth

# 5. View results
ls outputs/metrics/longest6/
ls outputs/videos/longest6/
```

---

## Deliverables

| # | Deliverable | Description | Phase |
|---|-------------|-------------|-------|
| 1 | Demo video | MP4 video of a Longest6 evaluation run | Phase 3.1 |
| 2 | User guide | Markdown guide (.md) for installation and usage | Phase 3.3 |
| 3 | Source code | Complete runnable code with Docker setup | Phase 1-3 |
| 4 | Metrics output | Route Completion, Driving Score, violation metrics | Phase 3.2 |

---

## Timeline

**Delivery Window:** June 30, 2026 – July 5, 2026 (5 days)

- **Day 1-2:** Phase 1 — Environment setup, Docker infra, baseline verification
- **Day 3:** Phase 2.1-2.2 — Codebase mapping, mmcv compatibility fix
- **Day 4:** Phase 2.3-2.4 — Evaluation script, model selection
- **Day 5:** Phase 3 — Video generation, metrics, documentation, packaging

---

## Key Technical Challenges

1. **mmcv GPU compatibility:** The official mmcv used by TransFuser doesn't support RTX 50-series (Blackwell architecture). Solution involves rebuilding mmcv from source, swapping to torch-native ops, or pinning to a compatible version triad.
2. **Headless rendering:** CARLA running in Docker on a headless server requires `-RenderOffScreen` flag or EGL configuration for off-screen rendering.
3. **Docker networking:** Two containers (CARLA server + evaluation client) must communicate reliably via Docker compose networking.
4. **Model architecture mismatch:** The official TransFuser model and CrossViT-Fusion `model50.pth` use different backbone architectures, requiring flexible model loading.
