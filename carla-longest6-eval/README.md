# CARLA Longest6 Evaluation — Autonomous Driving with TransFuser

## Environment

| Component | Detail |
|-----------|--------|
| **Server** | Cloud GPU instance (NVIDIA L4, 24 GB VRAM) |
| **OS** | Ubuntu Linux (headless) |
| **GPU Driver** | NVIDIA 580.159.03, CUDA 13.0 |
| **Container Runtime** | Docker 28.0.1 with NVIDIA Container Toolkit |
| **CARLA** | 0.9.16 running in Docker container (headless, off-screen) |
| **Client** | Separare Docker container with PyTorch 2.8.0 + TransFuser |
| **Hostname** | `carla-longest6-eval` project root |

Two Docker containers on a shared bridge network (`carla-longest6-net`):

```
┌─────────────────┐     TCP 2000-2002     ┌──────────────────┐
│  CARLA Server   │ ◄──────────────────►  │  Client Runner   │
│  carla-longest6  │    8000-8002 (TM)    │  transfuser:latest│
│  (headless)     │                      │  (GPU: L4)       │
└─────────────────┘                      └──────────────────┘
```

## What This Project Does

Evaluates an autonomous driving agent on the **Longest6 benchmark** in **CARLA 0.9.16** using the **TransFuser** neural network architecture. The Longest6 benchmark consists of 36 predefined driving routes across multiple CARLA towns, testing the agent's ability to navigate complex urban scenarios.

The pipeline:
1. CARLA server renders the simulation world headlessly
2. Client container runs the TransFuser model, which outputs steering/throttle/brake commands
3. The agent drives through each of the 36 routes
4. Metrics are collected (route completion %, driving score, infractions)
5. Camera frames from the run are stitched into a demo MP4 video

## The Issues

### Issue 1: Video Shows No Car Body (Fixed)

**What happened:** The video was capturing frames from `rgb_front` — a dashcam-style camera mounted at `(x=1.3, y=0, z=2.3)` on the car's hood pointing forward. Since it's a forward-facing camera on the car body, the car itself is never visible in the frame. The resulting video looks like a camera flying through the world with no visible vehicle.

**Fix:** Added a third-person chase camera (`rgb_chase`) positioned behind and above the car at `(x=-6.0, y=0.0, z=4.0)` with `pitch=-30°`, looking down at the car. The frame capture now reads from this camera instead of the dashcam. Also had to increase `MAX_ALLOWED_RADIUS_SENSOR` from 3.0 to 10.0 in `agent_wrapper.py` because the chase camera's mounting position exceeds the original validation radius.

### Issue 2: Car Keeps Crashing and Gets Stuck Forever (Diagnosed, Fix In Progress)

**Root cause — two bugs in the safety system (`submission_agent.py` + `config.py`):**

#### Bug A: Asymmetric Safety Box (detects obstacles on right side only)

The safety box filters LiDAR points in a coordinate system where `y *= -1` is applied first. The original bounds were:

```python
safety_box_y_min = -3.0   # y_inverted in [-3.0, 0.0]
safety_box_y_max = 0.0    # → original y in [0.0, 3.0] (right side only!)
```

After the `y *= -1` inversion, `y_min=-3, y_max=0` means only points with `y_inverted` in that range pass — which corresponds to original y values between 0 and +3 (the right side of the car only). Obstacles straight ahead (y≈0) or on the left (y<0) fall through undetected.

The x-range was also too narrow for useful obstacle detection:
```python
safety_box_x_min = -1.066  # barely past LiDAR position  
safety_box_x_max = 1.066   # only ~2.4m detection zone
```

#### Bug B: Emergency Stop Only Triggers During Stuck Creep (Death Spiral)

The emergency stop logic is:
```python
if (emergency_stop == True) and (is_stuck == True):  # only when already stuck!
    control.throttle = 0.0
    control.brake = True
```

This means during **normal driving**, the car never brakes for obstacles — it just crashes into them. Then when it gets stuck, the stuck detector triggers creep mode, AND THEN the safety box fires and brakes again. This creates a death spiral:

```
Car stops (stuck detector starts counting)
  → stuck_detector > threshold → creep mode activated
    → PID controller sets throttle=0.5, creep forward
      → safety box detects obstacle → emergency_stop=True → brake=1.0
        → car stops → stuck_detector stays high
          → creep activates again → brakes again → repeat forever
```

**Result:** The car either crashes into things during normal driving, or gets permanently stuck oscillating between creep and brake.

## What We Changed

| File | Change | Status |
|------|--------|--------|
| `submission_agent.py` | Added `rgb_chase` sensor (chase camera for video) | ✅ Done |
| `submission_agent.py` | Frame capture reads from `rgb_chase` instead of `rgb_front` | ✅ Done |
| `agent_wrapper.py` | `MAX_ALLOWED_RADIUS_SENSOR`: 3.0 → 10.0 (for chase cam) | ✅ Done |
| `config.py` | `safety_box_y_max`: 0.0 → 3.0 (symmetric: both sides) | 🔧 Pending |
| `config.py` | `safety_box_x_max`: 1.066 → 5.0 (longer reaction distance) | 🔧 Pending |
| `submission_agent.py` | Move emergency_stop check to fire during normal driving too | 🔧 Pending |
| `config.py` | `creep_duration`: 30 → 80 frames (4s creep) | 🔧 Pending |

## How to Run

```bash
# 1. Start CARLA server
cd docker/carla && docker-compose up -d

# 2. Run evaluation (generates metrics + video)
cd ../.. && bash scripts/quickstart_longest6.sh models/transfuser_official/model.pth

# 3. View results
ls outputs/metrics/longest6/
ls outputs/videos/longest6/
```

## Model

Currently using the official TransFuser checkpoint at `models/transfuser_official/model.pth` (backbone: `transFuser`, image/lidar architecture: `regnety_032`).

## Project Structure

```
carla-longest6-eval/
├── README.md
├── scripts/           # Evaluation entrypoints and helpers
├── config/            # YAML configs (evaluation, camera, models)
├── docker/
│   ├── carla/         # CARLA server container
│   └── client/        # Evaluation client container
├── transfuser/        # TransFuser source code
├── models/            # Model checkpoints (.pth)
└── outputs/
    ├── frames/        # Video frame PNGs
    ├── videos/        # Demo MP4 videos
    └── metrics/       # JSON + TXT results
```
