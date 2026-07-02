# Code Map — TransFuser + CrossViT-Fusion

## Architecture

```
transfuser/
├── transfuser-2022/          # Official TransFuser codebase
│   ├── team_code/
│   │   ├── run_evaluation.py         # Evaluation entrypoint
│   │   ├── run_transfuser.py         # Training entrypoint
│   │   ├── agent_runner.py           # Agent + sensor setup
│   │   ├── metrics.py                # Metrics computation
│   │   └── ...
│   ├── model/
│   │   ├── transfuser.py             # TransFuser backbone
│   │   ├── fusion.py                 # Fusion modules
│   │   └── ...
│   └── data/
│
└── bi-attenfusion/           # CrossViT-Fusion code (for model50.pth)
    ├── crossvit_fusion.py            # CrossViT backbone
    ├── model.py                      # Model wrapper
    └── ...
```

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Eval entrypoint | `transfuser-2022/team_code/run_evaluation.py` | Main evaluation script |
| Agent | `transfuser-2022/team_code/agent_runner.py` | CARLA agent with sensor I/O |
| TransFuser model | `transfuser-2022/model/transfuser.py` | Official TransFuser model |
| CrossViT model | `bi-attenfusion/crossvit_fusion.py` | CrossViT-Fusion backbone |
| Metrics | `transfuser-2022/team_code/metrics.py` | Metric computation |
| Args parser | `transfuser-2022/team_code/args_parser.py` | Configuration args |

## Data Flow

1. CARLA streams sensor data (RGB, LiDAR, GNSS, IMU) to client.
2. Agent runs model forward pass on sensor data to predict waypoints.
3. Controller converts waypoints to vehicle controls (steer, throttle, brake).
4. Metrics module tracks Route Completion, Driving Score, violations.
5. Results saved as JSON + TXT; optionally rendered as video.
