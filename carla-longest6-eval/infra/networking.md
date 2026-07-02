# Container Networking

## Architecture

```
┌──────────────────────┐      ┌──────────────────────────┐
│  CARLA Server        │      │  Evaluation Client       │
│  (carla-server:2000) │◄────►│  (transfuser-client)     │
│  CarlaUE4            │      │  run_longest6_eval.py    │
│  -RenderOffScreen    │      │                          │
└──────────────────────┘      └──────────────────────────┘
        │                               │
        └─────────── carla-net ─────────┘
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CARLA_HOST` | `carla-server` | CARLA container hostname |
| `CARLA_PORT` | `2000` | CARLA RPC port |
| `MODEL_PATH` | `/app/models/transfuser_official/model.pth` | Model checkpoint path |
