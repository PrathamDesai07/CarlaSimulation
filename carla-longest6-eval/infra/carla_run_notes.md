# CARLA Run Notes

## Starting CARLA

```bash
cd docker/carla && docker-compose up -d
bash scripts/run_carla_docker.sh up
```

## Stopping CARLA

```bash
cd docker/carla && docker-compose down
bash scripts/run_carla_docker.sh down
```

## Flags

- `-RenderOffScreen` — Run without display (headless servers)
- `-quality-level=Low` — Lower quality for faster eval; use `Epic` for better video
- `-carla-rpc-port=2000` — RPC port

## Ports

| Port | Usage |
|------|-------|
| 2000 | CARLA RPC (main) |
| 2001 | Streaming (sensors) |
| 2002 | Secondary |

## Troubleshooting

- If CARLA crashes: `docker logs carla-server`
- Empty sensor data: ensure `-RenderOffScreen` is set
- GPU OOM: lower `-quality-level` or sensor resolution
