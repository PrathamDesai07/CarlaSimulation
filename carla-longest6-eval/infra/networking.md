# Container Networking

## Architecture

The CARLA server and evaluation client run in **separate Docker compose files** that share a single **external Docker network** (`carla-longest6-net`). This allows both containers to communicate via service hostname while keeping the compose files independent.

```
Host (L40S / RTX 5090)
│
├── docker/carla/docker-compose.yml
│   └── service: carla (carla-server:2000)
│       ├── ports: 2000-2002 → host (accessible from host)
│       └── network: carla-longest6-net
│
├── docker/client/docker-compose.yml
│   └── service: client (transfuser-client)
│       └── network: carla-longest6-net (external)
│
└── Network: carla-longest6-net (bridge, created on first `up`)
```

## Why a Shared Network?

Both compose files declare the same network name (`carla-longest6-net`). The CARLA compose creates it with `driver: bridge`. The client compose declares it as `external: true`, meaning it expects the network to already exist.

This way:
- **`docker/carla/docker-compose up`** creates the network and starts CARLA
- **`docker/client/docker-compose run`** joins the same network and resolves `carla-server` via DNS

## Network Creation

The helper scripts (`run_carla_docker.sh`, `run_client_docker.sh`, `quickstart_longest6.sh`) automatically create the network if it doesn't exist:

```bash
docker network create carla-longest6-net --driver bridge
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CARLA_HOST` | `carla-server` | CARLA container hostname |
| `CARLA_PORT` | `2000` | CARLA RPC port |
| `MODEL_PATH` | `/app/models/transfuser_official/model.pth` | Model checkpoint path |

## Port Mapping

| Port | Mapping | Usage |
|------|---------|-------|
| 2000 | host:2000 → container:2000 | CARLA RPC (main) |
| 2001 | host:2001 → container:2001 | Streaming (sensors) |
| 2002 | host:2002 → container:2002 | Secondary |

## Running Without Compose

```bash
# Create network
docker network create carla-longest6-net

# Start CARLA
docker run -d --gpus all --network carla-longest6-net \
    --name carla-server -p 2000-2002:2000-2002 \
    carla-longest6:latest \
    ./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-rpc-port=2000

# Run client
docker run --rm --gpus all --network carla-longest6-net \
    -e CARLA_HOST=carla-server \
    -e CARLA_PORT=2000 \
    -e MODEL_PATH=/app/models/transfuser_official/model.pth \
    -v $(pwd)/models:/app/models \
    transfuser-client:latest
```

## Troubleshooting

```bash
# Check network exists
docker network ls | grep carla-longest6-net

# Inspect connected containers
docker network inspect carla-longest6-net

# Test connectivity from client container
docker run --rm --network carla-longest6-net alpine nc -zv carla-server 2000

# Recreate network (if needed)
docker network rm carla-longest6-net
docker network create carla-longest6-net --driver bridge
```
