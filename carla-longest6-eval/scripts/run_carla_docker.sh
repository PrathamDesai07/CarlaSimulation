#!/bin/bash
# Start CARLA server Docker container
# Usage: ./run_carla_docker.sh [up|down|restart]

set -e

CMD="${1:-up}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_DIR="$(cd "$SCRIPT_DIR/../docker/carla" && pwd)"
NETWORK_NAME="carla-longest6-net"

ensure_network() {
    docker network inspect "$NETWORK_NAME" &>/dev/null || \
        docker network create "$NETWORK_NAME" --driver bridge
}

case "$CMD" in
    up)
        echo "Starting CARLA server..."
        cd "$COMPOSE_DIR"
        docker compose up -d
        echo "CARLA server started. Port: 2000 (mapped)"
        ;;
    down)
        echo "Stopping CARLA server..."
        cd "$COMPOSE_DIR"
        docker compose down
        echo "CARLA server stopped. Network '$NETWORK_NAME' preserved for client use."
        ;;
    restart)
        "$0" down
        "$0" up
        ;;
    logs)
        cd "$COMPOSE_DIR"
        docker compose logs -f
        ;;
    *)
        echo "Usage: $0 [up|down|restart|logs]"
        exit 1
        ;;
esac
