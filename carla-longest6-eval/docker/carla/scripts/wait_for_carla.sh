#!/bin/bash
set -e

HOST=${1:-carla-server}
PORT=${2:-2000}
TIMEOUT=${3:-120}

echo "Waiting for CARLA at $HOST:$PORT..."

for i in $(seq 1 $TIMEOUT); do
  if nc -z "$HOST" "$PORT" 2>/dev/null; then
    echo "CARLA is ready after ${i}s"
    exit 0
  fi
  sleep 1
done

echo "TIMEOUT: CARLA did not become ready after ${TIMEOUT}s"
exit 1
