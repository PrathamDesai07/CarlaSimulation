#!/bin/bash
set -e

echo "Starting CARLA 0.9.16 in headless mode..."
echo "Command: $@"

exec "$@"
