#!/bin/bash
# Attach to running guided-transfer2.5 container with a new bash shell

if docker ps --format '{{.Names}}' | grep -q '^guided-transfer2.5$'; then
    echo "Attaching to guided-transfer2.5 container..."
    docker exec -it guided-transfer2.5 bash
else
    echo "Error: guided-transfer2.5 container is not running"
    echo ""
    echo "Start it first with:"
    echo "  ./run_guided_transfer2.5_docker.sh"
    exit 1
fi
