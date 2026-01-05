#!/bin/bash
set -e

echo "Running certificate integrity check..."
./check_integrity.sh

echo "Starting Docker Compose stack..."
docker compose -f compose/docker-compose.yml up -d
