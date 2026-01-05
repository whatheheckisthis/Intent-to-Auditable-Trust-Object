#!/bin/bash
set -e

# Generate TLS certs first
./generate_certs.sh

# Start full Docker Compose stack
docker-compose -f compose/docker-compose.yml up -d

echo "IATO stack started successfully with certificates mounted."
