#!/bin/bash
set -e

echo "Starting Prometheus..."
exec /bin/prometheus \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/prometheus
