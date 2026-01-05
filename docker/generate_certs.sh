#!/bin/bash
set -e

mkdir -p ./nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:4096 \
  -keyout ./nginx/ssl/iato.key \
  -out ./nginx/ssl/iato.crt \
  -subj "/CN=localhost"

echo "TLS certificates generated at ./nginx/ssl"

