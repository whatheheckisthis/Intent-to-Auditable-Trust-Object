#!/bin/bash
set -e

CERT_DIR="./nginx/ssl"
mkdir -p "$CERT_DIR"

echo "Generating self-signed TLS certs for IATO..."
openssl req -x509 -nodes -days 365 \
    -newkey rsa:2048 \
    -keyout "$CERT_DIR/iato.key" \
    -out "$CERT_DIR/iato.crt" \
    -subj "/C=NZ/ST=Auckland/L=Auckland/O=IATO/CN=localhost"

echo "Certificates generated in $CERT_DIR"
