#!/bin/bash
set -e

CERT_DIR="./nginx/ssl"
KEY_FILE="$CERT_DIR/iato.key"
CRT_FILE="$CERT_DIR/iato.crt"

echo "Verifying TLS certificates..."
if [[ ! -f "$KEY_FILE" || ! -f "$CRT_FILE" ]]; then
    echo "ERROR: TLS certificate or key not found."
    exit 1
fi

openssl x509 -noout -modulus -in "$CRT_FILE" | openssl md5 > /tmp/crt_md5.txt
openssl rsa -noout -modulus -in "$KEY_FILE" | openssl md5 > /tmp/key_md5.txt
if ! cmp -s /tmp/crt_md5.txt /tmp/key_md5.txt; then
    echo "ERROR: Certificate and key mismatch!"
    exit 1
fi
rm /tmp/crt_md5.txt /tmp/key_md5.txt
echo "Certificate integrity verified."

# Optional: check Docker volumes
for vol in redis-data kafka-data grafana-data trust-logs; do
    docker volume inspect "$vol" &>/dev/null || echo "Volume $vol will be created automatically."
done

echo "Integrity check complete."
