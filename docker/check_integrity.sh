#!/bin/bash
set -e

CERT_DIR="./nginx/ssl"
KEY_FILE="$CERT_DIR/iato.key"
CRT_FILE="$CERT_DIR/iato.crt"

echo "Verifying TLS certificates..."

# Check that files exist
if [[ ! -f "$KEY_FILE" || ! -f "$CRT_FILE" ]]; then
    echo "ERROR: TLS certificate or key not found in $CERT_DIR"
    exit 1
fi

# Check that certificate matches the private key
openssl x509 -noout -modulus -in "$CRT_FILE" | openssl md5 > /tmp/crt_md5.txt
openssl rsa -noout -modulus -in "$KEY_FILE" | openssl md5 > /tmp/key_md5.txt

if ! cmp -s /tmp/crt_md5.txt /tmp/key_md5.txt; then
    echo "ERROR: Certificate and private key do not match!"
    exit 1
fi

rm /tmp/crt_md5.txt /tmp/key_md5.txt

echo "Certificate integrity verified."

# Optional: verify Docker volumes exist
echo "Checking Docker volumes..."
for vol in redis-data kafka-data grafana-data trust-logs; do
    if ! docker volume inspect "$vol" &>/dev/null; then
        echo "WARNING: Docker volume '$vol' does not exist. It will be created automatically on stack start."
    fi
done

echo "Integrity check complete. You can safely start the stack with ./start_stack.sh"
