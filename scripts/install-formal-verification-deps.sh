#!/usr/bin/env bash
set -euo pipefail

# Installs toolchain components used by Armv9 CCA formal models:
# - TLC model checker (as tlc wrapper around tla2tools.jar)
# - Coq compiler (coqc)

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please run as root (or with sudo)." >&2
  exit 1
fi

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

echo "[1/4] Ensuring apt metadata is current..."
apt-get update -y

echo "[2/4] Installing runtime dependencies (Java + Coq)..."
apt-get install -y default-jre-headless coq curl ca-certificates

TLA_JAR="/usr/local/lib/tla2tools.jar"
TLC_WRAPPER="/usr/local/bin/tlc"

mkdir -p /usr/local/lib

echo "[3/4] Installing TLC (tla2tools.jar)..."
# Latest stable artifact published by the TLA+ project.
curl -fL "https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar" -o "$TLA_JAR"
chmod 0644 "$TLA_JAR"

cat > "$TLC_WRAPPER" <<'WRAP'
#!/usr/bin/env bash
set -euo pipefail
exec java -cp /usr/local/lib/tla2tools.jar tlc2.TLC "$@"
WRAP
chmod 0755 "$TLC_WRAPPER"

echo "[4/4] Verifying installations..."
tlc -version
coqc --version

echo "Formal verification dependencies installed successfully."
