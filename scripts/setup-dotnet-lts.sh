#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${HOME}/.dotnet"
INSTALL_SCRIPT="/tmp/dotnet-install.sh"
SCRIPT_URL="https://dot.net/v1/dotnet-install.sh"
DOTNET_VERSION="${DOTNET_VERSION:-8.0.404}"
PYTEST_VERSION="${PYTEST_VERSION:-9.0.2}"

echo "[dotnet-setup] target dotnet version: ${DOTNET_VERSION}"
echo "[dotnet-setup] expected pytest version: ${PYTEST_VERSION}"
echo "[dotnet-setup] downloading installer from ${SCRIPT_URL}"
curl -fsSL "${SCRIPT_URL}" -o "${INSTALL_SCRIPT}"

echo "[dotnet-setup] installing ${DOTNET_VERSION} to ${INSTALL_DIR}"
bash "${INSTALL_SCRIPT}" --version "${DOTNET_VERSION}" --install-dir "${INSTALL_DIR}"

if ! grep -q 'DOTNET_ROOT="$HOME/.dotnet"' "${HOME}/.bashrc"; then
  cat >> "${HOME}/.bashrc" <<'BASHRC_EOF'

# Added by scripts/setup-dotnet-lts.sh
export DOTNET_ROOT="$HOME/.dotnet"
export PATH="$DOTNET_ROOT:$DOTNET_ROOT/tools:$PATH"
BASHRC_EOF
fi

export DOTNET_ROOT="${INSTALL_DIR}"
export PATH="${DOTNET_ROOT}:${DOTNET_ROOT}/tools:${PATH}"

echo "[dotnet-setup] dotnet version:"
dotnet --version

echo "[dotnet-setup] pytest version:"
pytest --version
