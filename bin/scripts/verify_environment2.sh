#!/bin/bash
# Check for tools, Conda, and Python environment setup

TOOLS=(openssl git pandoc python3)

for tool in "${TOOLS[@]}"; do
  command -v "$tool" &> /dev/null && echo "$tool ✅" || echo "$tool ❌"
done

conda info && conda activate testenv || exit 1
python ci/verify_imports.py || exit 1


# Ensure core utilities are present
for pkg in git curl unzip jq; do
  if ! command -v $pkg &> /dev/null; then
    echo "Installing $pkg..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
      brew install $pkg
    else
      sudo apt-get install -y $pkg
    fi
  fi
done

echo "Environment setup complete."
