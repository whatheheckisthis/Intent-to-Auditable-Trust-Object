#!/bin/bash
# ------------------------------------------------------------------------------
# Script: run_all_checks.sh
# Location: ci/
#
# Description:
# Full CI environment validation script.
# Runs system tool checks, Conda validation, environment activation,
# and runtime Python import verification.
#
# Usage:
# bash ci/run_all_checks.sh
# ------------------------------------------------------------------------------

set -e  # Exit on any error

echo "Running system tool checks..."

# List of required CLI tools
TOOLS=(openssl git pandoc python3)

for tool in "${TOOLS[@]}"; do
  if ! command -v "$tool" &> /dev/null; then
    echo "ERROR: $tool not found in PATH"
    exit 1
  else
    echo "$tool is installed"
  fi
done

# Create directories if needed
mkdir -p files logs
echo "Directories ensured: files/, logs/"

echo "Checking for Conda..."

if ! command -v conda &> /dev/null; then
  echo "ERROR: Conda is not available"
  exit 1
fi

# Show Conda info
conda info
conda list

# Activate environment (assumes 'testenv' is created via setup-miniconda)
echo "Activating Conda environment: testenv"
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate testenv

# Confirm Python version
echo "Python version:"
python --version

echo "Verifying runtime imports..."

python - <<EOF
required_imports = [
    "os",
    "sys",
    "json",
    "datetime",
    "pathlib",
    "numpy",
    "pandas"
]

failures = []

for module in required_imports:
    try:
        __import__(module)
        print(f"OK: {module}")
    except ImportError:
        print(f"FAIL: {module} not found")
        failures.append(module)

if failures:
    print("\nMissing imports:", failures)
    exit(1)
else:
    print("\nAll required imports are available.")
EOF

echo "All CI checks passed."


