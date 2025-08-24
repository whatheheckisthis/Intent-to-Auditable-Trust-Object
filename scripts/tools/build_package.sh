#!/bin/bash
#
# scripts/build_package.sh
#
# Purpose:
#   Build a Python package from source using `setuptools` and validate its structure.
#
# Usage:
#   bash scripts/build_package.sh
#
# Requirements:
#   - pyproject.toml or setup.py must exist
#   - Python 3.7+, pip, and build module installed

echo "Building package..."

if [ ! -f "pyproject.toml" ] && [ ! -f "setup.py" ]; then
  echo "No build configuration found (pyproject.toml or setup.py missing)."
  exit 1
fi

# Ensure build module is available
python -m pip install --upgrade build > /dev/null

# Clean previous builds
rm -rf dist/ build/

# Build the package
python -m build

# Verify contents
echo "Contents of dist/:"
ls -lh dist/

echo "✅ Package build complete."

# End of script
