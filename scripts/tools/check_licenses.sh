#!/bin/bash
#
# scripts/check_licenses.sh
#
# Purpose:
#   Print license information for all installed Python packages.
#
# Usage:
#   bash scripts/check_licenses.sh
#
# Dependencies:
#   - pip-licenses (install via pip)
#
# Notes:
#   This is useful for compliance review before packaging or distribution.

echo "Checking licenses of installed Python packages..."

# Ensure tool is installed
if ! command -v pip-licenses &> /dev/null; then
  echo "pip-licenses not found. Installing it now..."
  pip install pip-licenses
fi

# Display license summary
pip-licenses --from=mixed --format=markdown --with-authors --with-urls --with-license-file

echo "✅ License check complete."

# End of script
