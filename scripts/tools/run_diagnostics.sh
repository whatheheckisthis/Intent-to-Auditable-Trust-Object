#!/bin/bash
#
# scripts/run_diagnostics.sh
#
# Purpose:
#   Run basic diagnostic checks on the local development environment.
#
# Checks:
#   - Python version
#   - Conda environment status
#   - Disk space and memory availability
#   - Internet connectivity
#
# Usage:
#   bash scripts/run_diagnostics.sh

echo "🔍 Running local environment diagnostics..."

# Check Python
echo "Python version:"
python --version || echo "Python not found"

# Check Conda
echo "Conda info:"
conda info || echo "Conda not found or not initialized"

# Disk space
echo "Disk space:"
df -h .

# Memory
echo "Memory usage:"
free -h || echo "free command not available"

# Internet check
echo "Internet connectivity (ping google.com):"
ping -c 2 google.com > /dev/null && echo "Internet OK" || echo "No internet connection"

echo "✅ Diagnostics complete."

# End of script
