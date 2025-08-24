#!/bin/bash
#
# scripts/clean_workspace.sh
#
# Purpose:
#   Remove build artifacts, Python caches, log archives, and temp files.
#
# Usage:
#   bash scripts/clean_workspace.sh
#
# Safe for: Local development cleanup before commits or packaging

echo "Cleaning workspace..."

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete

# Remove build artifacts
rm -rf dist/ build/ *.egg-info/

# Remove logs and archives
rm -rf logs/archive/*
rm -rf *.log

# Remove temporary files
find . -type f -name "*~" -delete

echo "✅ Workspace cleaned."

# End of scrip
