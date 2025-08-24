#!/bin/bash

echo "Activating testenv and installing required Python packages..."
conda activate testenv
conda install numpy flake8 pytest -c conda-forge -y
echo "✅ Environment is now fully equipped."
