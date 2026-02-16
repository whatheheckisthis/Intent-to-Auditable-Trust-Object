#!/bin/bash

echo "🔧 Starting Dev Tool Setup..."

# Determine OS type
if grep -qi microsoft /proc/version; then
    echo "🪟 Detected WSL"
    IS_WSL=true
else
    echo "🐧 Detected Linux"
    IS_WSL=false
fi

# Update and upgrade
echo "📦 Updating package lists..."
sudo apt-get update -y && sudo apt-get upgrade -y

# List of tools to install
TOOLS=(curl wget git vim nano htop unzip openssh-client build-essential python3 python3-pip)

echo "📦 Installing common dev tools..."
for tool in "${TOOLS[@]}"; do
    if ! command -v $tool &> /dev/null; then
        echo "➕ Installing $tool..."
        sudo apt-get install -y $tool
    else
        echo "✅ $tool already installed."
    fi
done

# Confirm Python and pip
echo "🐍 Python version:"
python3 --version

echo "📦 pip version:"
pip3 --version

# Optional: WSL install instructions for Windows (run from PowerShell/.bat, not Bash)
if [[ "$IS_WSL" = false && "$OSTYPE" == "msys" ]]; then
    echo "⚠️ This script should be run from within WSL or Linux."
    echo "If you're on Windows and want to install WSL, run this in PowerShell as admin:"
    echo 'wsl --install'
fi

echo "✅ Development tooling setup complete!"
