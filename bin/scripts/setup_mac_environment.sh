#!/bin/bash

echo "🍺 Starting macOS Dev Environment Setup..."

# Check for Homebrew and install if not found
if ! command -v brew &> /dev/null; then
    echo "🔧 Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✅ Homebrew is already installed."
fi

# Update Homebrew
echo "🔄 Updating Homebrew..."
brew update

# Core CLI tools to install
BREW_PACKAGES=(git python3 openssl wget curl htop vim nano zsh)

echo "📦 Installing core packages..."
for pkg in "${BREW_PACKAGES[@]}"; do
    if ! brew list $pkg &> /dev/null; then
        echo "➕ Installing $pkg..."
        brew install $pkg
    else
        echo "✅ $pkg is already installed."
    fi
done

# Check Python & pip
echo "🐍 Python version:"
python3 --version

echo "📦 pip version:"
pip3 --version

# Prompt to create a Python virtual environment
read -p "📦 Do you want to create a Python virtual environment here? (y/n): " CREATE_VENV
if [[ "$CREATE_VENV" == "y" ]]; then
    echo "📁 Creating 'venv' directory..."
    python3 -m venv venv
    echo "✅ Virtual environment created. Activate with: source venv/bin/activate"
fi

echo "🚀 macOS Dev Environment is ready!"
