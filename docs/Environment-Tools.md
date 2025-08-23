



### Terminals & Shells

#### **Bash – Primary Shell (Unix-like systems)**

Bash is the default shell on virtually all Linux distributions and supports extensive scripting capabilities.

* Bash powers automation and script execution in your environment.
* On Ubuntu or Debian, it's typically pre-installed. If not, you can install it via:

  ````bash
  sudo apt-get install bash
  ``` :contentReference[oaicite:1]{index=1}
  ````

#### **Zsh / Oh-My-Zsh – Optional Enhanced Shell (All OS)**

Zsh is a more interactive shell, often enhanced with **Oh-My-Zsh** for productivity improvements like themes, plugins, and auto-completion.



#### **WSL2 Terminal (Windows)**

On Windows, you’ll need to install WSL2 to use proper Linux shells and tooling:

* Enable and install WSL2 with one command:

  ```powershell
  wsl --install
  ```

  This installs the default Ubuntu distro. [learn.microsoft.com][3]
* For a modern terminal experience, Windows Terminal is highly recommended:

  * Supports tabs, Unicode, GPU-accelerated rendering, and more. [learn.microsoft.com][4], [Wikipedia][5]

---

### Summary Table

| Component            | Purpose & Installation Summary                                                                                           |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Bash**             | Default shell for Unix-like systems; pre-installed or installable via package manager [ioflood.com][6], [Wikipedia][1] |
| **Zsh / Oh-My-Zsh**  | Enhanced interactive shell with themes/plugins; install from Zsh or Oh-My-Zsh websites [Medium][2]                    |
| **WSL2 (Windows)**   | Runs Linux environment on Windows (Ubuntu default); install via `wsl --install` [learn.microsoft.com][3]               |
| **Windows Terminal** | Advanced terminal app for Windows with support for WSL, tabs, themes [learn.microsoft.com][4], [Wikipedia][5]         |

---



[1]: https://en.wikipedia.org/wiki/Bash_%28Unix_shell%29?utm_source=chatgpt.com "Bash (Unix shell)"
[2]: https://klauswong.medium.com/setup-guide-for-a-cool-terminal-on-windows-with-windows-terminal-and-wsl-d692fa8bdbac?utm_source=chatgpt.com "2022 Setup Guide for a cool terminal on Windows with ... - Klaus"
[3]: https://learn.microsoft.com/en-us/windows/wsl/install?utm_source=chatgpt.com "How to install Linux on Windows with WSL"
[4]: https://learn.microsoft.com/en-us/windows/wsl/setup/environment?utm_source=chatgpt.com "Set up a WSL development environment"
[5]: https://en.wikipedia.org/wiki/Windows_Terminal?utm_source=chatgpt.com "Windows Terminal"
[6]: https://ioflood.com/blog/install-bash-shell-linux/?utm_source=chatgpt.com "How to Install Bash in Linux: A Step-by-Step Guide"

---

#### 2. Resource Management

Resource management scripts help **monitor and optimize CPU, memory, GPU, and I/O usage** during oscillatory inference loops and PGD computations.

* **htop / atop** – real-time CPU/memory usage.
* **nvidia-smi** – GPU status and utilization.
* **watch / glances** – monitoring resource statistics in intervals.

*Purpose:* Ensures oscillatory loops and multi-device sharding run within optimal bounds, preventing crashes or starvation.

---

#### 3. Cmdlet & Script Management

Scripts in this category handle **installation, updates, environment readiness, and auxiliary automation**.

* `init_environment.sh` – sets up environment directories, makes scripts executable, and validates Conda/Python dependencies.
* `generate_auto_changelog.sh` – auto-generates changelog.
* `cleanup_changelog.sh` – cleans duplicates in logs.
* `backup_repo.sh` – copies repositories for reproducibility.
* `download_dependencies.sh` – installs all required Python packages, Conda env, and optional system utilities.

*Purpose:* Provides a **single source of truth** for dependencies, environment setup, and workflow automation, making onboarding and experimentation frictionless.

---

### Example Script: `download_dependencies.sh`

```bash
#!/bin/bash
# download_dependencies.sh
# Installs all required Python packages, Conda env, and essential system tools

echo "[*] Updating system packages..."
sudo apt update && sudo apt upgrade -y

echo "[*] Installing essential Linux utilities..."
sudo apt install -y git curl wget unzip htop atop ufw openssl

echo "[*] Installing Python & Conda..."
# Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
export PATH="$HOME/miniconda3/bin:$PATH"
conda init bash

echo "[*] Creating Conda environment..."
conda create -y -n kernel_env python=3.10
conda activate kernel_env

echo "[*] Installing Python packages..."
pip install --upgrade pip
pip install jax jaxlib numpy scipy matplotlib pandas scikit-learn shap lime cryptography scapy flake8 pylint mypy

echo "[*] Optional GPU support (if NVIDIA GPU present)..."
pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html || echo "Skipping GPU installation"

echo "[*] All dependencies installed successfully."
```

---

✅ **Key Notes:**

* This script can be extended to install additional **documentation, monitoring, or network analysis tools** automatically.
* Students or researchers can **run this script once** to ensure reproducible environment setup across multiple machines.
* It complements the existing `init_environment.sh`, creating a **full-stack dependency installation workflow**.

---


### Recommended Cybersecurity & Environment Tools

| Tool | Purpose | Install / Link |
|------|---------|----------------|
| **Miniconda** | Python environment manager | [Miniconda](https://docs.conda.io/en/latest/miniconda.html) |
| **Python 3.10+** | Required Python version for JAX and PGD kernels | [Python Downloads](https://www.python.org/downloads/) |
| **pip / pip-tools** | Manage additional Python dependencies | `sudo apt install python3-pip`<br>[pip Docs](https://pip.pypa.io/en/stable/) |
| **flake8** | Code linting | `pip install flake8`<br>[flake8 Docs](https://flake8.pycqa.org/en/latest/) |
| **pylint** | Advanced linting & code quality | `pip install pylint`<br>[pylint Docs](https://pylint.pycqa.org/en/latest/) |
| **bandit** | Python security linter | `pip install bandit`<br>[bandit Docs](https://bandit.readthedocs.io/en/latest/) |
| **mypy** | Optional static type checking | `pip install mypy`<br>[mypy Docs](https://mypy.readthedocs.io/en/stable/) |
| **git** | Version control | `sudo apt install git`<br>[git Docs](https://git-scm.com/doc) |
| **GPG / SSH Keys** | Signed commits & secure repo access | [GPG](https://gnupg.org/documentation/) / [SSH Keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) |
| **OpenSSL** | File encryption/decryption | `sudo apt install openssl`<br>[OpenSSL Docs](https://www.openssl.org/docs/) |
| **Python hashlib / cryptography** | Inline hashing & cryptographic operations | `pip install cryptography`<br>[cryptography Docs](https://cryptography.io/en/latest/) |
| **htop / atop** | CPU/memory monitoring | `sudo apt install htop atop`<br>[htop](https://htop.dev/) |
| **nvidia-smi** | GPU monitoring (NVIDIA) | [NVIDIA Docs](https://developer.nvidia.com/nvidia-system-management-interface) |
| **ufw / iptables** | Firewall for Linux | `sudo apt install ufw`<br>[ufw Docs](https://help.ubuntu.com/community/UFW) |
| **Wireshark** | Inspect residual traffic streams | `sudo apt install wireshark`<br>[Wireshark Docs](https://www.wireshark.org/docs/) |
| **tcpdump** | Command-line packet capture | `sudo apt install tcpdump`<br>[tcpdump Docs](https://www.tcpdump.org/) |
| **Scapy** | Python-based packet manipulation | `pip install scapy`<br>[Scapy Docs](https://scapy.readthedocs.io/en/latest/) |
| **Sphinx / MkDocs** | Project documentation | `pip install sphinx mkdocs`<br>[Sphinx Docs](https://www.sphinx-doc.org/) / [MkDocs Docs](https://www.mkdocs.org/) |



---

# Linux (Ubuntu/Debian)

```bash
# Update system
sudo apt update && sudo apt upgrade -y  

# Install Bash (already default on Ubuntu/Debian)
sudo apt install bash -y  

# Install Zsh
sudo apt install zsh -y  

# Install Git (needed for Oh-My-Zsh)
sudo apt install git -y  

# Install Oh-My-Zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

---

# macOS

```bash
# Install Homebrew (package manager if not installed)
if ! command -v brew &> /dev/null
then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install Bash (latest version)
brew install bash  

# Install Zsh (macOS comes with Zsh, this ensures latest)
brew install zsh  

# Install Git
brew install git  

# Install Oh-My-Zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

---

# Windows (PowerShell + WSL2)

**Step 1 – Enable WSL2**

```powershell
# Run in PowerShell as Administrator
wsl --install
```

➡️ This installs **WSL2** with default Ubuntu. Restart PC after.

**Step 2 – Inside WSL Ubuntu Terminal**

```bash
sudo apt update && sudo apt upgrade -y  
sudo apt install bash zsh git -y  

# Install Oh-My-Zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

**Step 3 – Optional: Windows Terminal**
Download from Microsoft Store:
👉 [https://aka.ms/terminal](https://aka.ms/terminal)

---

# Arch / Manjaro (extra for Arch users)

```bash
# Update system
sudo pacman -Syu  

# Install Bash (default already)
sudo pacman -S bash  

# Install Zsh
sudo pacman -S zsh git curl wget  

# Install Oh-My-Zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

---

👉 Links for reference:

* [Bash](https://www.gnu.org/software/bash/)
* [Zsh](https://www.zsh.org/)
* [Oh-My-Zsh](https://ohmyz.sh/)
* [Windows Terminal](https://aka.ms/terminal)
* [WSL2 Installation Guide](https://learn.microsoft.com/en-us/windows/wsl/install)

---


