#!/usr/bin/env python3
import os
import subprocess
import secrets

# -----------------------------
# Utility Functions
# -----------------------------
def write_file(path, content):
    with open(path, "w") as f:
        f.write(content.strip() + "\n")
    print(f"[+] Created {path}")

def run_cmd(cmd):
    print(f"[~] Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# -----------------------------
# Generate Config Files
# -----------------------------

# requirements.txt
requirements = """
jax
shap
lime
numpy
scipy
pytest
flake8
"""
write_file("requirements.txt", requirements)

# environment.yml (Conda)
environment_yml = """
name: ops-utilities-env
channels:
  - conda-forge
dependencies:
  - python=3.11
  - jax
  - shap
  - lime
  - numpy
  - scipy
  - pytest
  - flake8
  - pip
  - pip:
      - some-extra-pypi-package
"""
write_file("environment.yml", environment_yml)

# setup.cfg
setup_cfg = """
[flake8]
max-line-length = 88
exclude = .git,__pycache__,env,venv

[tool:pytest]
testpaths = tests
"""
write_file("setup.cfg", setup_cfg)

# pytest.ini
pytest_ini = """
[pytest]
minversion = 6.0
addopts = -ra -q
testpaths = tests
"""
write_file("pytest.ini", pytest_ini)

# .env with auto-generated API_KEY
api_key = secrets.token_hex(16)
env_example = f"""
API_KEY={api_key}
MODEL_PATH=path/to/model.pt
DEBUG=false
"""
write_file(".env", env_example)

# .gitignore
gitignore = """
__pycache__/
*.pyc
*.pyo
*.pyd
tests/
.git/
.env
.env.*
AUTO_CHANGELOG.md
"""
write_file(".gitignore", gitignore)

# Makefile
makefile = """
.PHONY: setup test lint

setup:
\tpip install -r requirements.txt

test:
\tpytest

lint:
\tflake8 src/ tests/
"""
write_file("Makefile", makefile)

# tests/test_entropy_pgd.py
os.makedirs("tests", exist_ok=True)
test_entropy = """
from src.kernel.entropy_pgd import pgd_optimize

def test_quadratic_minimum():
    def f(x):
        return (x - 2) ** 2
    result = pgd_optimize(f, 0.0, {"learning_rate": 0.1, "num_steps": 25})
    assert abs(result - 2.0) < 1e-2
"""
write_file("tests/test_entropy_pgd.py", test_entropy)

# Config README
os.makedirs("config", exist_ok=True)
config_readme = """
# Config Folder

This folder contains environment and configuration files:

- **requirements.txt** → Pip dependencies
- **environment.yml** → Conda environment file
- **setup.cfg** → Flake8 and pytest config
- **pytest.ini** → Pytest options
- **.env** → Environment variables (auto-generated API_KEY)
- **Makefile** → Simple automation commands
- **tests/** → Example unit test for PGD optimizer

Run the setup script once to generate and install everything.
"""
write_file("config/README.md", config_readme)

# -----------------------------
# Install Dependencies
# -----------------------------
print("\n[+] Installing dependencies...")
try:
    run_cmd("pip install -r requirements.txt")
except Exception:
    print("[!] Pip installation failed. Try manually or use conda: conda env create -f environment.yml")

print("\n✅ Setup complete! API_KEY saved in .env")
