#!/usr/bin/env python3
import subprocess
import sys
import os

def run(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        print(f"Command succeeded: {cmd}\n{out}")
        return out
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {cmd}\n{e.output}")
        return None

def check_env():
    # Check conda info and which python
    print("=== Conda Info ===")
    run("conda info")
    print("=== Which Python ===")
    run("which python")
    print("=== Python Version ===")
    run("python --version")

def check_tools():
    print("=== Checking flake8 ===")
    out = run("flake8 --version")
    if out is None:
        print("[ERROR] flake8 is not available in the environment.")
        sys.exit(1)
    print("=== Checking pytest ===")
    out = run("pytest --version")
    if out is None:
        print("[ERROR] pytest is not available in the environment.")
        sys.exit(1)

def check_env_file(env_file='environment.yml'):
    if not os.path.exists(env_file):
        print(f"[WARNING] {env_file} does not exist.")
        return
    print(f"=== Validating {env_file} ===")
    result = run(f"conda env update --file {env_file} --dry-run --name testenv")
    if result is None:
        print(f"[ERROR] Dependency resolution failed for {env_file}.")
        sys.exit(1)
    print("Environment file validated successfully.")

if __name__ == "__main__":
    check_env()
    check_tools()
    check_env_file()
    print("Conda environment and tools verified successfully.")
- name: Verify conda environment and tools
  shell: bash -l {0}
  run: python .github/scripts/verify_conda_env.py
