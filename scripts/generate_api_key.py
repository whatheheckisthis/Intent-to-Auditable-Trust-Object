#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
import getpass

KEY_DIR = Path("config/keys")
KEY_DIR.mkdir(parents=True, exist_ok=True)

def generate_key(filename: str):
    key_path = KEY_DIR / f"{filename}.enc"
    passphrase = getpass.getpass("Enter passphrase for encryption: ")
    subprocess.run([
        "openssl", "rand", "-base64", "32"
    ], capture_output=True, text=True, check=True)
    # Write raw key
    raw_key = subprocess.check_output(["openssl", "rand", "-base64", "32"]).decode().strip()
    with open("temp_key.txt", "w") as f:
        f.write(raw_key)
    # Encrypt key
    subprocess.run([
        "openssl", "enc", "-aes-256-cbc", "-salt",
        "-in", "temp_key.txt",
        "-out", str(key_path),
        "-pass", f"pass:{passphrase}"
    ], check=True)
    os.remove("temp_key.txt")
    print(f"[✓] API key generated and stored at {key_path}")

if __name__ == "__main__":
    key_name = input("Enter API key filename: ")
    generate_key(key_name)