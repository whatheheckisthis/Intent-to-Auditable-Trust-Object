#!/usr/bin/env python3
"""
scripts/generate_api_key.py
Generates a secure API key and stores it encrypted using OpenSSL AES-256-CBC.
"""

import os
import subprocess
from pathlib import Path
import getpass

# Create keys folder if it doesn't exist
KEY_DIR = Path("config/keys")
KEY_DIR.mkdir(parents=True, exist_ok=True)

def generate_key(filename: str):
    """
    Generates a 32-byte random key and encrypts it.
    """
    key_path = KEY_DIR / f"{filename}.enc"
    passphrase = getpass.getpass("Enter passphrase for encryption: ")

    # Generate raw random key
    raw_key = subprocess.check_output(["openssl", "rand", "-base64", "32"]).decode().strip()
    
    # Save temporarily
    temp_file = "temp_key.txt"
    with open(temp_file, "w") as f:
        f.write(raw_key)

    # Encrypt using AES-256-CBC
    subprocess.run([
        "openssl", "enc", "-aes-256-cbc", "-salt",
        "-in", temp_file,
        "-out", str(key_path),
        "-pass", f"pass:{passphrase}"
    ], check=True)

    os.remove(temp_file)
    print(f"[✓] API key generated and saved to {key_path}")

if __name__ == "__main__":
    filename = input("Enter filename for the API key: ").strip()
    if filename:
        generate_key(filename)
    else:
        print("[✗] No filename provided, exiting.")
