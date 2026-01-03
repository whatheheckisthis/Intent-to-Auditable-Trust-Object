#!/usr/bin/env python3
"""
config/key_generation_engine.py

Generates a secure API key, creates an HMAC for integrity, and encrypts the key using OpenSSL AES-256-CBC.
Includes verification of HMAC before use.
"""

import os
import subprocess
from pathlib import Path
import getpass
import hmac
import hashlib

# Create keys folder if it doesn't exist
KEY_DIR = Path("config/keys")
KEY_DIR.mkdir(parents=True, exist_ok=True)

def generate_hmac(raw_key: str, secret: str) -> str:
    """
    Generate HMAC for the given key using a secret passphrase.
    Returns a hex digest.
    """
    return hmac.new(secret.encode(), raw_key.encode(), hashlib.sha256).hexdigest()

def generate_key(filename: str):
    """
    Generates a 32-byte random key, creates HMAC, and encrypts it using AES-256-CBC.
    """
    key_path = KEY_DIR / f"{filename}.enc"
    hmac_path = KEY_DIR / f"{filename}.hmac"
    passphrase = getpass.getpass("Enter passphrase for encryption and HMAC: ")

    # Generate raw random key
    raw_key = subprocess.check_output(["openssl", "rand", "-base64", "32"]).decode().strip()
    
    # Generate and save HMAC
    hmac_digest = generate_hmac(raw_key, passphrase)
    with open(hmac_path, "w") as f:
        f.write(hmac_digest)
    print(f"[✓] HMAC saved to {hmac_path}")

    # Save raw key temporarily
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
    print(f"[✓] API key generated and encrypted at {key_path}")

    return key_path, hmac_path

def verify_hmac(raw_key: str, secret: str, hmac_file: Path) -> bool:
    """
    Verifies HMAC integrity of a raw key.
    """
    with open(hmac_file, "r") as f:
        stored_hmac = f.read().strip()
    calc_hmac = generate_hmac(raw_key, secret)
    return hmac.compare_digest(calc_hmac, stored_hmac)

def decrypt_key(enc_file: Path):
    """
    Decrypts the encrypted key and verifies HMAC.
    """
    passphrase = getpass.getpass("Enter passphrase for decryption: ")

    # Decrypt key
    raw_key = subprocess.check_output([
        "openssl", "enc", "-d", "-aes-256-cbc",
        "-in", str(enc_file),
        "-pass", f"pass:{passphrase}"
    ]).decode().strip()
    
    # Verify HMAC
    hmac_file = KEY_DIR / f"{enc_file.stem}.hmac"
    if verify_hmac(raw_key, passphrase, hmac_file):
        print("[✓] HMAC verified. Key integrity OK.")
    else:
        print("[✗] HMAC verification failed! Key may have been tampered.")
    
    return raw_key

if __name__ == "__main__":
    choice = input("Do you want to (G)enerate a key or (D)ecrypt a key? [G/D]: ").strip().upper()
    if choice == "G":
        filename = input("Enter filename for API key: ").strip()
        if filename:
            generate_key(filename)
        else:
            print("[✗] No filename provided, exiting.")
    elif choice == "D":
        enc_file_path = input("Enter path to encrypted key (.enc): ").strip()
        if enc_file_path and Path(enc_file_path).exists():
            decrypt_key(Path(enc_file_path))
        else:
            print("[✗] Invalid file path, exiting.")
    else:
        print("[✗] Invalid choice, exiting.")
