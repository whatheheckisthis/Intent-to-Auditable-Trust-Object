#!/usr/bin/env python3
"""
scripts/hmac_verify.py
Verifies HMAC integrity for an encrypted key.
"""

import argparse
import hmac
import hashlib
from pathlib import Path
import subprocess
import getpass

def verify_hmac(enc_file: Path, hmac_file: Path):
    passphrase = getpass.getpass("Enter passphrase for HMAC verification: ")
    
    # Decrypt the key to a temporary string
    raw_key = subprocess.check_output([
        "openssl", "enc", "-d", "-aes-256-cbc",
        "-in", str(enc_file),
        "-pass", f"pass:{passphrase}"
    ]).decode().strip()

    given_hmac = hmac_file.read_text().strip()
    computed_hmac = hmac.new(passphrase.encode(), raw_key.encode(), hashlib.sha256).hexdigest()

    if hmac.compare_digest(given_hmac, computed_hmac):
        print("[✓] HMAC verified successfully.")
        return True
    else:
        print("[✗] HMAC verification failed!")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify HMAC for encrypted API key.")
    parser.add_argument("enc_file", type=Path, help="Path to the .enc file")
    parser.add_argument("hmac_file", type=Path, help="Path to the .hmac file")
    args = parser.parse_args()

    verify_hmac(args.enc_file, args.hmac_file)
