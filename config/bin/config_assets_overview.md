Overview


The config/ folder contains configuration, environment, and security resources required to run and maintain the Ops Utilities & 18.X Kernel repository. These files manage environment setup, static configuration, API key management, and cryptographic operations.

Purpose:
	•	Centralize environment and workflow configurations.
	•	Store encrypted keys and trust-object metadata.
	•	Provide reproducible and auditable operational settings.
	•	Serve as a sandbox for secure and lawful experimentation.

⸻

Directory Structure

config/
├── environment.yml           # Conda environment definition for reproducibility
├── keys/                     # Encrypted API keys and HMAC verification files
├── setup_config.py           # Script to bootstrap environment, install dependencies
├── key_generation_engine.py  # Generates secure API keys and HMACs
├── flake8_config.ini         # Linting rules for Python code
├── pytest.ini                # Testing configuration
├── .env.example              # Example environment variables file
└── README.md                 # Folder overview


⸻

Key Components

File / Folder	Description
environment.yml	Conda environment file specifying Python version and key packages (JAX, SHAP, LIME, NumPy, SciPy, pytest, flake8). Ensures reproducible environments across systems.
keys/	Stores encrypted API keys and associated HMAC files generated via key_generation_engine.py. Each key is tamper-evident and auditable.
setup_config.py	Script to bootstrap dependencies, prepare folders, and verify environment readiness. Can install OS-specific dependencies.
key_generation_engine.py	Generates API keys, creates HMAC for integrity, and encrypts keys using OpenSSL AES-256-CBC. Includes key decryption and verification routines.
flake8_config.ini	Configures Python linting rules to ensure code quality and consistency.
pytest.ini	Centralized configuration for pytest, specifying test paths and options.
.env.example	Template for environment variables; includes placeholders for API keys, model paths, and debug flags.


⸻

Usage

1. Environment Setup

conda env create -f config/environment.yml
conda activate ops-utilities-env

2. Bootstrap Config

python config/setup_config.py

This prepares folders, installs additional dependencies if required, and verifies environment integrity.

3. Generate an API Key

python config/key_generation_engine.py

	•	Follow prompts to generate a secure, encrypted API key.
	•	HMAC is automatically created to ensure key integrity.

4. Decrypt and Verify an API Key

python config/key_generation_engine.py

	•	Select decrypt option and provide the .enc file path.
	•	Passphrase is required to decrypt and verify HMAC integrity.

⸻

Notes
	•	The config/ folder is central to reproducibility and security. Do not move or rename files without updating scripts that reference them.
	•	Encrypted keys are tamper-evident; always store the passphrase securely.
	•	Use .env files to manage environment-specific variables. Do not commit secret keys directly to the repository.

⸻

Quick Start: Initialize the Environment

1. Clone the Repository

git clone https://github.com/whatheheckisthis/ops-utilities.git
cd ops-utilities

2. Create the Conda Environment

conda env create -f config/environment.yml
conda activate ops-utilities-env

Update dependencies later with:

conda env update -f config/environment.yml --prune

3. Install Pip Dependencies

pip install -r requirements.txt

4. Run Setup Script

python config/setup_config.py

This script:
	•	Verifies Python & Conda versions
	•	Ensures required folders (config/keys/, config/logs/, config/output/) exist
	•	Checks for OpenSSL installation
	•	Prints next steps

⸻

Key Generation

Generate secure API keys with OpenSSL:

python config/key_generation_engine.py

	•	Prompts for a passphrase
	•	Generates a 32-byte random key
	•	Encrypts it with AES-256-CBC (OpenSSL)
	•	Saves under config/keys/
	•	Creates a .hmac integrity file

⸻

Key Verification & Decryption

After generating a key, verify and decrypt it.

1. Verify HMAC Signature

python scripts/hmac_verify.py config/keys/generated_key.enc config/keys/generated_key.hmac

Expected output:

[✓] HMAC signature verified successfully.


⸻

2. Decrypt the Key (OpenSSL CLI)

openssl enc -aes-256-cbc -d -in config/keys/generated_key.enc -out config/keys/generated_key.txt

It will ask for the same passphrase you set during key generation.
If correct, the decrypted key will appear in generated_key.txt.

⸻

3. Decrypt Programmatically (Python Snippet)

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import hashlib
import base64

# Example: load encrypted key + derive AES key from passphrase
with open("config/keys/generated_key.enc", "rb") as f:
    encrypted_key = f.read()

passphrase = b"my_secret_passphrase"
aes_key = hashlib.sha256(passphrase).digest()

cipher = Cipher(algorithms.AES(aes_key), modes.CBC(b"\x00"*16), backend=default_backend())
decryptor = cipher.decryptor()
decrypted_key = decryptor.update(encrypted_key) + decryptor.finalize()

print("Decrypted key:", base64.b64encode(decrypted_key).decode())


⸻

Folder Structure

config/
│
├── environment.yml        # Conda environment spec
├── setup_config.py        # Bootstrap script
├── key_generation_engine.py # Secure key generator
├── *.conf                 # App configuration files
├── keys/                  # Encrypted keys + HMAC signatures
├── logs/                  # Runtime logs
└── output/                # Generated files, reports


⸻
