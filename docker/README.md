# Docker Environment for OPS Utilities

This Docker setup provides a streamlined, reproducible environment for deploying and managing the OPS Utilities kernel, including secure Bash operations, modular Python inference modules, and PoC pipelines. It is designed to facilitate experimentation, development, and safe deployment without modifying the host system.

## Base Image

* Ubuntu 22.04 LTS
* Python 3.11-slim
* Essential system utilities: `bash`, `openssl`, `curl`, `ca-certificates`
* Optimized for secure, reproducible Python execution and containerized workflow automation.

*(Note: Unlike other legacy projects, this container does **not** include Apache/PHP unless explicitly added for legacy compatibility.)*

## Configuration

* The container mounts key project folders for live development:

  * `config/` → `/app/config`
  * `scripts/` → `/app/scripts`
  * `data/` → `/app/data`

* Entrypoint script (`docker/entrypoint.sh`) ensures:

  * Key directories are initialized (`keys`, `data`)
  * The container is ready for interactive usage or running scripts directly
  * Compliance with the root repo’s operational structure

* Dependency management uses the root `config/requirements.txt` for Python packages.

## Usage

```bash
# Build and start container
docker compose -f docker/docker-compose.yml up -d

# Enter interactive shell
docker exec -it ops-core bash

# Run scripts (example: generate encrypted API keys)
python scripts/generate_api_key.py
```

## Highlights

* **Reproducibility:** Ensures a consistent environment across Linux, macOS (via Docker Desktop), or Windows (via WSL2).
* **Separation of Concerns:** Keeps Docker assets isolated in `docker/` while mounting root repo folders.
* **Extensibility:** Easily integrates with CI/CD, GPU-enabled containers, or additional services in the future.
* **Security**: All API keys and trust objects can be generated and stored in encrypted form within the container.

| Feature                              | Description                                                                                                                  | Example / Code Snippet                                                                                        |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **File Integrity & Tamper Evidence** | Bash utilities and Python scripts maintain file integrity and tamper-evident logs.                                           | `bash\n# Generate encrypted API key\npython scripts/generate_api_key.py\n`                                    |
| **Minimal Base Image**               | Uses Ubuntu 22.04 LTS slim image with only essential services to reduce attack surface.                                      | `dockerfile\nFROM ubuntu:22.04\nRUN apt-get update && apt-get install -y bash openssl curl ca-certificates\n` |
| **Secure Mounting**                  | Supports mounting host directories securely for sensitive data while maintaining container isolation.                        | `yaml\nvolumes:\n  - ./config:/app/config:ro\n  - ./data:/app/data:rw\n`                                      |
| **Auditability**                     | Operations are traceable to root repo folders (`scripts/`, `config/`) with logs inside container directories.                | `bash\n# Example: tail audit logs\ntail -f /app/logs/operation.log\n`                                         |
| **Workflow Integration**             | Fully compatible with CI/CD pipelines and test scripts for reproducible execution of PGD cycles and trust-object validation. | `bash\n# Run pre-configured tests inside container\npytest tests/\n`                                          |


