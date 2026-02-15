# Docker assets

This directory contains container resources for local development and observability support.

## Includes

- `Dockerfile` and compose files for local services.
- `nginx/`, `grafana/`, and `rules/` configuration used by the local stack.
- helper scripts (`setup_stack.sh`, `check_integrity.sh`, `generate_certs.sh`).

## Usage

From repository root:

```bash
docker compose -f docker/docker-compose.yml up --build
```

## Scope

Docker resources here are for reproducible local execution and diagnostics.
