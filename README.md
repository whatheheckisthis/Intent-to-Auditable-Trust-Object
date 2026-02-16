# Auditable Trust Object: Enclave-Secured Local Stack

This repository now includes a complete Docker Compose environment that couples a private Drupal app stack with an Enclave-secured monitoring plane.

## Architecture Overview

- **Enclave Core** (`enclave_agent`) provides overlay connectivity and secure naming (for example `grafana.enclave` and `drupal.enclave`).
- **Host-visibility sidecar** (`node-exporter`) runs in the Enclave network namespace and reads host `/proc` and `/sys` for auditable host metrics.
- **Monitoring plane** (`prometheus` + `grafana`) runs behind Enclave networking and scrapes `localhost:9100` from Node Exporter.
- **Gateway plane** (`nginx`) terminates TLS on port `443` and routes:
  - `https://grafana.enclave` → Grafana (`grafana:3000`)
  - `https://drupal.enclave` → Drupal via Apache (`apache:8080`)
- **Private app stack** (`drupal`, `apache`, `mariadb`) is isolated on `app_net` and not directly exposed to the host.

## Files Included

- `docker-compose.yml`
- `prometheus.yml`
- `nginx.conf`

## Quick Start (WSL2)

1. **Prerequisites**
   - Docker Desktop with WSL2 integration enabled.
   - A valid Enclave enrolment key.

2. **Prepare environment file**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set:
   - `ENCLAVE_ENROLMENT_KEY`
   - `DRUPAL_DB_PASSWORD`
   - `MARIADB_ROOT_PASSWORD`
   - `GRAFANA_ADMIN_PASSWORD`

3. **Create TLS certificate for the Enclave Gateway**

   ```bash
   mkdir -p certs
   openssl req -x509 -nodes -newkey rsa:2048 -days 365 \
     -keyout certs/tls.key \
     -out certs/tls.crt \
     -subj "/CN=grafana.enclave"
   ```

4. **Start the full stack**

   ```bash
   docker compose up -d --build
   ```

## Verification Steps

### 1) Verify Enclave enrolment

```bash
docker logs enclave_agent
```

Look for successful enrolment and overlay connectivity messages.

### 2) Verify monitoring endpoints inside Enclave namespace

```bash
docker exec enclave_agent wget -qO- http://localhost:9100/metrics | head
```

```bash
docker exec enclave_agent wget -qO- http://localhost:9090/-/ready
```

### 3) Verify browser access through Enclave DNS

Open:

- `https://grafana.enclave`

(accept the self-signed certificate warning unless you provision a trusted cert).

### 4) Verify the Trust Object using Node Exporter dashboard

On first Grafana login:

1. Sign in with `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD`.
2. Confirm the **Prometheus** datasource is already provisioned at `http://localhost:9090`.
3. Open the pre-provisioned **Node Exporter Full** dashboard.
4. Confirm host-level panels (CPU, memory, filesystem, load) are populated.

If panels populate immediately after startup, your auditable host telemetry path is active:

**Host kernel metrics → Node Exporter sidecar → Prometheus scrape → Grafana dashboard (Trust Object evidence surface).**

## Stop and cleanup

```bash
docker compose down -v
```
