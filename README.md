# Intent-to-Auditable-Trust-Object (IATO)

IATO is a deterministic security assurance project designed to transform intent into auditable trust evidence through repeatable controls, telemetry, and policy-driven workflows.

## Multi-Step Security Architecture

### Step 1: Define control intent and trust boundaries
- Identify business-critical systems, data classifications, and trust zones.
- Map control objectives to:
  - **Essential Eight** maturity practices (hardening, patching, privilege restriction, MFA, backups).
  - **SOC 2** trust service criteria (Security, Availability, Processing Integrity, Confidentiality, Privacy).
- Create system boundaries for applications, supporting services, and third-party integrations.

### Step 2: Establish IAM governance as a first-class control plane
- Implement role-based access control (RBAC) and least privilege for users, services, and automation identities.
- Enforce lifecycle controls:
  - Joiner/Mover/Leaver provisioning and deprovisioning.
  - Credential rotation and secrets handling.
  - Break-glass access with approval and immutable audit trail.
- Require strong authentication (MFA where applicable) and policy-backed authorization.

### Step 3: Build telemetry foundations (logs, metrics, traces)
- Standardize structured logging at application, infrastructure, and security layers.
- Expose service and platform metrics for availability, latency, error rate, and saturation.
- Instrument distributed traces for request-path visibility across dependencies.
- Protect telemetry integrity with retention, access controls, and tamper-evident storage where required.

### Step 4: Implement monitoring (reactive)
**Monitoring answers: _"What is failing?"_**
- Define predefined, threshold-based metrics and SLO/SLI alerts (for example CPU, memory, 5xx rate, queue lag, cert expiry).
- Route alerts to on-call workflows with severity, ownership, and escalation policy.
- Use runbooks for deterministic triage and incident response.

### Step 5: Implement observability (proactive + investigative)
**Observability answers: _"Why is it failing?"_**
- Correlate logs, metrics, and traces to identify causal paths and hidden system interactions.
- Support exploratory investigation beyond predefined dashboards.
- Use high-cardinality dimensions and trace context to isolate blast radius and root cause quickly.

### Step 6: Close the assurance loop with evidence and audits
- Continuously map telemetry and control outputs to Essential Eight and SOC 2 control evidence.
- Capture:
  - Control execution logs.
  - IAM decision records.
  - Incident timelines and corrective actions.
- Produce auditable trust artifacts from reproducible pipelines.

---

## Repository Layout

- `config/` — environment defaults, runtime/security settings, and setup helpers.
- `scripts/` — core research and orchestration scripts.
- `tests/` — validation code, schemas, and sample data.
- `docs/` — project documentation and notes.
- `docker/` — local container and observability stack assets.
- `ci/` — CI checks and environment validation scripts.
- `bin/` — archived/legacy helper files not used in the primary workflow.

## Docker Images and Dev Environment Dependencies

The repository includes container assets for local execution and observability testing. For a PHP-based web control plane that supports both Apache and NGINX reverse proxy patterns, use the following baseline dependencies.

### PHP + Apache base image (application runtime)

```dockerfile
FROM php:8.2-apache

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    unzip \
    libzip-dev \
    libicu-dev \
    libonig-dev \
    libxml2-dev \
    && docker-php-ext-install pdo pdo_mysql intl zip \
    && a2enmod rewrite headers ssl status \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /var/www/html
COPY . /var/www/html
```

### NGINX sidecar/reverse proxy dependency
- `nginx:alpine` for request routing and static content acceleration in local development.
- Bind-mounted configuration from `docker/nginx/php-observability.nginx.conf` for PHP upstream routing.
- Route upstream traffic to the PHP/Apache service.

### Real workload compose example (`docker/compose/php-observability-stack.yml`)

```yaml
version: "3.9"

networks:
  iato-net:
    driver: bridge

volumes:
  redis-data:
  prometheus-data:
  grafana-data:

services:
  php-apache:
    build:
      context: ../..
      dockerfile: docker/php-apache.Dockerfile
    container_name: iato-php-apache
    environment:
      APP_ENV: dev
      APP_DEBUG: "true"
      REDIS_HOST: redis
    volumes:
      - ../../:/var/www/html
      - ../../docker/src/99-local.ini:/usr/local/etc/php/conf.d/99-local.ini:ro
      - ../../docker/src/status.conf:/etc/apache2/conf-enabled/status.conf:ro
    expose:
      - "80"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/server-status?auto"]
      interval: 30s
      timeout: 5s
      retries: 3
    networks:
      - iato-net

  nginx:
    image: nginx:alpine
    container_name: iato-nginx
    depends_on:
      php-apache:
        condition: service_healthy
    ports:
      - "8080:80"
    volumes:
      - ../../docker/nginx/php-observability.nginx.conf:/etc/nginx/nginx.conf:ro
      - ../../docker/nginx/ssl:/etc/nginx/ssl:ro
    networks:
      - iato-net

  redis:
    image: redis:7-alpine
    container_name: iato-redis
    command: ["redis-server", "--appendonly", "yes"]
    volumes:
      - redis-data:/data
    networks:
      - iato-net

  prometheus:
    image: prom/prometheus:v2.54.1
    container_name: iato-prometheus
    command: ["--config.file=/etc/prometheus/prometheus.yml", "--storage.tsdb.path=/prometheus"]
    ports:
      - "9090:9090"
    volumes:
      - ../../docker/rules/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    networks:
      - iato-net

  grafana:
    image: grafana/grafana:11.1.4
    container_name: iato-grafana
    depends_on:
      - prometheus
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana-data:/var/lib/grafana
    networks:
      - iato-net
```

Run it with:

```bash
docker compose -f docker/compose/php-observability-stack.yml up --build
```

## Getting Started

1. Create the Python environment:
   - `conda env create -f environment.yml`
2. Activate it:
   - `conda activate testenv`
3. Run repository checks:
   - `bash ci/tools/run_all_checks.sh`
4. (Optional) Start container stack:
   - `docker compose -f docker/compose/docker-compose.yml up --build`

## Scope Note

This repository includes both active workflows and historical artifacts. Prioritize `config/`, `scripts/`, `tests/`, `docs/`, `docker/`, and `ci/tools/` for current implementation paths.
