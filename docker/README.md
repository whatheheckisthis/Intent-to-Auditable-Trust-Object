# IATO Kernel — Unified Lawful Inference

This repository hosts the **Intent-to-Auditable-Trust-Object (IATO) kernel**, a containerized framework for deterministic, auditable inference in regulated or adversarial environments. It focuses on **trust-object propagation, audit logging, and secure multi-device computation**.

## Features

* **Oscillatory Inference Loops** — PGD updates with Bayesian feedback and RL-style adjustments.
* **Trust-Object Compliance** — Per-packet cryptographically signed audit artifacts (`HMAC/AES-256`) with RBAC and rate-limited access.
* **Entropy-Guided Exploration** — Controlled stochastic injections, bounded by `H_max`.
* **Multi-Device Execution** — Parallelized across CPUs, GPUs, or TPUs using JAX `pjit` / `mesh_utils`.
* **Explainability** — Inline per-cycle SHAP/LIME outputs for regulated domains.
* **Adaptive Resource Governance** — GPU, memory, and bandwidth dynamically adjusted per workload.
* **Cryptographic & Tamper Evidence** — Logs, DAG state, and Monte Carlo residual risk aggregates auditable via trust-object packets.
* **Mechanized Verification** — Optionally integrated with Isabelle/HOL for formal proof of invariants.

---

## Supported Platforms

| Platform      | Notes                                                                            |
| ------------- | -------------------------------------------------------------------------------- |
| `linux/amd64` | Standard x86_64 desktop/server/VM                                                |
| `linux/arm64` | ARM server, Apple M1/M2, Raspberry Pi 4+                                         |
| Multi-arch    | Built via **QEMU + Docker Buildx** for reproducible cross-architecture execution |

---

## Base Image & Utilities

* Ubuntu 22.04 LTS (slim)
* Python 3.11-slim
* Core utilities: `bash`, `openssl`, `curl`, `ca-certificates`
* Optional: `jq`, `git`, `vim` (for debugging / PoC environments)

This base ensures **secure, minimal, reproducible builds** optimized for cryptographic integrity and containerized workloads.

---

## Building the Docker Image

Enable multi-arch builds and QEMU support:

```bash
# Enable Buildx
docker buildx create --use

# Register QEMU emulation
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

# Build & push multi-arch image
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -t iato/kernel:latest \
    -f docker/Dockerfile \
    --push .
```

> This produces images runnable on both x86_64 and ARM architectures without modification.

---

## Stack Composition

The IATO Docker Compose stack includes:

* **iato-kernel** — Core inference engine
* **Redis** — DAG state storage and Monte Carlo aggregation
* **Kafka + Zookeeper** — Batch streaming of trust-object packets
* **Prometheus** — Metrics collection
* **Grafana** — Dashboarding and visualization
* **Nginx** — Reverse proxy and TLS termination
* **Node-exporter** — Host-level metrics for adaptive resource governance

All services are **networked on a dedicated bridge** and expose only necessary ports.

---

## Workflow Scripts

| Script               | Purpose                                                                            |
| -------------------- | ---------------------------------------------------------------------------------- |
| `generate_certs.sh`  | Generates TLS certificates for Nginx and intra-stack encryption                    |
| `start_stack.sh`     | Mounts certificates and runs Docker Compose                                        |
| `integrity_check.sh` | Validates Kafka topics, Redis DAG state, Prometheus targets, and trust-object logs |

---

## Configuration

### Prometheus

* `monitoring/prometheus.yml` → scrape targets for iato-kernel, Redis, Kafka, node-exporter
* `rules/` → Prometheus alerting rules

### Grafana

* `grafana/grafana.yml` → preconfigured dashboards for trust-object monitoring, DAG states, Monte Carlo aggregates

### Nginx

* `nginx/nginx.yml` → TLS termination and reverse proxy to iato-kernel

---

## Operational Notes

* **Execution Component**

  ```
  θₜ₊₁ = θₜ − ηₜ ∇𝓛(θₜ) + Bayesian feedback + RL-style adjustments
  ```

  Iterative, deterministic closed-loop inference; PGD updates respect KKT constraints and entropy bounds.

* **Resource Governance**

  ```
  resource_allocation = f(GPU, memory, bandwidth; workload)
  ```

  Adaptive allocation ensures reproducible PoC → production pipelines.

* **Explainability**

  ```
  explain_batch(θₜ) via SHAP/LIME per cycle
  ```

  Per-step interpretability for regulatory compliance and debugging.

* **Trust-Object Logging**
  Cryptographically bound per-packet artifacts, streamed via Kafka, stored in Redis DAG states.

---
