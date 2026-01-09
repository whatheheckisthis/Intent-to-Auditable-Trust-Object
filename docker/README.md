
# IATO Kernel — Deterministic Lawful Inference Engine

This repository hosts the **Intent-to-Auditable-Trust-Object (IATO) Kernel**:
a **deterministic inference and enforcement system** designed for **regulated, adversarial, and sovereign environments**.

IATO does **not** rely on probabilistic convergence, stochastic stabilization, or post-hoc justification.
All execution is gated by **pre-proven admissibility**.

> **Inference is permitted only if it is lawful, stable, and auditable *before* execution.**

---

## Architectural Scope (Corrected)

### What IATO Is

* A **pre-execution enforcement kernel** for inference systems
* A **trust-object admission engine**, not a model optimizer
* A **deterministic state transition system**
* A **kernel-adjacent architecture** (eBPF / XDP / integer-bounded execution)
* A system where **auditability is a consequence of correctness**, not a compensating control

---

### What IATO Is Not

* ❌ Not a probabilistic AI framework
* ❌ Not an online learning system
* ❌ Not a stochastic control loop
* ❌ Not container-authoritative
* ❌ Not dependent on explainability for correctness
* ❌ Not a convergence-based safety argument

---

## Core Principles

| Principle               | Meaning                                    |
| ----------------------- | ------------------------------------------ |
| **Admissibility First** | Unsafe transitions are excluded *a priori* |
| **Determinism**         | No stochastic elements in live enforcement |
| **Contractivity**       | State transitions are provably shrinking   |
| **Kernel Authority**    | Enforcement occurs below user space        |
| **Audit as Evidence**   | Logs seal already-valid execution          |

---

## Core Capabilities (Current)

### Deterministic Enforcement

* Execution paths are **pre-verified**
* State transitions must satisfy:

  * Lyapunov decrease
  * Contractive bounds
  * Safety invariants
  * Sovereignty constraints

> If a transition violates an invariant, **it is never executed**.

---

### Trust-Object Lifecycle

* Every action is represented as a **Trust Object**
* Trust Objects carry:

  * Deterministic state hash
  * Admissibility proof reference
  * Causal lineage
* Cryptography **seals** admissible execution — it does not decide it

---

### Kernel-Level Gating

* Enforcement occurs **pre-application**
* Eliminates:

  * TOCTOU vulnerabilities
  * Scheduler drift
  * Floating-point instability
* User-space and containers are **observers only**

---

### Audit & Explainability (Non-Control Path)

* Explainability exists **only for inspection**
* Used for:

  * Regulatory review
  * Human verification
  * Post-execution audit
* It **cannot influence execution**

---

## Containerization (Explicitly Non-Authoritative)

Docker exists **only** to support:

* Reproducible research
* Proof development
* Simulation
* Visualization
* Documentation

> **No containerized component participates in enforcement or trust admission.**

This is a **hard architectural boundary**, not a configuration choice.

---

## Supported Platforms (Research Harness Only)

| Platform      | Status                   |
| ------------- | ------------------------ |
| `linux/amd64` | Supported                |
| `linux/arm64` | Supported                |
| Multi-arch    | For reproducibility only |

---

## Docker Image Role

### Base Image

* Ubuntu 22.04 LTS (slim)
* Python 3.11
* Minimal cryptographic utilities

Chosen for **reproducibility**, not authority.

---

## Building the Research Image

```bash
docker buildx create --use
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t iato/research:latest \
  -f docker/Dockerfile \
  --push .
```

---

## Stack Components (Reclassified)

| Component  | Role                        |
| ---------- | --------------------------- |
| Jupyter    | Proof sketches, simulations |
| Redis      | Offline DAG inspection      |
| Kafka      | Audit artifact streaming    |
| Prometheus | Observation only            |
| Grafana    | Visualization               |
| Nginx      | TLS termination             |

None of these components can:

* Gate execution
* Enforce invariants
* Influence kernel decisions

---

## Removed Concepts (Post-Proof)

The following were **intentionally removed** after formal review:

* Online PGD enforcement
* Bayesian stabilization loops
* RL-style corrective feedback
* Probabilistic safety arguments
* Container-level trust
* Explainability-driven control

These were identified as **academically appealing but operationally unsound**.

---

## Summary

**Old framing:**

> Stress systems until they *appear* safe

**Current framing:**

> Prove safety so unsafe states **cannot exist**

IATO is now a **law-bound inference kernel**, not an experimental AI platform.

Determinism is not an optimization.
It is the **only admissible basis for trust**.


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

Below is a **clean, post-proof, non-academic rewrite** that aligns with the **current IATO architecture** and explicitly removes **probabilistic enforcement, Monte-Carlo logic, RL language, and container authority**.

This version can replace the section verbatim.

---

## Stack Composition (Reclassified)

The IATO Docker Compose stack exists **solely as a research, inspection, and reproducibility harness**.
No component in this stack participates in **execution gating, invariant enforcement, or trust admission**.

| Component                        | Role (Corrected)                                                            |
| -------------------------------- | --------------------------------------------------------------------------- |
| **iato-kernel (research build)** | Deterministic inference prototype and invariant demonstration environment   |
| **Redis**                        | Offline DAG inspection and trust-object lineage storage (non-authoritative) |
| **Kafka**                        | Streaming of signed audit artifacts and trust-object events                 |
| **Prometheus**                   | Observation-only metrics collection                                         |
| **Grafana**                      | Visualization of audit, lineage, and stability indicators                   |
| **Nginx**                        | TLS termination and read-only API exposure                                  |
| **Node Exporter**                | Host telemetry for inspection only                                          |

> **Important:**
> All enforcement in IATO occurs **below user space**.
> Containers and orchestration layers are **observers, not authorities**.

All services are networked on a **dedicated bridge** and expose **only minimal read-only ports** required for inspection.

---

## Workflow Scripts (Clarified Scope)

| Script               | Purpose                                                                 |
| -------------------- | ----------------------------------------------------------------------- |
| `generate_certs.sh`  | Generates TLS material for stack communication and inspection endpoints |
| `start_stack.sh`     | Launches the Docker Compose research stack                              |
| `integrity_check.sh` | Verifies log continuity, DAG consistency, and metric availability       |

These scripts **do not** initialize trust, enable execution, or override kernel decisions.

---

## Configuration (Observation Only)

### Prometheus

* `monitoring/prometheus.yml`
  Scrape targets for kernel metrics, Redis state visibility, Kafka throughput, and host telemetry
* `rules/`
  Alerting rules for **operational visibility**, not enforcement

### Grafana

* `grafana/grafana.yml`
  Preconfigured dashboards for:

  * Trust-object lineage
  * DAG evolution
  * Invariant violation attempts (rejected transitions)

### Nginx

* `nginx/nginx.yml`
  TLS termination and **read-only** proxying to inspection endpoints

---

## Operational Notes (Corrected)

### Deterministic Execution Model

The kernel no longer performs stochastic updates or online optimization.

**Removed:**

* PGD enforcement
* Bayesian feedback
* RL-style adjustments
* Entropy-driven execution

**Current model:**

```
xₜ₊₁ = f(xₜ)  subject to:
  ΔV(xₜ) ≤ 0
  ||xₜ₊₁ − xₜ|| ≤ ρ ||xₜ − xₜ₋₁||
```

Only **pre-verified contractive transitions** are admissible.

---

### Resource Governance (Non-Authoritative)

```
resource_allocation = observe(GPU, memory, bandwidth)
```

* Resource metrics are **observed**, not controlled
* No adaptive scheduling influences execution correctness
* Used strictly for capacity planning and reproducibility

---

### Explainability (Audit-Only)

```
explain(xₜ) → SHAP / LIME
```

* Generated **after** execution
* Used for:

  * Regulatory inspection
  * Human audit
  * Documentation
* Has **zero influence** on system behavior

---

### Trust-Object Logging

* Every admissible transition emits a **signed trust-object record**
* Artifacts are:

  * Streamed via Kafka
  * Indexed in Redis DAGs
  * Visualized through Grafana
* Cryptography **seals lawful execution**, it does not authorize it

---

## Summary

This stack:

* **Does not enforce safety**
* **Does not decide trust**
* **Does not gate execution**

It exists to **observe, inspect, reproduce, and audit** a kernel whose correctness is established **before runtime**.

The authority remains where it belongs:
**in proven invariants, not infrastructure.**

