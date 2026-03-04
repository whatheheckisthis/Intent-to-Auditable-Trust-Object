# Intent-to-Auditable-Trust-Object (IATO-V7)

[![Build](https://img.shields.io/badge/build-passing-brightgreen)](lean/iato_v7/lakefile.lean)
[![Essential%208](https://img.shields.io/badge/Essential%208-ML4-blue)](docs/cyber-risk-controls.md)
[![SOC2](https://img.shields.io/badge/SOC2-CC6.1%20%7C%20CC6.6-orange)](docs/ARCHITECTURE.md)
[![ISM](https://img.shields.io/badge/ISM-0457--0460-green)](docs/threat-model.md)


## Executive Summary:

**IĀTŌ‑V7** is a configuration-driven engine that scans host path file systems, applies predefined rules, and produces clean, standardised XML reports that capture exactly what was found.
The output is consistent and repeatable — the same config and files always produce the same result — giving teams reliable evidence for audits, compliance reviews, and secure system migrations, especially true for legacy JBoss EAP 7 worker environments.

## Overview: 

We ditched traditional filesystem parsers — which are slow, manual, and error-prone.
Instead, IĀTŌ-V7 uses Nmap (network scanner) as a fast, stateless, high-concurrency engine to audit local file paths.
Runs in WSL2, Minikube, or Ubuntu — works seamlessly in typical dev/test environments for Windows/Linux teams handling legacy JBoss migrations.

| Component | Role |
| --- | --- |
| **TOML Manifest** | Source of truth for paths, hash expectations, and audit rules. |
| **Nmap (Orchestrator)** | Executes path discovery and integrity checks via NSE scripts. |
| **XML Artifacts** | Canonical output format for downstream formal validation. |
| **IĀTŌ‑V7 Engine** | Consumes XML to validate observed state against the original TOML manifest. |

## Compliance Coverage:

| Framework | Control Scope | Coverage |
|---|---|---|
| Essential 8 (ML4) | Privilege separation, application control, patch governance | Worker isolation proofs and compatibility scanning |
| SOC2 TSC | `CC6.1`, `CC6.6`, `A1.2`, `PI1.3` | Access/control enforcement and change evidence workflows |
| ISM (ASD) | `0457`-`0460` privileged boundary and application control requirements | Administrative separation and migration control workflows |


## Control Mapping:

| Capability | Compliance Mapping | Evidence Surface |
|---|---|---|
| Worker compatibility proofing | Essential 8 ML4; SOC2 `CC6.1`; ISM `0457` isolation assurance | `lean/iato_v7/IATO/V7/Worker.lean` |
| Legacy conflict scanning | SOC2 `CC6.6`; SOC2 `PI1.3`; Essential 8 application control | `lean/iato_v7/IATO/V7/Scanner.lean` and scanner outputs |
| FEAT_RME migration planning | ISM `0457`, `0458`, `0460`; SOC2 change management | `scripts/migrate.sh` and architecture documentation |
| Privileged workflow hardening | ISM `0458`, `0459`; SOC2 `CC6.1` | Threat model and operational control procedures |
| Architecture invariants | SOC2/ISM control objectives; EAL7+ readiness support | `lean/iato_v7/IATO/V7/Architecture.lean` |

## Essential 8 Maturity (Level 4)

```text
[ ] Macro    [x] Privilege    [x] Application    [ ] Office
[x] Web      [x] User         [x] Patch           [x] Backup
```

Automation Focus:
- Privilege separation across worker domains.
- Application control through verification and migration gating.
- Patch compatibility checks for legacy worker migration inputs.

## Verified Architecture

```text
lean/iato_v7/IATO/V7/
  Basic.lean         # Lattice and security foundations
  Worker.lean        # Worker-domain non-interference
  Scanner.lean       # Dependency/conflict detection
  Architecture.lean  # System invariants aligned to SOC2 and ISM

workers/source/    # Legacy worker examples and source manifests
workers/target/    # FEAT_RME-aligned generated worker manifests
workers/reports/   # Compatibility scan reports and migration outputs

docs/ARCHITECTURE.md
  docs/WORKER_COMPAT.md  # Audit and implementation narrative
```

## IĀTŌ‑V7 Orchestration 

### 1) Purpose
The IĀTŌ‑V7 orchestration layer is a deterministic execution wrapper that converts a repository `config.toml` manifest into a canonical Nmap XML audit artifact for host-path filesystem assurance. It is optimized for WSL2/Minikube environments where metadata-heavy scans can amplify 9P I/O latency.

This section defines the target design, implementation expectations, and formal assurance behavior for the orchestration module.

### 2) Design Goals
- **Deterministic orchestration:** equivalent manifest inputs must produce equivalent command lines, policy payloads, and XML artifact paths.
- **Audit-only runtime posture:** no network discovery side effects.
- **In-process validation:** integrity and policy checks are performed by NSE scripts during scan execution.
- **Operational observability:** per-operation latency accounting is captured for forensic auditability.
- **Schema-compliant hand-off:** generated XML is validated against the IĀTŌ‑V7 audit schema before acceptance.

### 3) Architecture 
```text
config.toml
   │
   ▼
Manifest Parser ──► Deterministic Policy Builder ──► Nmap Command Planner
                                                      │
                                                      ▼
                                             Nmap + Custom NSE Scripts
                                                      │
                                                      ▼
                                          Canonical XML Artifact (-oX)
                                                      │
                                                      ▼
                                         Schema + Status Verification
                                                      │
                                                      ▼
                                         Clean/Dirty + Process Exit Code
```

#### 3.1 Module Responsibilities
1. **Manifest Parser**
   - Parse TOML and normalize explicit path targets.
   - Reject unspecified defaults that can introduce nondeterminism.
2. **Policy Builder**
   - Emit a stable, sorted policy structure used by NSE scripts (hash, mode, owner/group constraints, integrity predicates).
3. **Command Planner**
   - Build a constrained Nmap command that always includes:
     - `-Pn -sn` (state isolation / host-discovery bypass)
     - `-oX <artifact>` (canonical machine-readable output)
     - explicit NSE script invocation and script-args
   - Derive timing template (`-T2`..`-T4`) from measured filesystem latency.
4. **Execution + Telemetry**
   - Run via `subprocess` or `asyncio`.
   - Track operation-level latency (manifest load, policy write, scan, XML validation).
5. **Verification Gate**
   - Parse XML status fields produced by NSE and enforce Clean/Dirty semantics.
   - Return non-zero exit on any policy deviation.

### 4) Deterministic Behavior Requirements
- **Stable ordering:** all configured paths and rules must be lexicographically sorted before policy serialization.
- **Stable artifact mapping:** XML output path is derived from a deterministic tuple:
  - `(manifest fingerprint, target root, schema version)`.
- **Stable invocation:** command argument order must be fixed.
- **Stable environment:** pin Nmap version and disallow mutable runtime options that alter scan semantics.

### 5) Nmap / NSE Contract
#### 5.1 Mandatory Nmap Flags
- `-Pn -sn`: enforce local audit mode and skip ping/host discovery behavior.
- `-oX <path>`: emit canonical XML artifact.
- `--script <path_audit.nse>`: execute IĀTŌ‑V7 integrity checks in-process.
- `--script-args ...`: pass root path, policy file, schema version, project metadata.

#### 5.2 NSE Script Duties
Custom Lua NSE scripts are authoritative for:
- cryptographic hash checks,
- ownership and permission checks,
- structural integrity predicates,
- producing machine-readable `Clean` or `Dirty` status in XML.

Any mismatch (hash, ACL/mode/ownership, forbidden mutation) must set a failing status consumable by the orchestrator.

### 6) Timing & I/O Strategy (WSL2/9P-Aware)
- Run a short, bounded I/O latency probe at startup (sample reads/stats on configured target paths).
- Map probe results to Nmap timing template:
  - low latency → `-T4`,
  - medium latency → `-T3`,
  - high latency / 9P contention → `-T2`.
- Never exceed bounded scan scope:
  - **no unbounded recursive DFS traversal**,
  - only explicit manifest target paths and rule entries.

### 7) Progress and Latency Telemetry
The orchestrator must asynchronously emit progress records with operation timestamps and durations, for example:
- `manifest.parse.ms`
- `policy.write.ms`
- `nmap.exec.ms`
- `xml.validate.ms`

Telemetry output should be append-only JSONL or structured logs suitable for later evidence packaging.

### 8) Verification and Exit Semantics
- **Exit code 0:** XML is schema-valid and NSE reports `Clean` for all evaluated targets.
- **Non-zero exit:** any of the following:
  - XML schema mismatch,
  - NSE-reported `Dirty` condition,
  - policy serialization or deterministic mapping failure,
  - orchestrator runtime failure (Nmap unavailable, malformed config, etc.).

### 9) Implementation Constraints
- **Language:** Python (`subprocess` or `asyncio`) or Go.
- **No external parser dependency at hand-off:** final decision should be made from canonical XML/NSE outputs without additional ad hoc transforms.
- **Environment priority:** minimize filesystem interrupts and metadata churn in WSL2/Minikube.

### 10) Reference Integration Path
A practical Python entrypoint is expected at:
- `lean/iato_v7/scripts/nmap_path_audit_orchestrator.py`

NSE policy logic is expected at:
- `lean/iato_v7/nse/path_audit.nse`

This design section defines the acceptance baseline for those components.

### 11) Acceptance Criteria
The orchestration layer is accepted when it demonstrably:
1. Converts a valid TOML manifest into a deterministic Nmap invocation.
2. Produces canonical XML via `-oX` and validates it against the IĀTŌ‑V7 schema.
3. Returns `Clean`/`Dirty` based on NSE in-process checks only.
4. Emits a non-zero exit status for any integrity deviation.
5. Maintains bounded, explicit path scanning with latency-aware timing behavior.

## Repository Layout

- `lean/`: Lean 4 formal models and executable tests.
- `docs/`: Architecture and notebook documentation.
- `workers/`: Legacy/new worker implementations.
- `scripts/`: Automation and tooling helpers.
- `data/`: Static data manifests and reference files.

## Quick Start (WSL2)

Use this path if you are on Windows with WSL2 (Ubuntu recommended), especially for teams targeting JBoss EAP 7 worker migrations.

### 1) Open WSL2 and clone

```bash
cd ~
git clone https://github.com/<your-org>/Intent-to-Auditable-Trust-Object.git
cd Intent-to-Auditable-Trust-Object
```

### 2) Install base dependencies

```bash
sudo apt update
sudo apt install -y git curl python3 python3-pip build-essential nmap
```

### 3) Initialize deterministic audit manifest

```bash
cp config.toml config.local.toml
# edit config.local.toml and set root_path/targets for your environment
```

The `config.toml` file is the canonical schema template for IĀTŌ‑V7 path-audit orchestration. It defines all audit targets, expected hashes/ownership, timing profile (`T2`/`T3`), and XML artifact locations.

### 4) Install Lean toolchain dependencies

```bash
./scripts/install-formal-verification-deps.sh
./scripts/setup-lean-ci-deps.sh
```

### 4) Build + validate core models

```bash
# run Lean model tests
cd lean/iato_v7
lake test
cd ../..

# run worker compatibility scanner
python3 scripts/scan_workers.py data/legacy_workers.csv

# generate audit evidence artifacts
./scripts/lake_build.sh
```

### 5) Run orchestrator (dry-run then execute)

```bash
# view deterministic Nmap command and generated policy without scanning
python3 lean/iato_v7/scripts/nmap_path_audit_orchestrator.py --dry-run

# execute orchestrator to produce canonical XML artifact
python3 lean/iato_v7/scripts/nmap_path_audit_orchestrator.py
```

### 6) Target outcome (what success looks like)

- Lean tests complete successfully (`lake test` exits 0).
- Worker scan writes compatibility/risk output without errors.
- Audit evidence artifacts are generated by `scripts/lake_build.sh`.
- Orchestrator emits deterministic policy and canonical XML output via `-oX`.
- Validation outputs are ready to support JBoss EAP 7 migration planning workflows.

If a command fails in WSL2, run:

```bash
./scripts/normalize_ci_env.sh
```

Then rerun the failed step.


## Architecture Non-Goals

To keep the scope explicit, the architecture defines non-goals **NG-001** through **NG-005**:

- **NG-001**: Not a replacement for a full organizational SDLC/security program.
- **NG-002**: Not a guarantee of complete runtime hardening in every environment.
- **NG-003**: Not full formal verification of all third-party or external system behavior.
- **NG-004**: Not an automatic compliance attestation without organization-specific controls/evidence.
- **NG-005**: Not a claim of certification, endorsement, or affiliation with **Common Criteria**.

***IATO-V7 does not assert affiliation with Common Criteria***.

---


