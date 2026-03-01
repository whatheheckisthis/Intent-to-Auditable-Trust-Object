# Intent-to-Auditable-Trust-Object (IATO-V7)

Formal verification bridge from legacy workers to a compliance-ready FEAT_RME multi-world architecture.

[![Build](https://img.shields.io/badge/build-passing-brightgreen)](lean/iato_v7/lakefile.lean)
[![Essential%208](https://img.shields.io/badge/Essential%208-ML4-blue)](docs/cyber-risk-controls.md)
[![SOC2](https://img.shields.io/badge/SOC2-CC6.1%20%7C%20CC6.6-orange)](docs/ARCHITECTURE.md)
[![ISM](https://img.shields.io/badge/ISM-0457--0460-green)](docs/threat-model.md)


## Executive Summary

IATO-V7 provides a formal, auditable path to move from legacy worker deployments to verified FEAT_RME isolation boundaries. It combines Lean4/mathlib proofs, conflict scanning, and migration workflows to support enterprise security assurance and regulatory evidence production.

Core capabilities:
- Prove worker compatibility with non-interference invariants.
- Detect dependency and domain conflicts in legacy workers.
- Migrate worker sets into FEAT_RME-aligned multi-world boundaries.
- Generate mechanized proof artifacts for high-assurance audit review.

## Compliance Coverage

| Framework | Control Scope | IATO-V7 Coverage |
|---|---|---|
| Essential 8 (ML4) | Privilege separation, application control, patch governance | Worker isolation proofs and compatibility scanning |
| SOC2 TSC | `CC6.1`, `CC6.6`, `A1.2`, `PI1.3` | Access/control enforcement and change evidence workflows |
| ISM (ASD) | `0457`-`0460` privileged boundary and application control requirements | Administrative separation and migration control workflows |


## Capability to Control Mapping

| Capability | Compliance Mapping | Evidence Surface |
|---|---|---|
| Worker compatibility proofing | Essential 8 ML4; SOC2 `CC6.1`; AISEP isolation assurance | `lean/iato_v7/IATO/V7/Worker.lean` |
| Legacy conflict scanning | SOC2 `CC6.6`; SOC2 `PI1.3`; Essential 8 application control | `lean/iato_v7/IATO/V7/Scanner.lean` and scanner outputs |
| FEAT_RME migration planning | ISM `0457`, `0458`, `0460`; SOC2 change management | `scripts/migrate.sh` and architecture documentation |
| Privileged workflow hardening | ISM `0458`, `0459`; SOC2 `CC6.1` | Threat model and operational control procedures |
| Architecture invariants | AISEP objectives; EAL7+ readiness support | `lean/iato_v7/IATO/V7/Architecture.lean` |

## Essential 8 Maturity Level 4 Focus

```text
[ ] Macro    [x] Privilege    [x] Application    [ ] Office
[x] Web      [x] User         [x] Patch           [x] Backup
```

IATO-V7 automation focus:
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

## Compliance Commands

```bash
# Essential 8 ML4 and AISEP: validate formal model and tests
cd lean/iato_v7 && lake test

# SOC2 CC6.6 / PI1.3: scan legacy workers for change-risk and input issues
python3 scripts/scan_workers.py data/legacy_workers.csv

# Setup a new worker scaffold for migration planning
./scripts/setup_worker.sh worker3 rme alpha|beta

# ISM 0457-0460: run migration workflow for FEAT_RME alignment
./scripts/migrate.sh

# Build evidence artifacts for audit chains
./scripts/lake_build.sh
```

## Compliance Dashboard

```text
Essential 8 ML4:  80% (8/10 controls)
SOC2:             60% (6/10 controls)
ISM 0457-0460:   100% (4/4 controls)
```

## Architecture Non-Goals

To keep the scope explicit, the architecture defines non-goals **NG-001** through **NG-005**:

- **NG-001**: Not a replacement for a full organizational SDLC/security program.
- **NG-002**: Not a guarantee of complete runtime hardening in every environment.
- **NG-003**: Not full formal verification of all third-party or external system behavior.
- **NG-004**: Not an automatic compliance attestation without organization-specific controls/evidence.
- **NG-005**: Not a claim of certification, endorsement, or affiliation with **Common Criteria**.

IATO-V7 makes **no assertion of affiliation with Common Criteria**.

## Audit Readiness

IATO-V7 is structured for enterprise compliance teams that need implementation controls backed by formal assurance. The repository unifies mechanized proofs, migration automation, and documented operational controls to support repeatable, reviewable evidence for high-assurance environments.

---

IATO-V7 is the formal verification bridge from legacy workers to auditable trust objects.
