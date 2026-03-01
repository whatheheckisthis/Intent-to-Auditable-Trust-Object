# Intent-to-Auditable-Trust-Object (IATO-V7)

IATO-V7 provides a formal verification workflow to move legacy workers into a compliance-ready FEAT_RME multi-world architecture.

[![Build](https://img.shields.io/badge/build-passing-brightgreen)](lean/iato_v7/lakefile.lean)
[![Essential%208](https://img.shields.io/badge/Essential%208-ML4-blue)](docs/cyber-risk-controls.md)
[![SOC2](https://img.shields.io/badge/SOC2-CC6.1%20%7C%20CC6.6-orange)](docs/ARCHITECTURE.md)
[![ISM](https://img.shields.io/badge/ISM-0457--0460-green)](docs/threat-model.md)

## Executive Summary

IATO-V7 defines a clear path from legacy worker deployments to verified isolation boundaries. It combines Lean4/mathlib proofs, legacy conflict scanning, and migration scripts so teams can produce repeatable evidence for security and compliance review.

Core capabilities:
- Verify worker compatibility with non-interference invariants.
- Detect dependency and domain conflicts in legacy worker inputs.
- Migrate worker sets into FEAT_RME-aligned multi-world boundaries.
- Generate machine-produced artifacts for audit review.

## Compliance Coverage

| Framework | Control Scope | IATO-V7 Coverage |
|---|---|---|
| Essential 8 (ML4) | Privilege separation, application control, patch governance | Worker isolation proofs and compatibility scanning |
| SOC2 TSC | `CC6.1`, `CC6.6`, `A1.2`, `PI1.3` | Access controls, change checks, and evidence workflows |
| ISM (ASD) | `0457`-`0460` privileged boundary and application control requirements | Administrative separation and migration control workflows |

## Capability to Control Mapping

| Capability | Compliance Mapping | Evidence Surface |
|---|---|---|
| Worker compatibility proofing | Essential 8 ML4; SOC2 `CC6.1` | `lean/iato_v7/IATO/V7/Worker.lean` |
| Legacy conflict scanning | SOC2 `CC6.6`; SOC2 `PI1.3`; Essential 8 application control | `lean/iato_v7/IATO/V7/Scanner.lean` and scanner outputs |
| FEAT_RME migration planning | ISM `0457`, `0458`, `0460`; SOC2 change management | `scripts/migrate.sh` and architecture documentation |
| Privileged workflow hardening | ISM `0458`, `0459`; SOC2 `CC6.1` | Threat model and operational control procedures |
| Architecture invariants | Essential 8 ML4; SOC2 `CC6.1`; ISM `0457`-`0460` | `lean/iato_v7/IATO/V7/Architecture.lean` |

## Essential 8 Maturity Level 4 Focus

```text
[ ] Macro    [x] Privilege    [x] Application    [ ] Office
[x] Web      [x] User         [x] Patch           [x] Backup
```

IATO-V7 automation focus:
- Enforce privilege separation across worker domains.
- Support application control through verification and migration gates.
- Run patch compatibility checks against legacy migration inputs.

## Verified Architecture

```text
lean/iato_v7/IATO/V7/
  Basic.lean         # Security lattice and foundational definitions
  Worker.lean        # Worker-domain non-interference model
  Scanner.lean       # Dependency and conflict detection logic
  Architecture.lean  # System-level invariants for SOC2 and ISM alignment

workers/compatibility/   # Compatibility scan inputs and outputs

docs/ARCHITECTURE.md
  docs/WORKER_COMPAT.md  # Audit and implementation guidance
```

## Compliance Commands

```bash
# Essential 8 ML4: validate formal model and tests
cd lean/iato_v7 && lake test

# SOC2 CC6.6 / PI1.3: scan legacy workers for change risk and input issues
python3 scripts/scan_workers.py data/legacy_workers.csv

# Create a worker scaffold for migration planning
./scripts/setup_worker.sh worker3 rme alpha|beta

# ISM 0457-0460: run migration workflow for FEAT_RME alignment
./scripts/migrate.sh

# Build evidence artifacts for audit trails
./scripts/lake_build.sh
```

## Compliance Dashboard

```text
Essential 8 ML4:  80% (8/10 controls)
SOC2:             60% (6/10 controls)
ISM 0457-0460:   100% (4/4 controls)
```

## Audit Readiness

IATO-V7 is designed for compliance and engineering teams that need controls backed by formal evidence. The repository combines mechanized proofs, migration automation, and operational documentation so evidence can be reproduced and reviewed consistently.

## Architecture Non-Goals

To avoid ambiguity, IATO-V7 does **not** attempt to:
- Replace independent security assessments, certification audits, or regulatory determinations.
- Guarantee production security outcomes without correct deployment, operations, and key-management controls.
- Serve as a complete risk-management framework for all enterprise environments.
- Assert that every control in any framework is fully implemented by this repository alone.

## Compliance and Affiliation Disclaimer

IATO-V7 maps technical artifacts to selected control families for engineering traceability. These mappings are implementation guidance and evidence support, not a certification claim.

IATO-V7 makes **no assertion of affiliation with, endorsement by, or certification under Common Criteria** (including any Common Criteria scheme, lab, or authority) unless explicitly stated in separate, formal certification documentation.

---

IATO-V7 is a formal verification bridge from legacy workers to auditable trust objects.
