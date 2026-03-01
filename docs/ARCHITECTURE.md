# IATO-V7 Architecture

## Repository Layout (Restructured)

```text
lean/       # Lean 4 formal models and executable tests
docs/       # architecture and notebook documentation
workers/    # legacy/new worker implementations
scripts/    # automation and tooling helpers
data/       # static data manifests and reference files

Key internal paths:
  - lean/iato_v7/IATO/V7/
  - docs/notebooks/
  - workers/{source,target,reports}/
```

## Formal Verification Layers

```text
DepSet Lattice (⊆, ⊔, ⊥) ──┐
        ↓                   │ TS/SCI Type System
Worker Model ───────────────┼── Non-interference
        ↓                   │    Fundamental Theorem
FEAT_RME Axioms ────────────┘
```

## Migration Pipeline

```text
legacy_workers.csv ──[scan]──> workers/reports/scan_report.json
                                      ↓
                               [migrate] ──> workers/target/
```

## Control Mapping (SOC 2 + ISM)

- **Least privilege / segregation of duties**: domain-separated workers with explicit dependency sets.
- **Change management**: legacy workers are scanned, normalized, and migrated with reportable outcomes.
- **Integrity and traceability**: target manifests are deterministic artifacts suitable for audit evidence.
- **Configuration governance**: architecture invariants in Lean enforce compositional compatibility constraints.

## Architecture Non-Goals

To avoid over-claiming scope, IATO-V7 defines the following explicit non-goals:

- **NG-001**: This architecture does not claim to replace a full secure software development lifecycle (SDLC) program.
- **NG-002**: This architecture does not claim to provide complete runtime hardening for every deployment environment.
- **NG-003**: This architecture does not claim formal verification coverage for all third-party integrations or external systems.
- **NG-004**: This architecture does not claim regulatory or contractual compliance by default without organization-specific control implementation and evidence collection.
- **NG-005**: This architecture does not claim certification, endorsement, or affiliation with the **Common Criteria** scheme or any certification body.

### Certification and Affiliation Distinction

IATO-V7 may reference assurance concepts and evidence-readiness practices, but it makes **no assertion of affiliation with Common Criteria**, and no statement in this repository should be interpreted as a Common Criteria certification claim.
