# IATO-V7 Architecture

## Repository Layout (Restructured)

```text
lean/iato_v7/IATO/V7/
  Basic.lean         # Lattice and security foundations
  Worker.lean        # Worker-domain non-interference
  Scanner.lean       # Dependency/conflict detection
  Architecture.lean  # System invariants aligned to SOC2 and ISM

workers/source/      # Legacy worker examples and source manifests
workers/target/      # FEAT_RME-aligned generated worker manifests
workers/reports/     # Compatibility scan reports and migration outputs

docs/ARCHITECTURE.md
  docs/WORKER_COMPAT.md  # Audit and implementation narrative
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
