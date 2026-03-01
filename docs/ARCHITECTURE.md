# IATO-V7 Architecture

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


## Architecture Non-Goals

To avoid over-claiming scope, IATO-V7 defines the following explicit non-goals:

- **NG-001**: This architecture does not claim to replace a full secure software development lifecycle (SDLC) program.
- **NG-002**: This architecture does not claim to provide complete runtime hardening for every deployment environment.
- **NG-003**: This architecture does not claim formal verification coverage for all third-party integrations or external systems.
- **NG-004**: This architecture does not claim regulatory or contractual compliance by default without organization-specific control implementation and evidence collection.
- **NG-005**: This architecture does not claim certification, endorsement, or affiliation with the **Common Criteria** scheme or any certification body.

### Certification and Affiliation Distinction

IATO-V7 may reference assurance concepts and evidence-readiness practices, but it makes **no assertion of affiliation with Common Criteria**, and no statement in this repository should be interpreted as a Common Criteria certification claim.
