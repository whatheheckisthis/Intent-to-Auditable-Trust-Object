# OSINT Dispatcher High-Assurance Pipeline

This directory provides a three-layer assurance design for the OSINT Dispatcher.

## 1) Structural Modeling (Alloy)
- Module: `docs/notebooks/alloy/osint_dispatcher_command_mapping.als`
- Proves:
  - Command-token to memory-address mapping is injective (no aliasing).
  - Every `Signature` has `signedBy = TEEEnclave`.

## 2) Behavioral Verification (TLA+)
- Module: `docs/notebooks/tla/osint_dispatcher_rest_logging.tla`
- Models command execution and RESTful logging states.
- Liveness property (`NoSilentFailure`):
  - if command executes, eventually log transmitted OR state transitions to `Error`.
- Safety property (`SignedBeforeTransmit`):
  - transmitted logs are always Ed25519-signed.

## 3) Kernel Enforcement (eBPF/XDP)
- Program: `docs/notebooks/ebpf/osint_dispatcher_xdp_firewall.c`
- Enforces packet policy at XDP ingress:
  - traffic bound to `LOG_ENDPOINT_IP:LOG_ENDPOINT_PORT` must include
    `X-TEE-Security-Tag: TEE_ATTESTED` in payload.
  - unmatched endpoint traffic passes through unchanged.

## 4) Verification Tooling
- `Makefile` targets:
  - `make xdp-build` compiles the eBPF program.
  - `make verify-ebpf` runs Kani (preferred) or Control-Flag fallback.

## 5) Branchless Dijkstra Assurance + Static Disassembly
- Module: `docs/notebooks/tla/dijkstra_branchless_assurance.tla`
- Baseline config: `docs/notebooks/tla/dijkstra_branchless_assurance.cfg`
- Scenario pack under `docs/notebooks/tla/scenarios/` was purged during repository cleanup.
  - use `docs/notebooks/tla/dijkstra_branchless_assurance.cfg` as the retained baseline config.
- Static analysis workflow: `docs/notebooks/tla/STATIC_DISASSEMBLY_ANALYSIS.md`
- Tooling script: `docs/notebooks/tla/tools/static_disassembly_check.sh`
- ARMv9 reference target: `docs/notebooks/tla/tools/branchless_reduction_arm64_asm.S`

## 6) Armv9-A CCA Confidential Compute Control Formalization
- Spec document: `docs/notebooks/armv9_cca_confidential_compute_control_model.md`
- TLA+ safety models (x4): `docs/notebooks/tla/armv9_cca/`
- Coq non-interference proof scaffolds (x14): `docs/notebooks/coq/armv9_cca/`
- Focus: FEAT_SSBS + RME mapping, invariants, and ASL/TLA refinement obligations.


## 7) Armv9 CCA Formal Toolchain Bootstrap
- Script: `scripts/install-formal-verification-deps.sh`
- Installs:
  - TLC (`tlc`) via `tla2tools.jar`
  - Coq compiler (`coqc`)
- Verification:
  - `tlc -version`
  - `coqc --version`


## 8) Dependency Management Engine (HTTP 403/407 fallback)
- Script: `scripts/formal_deps_engine.py`
- Commands:
  - `python3 scripts/formal_deps_engine.py status`
  - `python3 scripts/formal_deps_engine.py install`
- Behavior:
  - runs pre-flight checks for `tlc` and `coqc`
  - wraps `./scripts/install-formal-verification-deps.sh` and captures STDOUT/STDERR
  - on HTTP `403` or `407`, falls back to `LOCAL_ARTIFACT_ROOT`
  - tracks Last Known Good / failed attempts in `deps_manifest.json`
- Retry behavior:
  - after an HTTP `403/407` failure, the engine records the failed `network_install` in `deps_manifest.json`
  - on the next run, if `LOCAL_ARTIFACT_ROOT` is still unset, network retry is skipped and local fallback guidance is shown

### Next steps
- Mirror artifacts into an offline folder and set `LOCAL_ARTIFACT_ROOT`:
  - `tla2tools.jar`
  - either `coq/bin/coqc`, `coqc`, or `debs/*.deb`
- Configure proxy and TLS inspection if required:
  - `http_proxy`, `https_proxy`, `NO_PROXY`
  - `CUSTOM_CA_BUNDLE=/path/to/ca.pem`
- Re-run install:
  - `python3 scripts/formal_deps_engine.py install`
- Verify tools:
  - `tlc -version`
  - `coqc --version`

## 9) Lean4/mathlib ARMv9-A RME Security Model + Compliance Artifact Generation
- Lean model: `lean/iato_v7/IATO/V7/RMEModel.lean`
- Lean tests: `lean/iato_v7/Test/RME.lean`
- Compliance generator: `scripts/generate_rme_compliance_artifacts.py`
- Generated artifacts:
  - `data/artifacts/compliance/armv9_rme_evidence.json`
  - `data/artifacts/compliance/armv9_rme_evidence.md`
- Run:
  - `python3 scripts/generate_rme_compliance_artifacts.py`
