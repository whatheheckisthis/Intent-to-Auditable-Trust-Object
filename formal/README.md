# OSINT Dispatcher High-Assurance Pipeline

This directory provides a three-layer assurance design for the OSINT Dispatcher.

## 1) Structural Modeling (Alloy)
- Module: `formal/alloy/osint_dispatcher_command_mapping.als`
- Proves:
  - Command-token to memory-address mapping is injective (no aliasing).
  - Every `Signature` has `signedBy = TEEEnclave`.

## 2) Behavioral Verification (TLA+)
- Module: `formal/tla/osint_dispatcher_rest_logging.tla`
- Models command execution and RESTful logging states.
- Liveness property (`NoSilentFailure`):
  - if command executes, eventually log transmitted OR state transitions to `Error`.
- Safety property (`SignedBeforeTransmit`):
  - transmitted logs are always Ed25519-signed.

## 3) Kernel Enforcement (eBPF/XDP)
- Program: `formal/ebpf/osint_dispatcher_xdp_firewall.c`
- Enforces packet policy at XDP ingress:
  - traffic bound to `LOG_ENDPOINT_IP:LOG_ENDPOINT_PORT` must include
    `X-TEE-Security-Tag: TEE_ATTESTED` in payload.
  - unmatched endpoint traffic passes through unchanged.

## 4) Verification Tooling
- `Makefile` targets:
  - `make xdp-build` compiles the eBPF program.
  - `make verify-ebpf` runs Kani (preferred) or Control-Flag fallback.
