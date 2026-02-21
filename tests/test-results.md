# Offline Bootstrap + Build Staging Results

Date (UTC): 2026-02-21T00:36:06Z

## Stage 0 — Fail-Fast SDK Bootstrap

### `scripts/ensure-dotnet.sh; echo EXIT_ENSURE:$?`
- **Exit code:** `42`
- **Result:** Correct fail-fast behavior when no staged SDK archive is available.
- **Output:**

```text
[ensure-dotnet] dotnet not found on PATH; invoking offline recovery
[dotnet-recover] detected architecture: linux-x64
[dotnet-recover][error] no staged SDK archive found for arch=linux-x64; searched: /opt/bootstrap, /root/bootstrap, /workspace/bootstrap
[ensure-dotnet][error] offline recovery failed: no staged SDK archive available.
[ensure-dotnet][error] searched: /opt/bootstrap, /root/bootstrap, /workspace/bootstrap
[timing-mitigation][audit] segment=ensure-dotnet rc=42 actual_ms=1808 target_ms=1800 overrun=0
EXIT_ENSURE:42
```

## Stage 1 — Toolchain Presence Check

### `dotnet --version; echo EXIT_DOTNET:$?`
- **Exit code:** `127`
- **Output:**

```text
bash: command not found: dotnet
EXIT_DOTNET:127
```

## Stage 3 — Build Attempt (No Restore / No Runtime)

### `dotnet build src/NfcReader/NfcReader.sln --no-restore -p:RestorePackages=false; echo EXIT_BUILD:$?`
- **Exit code:** `127`
- **Output:**

```text
bash: command not found: dotnet
EXIT_BUILD:127
```

## Stage 4 — Native EL2/NFC Validation Checks

### `make check`
- **Exit code:** `0`
- **Result:** EL2/NFC C test binaries built and passed (`test_enrollment`, `test_spdm_binding`, `test_two_factor_gate`, `test_expiry_sweep`, `test_replay_defense`).
- **Notes:** Build emitted non-fatal `-Wunused-parameter` warnings in `compat/mbedtls_stub.c` for `add` and `add_len`.

## Deterministic Workflow Assets
- `scripts/ensure-dotnet.sh`: CI guard with fail-fast exit code `42` for missing staged archive.
- `scripts/recover-dotnet-from-archive.sh`: architecture-aware offline recovery (`linux-x64` / `linux-arm64`) from:
  - `/opt/bootstrap`
  - `$HOME/bootstrap`
  - `/workspace/bootstrap`
- `scripts/activate-dotnet-offline.sh`: SDK-only environment activation (`DOTNET_ROOT`, `PATH`, telemetry disabled).
- `scripts/build-nfcreader-offline.sh`: compile-only build staging (`--no-restore`, no runtime execution).

## EL2 Isolation Notes
- Build staging performs compile-time validation only.
- No runtime entrypoints are executed during this flow.
- No device MMIO or hypervisor calls are triggered by the staging scripts.

## Final Status
- Worker remains **Not Tangible** for .NET build until a staged SDK archive is provided.
- Failure mode is deterministic and actionable with clear diagnostics.
- Native EL2/NFC validation path is currently passing in this environment.
