# IĀTŌ-V7 — Intent-to-Auditable-Trust-Object

A hardware-binding trust enforcement system for EL2-isolated environments.

> A device can perform DMA to a physical memory range if and only if a valid, unexpired, non-replayed, SPDM-attested credential for that device's StreamID has been validated by the EL2 hypervisor and the corresponding StreamTableEntry has been written to the SMMU by EL2 C code — not by any guest kernel or userspace process.

Targets QEMU virt with emulated SMMUv3, swtpm TPM 2.0, and libspdm, backed by a custom EL2 hypervisor in C with CPython embedded at EL2.

---

## Repository structure

**`el2/`** — EL2 hypervisor
- `boot.S` — reset vector, exception vector table
- `iato-el2.ld` — linker script (load address `0x40000000`)
- `main.c` — C entry point, exception dispatch
- `smmu_init.h/c` — SMMUv3 STRTAB initialisation and STE writes
- `smc_handler.h/c` — SMC trap handler, rate limiter, credential copy
- `py_embed.h/c` — CPython embedding, `iato_py_validate_credential()`

**`src/`** — Python validation layer
- `el2_validator.py` — ECDSA P-256 credential validator (149-byte wire format)
- `spdm_state_machine.py` — DMTF DSP0274 state machine
- `tpm_enrollment.py` — TPM PCR extend and seal (swtpm / sim)
- `hw_journal.py` — append-only hardware event journal (NDJSON)

**`scripts/`** — CI and bootstrap
- `timing-engine.sh` — constant-time execution engine
- `qemu-harness.sh` — full QEMU stack orchestrator
- `batch_runner.py` — 4×100 timing side-channel verification runs

---

## Trust chain

```
Guest kernel (EL1)
│  SMC #0xIATO (credential blob, stream_id)
▼
iato_smc_handle()                    [EL2 C]
│  rate check (I-07) → copy from guest PA → length check
▼
iato_py_validate_credential()        [CPython at EL2]
│  ECDSA P-256 verify (I-02)
│  nonce replay check (I-03)
│  SPDM session binding check (I-02)
▼
iato_smmu_write_ste()                [EL2 C]
│  STE binary layout per ARM IHI0070
│  CMD_CFGI_STE + CMD_SYNC + DSB SY (I-13)
▼
SMMU hardware                        [emulated SMMUv3]
   enforces PA range restriction for all DMA from stream_id
```

---

## Invariants

| ID | Claim | Enforced in |
|---|---|---|
| I-01 | No STE reaches PERMITTED without validated SteCredential | C + Python |
| I-02 | Validator verifies ECDSA P-256 AND SPDM session binding | Python |
| I-03 | Nonce consumed only after all checks pass | Python |
| I-07 | Rate check before validate — rejected calls never consume nonces | C + Python |
| I-13 | MMIO write followed by DSB SY before CMD_CFGI_STE | C |

Full invariant set (I-01 through I-14) is in [`docs/threat-model.md`](docs/threat-model.md).

---

## Timing side-channel mitigation

All bootstrap and build scripts are padded to fixed wall-clock targets by `timing-engine.sh`. An external observer cannot distinguish fast paths from slow paths by timing alone.

| Script | Target | Natural range |
|---|---|---|
| `ensure-dotnet.sh` | 45s | 0.01s – 2.5s |
| `recover-dotnet-from-archive.sh` | 90s | 0.05s – 40s |
| `build-nfcreader-offline.sh` | 120s | 0.01s – 60s |

Verified across 400 runs (4 batches × 100 tests), 100% TE window compliance. PRNG seed `0x4941544f2d5637` ("IATO-V7") — deterministic and auditable.

---

## Audit trail

Every trust decision is recorded in `build/hw-journal/{timestamp}.ndjson` across all four hardware concerns (TPM, SPDM, SMMU, CNTHP).

```bash
python3 scripts/hw-journal-inspect.py \
  --format timeline \
  build/hw-journal/latest.ndjson
```

`tests/test-results.md` is written after every CI run. `tests/hw-results.md` is written after every QEMU harness run. Both reference git SHA, branch, operator, and PRNG seed.

---

## EL2 memory map (QEMU virt)

| Address | Device |
|---|---|
| `0x09000000` | PL011 UART (boot diagnostics) |
| `0x09050000` | SMMUv3 MMIO (`IATO_SMMU_BASE`) |
| `0x40000000` | EL2 hypervisor load address |
| `/tmp/iato-spdm.sock` | libspdm responder socket |

---

## Non-goals

- **NG-01** Physical attacker with JTAG or direct memory access
- **NG-02** Compromised EL3 firmware
- **NG-03** Timing side-channel attacks on ECDSA verification
- **NG-05** Multi-core race conditions (single-CPU assumption, `-smp 1`)
