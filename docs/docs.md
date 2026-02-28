---

## Getting Started — Why It Won't Just Run in VS Code

If you cloned this repository and pressed Run, nothing visible happened. This is expected. Here is why and what to do instead.

---

### 1. Architecture mismatch

Your development machine is almost certainly x86-64 (Windows PC or Intel Mac). IĀTŌ-V7 is compiled for ARMv9-A targeting EL2 with SVE2. These are not compatible instruction sets. Your CPU cannot execute the kernel binary directly any more than it can run code compiled for a toaster.

VS Code is a text editor. It has no knowledge of ARM exception levels, SMMUv3 stream tables, or EL2 privilege. Pressing the green Run button sends your x86 OS looking for an executable it cannot run.

The fix is QEMU. `scripts/qemu-harness.sh` launches a software-emulated ARM CPU that supports the specific SMMUv3, EL2, and TPM features the kernel requires. Everything runs inside that virtual chip. Your host OS is irrelevant once the harness is running.

---

### 2. The EL2 privilege wall

Standard debuggers — including every VS Code debug adapter — are designed for one of two contexts: user-mode (EL0) or kernel-mode (EL1). IĀTŌ-V7 is a hypervisor. It runs at EL2, which is a higher privilege level than any standard OS kernel. No off-the-shelf debugger has a concept of EL2.

If you attach a standard debugger to the QEMU process without the correct configuration, the kernel appears dead. There is no process to attach to, no symbols loaded in a recognisable format, and no stack frames that make sense to a user-mode debug adapter.

To debug IĀTŌ-V7 properly you must connect VS Code to the QEMU GDB stub. QEMU exposes a GDB remote protocol server that speaks at the machine level and can see EL2 registers. This requires a `launch.json` configured as follows:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "IĀTŌ EL2 (QEMU GDB)",
      "type": "cppdbg",
      "request": "launch",
      "program": "${workspaceFolder}/build/el2/iato-el2.bin",
      "miDebuggerServerAddress": "localhost:1234",
      "miDebuggerPath": "aarch64-none-elf-gdb",
      "setupCommands": [
        {
          "text": "set architecture aarch64",
          "ignoreFailures": false
        },
        {
          "text": "set remote hardware-breakpoint-limit 4",
          "ignoreFailures": false
        }
      ],
      "stopAtConnect": true,
      "cwd": "${workspaceFolder}"
    }
  ]
}
```

Start QEMU with the GDB stub enabled by adding `-s -S` to the QEMU flags in `scripts/qemu-harness.sh` before launching. Then attach VS Code. Without this exact configuration VS Code will report that the target is not responding because it is looking for an EL0 or EL1 process that does not exist.

---

### 3. Missing hardware dependencies

The kernel has two hard external dependencies that must be running before it boots. If either is absent the boot sequence hangs at a specific point with no visible error unless you are watching the UART output.

**`/tmp/iato-spdm.sock` — libspdm responder**

The Python validator checks SPDM session binding as part of Invariant I-02. This check happens during the first credential validation after boot. If the libspdm responder process is not running and listening on this socket, `SpdmQemuTransport` will time out during `connect()`, the validator will return `IATO_PY_ERR_CALL`, and the SMC handler will return `IATO_SMC_ERR_VALIDATION` for every credential. The guest will be unable to provision any stream and the boot sequence will appear to complete normally but every DMA-capable device will remain in ABORT state permanently.

`scripts/qemu-harness.sh` starts the libspdm responder before QEMU boots and health-checks the socket before proceeding. If you are starting components manually, start the responder first:

```bash
./scripts/start-spdm-responder.sh
```

Equivalent raw command:

```bash
./build/libspdm/bin/spdm_responder_emu \
  --trans socket \
  --socket-path /tmp/iato-spdm.sock
```

**`swtpm` — TPM emulator**

`iato_py_embed_init()` calls `KeyCeremony.verify()` which calls `TpmEnrollmentSeal.verify_enrollment_unchanged()` which reads PCR 16 via the TPM device. If swtpm is not running, the TPM device handle fails to open, `iato_py_embed_init()` returns `IATO_PY_ERR_INIT`, and `iato_main()` prints `[iato][fatal] py_embed_init failed` to the UART and spins in an infinite loop. The kernel is alive but permanently halted. From VS Code this looks identical to a crash.

`scripts/qemu-harness.sh` starts swtpm before QEMU and health-checks `/tmp/iato-swtpm/swtpm.sock` before proceeding. If you are starting manually:

```bash
./scripts/start-swtpm.sh
```

Equivalent raw command:

```bash
mkdir -p /tmp/iato-swtpm
swtpm socket \
  --tpmstate dir=/tmp/iato-swtpm \
  --ctrl type=unixio,path=/tmp/iato-swtpm/swtpm.sock \
  --tpm2 --daemon \
  --pid /tmp/iato-swtpm/swtpm.pid
```

---

### 4. The SVE2 register gap

IĀTŌ-V7 uses SVE2 vector registers for initial state handling in `boot.S`. The standard VS Code Variables panel shows general-purpose registers x0–x30 and a stack view. It does not show SVE2 registers (Z0–Z31, P0–P15, FFR) unless you have installed the ARM Embedded Development extension pack and configured the register map explicitly.

If you open the Variables panel during a debug session and see nothing meaningful, this is why. The telemetry you are looking for may be sitting in Z16 or a predicate register rather than in a named variable or a general-purpose register.

To see SVE2 state in the GDB console during a debug session:

```
(gdb) info registers sve
(gdb) p/x $z16
(gdb) p/x $p0
```

The ARM Embedded Development extension for VS Code can surface these in the Registers panel if configured with the Cortex-A57 or Cortex-A72 register map, though SVE2 support in the extension is incomplete as of early 2026. The GDB console is the reliable path.

---

### The correct path to seeing it work

Do not use the VS Code Run button. Use the terminal.

**Step 1 — Launch the full hardware stack**

```bash
./scripts/run-hw-harness.sh
```

This wrapper sets metadata defaults (`IATO_OPERATOR`, `IATO_PRNG_SEED`) and then runs `scripts/qemu-harness.sh`.

```bash
# Optional: override metadata explicitly
IATO_OPERATOR="alice" IATO_PRNG_SEED="0x4941544f2d5637" ./scripts/run-hw-harness.sh
```

This starts swtpm, starts the libspdm responder, boots QEMU with SMMUv3 and TPM attached, waits for the guest ready signal, and runs the test suite. It handles the startup order and health checks automatically.

**Step 2 — Watch the UART**

The kernel writes all diagnostic output to the PL011 UART at `0x09000000`. QEMU maps this to its stdout by default when `-nographic` is set in the harness. You will see the boot sequence:

```
[iato] EL2 hypervisor starting
[iato] smmu_init OK
[iato] smc_init OK
[iato] cnthp_init OK
[iato] py_embed_init OK
[iato] EL2 hypervisor ready
[guest-ready]
```

If any of these lines is absent or followed by `[iato][fatal]`, the line tells you exactly which subsystem failed. Check the corresponding sidecar process log (`/tmp/iato-swtpm/` or `/tmp/iato-spdm.log`) for the root cause.

**Step 3 — Read the results**

```bash
cat tests/hw-results.md
```

This file is written by the harness after the test run completes. It contains the full pass/fail breakdown, hardware environment state (PCR 16 extended, SPDM session reached ATTESTED, SMMU STE write count, CNTHP intervals fired), and the git SHA for the run.

For detailed event-level inspection of what happened during a specific test:

```bash
python3 scripts/hw-journal-inspect.py \
  build/hw-journal/*.ndjson \
  --format timeline
```

This renders all hardware events (TPM, SPDM, SMMU, CNTHP) in causal order so you can see exactly what the kernel was doing at the moment any test passed or failed.
