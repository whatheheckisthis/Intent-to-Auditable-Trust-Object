# QEMU Hardware Harness Guide

This guide provisions a reproducible local environment where `make hw-test` boots the full IĀTŌ stack in QEMU and runs `pytest -m hw` with `IATO_HW_MODE=1`.

## 1) Install host dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
  qemu-system-aarch64 swtpm tpm2-tools \
  libvirt-clients virt-manager \
  qemu-utils libguestfs-tools netcat-openbsd python3-pip
```

## 2) Build libspdm responder

Use the flow in `docs/qemu-spdm.md` and ensure this output exists:

```text
build/libspdm/bin/spdm_responder_emu
```

## 3) Build custom EL2 hypervisor

Build your project-specific EL2 image and place it at:

```text
build/el2/iato-el2.bin
```

## 4) Fetch Python wheels (internet required)

```bash
make fetch-wheels
```

## 5) Build guest image (offline-capable after wheel fetch)

```bash
make build-guest-image
```

## 6) Run hardware tests

```bash
make hw-test
```

## 7) Review results

```bash
cat tests/hw-results.md
```

## Environment variables

- `IATO_GUEST_IMAGE`: path to guest qcow2 image (default: `build/guest/guest.img`)
- `IATO_EL2_BIN`: path to EL2 hypervisor binary (default: `build/el2/iato-el2.bin`)
- `IATO_SPDM_RESPONDER`: path to libspdm responder (default: `build/libspdm/bin/spdm_responder_emu`)
- `IATO_BOOT_TIMEOUT_S`: guest boot timeout in seconds (default: `120`)
- `IATO_TEST_TIMEOUT_S`: hardware test timeout in seconds (default: `300`)

## Notes

- `scripts/qemu-harness.sh` performs only local orchestration; it does not download assets.
- If preflight fails, the harness exits with setup pointers back to this document.
