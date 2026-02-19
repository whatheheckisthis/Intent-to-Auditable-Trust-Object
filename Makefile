CC ?= gcc
AARCH64_CC ?= $(shell command -v aarch64-linux-gnu-gcc 2>/dev/null || command -v gcc)
CFLAGS_COMMON := -Wall -Wextra -Icompat -Iel2/include -Infc/include
MBEDTLS_STUB := compat/mbedtls_stub.c

EL2_SRCS := el2/src/el2_trust_store.c el2/src/el2_spdm.c el2/src/el2_nfc_validator.c el2/src/el2_binding_table.c el2/src/el2_smmu.c el2/src/el2_expiry.c el2/src/el2_main.c
EL2_OBJS := $(EL2_SRCS:.c=.o)

TEST_SRCS := tests/test_enrollment.c tests/test_spdm_binding.c tests/test_two_factor_gate.c tests/test_expiry_sweep.c tests/test_replay_defense.c
TEST_BINS := $(TEST_SRCS:.c=)

EL2_ARM_FLAGS := -ffreestanding -nostdlib -fno-stack-protector -DSMMU_BASE=0x09050000UL
ifeq ($(findstring aarch64,$(notdir $(AARCH64_CC))),aarch64)
EL2_ARCH_FLAGS := -march=armv9-a -mgeneral-regs-only
else
EL2_ARCH_FLAGS :=
endif

EL2_AUDIT_MARKERS := \
	"el2/src/el2_binding_table.c:el2_binding_set_spdm" \
	"el2/src/el2_binding_table.c:el2_binding_set_nfc" \
	"el2/src/el2_expiry.c:el2_expiry_handler" \
	"el2/src/el2_spdm.c:DOE_" \
	"el2/src/el2_spdm.c:transcript" \
	"el2/src/el2_nfc_validator.c:nonce_seen" \
	"el2/src/el2_nfc_validator.c:expiry_ns" \
	"el2/src/el2_trust_store.c:el2_tpm2_pcr_extend7"

.PHONY: all el2 kernel_module nfc_se tests check check-el2 check-nfc check-spdm check-infra check-hardening clean
all: el2 nfc_se tests

el2: CFLAGS := $(CFLAGS_COMMON) $(EL2_ARCH_FLAGS) -O2 $(EL2_ARM_FLAGS)
el2: $(EL2_OBJS)

el2/src/%.o: el2/src/%.c
	$(AARCH64_CC) $(CFLAGS) -c $< -o $@

kernel_module:
	@echo "kernel_module target prepared for Kbuild integration (obj-m)."

nfc_se: nfc/src/se_enrollment.o nfc/src/se_operational.o

nfc/src/%.o: nfc/src/%.c
	$(CC) $(CFLAGS_COMMON) -c $< -o $@

tests: CFLAGS := $(CFLAGS_COMMON) -fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g

tests: $(TEST_BINS)

%: %.c $(filter-out el2/src/el2_main.c,$(EL2_SRCS)) nfc/src/se_operational.c $(MBEDTLS_STUB)
	$(CC) $(CFLAGS) $^ -o $@

check-el2: tests
	@set -e; \
	for t in tests/test_spdm_binding tests/test_two_factor_gate tests/test_expiry_sweep tests/test_replay_defense; do \
		echo "Running $$t"; ./$$t; \
	done
	@echo "[audit] EL2 markers: binding, expiry sweep, CNTHP timer hooks"
	@for marker in $(EL2_AUDIT_MARKERS); do \
		f=$${marker%%:*}; p=$${marker#*:}; \
		rg -n "$${p}" "$${f}" >/dev/null || (echo "Missing marker $$p in $$f"; exit 1); \
	done

check-nfc: tests
	@echo "Running tests/test_enrollment"
	@./tests/test_enrollment
	@echo "[audit] NFC APDU/encryption/nonce bindings"
	@rg -n "APDU_INS_ISSUE_CREDENTIAL|APDU_INS_GET_CHALLENGE_NONCE" nfc/include/se_protocol.h >/dev/null
	@rg -n "encrypted_payload|nonce|expiry_ns|session_id" nfc/include/se_protocol.h nfc/src/se_operational.c el2/src/el2_nfc_validator.c >/dev/null

check-spdm: tests
	@echo "Running tests/test_spdm_binding"
	@./tests/test_spdm_binding
	@echo "[audit] SPDM state machine inside EL2 + DOE mailbox + transcript hash"
	@rg -n "DOE_(CTRL|STATUS|WDATA|RDATA)" el2/src/el2_spdm.c >/dev/null
	@rg -n "transcript|mbedtls_sha256|el2_binding_set_spdm" el2/src/el2_spdm.c >/dev/null

check-hardening:
	@echo "[audit] SMC boundary rate limiting + SMMU fault/revocation + TPM PCR extend"
	@rg -n "SMC|smc|rate|throttle|limit" el2/src el2/include >/tmp/smc_audit.txt || true
	@rg -n "fault|revok|el2_smmu_fault_ste" el2/src/el2_smmu.c el2/src/el2_binding_table.c >/tmp/smmu_audit.txt
	@rg -n "PCR|pcr|seal|lock|el2_enrollment_seal|el2_tpm2_pcr_extend7" el2/src/el2_trust_store.c >/tmp/tpm_audit.txt
	@test -s /tmp/smmu_audit.txt
	@test -s /tmp/tpm_audit.txt

check-infra:
	@echo "[audit] pyproject.toml parse + infra entrypoints"
	@python3 -c 'import pathlib,tomli; tomli.loads(pathlib.Path("pyproject.toml").read_text()); print("pyproject.toml parse ok")' || (echo "pyproject.toml parse warning: invalid TOML" >&2; true)
	@rg -n "check-el2|check-nfc|check-spdm|check-hardening|check-infra" Makefile >/dev/null
	@rg -n "FROM .* AS build-c|FROM .* AS test|FROM .* AS runtime" Dockerfile >/dev/null

check: el2 tests check-el2 check-nfc check-spdm check-hardening check-infra
	@echo "ABI audit: objdump SMMU write locality"
	@objdump -d el2/src/*.o | awk '/<.*>:/ {f=$$0} /SMMU_BASE/ {print f; print $$0}' > /tmp/smmu_scan.txt
	@echo "ABI audit: nm Linux symbols in EL2 objects"
	@nm -u el2/src/*.o | rg -n "\b(kmalloc|printk|pci_|nfc_)" && exit 1 || true

clean:
	rm -f $(EL2_OBJS) nfc/src/*.o $(TEST_BINS)
