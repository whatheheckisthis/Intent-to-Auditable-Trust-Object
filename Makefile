CC ?= gcc
AARCH64_CC ?= aarch64-linux-gnu-gcc
CFLAGS_COMMON := -Wall -Wextra -Icompat -Iel2/include -Infc/include
MBEDTLS_STUB := compat/mbedtls_stub.c

EL2_SRCS := el2/src/el2_trust_store.c el2/src/el2_spdm.c el2/src/el2_nfc_validator.c el2/src/el2_binding_table.c el2/src/el2_smmu.c el2/src/el2_expiry.c el2/src/el2_main.c
EL2_OBJS := $(EL2_SRCS:.c=.o)

TEST_SRCS := tests/test_enrollment.c tests/test_spdm_binding.c tests/test_two_factor_gate.c tests/test_expiry_sweep.c tests/test_replay_defense.c
TEST_BINS := $(TEST_SRCS:.c=)

.PHONY: all el2 kernel_module nfc_se tests check clean
all: el2 nfc_se tests

el2: CFLAGS := $(CFLAGS_COMMON) -march=armv9-a -mgeneral-regs-only -ffreestanding -nostdlib -O2 -fno-stack-protector -DSMMU_BASE=0x09050000UL
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

check: tests
	@set -e; for t in $(TEST_BINS); do echo "Running $$t"; ./$$t; done
	@echo "ABI audit: objdump SMMU write locality"
	@objdump -d el2/src/*.o | awk '/<.*>:/ {f=$$0} /SMMU_BASE/ {print f; print $$0}' > /tmp/smmu_scan.txt || true
	@echo "ABI audit: nm Linux symbols in EL2 objects"
	@nm -u el2/src/*.o | rg -n "\b(kmalloc|printk|pci_|nfc_)" && exit 1 || true

clean:
	rm -f $(EL2_OBJS) nfc/src/*.o $(TEST_BINS)
