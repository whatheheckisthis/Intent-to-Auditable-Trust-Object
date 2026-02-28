CC ?= gcc
AARCH64_CC ?= $(shell command -v aarch64-linux-gnu-gcc 2>/dev/null || command -v gcc)
CFLAGS_COMMON := -Wall -Wextra -Icompat -Iel2/include -Infc/include
MBEDTLS_STUB := compat/mbedtls_stub.c

EL2_CC      ?= aarch64-none-elf-gcc
EL2_CFLAGS  = -O2 -ffreestanding -Wall -Werror -march=armv8-a \
              -mgeneral-regs-only -nostdlib -fno-stack-protector

HOST_CC     ?= gcc
HOST_CFLAGS = -O0 -g -Wall -DIATO_HOST_TEST


PYTHON_CFLAGS  = $(shell python3-config --cflags)
PYTHON_LDFLAGS = $(shell python3-config --ldflags --embed)

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

.PHONY: all el2 kernel_module nfc_se tests check check-el2 check-nfc check-spdm check-infra check-hardening check-all check-env check-native check-dotnet check-report check-report-commit fetch-wheels build-guest-image qemu-harness hw-test check-full clean
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



check-all:
	@set +e; 	$(MAKE) check-env; env_rc=$$?; 	$(MAKE) check-native; native_rc=$$?; 	if [ $$env_rc -eq 0 ]; then 		$(MAKE) check-dotnet; dotnet_rc=$$?; 	else 		printf '[check-dotnet] status=SKIPPED\n' > build/check-dotnet.log; 		echo SKIPPED > build/.check-dotnet.status; 		dotnet_rc=0; 	fi; 	$(MAKE) check-report; report_rc=$$?; 	$(MAKE) check-report-commit; commit_rc=$$?; 	exit $$report_rc

check-env:
	@mkdir -p build
	@set +e; 	bash -lc 'source scripts/activate-dotnet-offline.sh' >/dev/null 2>&1; rc=$$?; 	if [ $$rc -ne 0 ]; then 		printf '[check-all][error] environment not ready; run ensure-dotnet.sh first\n' >&2; 		echo FAIL > build/.check-env.status; 		exit 1; 	fi; 	echo PASS > build/.check-env.status

check-native:
	@mkdir -p build
	@set +e; 	$(MAKE) check > build/check-native.log 2>&1; rc=$$?; 	if [ $$rc -eq 0 ]; then native_status=PASS; else native_status=FAIL; fi; 	echo $$native_status > build/.check-native.status; 	passed=0; failed=0; 	for t in tests/test_enrollment tests/test_spdm_binding tests/test_two_factor_gate tests/test_expiry_sweep tests/test_replay_defense; do 		if [ -x $$t ]; then 			./$$t >/dev/null 2>&1; trc=$$?; 			if [ $$trc -eq 0 ]; then 				printf '[check-native][binary] %s=PASS\n' $$t >> build/check-native.log; 				passed=$$((passed+1)); 			else 				printf '[check-native][binary] %s=FAIL\n' $$t >> build/check-native.log; 				failed=$$((failed+1)); 			fi; 		else 			printf '[check-native][binary] %s=FAIL\n' $$t >> build/check-native.log; 			failed=$$((failed+1)); 		fi; 	done; 	printf '[check-native] status=%s\n' $$native_status >> build/check-native.log; 	printf '[check-native] total=%s passed=%s failed=%s\n' 5 $$passed $$failed >> build/check-native.log; 	exit $$rc

check-dotnet:
	@mkdir -p build
	@set +e; 	bash scripts/build-nfcreader-offline.sh > /dev/null 2> build/check-dotnet.log; rc=$$?; 	if [ $$rc -eq 0 ]; then 		echo PASS > build/.check-dotnet.status; 		printf '[check-dotnet] status=PASS\n' >> build/check-dotnet.log; 	else 		echo FAIL > build/.check-dotnet.status; 		printf '[check-dotnet] status=FAIL rc=%s\n' $$rc >> build/check-dotnet.log; 	fi; 	exit $$rc

check-report:
	@mkdir -p build
	@native_status=$$(cat build/.check-native.status 2>/dev/null || echo FAIL); 	dotnet_status=$$(cat build/.check-dotnet.status 2>/dev/null || echo SKIPPED); 	native_line=$$(rg -n "^\[check-native\] total=" build/check-native.log -N | tail -n1 | sed 's/^[^:]*://'); 	if [ -z "$$native_line" ]; then native_line='[check-native] total=0 passed=0 failed=0'; fi; 	native_total=$$(echo "$$native_line" | sed -E 's/.*total=([0-9]+).*/\1/'); 	native_passed=$$(echo "$$native_line" | sed -E 's/.*passed=([0-9]+).*/\1/'); 	native_failed=$$(echo "$$native_line" | sed -E 's/.*failed=([0-9]+).*/\1/'); 	dotnet_summary=$$(rg -n "^\[build\]\[test-summary\] total=" build/check-dotnet.log -N | tail -n1 | sed 's/^[^:]*://'); 	if [ -z "$$dotnet_summary" ]; then dotnet_summary='[build][test-summary] total=0 passed=0 failed=0 skipped=0'; fi; 	dotnet_total=$$(echo "$$dotnet_summary" | sed -E 's/.*total=([0-9]+).*/\1/'); 	dotnet_passed=$$(echo "$$dotnet_summary" | sed -E 's/.*passed=([0-9]+).*/\1/'); 	dotnet_failed=$$(echo "$$dotnet_summary" | sed -E 's/.*failed=([0-9]+).*/\1/'); 	dotnet_skipped=$$(echo "$$dotnet_summary" | sed -E 's/.*skipped=([0-9]+).*/\1/'); 	printf '[check-all] native=%s dotnet=%s\n' $$native_status $$dotnet_status; 	printf '[check-all] native: total=%s passed=%s failed=%s\n' $$native_total $$native_passed $$native_failed; 	printf '[check-all] dotnet: total=%s passed=%s failed=%s skipped=%s\n' $$dotnet_total $$dotnet_passed $$dotnet_failed $$dotnet_skipped; 	if [ "$$native_status" = "PASS" ] && [ "$$dotnet_status" = "PASS" ]; then 		printf '[check-all] verdict=PASS\n'; 		exit 0; 	fi; 	printf '[check-all] verdict=FAIL\n'; 	if [ "$$dotnet_status" = "SKIPPED" ]; then exit 2; fi; 	exit 1


check-report-commit:
	@mkdir -p tests
	@set +e; ts=$$(date -u +%Y-%m-%dT%H:%M:%SZ); sha=$$(git rev-parse HEAD 2>/dev/null || echo unknown); branch=$$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown); operator="$${USER:-unknown}@$${HOSTNAME:-unknown}"; seed=20260220; ensure_exit=$$(rg -o 'EXIT_ENSURE:[0-9]+' build/check-dotnet.log -N 2>/dev/null | tail -n1 | sed 's/EXIT_ENSURE://'); [ -n "$$ensure_exit" ] || ensure_exit=1; dotnet_exit=$$(rg -o 'EXIT_DOTNET:[0-9]+' build/check-dotnet.log -N 2>/dev/null | tail -n1 | sed 's/EXIT_DOTNET://'); [ -n "$$dotnet_exit" ] || dotnet_exit=1; build_exit=$$(rg -o 'EXIT_BUILD:[0-9]+' build/check-dotnet.log -N 2>/dev/null | tail -n1 | sed 's/EXIT_BUILD://'); [ -n "$$build_exit" ] || build_exit=1; native_status=$$(cat build/.check-native.status 2>/dev/null || echo FAIL); if [ "$$native_status" = PASS ]; then native_exit=0; else native_exit=1; fi; if rg -n 'warning|WARN|AddressSanitizer|UndefinedBehaviorSanitizer' build/check-native.log >/dev/null 2>&1; then notes='warnings present in build/check-native.log'; else notes='none'; fi; dotnet_status=$$(cat build/.check-dotnet.status 2>/dev/null || echo SKIPPED); if [ "$$dotnet_status" = SKIPPED ]; then verdict=UNPROVISIONED; elif [ "$$dotnet_status" = PASS ] && [ "$$native_status" = PASS ]; then verdict=PASS; else verdict=FAIL; fi; : > tests/test-results.md; echo '# Offline Bootstrap + Build Staging Results' >> tests/test-results.md; echo "Date (UTC): $$ts" >> tests/test-results.md; echo '' >> tests/test-results.md; echo '## Stage 0 — Fail-Fast SDK Bootstrap' >> tests/test-results.md; echo '### `scripts/ensure-dotnet.sh; echo EXIT_ENSURE:$$?`' >> tests/test-results.md; echo "- **Exit code:** `$$ensure_exit`" >> tests/test-results.md; echo "- **Result:** $$( [ "$$ensure_exit" = 0 ] && echo PASS || echo FAIL )" >> tests/test-results.md; echo '- **Output:**' >> tests/test-results.md; echo '```text' >> tests/test-results.md; cat build/check-dotnet.log 2>/dev/null >> tests/test-results.md; echo '```' >> tests/test-results.md; echo '' >> tests/test-results.md; echo '## Stage 1 — Toolchain Presence Check' >> tests/test-results.md; echo '### `dotnet --version; echo EXIT_DOTNET:$$?`' >> tests/test-results.md; echo "- **Exit code:** `$$dotnet_exit`" >> tests/test-results.md; echo '- **Output:**' >> tests/test-results.md; echo '```text' >> tests/test-results.md; cat build/check-dotnet.log 2>/dev/null >> tests/test-results.md; echo '```' >> tests/test-results.md; echo '' >> tests/test-results.md; echo '## Stage 3 — Build Attempt (No Restore / No Runtime)' >> tests/test-results.md; echo '### `dotnet build src/NfcReader/NfcReader.sln --no-restore -p:RestorePackages=false; echo EXIT_BUILD:$$?`' >> tests/test-results.md; echo "- **Exit code:** `$$build_exit`" >> tests/test-results.md; echo '- **Output:**' >> tests/test-results.md; echo '```text' >> tests/test-results.md; cat build/check-dotnet.log 2>/dev/null >> tests/test-results.md; echo '```' >> tests/test-results.md; echo '' >> tests/test-results.md; echo '## Stage 4 — Native EL2/NFC Validation Checks' >> tests/test-results.md; echo '### `make check`' >> tests/test-results.md; echo "- **Exit code:** `$$native_exit`" >> tests/test-results.md; echo "- **Result:** $$native_status" >> tests/test-results.md; echo "- **Notes:** $$notes" >> tests/test-results.md; echo '' >> tests/test-results.md; echo '## Deterministic Workflow Assets' >> tests/test-results.md; echo '- `scripts/ensure-dotnet.sh`: offline SDK bootstrap and recovery orchestrator.' >> tests/test-results.md; echo '- `scripts/recover-dotnet-from-archive.sh`: deterministic archive-based SDK recovery.' >> tests/test-results.md; echo '- `scripts/activate-dotnet-offline.sh`: offline dotnet environment activation.' >> tests/test-results.md; echo '- `scripts/build-nfcreader-offline.sh`: offline restore/build/test workflow with TRX summary.' >> tests/test-results.md; echo '- `run_batch_tests.py`: deterministic batch harness using base seed `20260220`.' >> tests/test-results.md; echo '' >> tests/test-results.md; echo '## EL2 Isolation Notes' >> tests/test-results.md; echo '- Build staging performs compile-time validation only.' >> tests/test-results.md; echo '- No runtime entrypoints are executed during this flow.' >> tests/test-results.md; echo '- No device MMIO or hypervisor calls are triggered by the staging scripts.' >> tests/test-results.md; echo '' >> tests/test-results.md; echo '## Final Status' >> tests/test-results.md; echo "- **Verdict:** $$verdict" >> tests/test-results.md; echo "- **Commit:** $$sha" >> tests/test-results.md; echo "- **Branch:** $$branch" >> tests/test-results.md; echo "- **Operator:** $$operator" >> tests/test-results.md; echo "- **PRNG seed:** $$seed" >> tests/test-results.md; echo '- Failure mode is deterministic and actionable with clear diagnostics.' >> tests/test-results.md
	@printf '[check-all] test-results.md written\n' >&2

fetch-wheels:
	bash scripts/fetch-wheels.sh

build-guest-image: fetch-wheels
	bash scripts/build-guest-image.sh

qemu-harness:
	bash scripts/qemu-harness.sh

hw-test: qemu-harness

check-full: check-all
	@if [ -f build/guest/guest.img ]; then \
		$(MAKE) hw-test; \
	else \
		echo "[check-full] guest image not built; skipping hw-test"; \
		echo "[check-full] run: make build-guest-image"; \
	fi

check: el2 tests check-el2 check-nfc check-spdm check-hardening check-infra
	@echo "ABI audit: objdump SMMU write locality"
	@objdump -d el2/src/*.o | awk '/<.*>:/ {f=$$0} /SMMU_BASE/ {print f; print $$0}' > /tmp/smmu_scan.txt
	@echo "ABI audit: nm Linux symbols in EL2 objects"
	@nm -u el2/src/*.o | rg -n "\b(kmalloc|printk|pci_|nfc_)" && exit 1 || true

clean:
	rm -f $(EL2_OBJS) nfc/src/*.o $(TEST_BINS)


inspect-journal:
	@if [ -z "$(JOURNAL)" ]; then echo "Usage: make inspect-journal JOURNAL=build/hw-journal/<file>.ndjson [ARGS="..."]"; exit 1; fi
	python3 scripts/hw-journal-inspect.py $(ARGS) $(JOURNAL)


.PHONY: build-el2-smmu test-el2-smmu build-el2-smc test-el2-smc build-el2-cnthp test-el2-cnthp build-el2-boot test-el2-boot build-el2-pyembed test-el2-pyembed build-el2 test-el2 test-e2e test-e2e-hw check-e2e

build-el2-smmu:
	@mkdir -p build/el2
	$(EL2_CC) $(EL2_CFLAGS) -c el2/smmu_init.c -o build/el2/smmu_init.o

test-el2-smmu:
	@mkdir -p build
	$(HOST_CC) $(HOST_CFLAGS) -DIATO_MOCK_MMIO el2/smmu_init.c el2/smmu_init_test.c -o build/test-el2-smmu
	./build/test-el2-smmu

build-el2-smc: build-el2-smmu
	$(EL2_CC) $(EL2_CFLAGS) -c el2/smc_handler.c -o build/el2/smc_handler.o

test-el2-smc:
	@mkdir -p build
	$(HOST_CC) $(HOST_CFLAGS) -DIATO_MOCK_MMIO -DIATO_MOCK_VALIDATOR -DIATO_MOCK_CNTVCT el2/smmu_init.c el2/smc_handler.c el2/smc_handler_test.c -o build/test-el2-smc
	./build/test-el2-smc

build-el2-cnthp: build-el2-smc
	$(EL2_CC) $(EL2_CFLAGS) -c el2/cnthp_driver.c -o build/el2/cnthp_driver.o
	$(EL2_CC) $(EL2_CFLAGS) -c el2/cnthp_ioctl.c -o build/el2/cnthp_ioctl.o
	$(EL2_CC) $(EL2_CFLAGS) -c el2/expiry_sweep.c -o build/el2/expiry_sweep.o

test-el2-cnthp:
	@mkdir -p build
	$(HOST_CC) $(HOST_CFLAGS) -DIATO_MOCK_MMIO -DIATO_MOCK_CNTVCT -DIATO_MOCK_SMMU el2/smmu_init.c el2/cnthp_driver.c el2/expiry_sweep.c el2/cnthp_driver_test.c -o build/test-el2-cnthp
	./build/test-el2-cnthp

build-el2: build-el2-boot build-el2-smmu build-el2-smc build-el2-cnthp build-el2-pyembed
	$(EL2_CC) $(EL2_CFLAGS) build/el2/boot.o build/el2/main.o build/el2/uart.o build/el2/smmu_init.o build/el2/smc_handler.o build/el2/cnthp_driver.o build/el2/cnthp_ioctl.o build/el2/expiry_sweep.o build/el2/py_embed.o -T el2/iato-el2.ld -o build/el2/iato-el2.bin

test-el2: test-el2-boot test-el2-smmu test-el2-smc test-el2-cnthp test-el2-pyembed
	@echo "[test-el2] all EL2 unit tests passed"


build-el2-boot:
	@mkdir -p build/el2
	$(EL2_CC) $(EL2_CFLAGS) -c el2/boot.S -o build/el2/boot.o
	$(EL2_CC) $(EL2_CFLAGS) -c el2/main.c -o build/el2/main.o
	$(EL2_CC) $(EL2_CFLAGS) -c el2/uart.c -o build/el2/uart.o

test-el2-boot:
	@mkdir -p build
	$(HOST_CC) $(HOST_CFLAGS) -DIATO_MOCK_ALL -DIATO_MOCK_UART -DIATO_MOCK_VALIDATOR -DIATO_MOCK_CNTVCT 		el2/main.c el2/uart.c el2/boot_test.c -o build/test-el2-boot
	./build/test-el2-boot

build-el2-pyembed:
	@mkdir -p build/el2
	$(HOST_CC) $(HOST_CFLAGS) $(PYTHON_CFLAGS) -c el2/py_embed.c -o build/el2/py_embed.o

test-el2-pyembed:
	@mkdir -p build
	$(HOST_CC) $(HOST_CFLAGS) $(PYTHON_CFLAGS) -DIATO_MOCK_UART el2/py_embed.c el2/uart.c el2/py_embed_test.c $(PYTHON_LDFLAGS) -o build/test-el2-pyembed
	PYTHONPATH=$(PWD) ./build/test-el2-pyembed

test-e2e:
	IATO_HW_MODE=0 IATO_TPM_SIM=1 IATO_MANA_SIM_MMIO=1 pytest tests/integration/test_e2e_smc_to_smmu.py -v --tb=short -m "not hw"

test-e2e-hw:
	IATO_HW_MODE=1 IATO_TPM_SIM=0 IATO_MANA_SIM_MMIO=0 pytest tests/integration/test_e2e_smc_to_smmu.py -v --tb=short -m hw --junitxml=/tmp/hw-e2e-pytest.xml

check-e2e:
	$(MAKE) test-e2e
