# Load variables from .env if present
-include .env

CC ?= gcc
NASM ?= nasm
CLANG ?= clang
CFLAGS ?= -O2 -Wall -Wextra
IATO_SVE2_CFLAGS ?= -O2 -Wall -Wextra -march=armv9-a+sve2+sha3
BPF_ARCH_INCLUDE ?= /usr/include/x86_64-linux-gnu
XDP_CFLAGS ?= -O2 -g -target bpf -D__TARGET_ARCH_x86 -I$(BPF_ARCH_INCLUDE)
LDFLAGS ?=
KANI ?= kani
CONTROL_FLAG ?= control-flag

HOST_ARCH := $(shell uname -m)
ENABLE_IATO_SVE2 ?= $(if $(filter aarch64 arm64,$(HOST_ARCH)),1,0)

API_KEY ?=
REST_URL ?=
TARGET_POLICY ?=

DISPATCHER = dispatcher
ASM_OBJ = fast_mask.o
C_OBJ = dispatcher.o pkcs11_signer.o osint_audit_log.o mask_v7_sve2.o tinycbor.o
IATO_SVE2_OBJ = iato_mask_sve2.o
IATO_BRIDGE_OBJ = iato_hsm_bridge.o
RESULT_JSON = result.json
XDP_SRC = formal/ebpf/osint_dispatcher_xdp_firewall.c
XDP_OBJ = formal/ebpf/osint_dispatcher_xdp_firewall.o
SNARK_XDP_SRC = ebpf/osint_snark_bridge.bpf.c
SNARK_XDP_OBJ = ebpf/osint_snark_bridge.bpf.o

.PHONY: all clean run parse-json log-restful xdp-build snark-xdp-build verify-ebpf

all: $(DISPATCHER)

ifeq ($(ENABLE_IATO_SVE2),1)
C_OBJ += $(IATO_BRIDGE_OBJ)
EXTRA_LINK_OBJS += $(IATO_SVE2_OBJ)
endif

$(ASM_OBJ): fast_mask.asm
	$(NASM) -f elf64 -o $@ $<

dispatcher.o: dispatcher.c pkcs11_signer.h osint_audit_log.h
	$(CC) $(CFLAGS) -c -o $@ $<


pkcs11_signer.o: pkcs11_signer.c pkcs11_signer.h
	$(CC) $(CFLAGS) -c -o $@ $<

osint_audit_log.o: osint_audit_log.c osint_audit_log.h tinycbor.h
	$(CC) $(CFLAGS) -c -o $@ $<

mask_v7_sve2.o: mask_v7_sve2.c osint_audit_log.h
	$(CC) $(CFLAGS) -c -o $@ $<

iato_hsm_bridge.o: src/iato_hsm_bridge.c include/iato/rme_mgmt.h include/iato/sve_mask.h pkcs11_signer.h
	$(CC) $(CFLAGS) -Iinclude -c -o $@ $<

iato_mask_sve2.o: src/iato_mask_sve2.S
	$(CC) $(IATO_SVE2_CFLAGS) -c -o $@ $<

tinycbor.o: tinycbor.c tinycbor.h
	$(CC) $(CFLAGS) -c -o $@ $<

$(DISPATCHER): $(ASM_OBJ) $(C_OBJ) $(EXTRA_LINK_OBJS)
	$(CC) -o $@ $(ASM_OBJ) $(C_OBJ) $(EXTRA_LINK_OBJS) $(LDFLAGS) -ldl -lcrypto

parse-json:
	@test -n "$(TARGET_POLICY)" || (echo "ERROR: TARGET_POLICY is required" && exit 1)
	python3 iam_parser.py --file "$(TARGET_POLICY)" > $(RESULT_JSON)

run: all
	@test -n "$(TARGET_POLICY)" || (echo "ERROR: TARGET_POLICY is required" && exit 1)
	./$(DISPATCHER) parse-iam "$(TARGET_POLICY)"

log-restful: run

xdp-build: $(XDP_OBJ)

snark-xdp-build: $(SNARK_XDP_OBJ)

$(XDP_OBJ): $(XDP_SRC)
	$(CLANG) $(XDP_CFLAGS) -c -o $@ $<

$(SNARK_XDP_OBJ): $(SNARK_XDP_SRC) ebpf/snark_fpga_mmio.h
	$(CLANG) $(XDP_CFLAGS) -c -o $@ $<

verify-ebpf: $(XDP_SRC)
	@if command -v $(KANI) >/dev/null 2>&1; then \
		echo "Running Kani Rust Verifier over C source for anomaly scanning"; \
		$(KANI) c $<; \
	elif command -v $(CONTROL_FLAG) >/dev/null 2>&1; then \
		echo "Running ControlFlag over eBPF source for anomaly scanning"; \
		$(CONTROL_FLAG) check $<; \
	else \
		echo "ERROR: neither $(KANI) nor $(CONTROL_FLAG) is installed"; \
		exit 1; \
	fi

clean:
	rm -f $(DISPATCHER) $(ASM_OBJ) dispatcher.o pkcs11_signer.o osint_audit_log.o mask_v7_sve2.o tinycbor.o $(IATO_BRIDGE_OBJ) $(IATO_SVE2_OBJ) $(RESULT_JSON) $(XDP_OBJ) $(SNARK_XDP_OBJ)
