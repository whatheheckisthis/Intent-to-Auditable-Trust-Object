# Load variables from .env if present
-include .env

CC ?= gcc
NASM ?= nasm
CFLAGS ?= -O2 -Wall -Wextra
LDFLAGS ?=

API_KEY ?=
REST_URL ?=
TARGET_POLICY ?=

DISPATCHER = dispatcher
ASM_OBJ = fast_mask.o
C_OBJ = dispatcher.o pkcs11_signer.o
RESULT_JSON = result.json

.PHONY: all clean run parse-json log-restful

all: $(DISPATCHER)

$(ASM_OBJ): fast_mask.asm
	$(NASM) -f elf64 -o $@ $<

dispatcher.o: dispatcher.c pkcs11_signer.h
	$(CC) $(CFLAGS) -c -o $@ $<

pkcs11_signer.o: pkcs11_signer.c pkcs11_signer.h
	$(CC) $(CFLAGS) -c -o $@ $<

$(DISPATCHER): $(ASM_OBJ) $(C_OBJ)
	$(CC) -o $@ $(ASM_OBJ) $(C_OBJ) $(LDFLAGS) -ldl -lcrypto

parse-json:
	@test -n "$(TARGET_POLICY)" || (echo "ERROR: TARGET_POLICY is required" && exit 1)
	python3 iam_parser.py --file "$(TARGET_POLICY)" > $(RESULT_JSON)

run: all
	@test -n "$(TARGET_POLICY)" || (echo "ERROR: TARGET_POLICY is required" && exit 1)
	./$(DISPATCHER) parse-iam "$(TARGET_POLICY)"

log-restful: run

clean:
	rm -f $(DISPATCHER) $(ASM_OBJ) dispatcher.o pkcs11_signer.o $(RESULT_JSON)
