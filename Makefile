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
C_OBJ = dispatcher.o
RESULT_JSON = result.json

.PHONY: all clean run parse-json log-restful

all: $(DISPATCHER)

$(ASM_OBJ): fast_mask.asm
	$(NASM) -f elf64 -o $@ $<

$(C_OBJ): dispatcher.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(DISPATCHER): $(ASM_OBJ) $(C_OBJ)
	$(CC) -o $@ $(ASM_OBJ) $(C_OBJ) $(LDFLAGS)

parse-json:
	@test -n "$(TARGET_POLICY)" || (echo "ERROR: TARGET_POLICY is required" && exit 1)
	python3 iam_parser.py --file "$(TARGET_POLICY)" > $(RESULT_JSON)

run: all
	@test -n "$(TARGET_POLICY)" || (echo "ERROR: TARGET_POLICY is required" && exit 1)
	./$(DISPATCHER) "$(TARGET_POLICY)"

log-restful: parse-json
	@test -n "$(API_KEY)" || (echo "ERROR: API_KEY is required" && exit 1)
	@test -n "$(REST_URL)" || (echo "ERROR: REST_URL is required" && exit 1)
	curl --fail --silent --show-error -X POST "$(REST_URL)" \
		-H "Authorization: Bearer $(API_KEY)" \
		-H "Content-Type: application/json" \
		-d @$(RESULT_JSON)

clean:
	rm -f $(DISPATCHER) $(ASM_OBJ) $(C_OBJ) $(RESULT_JSON)
