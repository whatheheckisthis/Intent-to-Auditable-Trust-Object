FROM debian:bookworm-slim AS build-c

WORKDIR /src

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc-aarch64-linux-gnu make python3 python3-venv python3-pip ca-certificates swtpm tpm2-tools \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN make tests

FROM build-c AS test

WORKDIR /src
RUN make check-el2 check-nfc check-spdm check-hardening check-infra

FROM gcr.io/distroless/cc-debian12 AS runtime

WORKDIR /app
COPY --from=build-c /src/tests/test_enrollment /app/test_enrollment
COPY --from=build-c /src/tests/test_spdm_binding /app/test_spdm_binding
COPY --from=build-c /src/tests/test_two_factor_gate /app/test_two_factor_gate
COPY --from=build-c /src/tests/test_expiry_sweep /app/test_expiry_sweep
COPY --from=build-c /src/tests/test_replay_defense /app/test_replay_defense

ENTRYPOINT ["/app/test_enrollment"]
