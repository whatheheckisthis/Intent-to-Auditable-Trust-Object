FROM ubuntu:24.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential clang llvm nasm libbpf-dev libelf-dev libssl-dev libpkcs11-helper1-dev pkg-config ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ato
COPY dispatcher.c pkcs11_signer.c pkcs11_signer.h fast_mask.asm Makefile ./

RUN make dispatcher LDFLAGS="-lbpf"

RUN mkdir -p /var/log/audit && touch /var/log/audit/signed_evidence.log

ENV FLOW_MAP_PATH=/sys/fs/bpf/flow_map
ENV SIGNED_EVIDENCE_LOG=/var/log/audit/signed_evidence.log
ENV DISPATCH_INTERVAL_S=5

CMD ["/opt/ato/dispatcher"]
