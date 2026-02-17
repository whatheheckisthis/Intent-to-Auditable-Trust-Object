FROM golang:1.22-bookworm AS builder
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends \
    clang llvm libbpf-dev libelf-dev pkg-config gcc make \
    && rm -rf /var/lib/apt/lists/*
COPY netflow-sidecar/go.mod ./go.mod
COPY netflow-sidecar/main.go ./main.go
RUN go mod download
RUN CGO_ENABLED=1 go build -o /out/netflow-sidecar ./main.go

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    clang \
    llvm \
    bpftool \
    libbpf-dev \
    iproute2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/netflow
COPY xdp_audit.c /opt/netflow/xdp_audit.c
COPY netflow-sidecar/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
COPY --from=builder /out/netflow-sidecar /usr/local/bin/netflow-sidecar

ENTRYPOINT ["/entrypoint.sh"]
