FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bpfcc-tools \
    python3 \
    python3-bpfcc \
    iproute2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/xdp
COPY xdp_audit.c /opt/xdp/xdp_audit.c
COPY agent.py /opt/xdp/agent.py

CMD ["python3", "/opt/xdp/agent.py"]
