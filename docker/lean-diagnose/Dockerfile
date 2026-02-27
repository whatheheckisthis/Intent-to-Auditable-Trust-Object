FROM alpine:3.20

RUN apk add --no-cache bash ripgrep

WORKDIR /workspace
ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["/workspace/formal/lean/scripts/virtual_lean_diagnose.sh"]
