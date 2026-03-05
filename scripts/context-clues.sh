#!/usr/bin/env bash
set -euo pipefail

echo "IĀTŌ-V7 Linux context"
echo "- Primary runtime: Linux shell (Ubuntu/WSL2/Minikube host shell)"
echo "- Common Linux targets: /etc, /opt, /var/lib"
echo "- Ensure tools exist: python3 and nmap"
echo "- Entrypoint from repo root: iato-scan --config config.local.toml"
echo "- Output artifact: lean/iato_v7/nmap-path-state.xml"
echo "- Use --dry-run first to validate deterministic command generation"
