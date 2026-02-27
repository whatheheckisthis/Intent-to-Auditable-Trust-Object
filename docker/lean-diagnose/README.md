# Lean Diagnose Compose Stack

This compose stack runs Lean diagnostics in a background container and exposes the
latest report through a reverse proxy web server.

## Services

- `lean-diagnose`: executes `formal/lean/scripts/virtual_lean_diagnose.sh` every 5 minutes and writes logs to a shared volume.
- `lean-diagnose-web`: serves the shared report volume over HTTP on internal port `3000`.
- `lean-diagnose-proxy`: nginx reverse proxy exposing the web service on `http://localhost:8088`.

## Run (detached)

```bash
./scripts/run-lean-diagnose-compose.sh
```

Then open:

- `http://127.0.0.1:8088/`
- `http://127.0.0.1:8088/latest.log`
- `http://127.0.0.1:8088/healthz`

## Stop

```bash
docker compose -f docker/lean-diagnose/docker-compose.yml down
```
