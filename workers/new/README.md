# New Workers

IATO-V7-compatible worker outputs are written here.

## Setup a new worker

Use the repository helper script to scaffold a worker manifest:

```bash
./scripts/setup_worker.sh <worker_id> <world:rme|normal> [deps_pipe_separated]
```

Example:

```bash
./scripts/setup_worker.sh worker3 rme alpha|beta
```

This creates `workers/new/<worker_id>/worker.json`.

