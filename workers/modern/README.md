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

This creates `workers/modern/<worker_id>/worker.json`.

## Predefined producer worker

`workers/modern/readme_conflict_producer/worker.json` defines a producer worker that marks `README.md`
changes as non-blocking by using the Git union merge policy (`README.md merge=union`).
This worker explicitly does not require environment variables.
