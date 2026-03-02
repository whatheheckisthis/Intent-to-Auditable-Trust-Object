# IATO_V7 Lean Package

## Install Lean/Lake locally

Use the helper script:

```bash
./scripts/install_lake.sh
```

It attempts:
1. Reuse existing `lake` if installed.
2. Install `elan` from GitHub release.
3. Fallback to `apt-get install elan`.
4. Install Lean stable toolchain and verify `lean`/`lake`.

## Build and test

```bash
lake update
lake build
lake test
lake exe cache
lake aot
```


## Build environment variables

Before building, load the environment helper:

```bash
source scripts/env.sh
```

This exports:
- `ELAN_HOME` and prepends `$ELAN_HOME/bin` to `PATH`
- `LAKE` pointing to the elan-managed `lake` binary
- `LEAN_ABORT_ON_PANIC=1` for fail-fast behavior
- `LEAN_PATH` defaulting to the local `.lake/packages`
- `LAKE_NO_CACHE=0` (cache enabled)


## Install Git GUI

Use:

```bash
./scripts/install_git_gui.sh
```

What it does:
- Installs `git` and `git-gui` via apt.
- Prints proxy-related diagnostics if apt fails (common in restricted environments).
- Adds `alias ggui="git gui"` to `~/.bashrc` for convenience.


## Install Podman

Use:

```bash
./scripts/install_podman.sh
```

What it does:
- Installs `podman`, `podman-compose`, and rootless container dependencies.
- Prints proxy diagnostics when apt is blocked.
- Ensures `scripts/env.sh` includes Podman defaults for portability.

## Podman environment variables for portability

Load env variables before running container builds:

```bash
source scripts/env.sh
```

Important variables:
- `PODMAN_USERNS=keep-id` to reduce host/container UID mismatch issues.
- `PODMAN_SYSTEMD_UNIT=false` to avoid systemd unit expectations in simple sessions.
- `COMPOSE_DOCKER_CLI_BUILD=1` for compatible compose build behavior.

## Kubernetes fallback job (proxy-conflict mitigation)

For environments where local container networking/proxy behavior differs from Podman,
you can run the same build as a standard Kubernetes Job:

```bash
kubectl apply -f k8s-build-job.yaml
kubectl logs -f job/iato-v7-lean-build
```

The job includes `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` fields so cluster-level
proxy values can be managed consistently.


## Fail-forward Defensive DevOps strategy

This project intentionally prioritizes availability and execution over architectural purity:

- **Normalization as a Filter**: `scripts/normalize_ci_env.sh` acts as the source of truth for CI proxy state.
  It maps upper/lower-case proxy variables and writes deterministic values to `GITHUB_ENV` so Lean/Lake
  and container tooling do not inherit host-specific ambiguity.
- **Podman as the low-cost path**: `podman-compose.yml` is the first, fastest daemonless execution path
  for terminal-centric workflows.
- **Kubernetes as the guaranteed path**: `k8s-build-job.yaml` is the fallback when local DNS/proxy
  interception causes Podman networking mismatches, using cluster-native service discovery and egress policy.
