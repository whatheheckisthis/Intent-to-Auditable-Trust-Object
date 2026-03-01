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
