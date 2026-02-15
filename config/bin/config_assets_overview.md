# Config assets overview

This document summarizes configuration-related assets under `config/`.

## Active folders

- `config/env/` — environment defaults, runtime settings, security/logging configuration, and key-generation helpers.
- `config/py/` — Python bootstrap/backend support modules.
- `config/scripts/` — setup and environment check scripts.
- `config/bin/` — shell/bootstrap helpers and this overview.

## Notes

- Treat `config/env/defaults.yaml`, `config/env/security.yaml`, and `config/env/logging.yaml` as baseline configuration sources.
- `config/env/runtime.env` contains runtime environment-style values.
- Keep file names and references synchronized with scripts when moving or renaming config files.
