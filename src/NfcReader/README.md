# NFC Reader Worker (.NET 8)

This solution provides a Clean Architecture-based NFC reader using PC/SC and SignalR.

## Projects

- **NfcReader.Domain**: Domain model (`NfcTagRead`).
- **NfcReader.Application**: Abstractions and orchestration (`INfcReader`, in-memory stream, hosted service).
- **NfcReader.Infrastructure**: PC/SC implementation (`PcscNfcReader`) that reads card UIDs.
- **NfcReader.Worker**: Console/worker host with ASP.NET Core SignalR hub and static browser UI.

## Runtime flow

1. `PcscNfcReader` polls PC/SC readers and emits tag reads.
2. `NfcReaderHostedService` publishes tag events into an application stream.
3. `SignalRTagBroadcastService` consumes stream events and broadcasts `TagRead` to all hub clients.
4. Browser clients connect to `/hubs/nfc` and render incoming tag data.

## Run

```bash
dotnet restore src/NfcReader/NfcReader.sln
dotnet run --project src/NfcReader/NfcReader.Worker/NfcReader.Worker.csproj
```

Open http://localhost:5000 to view live events.


## Environment bootstrap (Microsoft install-script flow)

If `dotnet` is missing, bootstrap a **pinned** .NET SDK using the same approach documented by Microsoft (`dotnet-install.sh`):

```bash
DOTNET_VERSION=8.0.404 PYTEST_VERSION=9.0.2 bash scripts/setup-dotnet-lts.sh
source ~/.bashrc
```

Manual equivalent:

```bash
curl -fsSL https://dot.net/v1/dotnet-install.sh -o /tmp/dotnet-install.sh
bash /tmp/dotnet-install.sh --version 8.0.404 --install-dir "$HOME/.dotnet"
export DOTNET_ROOT="$HOME/.dotnet"
export PATH="$DOTNET_ROOT:$DOTNET_ROOT/tools:$PATH"
```

Verify toolchain:

```bash
dotnet --version
pytest --version
dotnet test src/NfcReader/NfcReader.sln
```

If downloads fail with `403 Forbidden`, the environment is behind a restricted proxy and the .NET installer must be allowlisted or mirrored internally.


## Zero-network CI bootstrap (staged archive)

For restricted workers, stage a `.NET SDK` archive at one of:

- `/opt/bootstrap/dotnet-sdk-<version>-linux-x64.tar.gz` or `...-linux-arm64.tar.gz`
- `$HOME/bootstrap/dotnet-sdk-<version>-linux-x64.tar.gz` or `...-linux-arm64.tar.gz`
- `/workspace/bootstrap/dotnet-sdk-<version>-linux-x64.tar.gz` or `...-linux-arm64.tar.gz`

Then run:

```bash
bash scripts/ensure-dotnet.sh

dotnet --version
dotnet build src/NfcReader/NfcReader.sln
```

### Example CI snippet

```bash
set -euo pipefail

# No external downloads performed by this flow.
bash scripts/ensure-dotnet.sh

dotnet --version
dotnet build src/NfcReader/NfcReader.sln
```

Notes:
- Architecture-aware archive selection supports `linux-x64` and `linux-arm64` archive names.
- Optional integrity verification is enabled automatically when `<archive>.sha512` is present.
- Fail-fast behavior uses exit code `42` when no staged archive is available.


## Offline build staging (compile-only, EL2-safe)

Use the staging workflow when the worker must validate compile readiness only:

```bash
bash scripts/ensure-dotnet.sh
source scripts/activate-dotnet-offline.sh
bash scripts/build-nfcreader-offline.sh
```

Behavior:
- Never downloads SDKs or system packages.
- Uses staged SDK archive recovery only.
- Builds with `--no-restore -p:RestorePackages=false` to avoid package fetches.
- Compile-only validation: no runtime execution, no MMIO access, no hypervisor calls.
- EL2/runtime paths must remain gated (for C/C++ modules, use `#ifdef ENABLE_EL2_RUNTIME`; for managed projects, keep runtime entrypoints disabled in build staging).
