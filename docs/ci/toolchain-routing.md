# .NET Toolchain Routing for Hardened CI Workers

This repository supports a two-worker CI model that treats `dotnet` exit code `127` on hardened workers as an **expected** signal and routes builds to a dedicated toolchain worker.

## Trust Boundaries

- **Primary worker (hardened-primary label)**
  - No SDK installation.
  - No package installation.
  - No outbound dependency on external proxy/network services.
  - Performs detection and secure pipeline controls only.
  - Must not execute EL2 or privileged runtime instructions during build stages.

- **Secondary worker (toolchain-dotnet label)**
  - Hosts the .NET SDK and internal/offline NuGet cache.
  - Performs restore/build/publish/pack deterministically.
  - Returns artifacts to the secure pipeline as untrusted outputs.

- **Secure continuation stage**
  - Runs on hardened worker.
  - Validates archive integrity before trust elevation.
  - Leaves signing, attestation, and EL2 runtime validation in dedicated privileged stages.

## Stage Flow

1. **Detect toolchain availability**
   - `scripts/ci/detect-dotnet-and-route.sh` returns:
     - `0` when `dotnet` is available.
     - `127` when blocked/missing (expected routing path).
   - Any other exit code is treated as unexpected and fails fast.

2. **Route build to toolchain worker**
   - Workflow output `route_to_toolchain=true` gates the routed build job.
   - No retries against external package feeds are required for routing.

3. **Run deterministic build on secondary worker**
   - `scripts/ci/build-on-toolchain-worker.sh` performs:
     - `dotnet restore --ignore-failed-sources` with local package cache.
     - `dotnet build` in `Release` with deterministic settings.
     - `dotnet publish` output to `artifacts/toolchain-output/output/`.
     - optional package builds for `NfcReader` and `NfcReader.Tools` when project files exist.

4. **Package and return artifacts**
   - Generated layout:

     ```text
     artifacts/toolchain-output/
       output/
         *.dll
         *.deps.json
         *.runtimeconfig.json
       packages/
         *.nupkg
       SHA256SUMS.txt
       package.tar.gz
     ```

   - Artifact is uploaded from the toolchain worker and downloaded in the secure continuation job.

5. **Resume secure pipeline**
   - Hardened stage validates archive structure.
   - Signing/attestation and EL2 checks remain separate from build routing.

## Local Validation

You can test detection behavior without installing SDKs:

```bash
./scripts/ci/detect-dotnet-and-route.sh; echo "$?"
```

Expected behavior in hardened environments: `127` and a `[router][error] dotnet missing...` diagnostic.
