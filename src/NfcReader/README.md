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
