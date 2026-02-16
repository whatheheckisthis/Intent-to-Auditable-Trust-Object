using NfcReader.Domain.Entities;

namespace NfcReader.Application.Abstractions;

public interface INfcReader
{
    event Func<NfcTagRead, CancellationToken, Task>? TagRead;

    Task StartAsync(CancellationToken cancellationToken);

    Task StopAsync(CancellationToken cancellationToken);
}
