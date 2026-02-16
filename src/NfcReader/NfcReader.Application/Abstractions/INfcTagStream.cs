using NfcReader.Domain.Entities;

namespace NfcReader.Application.Abstractions;

public interface INfcTagStream
{
    ValueTask PublishAsync(NfcTagRead tag, CancellationToken cancellationToken);

    IAsyncEnumerable<NfcTagRead> SubscribeAsync(CancellationToken cancellationToken);
}
