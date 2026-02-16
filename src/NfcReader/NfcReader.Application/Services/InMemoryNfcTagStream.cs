using System.Threading.Channels;
using NfcReader.Application.Abstractions;
using NfcReader.Domain.Entities;

namespace NfcReader.Application.Services;

public sealed class InMemoryNfcTagStream : INfcTagStream
{
    private readonly Channel<NfcTagRead> _channel = Channel.CreateUnbounded<NfcTagRead>(new UnboundedChannelOptions
    {
        SingleReader = false,
        SingleWriter = false
    });

    public ValueTask PublishAsync(NfcTagRead tag, CancellationToken cancellationToken)
        => _channel.Writer.WriteAsync(tag, cancellationToken);

    public IAsyncEnumerable<NfcTagRead> SubscribeAsync(CancellationToken cancellationToken)
        => _channel.Reader.ReadAllAsync(cancellationToken);
}
