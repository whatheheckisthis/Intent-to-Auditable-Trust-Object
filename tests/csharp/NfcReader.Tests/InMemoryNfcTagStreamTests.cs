using NfcReader.Application.Services;
using NfcReader.Domain.Entities;

namespace NfcReader.Tests;

public class InMemoryNfcTagStreamTests
{
    [Fact]
    public async Task PublishAsync_ShouldExposePublishedTagToSubscriber()
    {
        var stream = new InMemoryNfcTagStream();
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));

        var expected = BuildTag("reader-a", "04AABBCCDD", "11223344");

        await stream.PublishAsync(expected, cts.Token);

        var actual = await ReadNextAsync(stream, cts.Token);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public async Task PublishAsync_ShouldPreserveOrderAcrossMultipleMessages()
    {
        var stream = new InMemoryNfcTagStream();
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));

        var first = BuildTag("reader-a", "UID-001", "A1");
        var second = BuildTag("reader-a", "UID-002", "A2");

        await stream.PublishAsync(first, cts.Token);
        await stream.PublishAsync(second, cts.Token);

        var observedFirst = await ReadNextAsync(stream, cts.Token);
        var observedSecond = await ReadNextAsync(stream, cts.Token);

        Assert.Equal(first, observedFirst);
        Assert.Equal(second, observedSecond);
    }

    private static NfcTagRead BuildTag(string readerName, string uid, string rawDataHex) =>
        new(
            ReaderName: readerName,
            Uid: uid,
            Atr: null,
            RawDataHex: rawDataHex,
            ReadAtUtc: DateTimeOffset.UtcNow);

    private static async Task<NfcTagRead> ReadNextAsync(InMemoryNfcTagStream stream, CancellationToken cancellationToken)
    {
        await foreach (var tag in stream.SubscribeAsync(cancellationToken))
        {
            return tag;
        }

        throw new InvalidOperationException("No tag was published before the subscription completed.");
    }
}
