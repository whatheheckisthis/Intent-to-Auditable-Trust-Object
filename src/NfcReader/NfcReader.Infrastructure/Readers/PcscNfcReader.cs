using Microsoft.Extensions.Logging;
using NfcReader.Application.Abstractions;
using NfcReader.Domain.Entities;
using PCSC;

namespace NfcReader.Infrastructure.Readers;

public sealed class PcscNfcReader(ILogger<PcscNfcReader> logger) : INfcReader
{
    private readonly ILogger<PcscNfcReader> _logger = logger;
    private readonly CancellationTokenSource _internalCts = new();
    private Task? _worker;

    public event Func<NfcTagRead, CancellationToken, Task>? TagRead;

    public Task StartAsync(CancellationToken cancellationToken)
    {
        if (_worker is not null)
        {
            return Task.CompletedTask;
        }

        var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, _internalCts.Token);
        _worker = Task.Run(() => PollLoopAsync(linkedCts.Token), linkedCts.Token);
        return Task.CompletedTask;
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        _internalCts.Cancel();

        if (_worker is null)
        {
            return;
        }

        await Task.WhenAny(_worker, Task.Delay(TimeSpan.FromSeconds(5), cancellationToken));
    }

    private async Task PollLoopAsync(CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                await PollReadersOnceAsync(cancellationToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Unexpected error while polling NFC readers.");
                await Task.Delay(TimeSpan.FromSeconds(2), cancellationToken);
            }
        }
    }

    private async Task PollReadersOnceAsync(CancellationToken cancellationToken)
    {
        using var context = ContextFactory.Instance.Establish(SCardScope.System);
        var readers = context.GetReaders();

        if (readers is null || readers.Length == 0)
        {
            _logger.LogDebug("No PC/SC readers available.");
            await Task.Delay(TimeSpan.FromSeconds(1), cancellationToken);
            return;
        }

        var states = readers
            .Select(reader => new SCardReaderState
            {
                ReaderName = reader,
                CurrentStateValue = SCRState.Unaware
            })
            .ToArray();

        var rc = context.GetStatusChange(SCardReader.Infinite, states);
        if (rc != SCardError.Success)
        {
            _logger.LogWarning("GetStatusChange failed: {Error}", SCardHelper.StringifyError(rc));
            await Task.Delay(TimeSpan.FromMilliseconds(500), cancellationToken);
            return;
        }

        foreach (var state in states)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var cardPresent = state.EventState.HasFlag(SCRState.Present);
            var stateChanged = state.EventState.HasFlag(SCRState.Changed);
            if (!cardPresent || !stateChanged)
            {
                continue;
            }

            var tag = TryReadTag(context, state.ReaderName);
            if (tag is null || TagRead is null)
            {
                continue;
            }

            await TagRead.Invoke(tag, cancellationToken);
        }
    }

    private NfcTagRead? TryReadTag(ISCardContext context, string readerName)
    {
        using var cardReader = new SCardReader(context);

        var connectRc = cardReader.Connect(readerName, SCardShareMode.Shared, SCardProtocol.Any);
        if (connectRc != SCardError.Success)
        {
            _logger.LogWarning(
                "Failed connecting to reader {ReaderName}. Error: {Error}",
                readerName,
                SCardHelper.StringifyError(connectRc));
            return null;
        }

        var getUidApdu = new byte[] { 0xFF, 0xCA, 0x00, 0x00, 0x00 };
        var receiveBuffer = new byte[256];

        var sendPci = SCardPCI.GetPci(cardReader.ActiveProtocol);
        var transmitRc = cardReader.Transmit(sendPci, getUidApdu, receiveBuffer, out var receivedLength);

        if (transmitRc != SCardError.Success || receivedLength < 2)
        {
            _logger.LogWarning(
                "Failed reading UID from reader {ReaderName}. Error: {Error}",
                readerName,
                SCardHelper.StringifyError(transmitRc));
            return null;
        }

        var payloadLength = receivedLength - 2;
        var uidBytes = receiveBuffer.Take(payloadLength).ToArray();

        cardReader.Status(out _, out _, out var protocol, out var atr);

        return new NfcTagRead(
            readerName,
            ToHex(uidBytes),
            atr is { Length: > 0 } ? ToHex(atr) : null,
            ToHex(receiveBuffer.AsSpan(0, receivedLength)),
            DateTimeOffset.UtcNow);
    }

    private static string ToHex(ReadOnlySpan<byte> data)
        => Convert.ToHexString(data);
}
