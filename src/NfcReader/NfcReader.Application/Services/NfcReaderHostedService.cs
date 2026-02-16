using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using NfcReader.Application.Abstractions;
using NfcReader.Domain.Entities;

namespace NfcReader.Application.Services;

public sealed class NfcReaderHostedService(
    INfcReader nfcReader,
    INfcTagStream tagStream,
    ILogger<NfcReaderHostedService> logger) : IHostedService
{
    public async Task StartAsync(CancellationToken cancellationToken)
    {
        nfcReader.TagRead += OnTagReadAsync;
        await nfcReader.StartAsync(cancellationToken);
        logger.LogInformation("NFC reader service started.");
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        nfcReader.TagRead -= OnTagReadAsync;
        await nfcReader.StopAsync(cancellationToken);
        logger.LogInformation("NFC reader service stopped.");
    }

    private async Task OnTagReadAsync(NfcTagRead tag, CancellationToken cancellationToken)
    {
        logger.LogInformation(
            "NFC tag read from {ReaderName}: UID={Uid}, ATR={Atr}",
            tag.ReaderName,
            tag.Uid,
            tag.Atr ?? "<none>");

        await tagStream.PublishAsync(tag, cancellationToken);
    }
}
