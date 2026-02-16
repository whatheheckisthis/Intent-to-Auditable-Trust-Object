using Microsoft.AspNetCore.SignalR;
using NfcReader.Application.Abstractions;
using NfcReader.Worker.Hubs;

namespace NfcReader.Worker.Services;

public sealed class SignalRTagBroadcastService(
    INfcTagStream tagStream,
    IHubContext<NfcHub> hubContext,
    ILogger<SignalRTagBroadcastService> logger) : BackgroundService
{
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        await foreach (var tag in tagStream.SubscribeAsync(stoppingToken))
        {
            await hubContext.Clients.All.SendAsync("TagRead", tag, stoppingToken);
            logger.LogInformation("Broadcasted tag {Uid} from reader {ReaderName}", tag.Uid, tag.ReaderName);
        }
    }
}
