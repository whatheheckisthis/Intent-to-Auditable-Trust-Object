using NfcReader.Application.Extensions;
using NfcReader.Infrastructure.Extensions;
using NfcReader.Worker.Hubs;
using NfcReader.Worker.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services
    .AddNfcApplication()
    .AddNfcInfrastructure();

builder.Services.AddSignalR();
builder.Services.AddHostedService<SignalRTagBroadcastService>();

var app = builder.Build();

app.UseDefaultFiles();
app.UseStaticFiles();

app.MapHub<NfcHub>("/hubs/nfc");
app.MapGet("/health", () => Results.Ok(new { status = "ok" }));

app.Run();
