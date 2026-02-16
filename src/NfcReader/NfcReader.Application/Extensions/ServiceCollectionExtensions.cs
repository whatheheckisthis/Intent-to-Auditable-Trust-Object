using Microsoft.Extensions.DependencyInjection;
using NfcReader.Application.Abstractions;
using NfcReader.Application.Services;

namespace NfcReader.Application.Extensions;

public static class ServiceCollectionExtensions
{
    public static IServiceCollection AddNfcApplication(this IServiceCollection services)
    {
        services.AddSingleton<INfcTagStream, InMemoryNfcTagStream>();
        services.AddHostedService<NfcReaderHostedService>();
        return services;
    }
}
