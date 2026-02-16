using Microsoft.Extensions.DependencyInjection;
using NfcReader.Application.Abstractions;
using NfcReader.Infrastructure.Readers;

namespace NfcReader.Infrastructure.Extensions;

public static class ServiceCollectionExtensions
{
    public static IServiceCollection AddNfcInfrastructure(this IServiceCollection services)
    {
        services.AddSingleton<INfcReader, PcscNfcReader>();
        return services;
    }
}
