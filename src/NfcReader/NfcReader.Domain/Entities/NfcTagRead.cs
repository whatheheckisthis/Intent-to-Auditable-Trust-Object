namespace NfcReader.Domain.Entities;

public sealed record NfcTagRead(
    string ReaderName,
    string Uid,
    string? Atr,
    string? RawDataHex,
    DateTimeOffset ReadAtUtc);
