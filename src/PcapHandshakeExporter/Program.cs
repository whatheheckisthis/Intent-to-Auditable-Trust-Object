using System.Globalization;
using PacketDotNet;
using SharpPcap;

if (args.Length == 0)
{
    Console.WriteLine("Usage: dotnet run -- <path-to-pcap-file>");
    return;
}

var pcapFilePath = args[0];
if (!File.Exists(pcapFilePath))
{
    Console.Error.WriteLine($"PCAP file not found: {pcapFilePath}");
    return;
}

const string outputFileName = "handshake_data.csv";

using var writer = new StreamWriter(outputFileName, false);
writer.WriteLine("Timestamp,SourceIP,DestinationIP,SequenceNumber,AcknowledgmentNumber,HandshakeType");

using var device = new CaptureFileReaderDevice(pcapFilePath);
device.Open();

RawCapture? rawCapture;
while ((rawCapture = device.GetNextPacket()) != null)
{
    var packet = Packet.ParsePacket(rawCapture.LinkLayerType, rawCapture.Data);
    var ipPacket = packet.Extract<IPPacket>();
    var tcpPacket = packet.Extract<TcpPacket>();

    if (ipPacket is null || tcpPacket is null)
    {
        continue;
    }

    var handshakeType = GetHandshakeType(tcpPacket);
    if (handshakeType is null)
    {
        continue;
    }

    var timestamp = rawCapture.Timeval.Date.ToString("O", CultureInfo.InvariantCulture);
    var row = string.Join(",",
        CsvEscape(timestamp),
        CsvEscape(ipPacket.SourceAddress.ToString()),
        CsvEscape(ipPacket.DestinationAddress.ToString()),
        tcpPacket.SequenceNumber.ToString(CultureInfo.InvariantCulture),
        tcpPacket.AcknowledgmentNumber.ToString(CultureInfo.InvariantCulture),
        handshakeType);

    writer.WriteLine(row);
}

Console.WriteLine($"Handshake packet data exported to {Path.GetFullPath(outputFileName)}");

static string? GetHandshakeType(TcpPacket tcpPacket)
{
    if (tcpPacket.Syn && tcpPacket.Ack)
    {
        return "SYN-ACK";
    }

    if (tcpPacket.Syn && !tcpPacket.Ack)
    {
        return "SYN";
    }

    if (!tcpPacket.Syn && tcpPacket.Ack)
    {
        return "ACK";
    }

    return null;
}

static string CsvEscape(string input)
{
    if (!input.Contains(',') && !input.Contains('"') && !input.Contains('\n'))
    {
        return input;
    }

    var escaped = input.Replace("\"", "\"\"");
    return $"\"{escaped}\"";
}
