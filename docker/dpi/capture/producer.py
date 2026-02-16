import json
import os
import signal
import threading
import time
from hashlib import blake2b
from typing import Any, Dict, Optional

from kafka import KafkaProducer
from scapy.all import IP, IPv6, TCP, UDP, ICMP, ICMPv6EchoRequest, Raw, sniff

BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "raw.packets")
INTERFACE = os.getenv("CAPTURE_INTERFACE", "any")
BPF_FILTER = os.getenv("BPF_FILTER", "ip or ip6")
BATCH_SIZE = int(os.getenv("KAFKA_BATCH_SIZE", "131072"))
LINGER_MS = int(os.getenv("KAFKA_LINGER_MS", "5"))
COMPRESSION = os.getenv("KAFKA_COMPRESSION", "lz4")

RUNNING = True


def normalize_transport(packet: Any) -> tuple[int, int, str]:
    if TCP in packet:
        layer = packet[TCP]
        return int(layer.sport), int(layer.dport), "TCP"
    if UDP in packet:
        layer = packet[UDP]
        return int(layer.sport), int(layer.dport), "UDP"
    if ICMP in packet or ICMPv6EchoRequest in packet:
        return 0, 0, "ICMP"
    return 0, 0, "OTHER"


def packet_to_record(packet: Any) -> Optional[Dict[str, Any]]:
    ip_layer = None
    version = 0
    if IP in packet:
        ip_layer = packet[IP]
        version = 4
    elif IPv6 in packet:
        ip_layer = packet[IPv6]
        version = 6

    if ip_layer is None:
        return None

    src = str(ip_layer.src)
    dst = str(ip_layer.dst)
    sport, dport, transport = normalize_transport(packet)
    proto = transport if transport != "OTHER" else str(getattr(ip_layer, "proto", getattr(ip_layer, "nh", "UNK")))
    length = int(len(packet))
    flow_key = f"{src}|{dst}|{sport}|{dport}|{proto}|{version}"
    flow_id = blake2b(flow_key.encode("utf-8"), digest_size=12).hexdigest()

    return {
        "ts_ns": time.time_ns(),
        "flow_id": flow_id,
        "src_ip": src,
        "dst_ip": dst,
        "src_port": sport,
        "dst_port": dport,
        "protocol": proto,
        "ip_version": version,
        "packet_bytes": length,
        "payload_bytes": len(packet[Raw].load) if Raw in packet else 0,
        "ttl": int(getattr(ip_layer, "ttl", getattr(ip_layer, "hlim", 0))),
    }


def on_send_error(exc: BaseException) -> None:
    print(f"[capture] kafka send error: {exc}", flush=True)


def packet_handler(packet: Any, producer: KafkaProducer) -> None:
    record = packet_to_record(packet)
    if not record:
        return

    future = producer.send(TOPIC, value=record)
    future.add_errback(on_send_error)


def stop_signal_handler(signum: int, _frame: Any) -> None:
    global RUNNING
    RUNNING = False
    print(f"[capture] received signal {signum}, stopping capture loop...", flush=True)


def capture_loop(producer: KafkaProducer) -> None:
    while RUNNING:
        sniff(
            iface=INTERFACE,
            filter=BPF_FILTER,
            store=False,
            prn=lambda pkt: packet_handler(pkt, producer),
            timeout=1,
        )


def main() -> None:
    signal.signal(signal.SIGINT, stop_signal_handler)
    signal.signal(signal.SIGTERM, stop_signal_handler)

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v, separators=(",", ":")).encode("utf-8"),
        acks=1,
        linger_ms=LINGER_MS,
        batch_size=BATCH_SIZE,
        compression_type=COMPRESSION,
        max_in_flight_requests_per_connection=5,
        retries=5,
    )

    print(
        f"[capture] starting scapy on iface={INTERFACE}, filter='{BPF_FILTER}', topic={TOPIC}, kafka={BOOTSTRAP_SERVERS}",
        flush=True,
    )

    thread = threading.Thread(target=capture_loop, args=(producer,), daemon=True)
    thread.start()

    try:
        while RUNNING:
            time.sleep(1)
    finally:
        producer.flush(timeout=10)
        producer.close(timeout=10)
        print("[capture] producer closed", flush=True)


if __name__ == "__main__":
    main()
