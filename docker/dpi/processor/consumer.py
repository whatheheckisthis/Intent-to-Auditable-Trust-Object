import json
import os
import signal
import time
from collections import defaultdict
from typing import Dict, Tuple

import redis
from kafka import KafkaConsumer
from redis.exceptions import ResponseError

BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "raw.packets")
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "flow-metrics-v1")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
FLUSH_INTERVAL_SECONDS = float(os.getenv("FLUSH_INTERVAL_SECONDS", "1"))
MAX_POLL_RECORDS = int(os.getenv("MAX_POLL_RECORDS", "10000"))
RETENTION_MS = int(os.getenv("RETENTION_MS", "86400000"))

RUNNING = True
SERIES_CACHE = set()


def ensure_series(rdb: redis.Redis, key: str, labels: Dict[str, str]) -> None:
    if key in SERIES_CACHE:
        return
    cmd = ["TS.CREATE", key, "RETENTION", RETENTION_MS, "DUPLICATE_POLICY", "SUM", "LABELS"]
    for label_key, label_value in labels.items():
        cmd.extend([label_key, label_value])
    try:
        rdb.execute_command(*cmd)
    except ResponseError as exc:
        if "already exists" not in str(exc).lower():
            raise
    SERIES_CACHE.add(key)


def safe_flow_metadata(value: Dict[str, object]) -> Dict[str, str]:
    return {
        "src_ip": str(value.get("src_ip", "")),
        "dst_ip": str(value.get("dst_ip", "")),
        "src_port": str(value.get("src_port", "0")),
        "dst_port": str(value.get("dst_port", "0")),
        "protocol": str(value.get("protocol", "UNKNOWN")),
        "ip_version": str(value.get("ip_version", "0")),
    }


def flush_aggregates(
    rdb: redis.Redis,
    flow_aggregate: Dict[str, Dict[str, int]],
    flow_meta: Dict[str, Dict[str, str]],
    protocol_aggregate: Dict[str, Dict[str, int]],
) -> None:
    if not flow_aggregate and not protocol_aggregate:
        return

    timestamp_ms = int(time.time() * 1000)
    pipe = rdb.pipeline(transaction=False)

    global_packets = 0
    global_bytes = 0

    for flow_id, metrics in flow_aggregate.items():
        packets = metrics["packets"]
        bytes_count = metrics["bytes"]
        global_packets += packets
        global_bytes += bytes_count

        metadata = flow_meta[flow_id]
        flow_hash_key = f"flow:{flow_id}:meta"
        pipe.hset(flow_hash_key, mapping=metadata)
        pipe.hincrby(flow_hash_key, "total_packets", packets)
        pipe.hincrby(flow_hash_key, "total_bytes", bytes_count)
        pipe.expire(flow_hash_key, max(int(RETENTION_MS / 1000), 60))

        pps_key = f"ts:flow:{flow_id}:pps"
        bps_key = f"ts:flow:{flow_id}:bps"
        ensure_series(rdb, pps_key, {"scope": "flow", "flow_id": flow_id, **metadata, "metric": "pps"})
        ensure_series(rdb, bps_key, {"scope": "flow", "flow_id": flow_id, **metadata, "metric": "bps"})

        pipe.execute_command("TS.INCRBY", pps_key, packets, "TIMESTAMP", timestamp_ms)
        pipe.execute_command("TS.INCRBY", bps_key, bytes_count, "TIMESTAMP", timestamp_ms)
        pipe.zincrby("flows:top:bytes", bytes_count, flow_id)

    for protocol, metrics in protocol_aggregate.items():
        proto_pps_key = f"ts:protocol:{protocol}:pps"
        proto_bps_key = f"ts:protocol:{protocol}:bps"
        ensure_series(rdb, proto_pps_key, {"scope": "protocol", "protocol": protocol, "metric": "pps"})
        ensure_series(rdb, proto_bps_key, {"scope": "protocol", "protocol": protocol, "metric": "bps"})

        pipe.execute_command("TS.INCRBY", proto_pps_key, metrics["packets"], "TIMESTAMP", timestamp_ms)
        pipe.execute_command("TS.INCRBY", proto_bps_key, metrics["bytes"], "TIMESTAMP", timestamp_ms)

    ensure_series(rdb, "ts:global:pps", {"scope": "global", "metric": "pps"})
    ensure_series(rdb, "ts:global:bps", {"scope": "global", "metric": "bps"})
    pipe.execute_command("TS.INCRBY", "ts:global:pps", global_packets, "TIMESTAMP", timestamp_ms)
    pipe.execute_command("TS.INCRBY", "ts:global:bps", global_bytes, "TIMESTAMP", timestamp_ms)

    pipe.execute()


def stop_signal_handler(signum: int, _frame: object) -> None:
    global RUNNING
    RUNNING = False
    print(f"[processor] received signal {signum}, draining buffer...", flush=True)


def main() -> None:
    signal.signal(signal.SIGINT, stop_signal_handler)
    signal.signal(signal.SIGTERM, stop_signal_handler)

    rdb = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id=GROUP_ID,
        enable_auto_commit=False,
        auto_offset_reset="latest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        max_poll_records=MAX_POLL_RECORDS,
        consumer_timeout_ms=1000,
        fetch_max_bytes=52428800,
        max_partition_fetch_bytes=10485760,
    )

    print(
        f"[processor] consuming topic={TOPIC} kafka={BOOTSTRAP_SERVERS}, redis={REDIS_HOST}:{REDIS_PORT}",
        flush=True,
    )

    flow_aggregate: Dict[str, Dict[str, int]] = defaultdict(lambda: {"packets": 0, "bytes": 0})
    flow_meta: Dict[str, Dict[str, str]] = {}
    protocol_aggregate: Dict[str, Dict[str, int]] = defaultdict(lambda: {"packets": 0, "bytes": 0})

    next_flush = time.time() + FLUSH_INTERVAL_SECONDS

    while RUNNING:
        records_pack = consumer.poll(timeout_ms=250, max_records=MAX_POLL_RECORDS)
        for _topic_partition, records in records_pack.items():
            for record in records:
                packet = record.value
                flow_id = str(packet.get("flow_id", "unknown"))
                packet_bytes = int(packet.get("packet_bytes", 0))
                protocol = str(packet.get("protocol", "UNKNOWN")).upper()

                flow_aggregate[flow_id]["packets"] += 1
                flow_aggregate[flow_id]["bytes"] += packet_bytes
                flow_meta.setdefault(flow_id, safe_flow_metadata(packet))

                protocol_aggregate[protocol]["packets"] += 1
                protocol_aggregate[protocol]["bytes"] += packet_bytes

        now = time.time()
        if now >= next_flush:
            flush_aggregates(rdb, flow_aggregate, flow_meta, protocol_aggregate)
            if records_pack:
                consumer.commit()
            flow_aggregate.clear()
            flow_meta.clear()
            protocol_aggregate.clear()
            next_flush = now + FLUSH_INTERVAL_SECONDS

    flush_aggregates(rdb, flow_aggregate, flow_meta, protocol_aggregate)
    if flow_aggregate:
        consumer.commit()
    consumer.close()


if __name__ == "__main__":
    main()
