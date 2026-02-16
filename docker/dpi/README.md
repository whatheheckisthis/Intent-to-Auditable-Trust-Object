# Real-Time Docker DPI Pipeline (Scapy + Kafka + RedisTimeSeries + Grafana)

This stack builds a high-throughput, real-time deep packet inspection (DPI) pipeline:

1. **Scapy capture service** sniffs packets from a host interface.
2. Packets are serialized and pushed into **Kafka** (`raw.packets` topic).
3. A **flow processor** consumes Kafka in batches, computes flow-level metrics, and writes counters to **Redis TimeSeries**.
4. **Grafana** visualizes global throughput, packet rate, protocol trends, and top talkers.

## Architecture

```text
[Host NIC / any iface]
        |
        v
  packet-capture (Scapy)
        |
        v
      Kafka topic: raw.packets (12 partitions)
        |
        v
    flow-processor (batch + aggregation)
        |
        v
 Redis Stack (RedisTimeSeries + sorted sets + hashes)
        |
        v
      Grafana dashboard (auto provisioned)
```

## Start the pipeline

```bash
cd docker/dpi
docker compose up -d --build
```

Open Grafana:

- URL: `http://localhost:3000`
- User: `admin`
- Password: `admin`
- Dashboard: **DPI Real-Time Flow Metrics** (folder: `DPI`)

## Optional high-throughput traffic generator

A synthetic UDP generator is included under a profile so the stack can be stress-tested quickly.

```bash
docker compose --profile load-test up -d traffic-generator
```

Tune generated load via env vars:

- `RATE_PER_SECOND` (default `5000`)
- `PAYLOAD_SIZE` (default `1300`)

## Stored metrics

### Global series

- `ts:global:bps` (bytes/sec)
- `ts:global:pps` (packets/sec)

### Per-protocol series

- `ts:protocol:<PROTO>:bps`
- `ts:protocol:<PROTO>:pps`

### Per-flow series

- `ts:flow:<FLOW_ID>:bps`
- `ts:flow:<FLOW_ID>:pps`

Flow metadata and rollups:

- `flow:<FLOW_ID>:meta` hash (5-tuple metadata + cumulative byte/packet totals)
- `flows:top:bytes` sorted set (top flows by total bytes)

## High-throughput implementation details

- Kafka producer batching (`linger.ms`, `batch_size`, compression).
- Kafka topic configured with **12 partitions** for parallelism.
- Consumer polls in large batches (`MAX_POLL_RECORDS=10000`).
- In-memory per-second aggregation before Redis writes.
- Redis writes are pipelined and use `TS.INCRBY` for low-overhead append/update semantics.
- Time series are created lazily with labels and `DUPLICATE_POLICY SUM`.

## Notes

- `packet-capture` runs with `network_mode: host` to sniff host-visible traffic on Linux.
- Requires Docker Engine with access to host networking + packet capture capabilities.
- To capture only specific traffic classes, adjust `BPF_FILTER`.
