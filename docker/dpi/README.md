# Docker Monitoring Stack (Kafka + Redis + Prometheus + Grafana)

This compose stack deploys:

- Zookeeper
- Kafka (with JMX exposed)
- Redis
- kafka_exporter
- redis_exporter
- Prometheus
- Grafana (auto-provisioned with Prometheus datasource)

## Run

```bash
cd docker/dpi
docker compose up -d
```

## Exposed ports

- Zookeeper: `2181`
- Kafka broker: `9092`
- Kafka JMX: `9999`
- Redis: `6379`
- kafka_exporter: `9308`
- redis_exporter: `9121`
- Prometheus: `9090`
- Grafana: `3000`

## Prometheus scraping

Prometheus is configured to scrape:

- `kafka_exporter:9308`
- `redis_exporter:9121`

See `prometheus/prometheus.yml`.

## Grafana

Grafana starts with a pre-provisioned Prometheus datasource.

- URL: `http://localhost:3000`
- User: `admin`
- Password: `admin`
