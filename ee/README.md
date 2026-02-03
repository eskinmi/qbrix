# Qbrix Enterprise Edition

This directory contains Enterprise Edition (EE) features for Qbrix.

## Components

### tracesvc

Event persistence service that writes selection and feedback events to ClickHouse for analytics and insights.

**Features:**
- Consumes selection events from `qbrix:selection` Redis stream
- Consumes feedback events from `qbrix:feedback` Redis stream
- Batch writes to ClickHouse for efficient storage
- Horizontally scalable (append-only writes, no race conditions)

**Architecture:**
```
proxysvc ─────► qbrix:selection ─────► tracesvc ─────► ClickHouse
                                           │
proxysvc ─────► qbrix:feedback  ─────►─────┘
                    │
                    └─────► cortexsvc (training)
```

## Enabling EE Features

### Docker Compose

```bash
# run with ee profile
docker compose --profile ee up
```

### Helm

```bash
helm install qbrix ./helm/qbrix --set ee.enabled=true
```

## ClickHouse Schema

Events are stored with tenant-aware partitioning for efficient multi-tenant queries:

- **selection_events**: Partitioned by `(tenant_id, toYYYYMM(event_date))`
- **feedback_events**: Partitioned by `(tenant_id, toYYYYMM(event_date))`

Both tables have a 90-day TTL by default (configurable).

## Environment Variables

### tracesvc

| Variable | Default | Description |
|----------|---------|-------------|
| `TRACE_GRPC_HOST` | `0.0.0.0` | gRPC server host |
| `TRACE_GRPC_PORT` | `50053` | gRPC server port |
| `TRACE_REDIS_HOST` | `localhost` | Redis host |
| `TRACE_REDIS_PORT` | `6379` | Redis port |
| `TRACE_FEEDBACK_STREAM_NAME` | `qbrix:feedback` | Feedback stream name |
| `TRACE_SELECTION_STREAM_NAME` | `qbrix:selection` | Selection stream name |
| `TRACE_CONSUMER_GROUP` | `trace` | Redis consumer group |
| `TRACE_CONSUMER_NAME` | `worker-0` | Consumer instance name |
| `TRACE_CLICKHOUSE_HOST` | `localhost` | ClickHouse host |
| `TRACE_CLICKHOUSE_PORT` | `8123` | ClickHouse HTTP port |
| `TRACE_CLICKHOUSE_USER` | `default` | ClickHouse user |
| `TRACE_CLICKHOUSE_PASSWORD` | `` | ClickHouse password |
| `TRACE_CLICKHOUSE_DATABASE` | `qbrix` | ClickHouse database |
| `TRACE_BATCH_SIZE` | `500` | Batch size for writes |
| `TRACE_FLUSH_INTERVAL_SEC` | `5.0` | Flush interval in seconds |

### proxysvc (EE settings)

| Variable | Default | Description |
|----------|---------|-------------|
| `PROXY_EE_ENABLED` | `false` | Enable EE features |
| `PROXY_SELECTION_STREAM_NAME` | `qbrix:selection` | Selection stream name |
