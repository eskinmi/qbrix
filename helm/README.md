# Qbrix Helm Charts

Helm charts for deploying Qbrix - a distributed multi-armed bandit system for site variant optimization.

## Architecture Overview

```
                              ┌─────────────────────────────────────┐
                              │            proxysvc                 │
                              │  - Request routing (stateless)      │
                              │  - Experiment/pool management       │
                              │  - Feature gates                    │
                              │  - Horizontally scalable            │
                              └──────────┬──────────────────────────┘
                                         │ gRPC
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
            ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
            │   motorsvc    │    │   motorsvc    │    │   motorsvc    │
            │  (stateless)  │    │  (stateless)  │    │  (stateless)  │
            │  HOT PATH     │    │  HOT PATH     │    │  HOT PATH     │
            └───────┬───────┘    └───────┬───────┘    └───────┬───────┘
                    │                    │                    │
                    └────────────────────┼────────────────────┘
                                         │ Read (TTL cache)
                                         ▼
                              ┌─────────────────────┐
                              │       Redis         │
                              │   (params cache)    │
                              └─────────────────────┘
                                         ▲
                                         │ Write (batch)
                              ┌─────────────────────┐
                              │     cortexsvc       │
                              │  (single instance)  │
                              │   TRAINING PATH     │
                              │  event sourcing     │
                              └──────────┬──────────┘
                                         │ Consume
                              ┌─────────────────────┐
                              │   Redis Streams     │
                              │   (feedback queue)  │
                              └─────────────────────┘
                                         ▲
                                         │ Publish
                              ┌─────────────────────┐
                              │      proxysvc       │
                              └──────────┬──────────┘
                                         │
                              ┌─────────────────────┐
                              │     Postgres        │
                              │ (experiments/pools) │
                              └─────────────────────┘
```

## Service Scaling

| Service | Scaling | Reason |
|---------|---------|--------|
| **proxy** | Horizontal (HPA) | Stateless gateway, cache with Redis fallback |
| **motor** | Horizontal (HPA) | Stateless selection, TTL cache |
| **cortex** | Single instance | By design - see below |

### Why Cortex is Single Instance

Cortex uses the **event sourcing pattern** - a single consumer processes all training events:

```
Feedback → Redis Streams (buffer) → Cortex (single) → Params in Redis
```

This is a deliberate design choice:

1. **Not on the hot path** - Training doesn't affect selection latency
2. **Correct by construction** - No race conditions in parameter updates
3. **Back-pressure via Redis Streams** - Buffers traffic spikes automatically
4. **Computationally cheap** - Single instance handles thousands of events/sec

If cortex becomes a bottleneck:
1. Vertical scaling (more CPU/memory)
2. Increase batch size
3. Shard by experiment (each cortex owns subset)

## Prerequisites

- Kubernetes 1.25+
- Helm 3.10+
- PostgreSQL and Redis (external or in-cluster for dev)

## Quick Start

### Development

```bash
# install with in-cluster postgres/redis
helm install qbrix ./qbrix -f qbrix/values-dev.yaml

# check status
kubectl get pods -l app.kubernetes.io/part-of=qbrix
```

### Production

```bash
# create secrets first
kubectl create secret generic qbrix-postgres-credentials \
  --from-literal=username=qbrix \
  --from-literal=password=<password>

kubectl create secret generic qbrix-redis-credentials \
  --from-literal=password=<password>

kubectl create secret generic qbrix-jwt-secret \
  --from-literal=jwt-secret=<secret>

# install (uses external postgres/redis)
helm install qbrix ./qbrix \
  --set global.postgres.existingSecret=qbrix-postgres-credentials \
  --set global.redis.existingSecret=qbrix-redis-credentials \
  --set proxy.config.jwt.existingSecret=qbrix-jwt-secret
```

## Chart Structure

```
helm/
└── qbrix/
    ├── Chart.yaml
    ├── values.yaml          # production-ready defaults
    ├── values-dev.yaml      # development overrides
    ├── templates/           # shared resources (secrets, infra)
    └── charts/
        ├── proxy/           # gateway (HPA enabled)
        ├── motor/           # selection (HPA enabled)
        └── cortex/          # training (single replica)
```

## Configuration

### Global

| Parameter | Description | Default |
|-----------|-------------|---------|
| `global.imageRegistry` | Registry prefix | `""` |
| `global.postgres.host` | PostgreSQL host | `qbrix-postgres` |
| `global.postgres.existingSecret` | Secret with credentials | `""` |
| `global.redis.host` | Redis host | `qbrix-redis` |
| `global.redis.existingSecret` | Secret with password | `""` |
| `global.logging.format` | json or text | `json` |
| `global.logging.level` | Log level | `WARNING` |

### Infrastructure (Dev Only)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `infrastructure.postgres.enabled` | Deploy in-cluster PostgreSQL | `false` |
| `infrastructure.redis.enabled` | Deploy in-cluster Redis | `false` |

### Services

Each service (proxy, motor, cortex) supports:

| Parameter | Description |
|-----------|-------------|
| `<svc>.enabled` | Enable/disable service |
| `<svc>.replicaCount` | Replicas (if HPA disabled) |
| `<svc>.autoscaling.enabled` | Enable HPA |
| `<svc>.autoscaling.minReplicas` | Min replicas |
| `<svc>.autoscaling.maxReplicas` | Max replicas |
| `<svc>.resources` | CPU/memory requests/limits |
| `<svc>.config.*` | Service-specific config |

## Secrets

### Using Existing Secrets (Production)

```yaml
global:
  postgres:
    existingSecret: "my-postgres-secret"  # keys: username, password
  redis:
    existingSecret: "my-redis-secret"     # key: password

proxy:
  config:
    jwt:
      existingSecret: "my-jwt-secret"     # key: jwt-secret
```

### Inline Secrets (Dev Only)

```yaml
global:
  postgres:
    username: "qbrix"
    password: "dev-password"

proxy:
  config:
    jwt:
      secretKey: "dev-jwt-secret"
```

## Upgrading

```bash
helm upgrade qbrix ./qbrix -f qbrix/values-dev.yaml
helm rollback qbrix 1  # if needed
```

## Uninstalling

```bash
helm uninstall qbrix
kubectl delete pvc -l app.kubernetes.io/instance=qbrix  # if using in-cluster infra
```
