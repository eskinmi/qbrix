# Qbrix - Distributed Multi-Armed Bandit System

## Project Overview

Qbrix is a distributed system for multi-armed bandit (MAB) optimizations. It separates the hot path (selection) from the learning path (training) to achieve low-latency decisions with eventual consistency in parameter updates.

## Architecture

```
                              ┌─────────────────────────────────────┐
                              │            proxysvc                 │
                              │  - Request routing                  │
                              │  - Experiment/pool management       │
                              │  - Feature gates                    │
                              └──────────┬──────────────────────────┘
                                         │ gRPC
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
            ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
            │   motorsvc    │    │   motorsvc    │    │   motorsvc    │
            │  (selection)  │    │  (selection)  │    │  (selection)  │
            └───────┬───────┘    └───────┬───────┘    └───────┬───────┘
                    │                    │                    │
                    └────────────────────┼────────────────────┘
                                         │ Read (TTL cache via cachebox)
                                         ▼
                              ┌─────────────────────┐
                              │       Redis         │
                              │   (params cache)    │
                              └─────────────────────┘
                                         ▲
                                         │ Write (batch)
                              ┌─────────────────────┐
                              │     cortexsvc       │
                              │    (training)       │
                              └──────────┬──────────┘
                                         │ Consume
                              ┌─────────────────────┐
                              │   Redis Streams     │
                              │   (feedback queue)  │
                              └─────────────────────┘
                                         ▲
                                         │ Publish (feedback)
                              ┌─────────────────────┐
                              │      proxysvc       │
                              └──────────┬──────────┘
                                         │
                              ┌─────────────────────┐
                              │     Postgres        │
                              │ (experiments/pools) │
                              └─────────────────────┘
```

## Services

### proxysvc (Gateway/Control Plane) - Port 50050
- Entry point for all client requests
- Manages experiments and pools in Postgres
- Routes selection requests to motorsvc via gRPC
- Publishes feedback events to Redis Streams
- Feature gates and targeting

### motorsvc (Selection Service) - Port 50051
- Hot path for arm selection only
- Stateless, horizontally scalable
- Reads params from Redis with TTL-based caching (cachebox)
- Receives routed requests from proxysvc

### cortexsvc (Training Service) - Port 50052
- Consumes feedback from Redis Streams
- Batch training of bandit algorithms
- Writes updated params to Redis for motorsvc

## Libraries

### qbrixcore (lib/core)
Core MAB algorithms. Published to envelope registry.

**Protocols**:
- Stochastic: BetaTSProtocol, GaussianTSProtocol, UCB1TunedProtocol, KLUCBProtocol, KLUCBPlusProtocol, EpsilonProtocol, MOSSProtocol, MOSSAnyTimeProtocol
- Contextual: LinUCBProtocol, LinTSProtocol
- Adversarial: EXP3Protocol, FPLProtocol

**Key abstractions**:
- `BaseProtocol`: Interface with `name`, `select()`, `train()`, `init_params()`
- `BaseParamState`: Pydantic model for parameter state
- `BaseParamBackend`: Abstract backend for param storage (InMemoryParamBackend, RedisParamBackend)
- `Agent`: Orchestrates protocol execution with callbacks
- `Pool` / `Arm`: Experiment structure
- `Context`: Request context with id, vector, metadata

### qbrixstore (lib/store)
Storage layer for Postgres and Redis. Published to envelope registry.

**Postgres**:
- `Pool`, `Arm`, `Experiment`, `FeatureGate` SQLAlchemy models (SQLAlchemy 2.0 + asyncpg)
- Async session management with `get_session()`, `init_db()`

**Redis**:
- `RedisClient`: Params and experiment caching
- `RedisStreamPublisher` / `RedisStreamConsumer`: Feedback queue with consumer groups
- `FeedbackEvent`: Dataclass with experiment_id, request_id, arm_index, reward, context

### qbrixproto (lib/proto)
Generated gRPC stubs from proto definitions.

- `common_pb2`: Base types (Context, Arm, Pool, Experiment)
- `motor_pb2` / `motor_pb2_grpc`: MotorService stubs
- `proxy_pb2` / `proxy_pb2_grpc`: ProxyService stubs
- `cortex_pb2` / `cortex_pb2_grpc`: CortexService stubs

## Project Structure

```
qbrix/
├── proto/                    # gRPC proto definitions
│   ├── buf.yaml              # Buf configuration
│   ├── buf.gen.yaml          # Buf generation config
│   ├── common.proto
│   ├── motor.proto
│   ├── cortex.proto
│   └── proxy.proto
├── lib/
│   ├── core/                 # qbrixcore - MAB algorithms
│   ├── store/                # qbrixstore - storage layer
│   └── proto/                # qbrixproto - generated gRPC stubs
├── svc/
│   ├── proxy/                # proxysvc - gateway
│   ├── motor/                # motorsvc - selection
│   └── cortex/               # cortexsvc - training
├── docker-compose.yml        # Local deployment
├── Makefile                  # Development commands
└── bin/                      # Scripts (proto generation)
```

## Tech Stack

- **Language**: Python 3.10+
- **Package manager**: uv (workspace mode)
- **Persistent storage**: Postgres (experiments, pools)
- **Hot storage**: Redis (params cache)
- **Message queue**: Redis Streams (feedback)
- **Caching**: cachebox (in-memory TTL cache)
- **Inter-service**: gRPC
- **Container orchestration**: Docker Compose (local), Kubernetes (production)

## Development

### Setup
```bash
make install
# or: uv sync
```

### Run tests
```bash
make test
# or: uv run pytest
```

### Local infrastructure (Postgres + Redis)
```bash
make infra
```

### Run services locally
```bash
make dev              # All services
make dev-proxy        # proxysvc only
make dev-motor        # motorsvc only
make dev-cortex       # cortexsvc only
```

### Full containerized deployment
```bash
make docker
# or: docker compose up --build
```

### Generate proto stubs
```bash
make proto
# or: cd proto && buf generate
```

### Linting and formatting
```bash
make lint             # mypy type checking
make fmt              # black formatting
```

### Database reset
```bash
make db-reset
```

## Coding Conventions

### Logging
- Always use lowercase for log messages
- Do not capitalize the first letter of log statements
```python
# Good
logger.info("starting motor service on port 50051")
logger.error("failed to connect to redis")

# Bad
logger.info("Starting motor service on port 50051")
logger.error("Failed to connect to Redis")
```

### Comments
- Avoid unnecessary comments; code should be self-explanatory
- When comments are needed, use lowercase (no capitalization)
- Do not state the obvious
```python
# Good
# handles edge case when pool has no active arms
if not active_arms:
    return default_arm

# Bad
# This function selects an arm
def select_arm():
    ...
```

### Imports
- Import each module on a separate line for clarity
- Group imports: standard library, third-party, local
- Use absolute imports for local modules
```python
# Good
from qbrixcore.protoc.stochastic.ts import BetaTSProtocol
from qbrixcore.protoc.stochastic.ucb import UCB1TunedProtocol
from qbrixcore.agent import Agent

# Bad
from qbrixcore.protoc.stochastic.ts import BetaTSProtocol, GaussianTSProtocol
from qbrixcore.protoc.stochastic.ucb import UCB1TunedProtocol, KLUCBProtocol
```

### Type Hints
- Always use type hints for function signatures
- Use `from __future__ import annotations` for forward references

### Async Code
- Prefer async/await for I/O operations
- Use `asyncio.gather` for concurrent operations
- Never block the event loop with synchronous calls

## Dependency Management

### Package Manager
- Always use `uv` for dependency management (not pip)
- The project uses uv workspace mode with multiple packages

### Adding Dependencies
```bash
# Add to root workspace
uv add <package>

# Add to specific package
uv add <package> --package <package-name>

# Add dev dependency
uv add --dev <package>

# Example: add redis to motorsvc
uv add redis --package motorsvc
```

### Removing Dependencies
```bash
uv remove <package>
uv remove <package> --package <package-name>
```

### Syncing Environment
```bash
# Sync all workspace packages
uv sync

# Update lockfile
uv lock
```

### Version Constraints
- Pin major versions for stability: `package>=1.0,<2.0`
- Use exact versions only when necessary for reproducibility

## Request Flow

1. **Create Pool**: `proxysvc` → Postgres
2. **Create Experiment**: `proxysvc` → Postgres → sync to Redis
3. **Select**: Client → `proxysvc` (feature gates) → `motorsvc` (gRPC) → Redis (cached params) → response
4. **Feedback**: Client → `proxysvc` → Redis Streams → `cortexsvc` (batch train) → Redis (updated params)

## Environment Variables

### proxysvc
- `PROXY_GRPC_HOST`, `PROXY_GRPC_PORT`
- `PROXY_POSTGRES_HOST`, `PROXY_POSTGRES_PORT`, `PROXY_POSTGRES_USER`, `PROXY_POSTGRES_PASSWORD`, `PROXY_POSTGRES_DATABASE`
- `PROXY_REDIS_HOST`, `PROXY_REDIS_PORT`
- `PROXY_MOTOR_HOST`, `PROXY_MOTOR_PORT`

### motorsvc
- `MOTOR_GRPC_HOST`, `MOTOR_GRPC_PORT`
- `MOTOR_REDIS_HOST`, `MOTOR_REDIS_PORT`
- `MOTOR_PARAM_CACHE_TTL`, `MOTOR_AGENT_CACHE_TTL`

### cortexsvc
- `CORTEX_GRPC_HOST`, `CORTEX_GRPC_PORT`
- `CORTEX_REDIS_HOST`, `CORTEX_REDIS_PORT`
- `CORTEX_CONSUMER_NAME`, `CORTEX_BATCH_SIZE`
