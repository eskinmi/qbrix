# Tenant Identification in Qbrix

## Overview

Qbrix implements multi-tenancy with a 1:1 user-tenant relationship. Each user belongs to exactly one tenant, and tenant resolution is implicit through authentication.

## Design Decisions

### User-Tenant Relationship
- **1:1 mapping**: Each user belongs to exactly one tenant
- **API Keys inherit tenant**: API keys belong to users and inherit their tenant
- **No tenant switching**: Users cannot switch tenants; they are bound to their tenant

### Tenant Resolution

Tenant resolution is **implicit** - no `X-Tenant-ID` header is required. The tenant is determined automatically from the authenticated user.

#### Resolution Flow

1. **API Key Authentication**:
   ```
   X-API-Key: optiq_abc123... → User lookup → user.tenant_id
   ```

2. **JWT Token Authentication**:
   ```
   Authorization: Bearer eyJ... → Token decode → payload.tenant_id
   ```

3. **Development Mode**:
   ```
   RUNENV=dev → tenant_id = "dev-tenant"
   ```

### JWT Token Structure

Access tokens include the tenant_id claim:

```json
{
  "sub": "user_id",
  "tenant_id": "tenant_id",
  "email": "user@example.com",
  "role": "member",
  "plan_tier": "free",
  "exp": 1234567890,
  "iat": 1234567890,
  "type": "access"
}
```

## Data Isolation

### Database Layer

All tenant-scoped tables have a `tenant_id` foreign key:

- `pools.tenant_id` → `tenants.id`
- `experiments.tenant_id` → `tenants.id`
- `users.tenant_id` → `tenants.id`

Unique constraints are tenant-scoped:
- Pool names are unique within a tenant: `(tenant_id, name)`
- Experiment names are unique within a tenant: `(tenant_id, name)`

### Redis Layer

All Redis keys include the tenant_id:

```
qbrix:tenant:{tenant_id}:params:{experiment_id}
qbrix:tenant:{tenant_id}:experiment:{experiment_id}
qbrix:tenant:{tenant_id}:gate:{experiment_id}
```

### Feedback Events

The `FeedbackEvent` includes `tenant_id`:

```python
@dataclass
class FeedbackEvent:
    tenant_id: str
    experiment_id: str
    request_id: str
    arm_index: int
    reward: float
    context_id: str
    context_vector: list[float]
    context_metadata: dict
    timestamp_ms: int
```

### Selection Tokens

Selection tokens encode the tenant_id:

```json
{
  "tnt_id": "tenant_id",
  "exp_id": "experiment_id",
  "arm_idx": 0,
  "ctx_id": "context_id",
  "ctx_vec": [],
  "ctx_meta": {},
  "ts": 1234567890000
}
```

## Service Layer

### ProxyService

All tenant-scoped methods require `tenant_id` as the first parameter:

```python
async def create_pool(self, tenant_id: str, name: str, arms: list[dict]) -> dict
async def get_pool(self, tenant_id: str, pool_id: str) -> dict | None
async def create_experiment(self, tenant_id: str, name: str, ...) -> dict
async def select(self, tenant_id: str, experiment_id: str, ...) -> dict
```

### MotorService

Selection requests include tenant_id from the gRPC request:

```protobuf
message SelectRequest {
  string tenant_id = 1;
  string experiment_id = 2;
  qbrix.common.Context context = 3;
}
```

### CortexService

Training groups events by `(tenant_id, experiment_id)` tuple to maintain isolation.

## User Registration

When a user registers without specifying a tenant:

1. A new tenant is created with:
   - `name`: "{email}'s Workspace"
   - `slug`: Derived from email prefix (e.g., "john" from "john@example.com")

2. The user is assigned to this new tenant

```python
await auth_service.register_user(
    email="john@example.com",
    password="...",
    # tenant_id=None means create new tenant
)
```

## Admin Access

For now, tenant administration is done via direct database access. The system creates a "default" tenant during migration for existing data.

## Migration

Run the migration script to add multi-tenancy to an existing installation:

```bash
python scripts/migrations/001_add_multitenancy.py
```

This script:
1. Creates the `tenants` table
2. Creates a "default" tenant
3. Adds `tenant_id` to pools, experiments, and users
4. Backfills existing records with the default tenant
5. Updates unique constraints to be tenant-scoped
