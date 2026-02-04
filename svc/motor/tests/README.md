# motorsvc unit tests

comprehensive unit test suite for the motor selection service (motorsvc).

## overview

motorsvc is the hot path selection service for qbrix. it is stateless, horizontally scalable, and designed for low-latency arm selection. the tests ensure reliability of selection logic, caching, redis integration, and grpc handlers.

## test structure

```
tests/
├── unit/
│   ├── conftest.py           # shared fixtures and mocks
│   ├── test_cache.py         # motor cache tests
│   ├── test_param_backend.py # redis-backed parameter backend tests
│   ├── test_agent_factory.py # agent factory tests
│   ├── test_service.py       # motor service tests
│   └── test_server.py        # grpc servicer tests
└── README.md
```

## running tests

### all tests
```bash
cd /Users/eskinmi/Dev/qbrix/svc/motor
uv run pytest
```

### with coverage
```bash
uv run pytest --cov-report=term-missing
```

### specific test file
```bash
uv run pytest tests/unit/test_service.py -v
```

### specific test class
```bash
uv run pytest tests/unit/test_agent_factory.py::TestAgentFactoryGetOrCreate -v
```

### specific test
```bash
uv run pytest tests/unit/test_service.py::TestMotorServiceSelect::test_select_returns_selected_arm_info -v
```

## test coverage

current coverage: **82%**

| module | coverage |
|--------|----------|
| agent_factory.py | 100% |
| cache.py | 100% |
| param_backend.py | 100% |
| service.py | 86% |
| config.py | 88% |
| server.py | 64% |
| cli.py | 0% (excluded) |

## test categories

### cache tests (test_cache.py)

tests for `MotorCache` which provides ttl-based caching for params and agents using cachebox.

**key tests:**
- ttl cache initialization
- param storage and retrieval
- agent storage and retrieval
- cache invalidation
- cache clearing
- ttl expiration behavior
- maxsize limits

### param backend tests (test_param_backend.py)

tests for `RedisBackedInMemoryParamBackend` which integrates redis with in-memory caching.

**key tests:**
- get returns cached params
- set caches params
- update_params fetches from redis and caches
- update_params returns none when redis is empty
- policy-specific param state classes
- param dict validation

### agent factory tests (test_agent_factory.py)

tests for `AgentFactory` which creates and caches agents for experiments.

**key tests:**
- policy map contains all policys
- build pool from experiment data
- get or create agent (cached vs new)
- agent caching
- param initialization
- policy params from experiment
- unknown policy handling
- param fetching from backend

### service tests (test_service.py)

tests for `MotorService` which orchestrates selection logic.

**key tests:**
- lifecycle (start/stop)
- redis connection management
- select retrieves experiment from redis
- select raises error when experiment not found
- select gets or creates agent
- select calls agent.select with context
- select returns arm info
- select handles empty context vector and metadata
- health check (redis ping)

### server tests (test_server.py)

tests for `MotorGRPCServicer` which handles grpc requests.

**key tests:**
- select calls service.select
- select returns arm in response
- select handles valueerror with not_found status
- select handles general exception with internal status
- select returns empty response on error
- select handles empty context vector
- select converts protobuf repeated to list
- health returns serving/not_serving status

## mocking strategy

### what is mocked

- **redis client**: mock all redis operations (get_experiment, get_params, ping)
- **grpc context**: mock grpc context for servicer tests
- **agent backend**: mock param backend for agent factory tests
- **time-sensitive**: avoid actual ttl waits in tests

### what is NOT mocked

- **policy logic**: use real qbrixcore policys (BetaTSPolicy, etc.)
- **cache implementation**: use real MotorCache with cachebox
- **data structures**: use real Pool, Arm, Context objects
- **param state**: use real pydantic param state models

## fixtures (conftest.py)

### settings and cache
- `motor_settings`: default motor settings
- `motor_cache`: motor cache instance

### mocks
- `mock_redis_client`: mocked redis client
- `mock_grpc_context`: mocked grpc context

### data structures
- `pool_with_three_arms`: pool with 3 arms
- `pool_data_dict`: pool as dict (experiment format)
- `experiment_data_dict`: experiment as dict (redis format)

### param states
- `beta_ts_params`: initialized beta ts params
- `gaussian_ts_params`: initialized gaussian ts params

### agents
- `mock_agent`: mocked agent with beta ts policy

## edge cases tested

- empty context vector
- empty metadata
- missing experiment in redis
- missing params in redis and cache
- unknown policy
- redis connection failure
- agent cache expiration
- param cache expiration
- protobuf float precision
- concurrent agent creation race condition (documented)

## async testing

tests use `pytest-asyncio` with auto mode enabled in `pytest.ini`.

all async service and backend methods are tested with `@pytest.mark.asyncio` decorator.

## best practices

1. **test independence**: each test is independent, no shared mutable state
2. **clear naming**: test names follow `test_<function>_<scenario>_<expected>` pattern
3. **aaa pattern**: arrange, act, assert structure
4. **mock verification**: verify mock calls when testing side effects
5. **edge cases**: test empty inputs, none values, missing data
6. **error handling**: test exception paths and error codes

## continuous integration

tests run automatically on:
- pre-commit hooks
- github actions (if configured)
- local development with `make test`

## adding new tests

1. identify the function/class to test
2. create test class following naming convention
3. use existing fixtures from conftest.py
4. mock external dependencies (redis, grpc)
5. verify behavior with assertions
6. test edge cases and error paths

example:
```python
class TestNewFeature:
    @pytest.mark.asyncio
    async def test_feature_with_valid_input_succeeds(self, motor_cache):
        # arrange
        feature = NewFeature(motor_cache)

        # act
        result = await feature.process("valid-input")

        # assert
        assert result is not None
        assert result.status == "success"
```
