# Qbrix Load Testing

Load testing suite for qbrix using [Locust](https://locust.io/).

## Scenarios

### Single Experiment (`-s single`)

All users interact with a single experiment/pool. This simulates high-concurrency on a single bandit.

**Behavior:**
- Realistic async feedback (70% of selections get feedback, with 100-2000ms delay)
- Probabilistic rewards (30% success rate)
- Select:Health request ratio is 10:1

### Multi Experiment (`-s multi`)

Users are distributed across multiple experiments. This simulates multi-tenant scenarios.

**Behavior:**
- Configurable number of experiments (`--num-experiments`, default: 10)
- Max users per experiment cap (`--max-users-per-experiment`, default: 100)
- Users are assigned round-robin with capacity limits
- Same realistic feedback behavior as single experiment

## Usage

### Via Makefile

```bash
# single experiment with web UI
make loadtest-web

# single experiment headless (60s, 10 users)
make loadtest

# multi-experiment with web UI
make loadtest-multi-web

# multi-experiment headless (60s, 50 users, 5 experiments)
make loadtest-multi
```

### Via CLI

```bash
# web interface (single experiment)
uv run loadtest --web

# headless with custom params
uv run loadtest -u 100 -r 10 -t 60s

# multi-experiment scenario
uv run loadtest -s multi --num-experiments 10 -u 200 -r 20 -t 5m

# connect to remote proxy
uv run loadtest -h proxy.example.com -p 50050 --web
```

### CLI Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--scenario` | `-s` | `single` | Scenario: `single` or `multi` |
| `--web` | | `false` | Start with web UI |
| `--host` | `-h` | `localhost` | Proxy service host |
| `--port` | `-p` | `50050` | Proxy service port |
| `--users` | `-u` | `10` | Number of concurrent users |
| `--spawn-rate` | `-r` | `1` | Users to spawn per second |
| `--run-time` | `-t` | | Run duration (e.g., `60s`, `5m`, `1h`) |
| `--web-host` | | `localhost` | Web UI host |
| `--web-port` | | `8089` | Web UI port |
| `--num-experiments` | | `10` | Number of experiments (multi scenario) |
| `--max-users-per-experiment` | | `100` | Max users per experiment (multi scenario) |

## Configuration

All settings can be configured via environment variables with `LOADTEST_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `LOADTEST_PROXY_HOST` | `localhost` | Proxy service host |
| `LOADTEST_PROXY_PORT` | `50050` | Proxy service port |
| `LOADTEST_FEEDBACK_PROBABILITY` | `0.7` | Probability of sending feedback after select |
| `LOADTEST_FEEDBACK_DELAY_MIN_MS` | `100` | Min delay before sending feedback |
| `LOADTEST_FEEDBACK_DELAY_MAX_MS` | `2000` | Max delay before sending feedback |
| `LOADTEST_REWARD_SUCCESS_PROBABILITY` | `0.3` | Probability of positive reward |
| `LOADTEST_NUM_EXPERIMENTS` | `10` | Number of experiments (multi scenario) |
| `LOADTEST_MAX_USERS_PER_EXPERIMENT` | `100` | Max users per experiment |
| `LOADTEST_DEFAULT_NUM_ARMS` | `5` | Number of arms per pool |
| `LOADTEST_DEFAULT_PROTOCOL` | `beta_ts` | Bandit policy to use |

## Prerequisites

Services must be running before load testing:

```bash
# start infrastructure and services
make dev

# in another terminal, run load tests
make loadtest-web
```

## Web Interface

When using `--web`, Locust starts a web interface at `http://localhost:8089` where you can:

- Configure number of users and spawn rate
- Start/stop tests
- View real-time charts (RPS, response times, failures)
- Download test results as CSV