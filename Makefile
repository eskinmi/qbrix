.PHONY: help proto install test lint fmt clean \
        infra infra-down infra-ee infra-ee-down \
        dev dev-proxy dev-motor dev-cortex dev-trace \
        docker docker-build docker-build-no-cache docker-up docker-down docker-logs docker-ps \
        docker-ee docker-ee-build docker-ee-up docker-ee-down docker-ee-logs \
        db-reset clickhouse-reset \
        loadtest loadtest-web loadtest-multi loadtest-multi-web

help:
	@echo "Qbrix Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install              Install all dependencies"
	@echo "  make proto                Generate protobuf stubs"
	@echo ""
	@echo "Development:"
	@echo "  make infra                Start infrastructure (postgres + redis)"
	@echo "  make infra-down           Stop infrastructure"
	@echo "  make infra-ee             Start EE infrastructure (postgres + redis + clickhouse)"
	@echo "  make infra-ee-down        Stop EE infrastructure"
	@echo "  make dev                  Start all services locally (requires infra)"
	@echo "  make dev-proxy            Start proxy service only"
	@echo "  make dev-motor            Start motor service only"
	@echo "  make dev-cortex           Start cortex service only"
	@echo "  make dev-trace            Start trace service only (EE)"
	@echo ""
	@echo "Docker (core services):"
	@echo "  make docker               Build and start core containers"
	@echo "  make docker-build         Build core containers"
	@echo "  make docker-up            Start core containers"
	@echo "  make docker-down          Stop all containers"
	@echo "  make docker-logs          Tail logs from all containers"
	@echo "  make docker-ps            Show running containers"
	@echo ""
	@echo "Docker EE (with trace + clickhouse):"
	@echo "  make docker-ee            Build and start all containers (including EE)"
	@echo "  make docker-ee-build      Build all containers (including EE)"
	@echo "  make docker-ee-up         Start all containers (including EE)"
	@echo "  make docker-ee-down       Stop all containers (including EE)"
	@echo "  make docker-ee-logs       Tail logs from EE containers"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test                 Run all tests"
	@echo "  make lint                 Run linters"
	@echo "  make fmt                  Format code"
	@echo ""
	@echo "Load Testing:"
	@echo "  make loadtest             Run single experiment load test (headless, 60s)"
	@echo "  make loadtest-web         Run single experiment load test (web UI)"
	@echo "  make loadtest-multi       Run multi-experiment load test (headless, 60s)"
	@echo "  make loadtest-multi-web   Run multi-experiment load test (web UI)"
	@echo ""
	@echo "Utilities:"
	@echo "  make db-reset             Reset postgres database"
	@echo "  make clickhouse-reset     Reset clickhouse database (EE)"
	@echo "  make clean                Clean generated files and caches"

# ============================================================================
# Setup
# ============================================================================

install:
	uv sync

proto:
	./bin/generate-proto.sh

# ============================================================================
# Development (local services with docker infra)
# ============================================================================

infra:
	docker compose up -d postgres redis
	@echo "Waiting for services to be healthy..."
	@sleep 3
	@docker compose ps

infra-down:
	docker compose down postgres redis

infra-ee:
	docker compose --profile ee up -d postgres redis clickhouse
	@echo "Waiting for services to be healthy..."
	@sleep 5
	@docker compose --profile ee ps

infra-ee-down:
	docker compose --profile ee down postgres redis clickhouse

dev: infra
	@echo "Starting all services..."
	@trap 'kill 0' EXIT; \
	PROXY_POSTGRES_HOST=localhost PROXY_REDIS_HOST=localhost PROXY_MOTOR_HOST=localhost \
	uv run python -m proxysvc.cli & \
	MOTOR_REDIS_HOST=localhost \
	uv run python -m motorsvc.cli & \
	CORTEX_REDIS_HOST=localhost \
	uv run python -m cortexsvc.cli & \
	wait

dev-proxy:
	PROXY_POSTGRES_HOST=localhost PROXY_REDIS_HOST=localhost PROXY_MOTOR_HOST=localhost \
	uv run python -m proxysvc.cli

dev-motor:
	MOTOR_REDIS_HOST=localhost \
	uv run python -m motorsvc.cli

dev-cortex:
	CORTEX_REDIS_HOST=localhost \
	uv run python -m cortexsvc.cli

dev-trace:
	TRACE_REDIS_HOST=localhost TRACE_CLICKHOUSE_HOST=localhost \
	uv run python -m tracesvc.cli

# ============================================================================
# Docker Compose (full containerized setup)
# ============================================================================

docker: docker-build docker-up

docker-build:
	docker compose build

docker-build-no-cache:
	docker compose build --no-cache

docker-up:
	docker compose up -d
	@docker compose ps

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-ps:
	docker compose ps

# ============================================================================
# Docker Compose EE (with trace + clickhouse)
# ============================================================================

docker-ee: docker-ee-build docker-ee-up

docker-ee-build:
	QBRIX_EE_ENABLED=true docker compose --profile ee build

docker-ee-up:
	QBRIX_EE_ENABLED=true docker compose --profile ee up -d
	@docker compose --profile ee ps

docker-ee-down:
	docker compose --profile ee down

docker-ee-logs:
	docker compose --profile ee logs -f trace clickhouse

# ============================================================================
# Testing & Quality
# ============================================================================

test:
	uv run pytest

lint:
	uv run black --check .
	uv run mypy .

fmt:
	uv run black .

# ============================================================================
# Utilities
# ============================================================================

db-reset:
	docker compose down postgres
	docker volume rm qbrix_postgres_data 2>/dev/null || true
	docker compose up -d postgres
	@echo "Waiting for postgres to be ready..."
	@sleep 3

clickhouse-reset:
	docker compose --profile ee down clickhouse
	docker volume rm qbrix_clickhouse_data 2>/dev/null || true
	docker compose --profile ee up -d clickhouse
	@echo "Waiting for clickhouse to be ready..."
	@sleep 5

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/ dist/ build/

# ============================================================================
# Load Testing
# ============================================================================

loadtest:
	uv run python -m tests.loadtest.cli -s single -u 10 -r 2 -t 60s

loadtest-web:
	uv run python -m tests.loadtest.cli -s single --web

loadtest-multi:
	uv run python -m tests.loadtest.cli -s multi -u 50 -r 5 -t 60s --num-experiments 5

loadtest-multi-web:
	uv run python -m tests.loadtest.cli -s multi --web --num-experiments 5