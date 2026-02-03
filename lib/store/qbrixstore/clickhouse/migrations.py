from __future__ import annotations

from clickhouse_connect.driver.client import Client

from qbrixstore.config import ClickHouseSettings


SELECTION_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS selection_events (
    tenant_id String,
    experiment_id String,
    request_id String,
    arm_id String,
    arm_name String,
    arm_index UInt16,
    is_default Bool,
    context_id String,
    context_vector Array(Float64),
    context_metadata String,
    timestamp_ms Int64,
    protocol String,
    event_date Date DEFAULT toDate(fromUnixTimestamp64Milli(timestamp_ms))
)
ENGINE = MergeTree()
PARTITION BY (tenant_id, toYYYYMM(event_date))
ORDER BY (tenant_id, experiment_id, timestamp_ms, request_id)
TTL event_date + INTERVAL {ttl_days} DAY
SETTINGS index_granularity = 8192
"""

FEEDBACK_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS feedback_events (
    tenant_id String,
    experiment_id String,
    request_id String,
    arm_index UInt16,
    reward Float64,
    context_id String,
    context_vector Array(Float64),
    context_metadata String,
    timestamp_ms Int64,
    event_date Date DEFAULT toDate(fromUnixTimestamp64Milli(timestamp_ms))
)
ENGINE = MergeTree()
PARTITION BY (tenant_id, toYYYYMM(event_date))
ORDER BY (tenant_id, experiment_id, timestamp_ms, request_id)
TTL event_date + INTERVAL {ttl_days} DAY
SETTINGS index_granularity = 8192
"""

DROP_SELECTION_EVENTS = "DROP TABLE IF EXISTS selection_events"
DROP_FEEDBACK_EVENTS = "DROP TABLE IF EXISTS feedback_events"


def create_tables(client: Client, ttl_days: int = 90) -> None:
    """create clickhouse tables for event storage."""
    client.command(SELECTION_EVENTS_TABLE.format(ttl_days=ttl_days))
    client.command(FEEDBACK_EVENTS_TABLE.format(ttl_days=ttl_days))


def drop_tables(client: Client) -> None:
    """drop clickhouse tables."""
    client.command(DROP_SELECTION_EVENTS)
    client.command(DROP_FEEDBACK_EVENTS)


def create_database(client: Client, database: str) -> None:
    """create database if it doesn't exist."""
    client.command(f"CREATE DATABASE IF NOT EXISTS {database}")
