-- selection events table for tracking arm selections
-- partitioned by tenant and month for efficient queries and data retention

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
    policy String,
    event_date Date DEFAULT toDate(fromUnixTimestamp64Milli(timestamp_ms))
)
ENGINE = MergeTree()
PARTITION BY (tenant_id, toYYYYMM(event_date))
ORDER BY (tenant_id, experiment_id, timestamp_ms, request_id)
TTL event_date + INTERVAL 90 DAY
SETTINGS index_granularity = 8192;
