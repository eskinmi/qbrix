-- feedback events table for tracking rewards received for selections
-- partitioned by tenant and month for efficient queries and data retention

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
TTL event_date + INTERVAL 90 DAY
SETTINGS index_granularity = 8192;
