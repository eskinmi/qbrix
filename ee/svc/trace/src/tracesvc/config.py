from pydantic_settings import BaseSettings, SettingsConfigDict


class TraceSettings(BaseSettings):

    model_config = SettingsConfigDict(env_prefix="TRACE_")

    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50053

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str | None = None
    redis_db: int = 0

    feedback_stream_name: str = "qbrix:feedback"
    selection_stream_name: str = "qbrix:selection"
    consumer_group: str = "trace"
    consumer_name: str = "worker-0"

    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_database: str = "qbrix"

    batch_size: int = 500
    flush_interval_sec: float = 5.0

    @property
    def redis_url(self) -> str:
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"
