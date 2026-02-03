from pydantic_settings import BaseSettings, SettingsConfigDict


class PostgresSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POSTGRES_")

    host: str = "localhost"
    port: int = 5432
    user: str = "qbrix"
    password: str = "qbrix"
    database: str = "qbrix"

    @property
    def dsn(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_")

    host: str = "localhost"
    port: int = 6379
    password: str | None = None
    db: int = 0

    # Stream settings
    stream_name: str = "qbrix:feedback"
    consumer_group: str = "cortex"
    stream_max_len: int = 1_000_000  # safety cap to prevent unbounded growth

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class ClickHouseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLICKHOUSE_")

    host: str = "localhost"
    port: int = 8123
    user: str = "default"
    password: str = ""
    database: str = "qbrix"

    # batch settings for tracesvc
    batch_size: int = 1000
    flush_interval_sec: float = 5.0
    ttl_days: int = 90


class StoreSettings(BaseSettings):
    postgres: PostgresSettings = PostgresSettings()
    redis: RedisSettings = RedisSettings()
    clickhouse: ClickHouseSettings = ClickHouseSettings()
