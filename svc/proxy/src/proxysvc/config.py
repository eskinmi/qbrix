from pydantic_settings import BaseSettings, SettingsConfigDict


class ProxySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PROXY_")

    runenv: str = "dev"

    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50050

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "qbrix"
    postgres_password: str = "qbrix"
    postgres_database: str = "qbrix"

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str | None = None
    redis_db: int = 0

    motor_host: str = "localhost"
    motor_port: int = 50051

    stream_name: str = "qbrix:feedback"

    token_secret: str = "change-me-in-production"  # need to come from secrets.
    token_max_age_ms: int | None = None

    gate_cache_maxsize: int = 1000
    gate_cache_ttl: float = 30.0  # seconds
    gate_redis_ttl: int = 300  # seconds

    # jwt settings
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # http settings
    http_host: str = "0.0.0.0"
    http_port: int = 8080

    # ee settings
    ee_enabled: bool = False
    selection_stream_name: str = "qbrix:selection"

    # ee clickhouse settings (for analytics endpoints)
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_database: str = "qbrix"

    @property
    def postgres_dsn(self) -> str:
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"

    @property
    def redis_url(self) -> str:
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def motor_address(self) -> str:
        return f"{self.motor_host}:{self.motor_port}"

    @property
    def token_secret_bytes(self) -> bytes:
        return self.token_secret.encode()


settings = ProxySettings()
