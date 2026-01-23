from pydantic_settings import BaseSettings, SettingsConfigDict


class CortexSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CORTEX_")

    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50052

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str | None = None
    redis_db: int = 0

    stream_name: str = "qbrix:feedback"
    consumer_group: str = "cortex"
    consumer_name: str = "worker-0"

    batch_size: int = 256
    batch_timeout_ms: int = 100
    flush_interval_sec: int = 10

    @property
    def redis_url(self) -> str:
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"
