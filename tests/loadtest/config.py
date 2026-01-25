from __future__ import annotations

from pydantic_settings import BaseSettings


class LoadTestSettings(BaseSettings):
    proxy_host: str = "localhost"
    proxy_port: int = 8080
    api_key: str = ""

    # single experiment scenario defaults
    default_pool_name: str = "loadtest-pool"
    default_experiment_name: str = "loadtest-experiment"
    default_protocol: str = "BetaTSProtocol"
    default_num_arms: int = 5

    # multi experiment scenario defaults
    num_experiments: int = 10
    max_users_per_experiment: int = 100

    # behavior tuning
    feedback_probability: float = 0.7  # probability of sending feedback after select
    feedback_delay_min_ms: int = 100  # min delay before sending feedback
    feedback_delay_max_ms: int = 2000  # max delay before sending feedback
    reward_success_probability: float = 0.3  # probability of positive reward

    # context generation
    context_vector_dim: int = 10  # dimension of context vector
    context_metadata_keys: list[str] = ["device", "region", "user_tier"]

    model_config = {"env_prefix": "LOADTEST_"}

    @property
    def proxy_address(self) -> str:
        return f"http://{self.proxy_host}:{self.proxy_port}"  # noqa


settings = LoadTestSettings()
