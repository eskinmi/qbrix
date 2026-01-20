import time

from qbrixstore.postgres.session import init_db, get_session, create_tables
from qbrixstore.postgres.models import Pool, Experiment
from qbrixstore.redis.client import RedisClient
from qbrixstore.redis.streams import RedisStreamPublisher, FeedbackEvent
from qbrixstore.config import PostgresSettings, RedisSettings

from proxysvc.config import ProxySettings
from proxysvc.repository import (
    PoolRepository,
    ExperimentRepository,
    FeatureGateRepository,
)
from proxysvc.motor_client import MotorClient
from proxysvc.token import SelectionToken
from proxysvc.gate import GateService


class ProxyService:
    def __init__(self, settings: ProxySettings):
        self._settings = settings
        self._redis: RedisClient | None = None
        self._publisher: RedisStreamPublisher | None = None
        self._motor_client: MotorClient | None = None
        self._gate_service: GateService | None = None

    async def start(self) -> None:
        pg_settings = PostgresSettings(
            host=self._settings.postgres_host,
            port=self._settings.postgres_port,
            user=self._settings.postgres_user,
            password=self._settings.postgres_password,
            database=self._settings.postgres_database,
        )
        init_db(pg_settings)
        await create_tables()

        redis_settings = RedisSettings(
            host=self._settings.redis_host,
            port=self._settings.redis_port,
            password=self._settings.redis_password,
            db=self._settings.redis_db,
            stream_name=self._settings.stream_name,
        )
        self._redis = RedisClient(redis_settings)
        await self._redis.connect()

        self._publisher = RedisStreamPublisher(redis_settings)
        await self._publisher.connect()

        self._motor_client = MotorClient(self._settings.motor_address)
        await self._motor_client.connect()

        self._gate_service = GateService(self._redis, self._settings)

    async def stop(self) -> None:
        if self._motor_client:
            await self._motor_client.close()
        if self._publisher:
            await self._publisher.close()
        if self._redis:
            await self._redis.close()

    async def create_pool(self, name: str, arms: list[dict]) -> dict:
        async with get_session() as session:
            repo = PoolRepository(session)
            pool = await repo.create(name, arms)
            return self._pool_to_dict(pool)

    async def get_pool(self, pool_id: str) -> dict | None:
        async with get_session() as session:
            repo = PoolRepository(session)
            pool = await repo.get(pool_id)
            if pool is None:
                return None
            return self._pool_to_dict(pool)

    @staticmethod
    async def delete_pool(pool_id: str) -> bool:
        async with get_session() as session:
            repo = PoolRepository(session)
            return await repo.delete(pool_id)

    async def create_experiment(
        self,
        name: str,
        pool_id: str,
        protocol: str,
        protocol_params: dict,
        enabled: bool,
        feature_gate_config: dict | None = None,
    ) -> dict:
        async with get_session() as session:
            repo = ExperimentRepository(session)
            experiment = await repo.create(
                name=name,
                pool_id=pool_id,
                protocol=protocol,
                protocol_params=protocol_params,
                enabled=enabled,
                feature_gate_config=feature_gate_config,
            )
            exp_dict = self._experiment_to_dict(experiment)
            experiment_id = experiment.id

            # sync gate config to redis if provided
            if feature_gate_config:
                gate_repo = FeatureGateRepository(session)
                gate = await gate_repo.get(experiment_id)
                if gate:
                    gate_config = gate_repo.to_config(gate)
                    await self._gate_service.set_config(experiment_id, gate_config)

        await self._sync_experiment_to_redis(experiment_id, pool_id)
        return exp_dict

    async def get_experiment(self, experiment_id: str) -> dict | None:
        async with get_session() as session:
            repo = ExperimentRepository(session)
            experiment = await repo.get(experiment_id)
            if experiment is None:
                return None
            return self._experiment_to_dict(experiment)

    async def update_experiment(self, experiment_id: str, **kwargs) -> dict | None:
        async with get_session() as session:
            repo = ExperimentRepository(session)
            experiment = await repo.update(experiment_id, **kwargs)
            if experiment is None:
                return None
            exp_dict = self._experiment_to_dict(experiment)
            pool_id = experiment.pool_id
        await self._sync_experiment_to_redis(experiment_id, pool_id)
        return exp_dict

    async def delete_experiment(self, experiment_id: str) -> bool:
        async with get_session() as session:
            repo = ExperimentRepository(session)
            deleted = await repo.delete(experiment_id)
        if deleted:
            await self._redis.delete_experiment(experiment_id)
            await self._gate_service.delete_config(experiment_id)
        return deleted

    async def create_gate_config(self, experiment_id: str, config: dict) -> dict | None:
        """create feature gate config for an experiment."""
        async with get_session() as session:
            repo = FeatureGateRepository(session)
            await repo.create(experiment_id, config)
            # reload to get default_arm relationship
            gate = await repo.get(experiment_id)
            gate_config = repo.to_config(gate)
        await self._gate_service.set_config(experiment_id, gate_config)
        return gate_config.model_dump(mode="json")

    async def get_gate_config(self, experiment_id: str) -> dict | None:
        """get feature gate config for an experiment."""
        config = await self._gate_service.get_config(experiment_id)
        if config is None:
            return None
        return config.model_dump(mode="json")

    async def update_gate_config(self, experiment_id: str, config: dict) -> dict | None:
        """update feature gate config for an experiment."""
        async with get_session() as session:
            repo = FeatureGateRepository(session)
            updated = await repo.update(experiment_id, config)
            if updated is None:
                return None
            # reload to get default_arm relationship
            gate = await repo.get(experiment_id)
            gate_config = repo.to_config(gate)
        self._gate_service.invalidate(experiment_id)
        await self._gate_service.set_config(experiment_id, gate_config)
        return gate_config.model_dump(mode="json")

    async def delete_gate_config(self, experiment_id: str) -> bool:
        """delete feature gate config for an experiment."""
        async with get_session() as session:
            repo = FeatureGateRepository(session)
            deleted = await repo.delete(experiment_id)
        if deleted:
            await self._gate_service.delete_config(experiment_id)
        return deleted

    async def select(
        self,
        experiment_id: str,
        context_id: str,
        context_vector: list[float],
        context_metadata: dict,
    ) -> dict:
        # evaluate feature gate first
        committed_arm = await self._gate_service.evaluate(
            experiment_id=experiment_id,
            context_id=context_id,
            context_metadata=context_metadata,
        )

        if committed_arm is not None and committed_arm.index is not None:
            # gate determined the arm - skip bandit
            token = SelectionToken.encode(
                secret=self._settings.token_secret_bytes,
                experiment_id=experiment_id,
                arm_index=committed_arm.index,
                context_id=context_id,
                context_vector=context_vector,
                context_metadata=context_metadata,
            )
            return {
                "arm": {
                    "id": committed_arm.id,
                    "name": committed_arm.name,
                    "index": committed_arm.index,
                },
                "request_id": token,
                "is_default": True,
            }

        # bandit selection via motorsvc
        response = await self._motor_client.select(
            experiment_id=experiment_id,
            context_id=context_id,
            context_vector=context_vector,
            context_metadata=context_metadata,
        )

        token = SelectionToken.encode(
            secret=self._settings.token_secret_bytes,
            experiment_id=experiment_id,
            arm_index=response["arm"]["index"],
            context_id=context_id,
            context_vector=context_vector,
            context_metadata=context_metadata,
        )
        response["request_id"] = token
        response["is_default"] = False
        return response

    async def feed(self, request_id: str, reward: float) -> bool:
        """
        process feedback for a prior selection.

        args:
            request_id: signed token from select() containing selection context
            reward: observed reward value

        returns:
            True if feedback was accepted

        raises:
            TokenError: if token is invalid or expired
        """
        selection = SelectionToken.decode(
            secret=self._settings.token_secret_bytes,
            token=request_id,
            max_age_ms=self._settings.token_max_age_ms,
        )

        event = FeedbackEvent(
            experiment_id=selection.experiment_id,
            request_id=request_id,
            arm_index=selection.arm_index,
            reward=reward,
            context_id=selection.context_id,
            context_vector=selection.context_vector,
            context_metadata=selection.context_metadata,
            timestamp_ms=int(time.time() * 1000),
        )
        await self._publisher.publish(event)
        return True

    async def _sync_experiment_to_redis(self, experiment_id: str, pool_id: str) -> None:
        """Sync experiment with full pool data to Redis for motorsvc."""
        async with get_session() as session:
            pool_repo = PoolRepository(session)
            pool = await pool_repo.get(pool_id)
            experiment_repo = ExperimentRepository(session)
            experiment = await experiment_repo.get(experiment_id)

            redis_data = {
                "id": experiment.id,
                "name": experiment.name,
                "pool_id": experiment.pool_id,
                "pool": self._pool_to_dict(pool),
                "protocol": experiment.protocol,
                "protocol_params": experiment.protocol_params,
                "enabled": experiment.enabled,
            }
            await self._redis.set_experiment(experiment_id, redis_data)

    async def health(self) -> bool:
        try:
            await self._redis.client.ping()
            return True
        except Exception:  # noqa
            return False

    @staticmethod
    def _pool_to_dict(pool: Pool) -> dict:
        return {
            "id": pool.id,
            "name": pool.name,
            "arms": [
                {
                    "id": arm.id,
                    "name": arm.name,
                    "index": arm.index,
                    "is_active": arm.is_active,
                }
                for arm in sorted(pool.arms, key=lambda a: a.index)
            ],
        }

    @staticmethod
    def _experiment_to_dict(experiment: Experiment) -> dict:
        return {
            "id": experiment.id,
            "name": experiment.name,
            "pool_id": experiment.pool_id,
            "protocol": experiment.protocol,
            "protocol_params": experiment.protocol_params,
            "enabled": experiment.enabled,
        }
