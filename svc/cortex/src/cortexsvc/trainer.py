from collections import defaultdict
import numpy as np

from qbrixcore.context import Context
from qbrixcore.protoc.base import BaseProtocol
from qbrixstore.redis.client import RedisClient
from qbrixstore.redis.streams import FeedbackEvent


def _build_protocol_map() -> dict[str, type[BaseProtocol]]:
    registry = {}

    def collect(cls):
        for subclass in cls.__subclasses__():
            if hasattr(subclass, "name") and subclass.name:
                registry[subclass.name] = subclass
            collect(subclass)

    collect(BaseProtocol)
    return registry


PROTOCOL_MAP = _build_protocol_map()


class BatchTrainer:
    def __init__(self, redis_client: RedisClient):
        self._redis = redis_client

    async def train_batch(self, events: list[FeedbackEvent]) -> dict[str, int]:
        events_by_experiment = defaultdict(list)
        for event in events:
            events_by_experiment[event.experiment_id].append(event)

        ledger = {}
        for experiment_id, experiment_events in events_by_experiment.items():
            count = await self._train_experiment(experiment_id, experiment_events)
            ledger[experiment_id] = count

        return ledger

    async def _train_experiment(
        self, experiment_id: str, events: list[FeedbackEvent]
    ) -> int:
        experiment_data = await self._redis.get_experiment(experiment_id)
        if experiment_data is None:
            return 0

        protocol_name = experiment_data["protocol"]
        protocol_cls = PROTOCOL_MAP.get(protocol_name)
        if protocol_cls is None:
            return 0

        params = await self._redis.get_params(experiment_id)
        if params is None:
            num_arms = len(experiment_data["pool"]["arms"])
            param_state = protocol_cls.init_params(
                num_arms=num_arms, **experiment_data.get("protocol_params", {})
            )
        else:
            param_state = protocol_cls.param_state_cls.model_validate(params)

        for event in events:
            context = Context(
                id=event.context_id,
                vector=np.array(event.context_vector, dtype=np.float16),
                metadata=event.context_metadata,
            )

            param_state = protocol_cls.train(
                ps=param_state,
                context=context,
                choice=event.arm_index,
                reward=event.reward,
            )

        await self._redis.set_params(experiment_id, param_state.model_dump())
        return len(events)
