from __future__ import annotations

import random
import uuid
from collections import deque
from dataclasses import dataclass
from typing import ClassVar

import gevent
from locust import User
from locust import between
from locust import events
from locust import task

from tests.loadtest.client import ProxyClient
from tests.loadtest.client import SelectResult
from tests.loadtest.config import settings


@dataclass
class PendingFeedback:
    request_id: str


class SingleExperimentUser(User):
    """
    simulates users interacting with a single experiment/pool.

    behavior:
    - users make select requests continuously
    - feedback is sent asynchronously with realistic delay
    - not all selections result in feedback (configurable probability)
    - rewards are probabilistic (success/failure)

    this simulates real-world behavior where:
    - selection is synchronous (user sees the arm immediately)
    - conversion/feedback happens later (user interacts, then converts or not)
    """

    wait_time = between(0.1, 0.5)

    # shared state across all users (set during test init)
    experiment_id: ClassVar[str | None] = None
    pool_id: ClassVar[str | None] = None
    _setup_done: ClassVar[bool] = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client: ProxyClient | None = None
        self.pending_feedbacks: deque[PendingFeedback] = deque(maxlen=100)

    def on_start(self) -> None:
        self.client = ProxyClient()
        self.client.connect()

    def on_stop(self) -> None:
        if self.client:
            self.client.close()

    @task(10)
    def select_arm(self) -> None:
        """make a selection request and potentially queue feedback."""
        if not self.experiment_id or not self.client:
            return

        context_id = str(uuid.uuid4())
        context_vector = [random.random() for _ in range(settings.context_vector_dim)]
        context_metadata = {
            "device": random.choice(["mobile", "desktop", "tablet"]),
            "region": random.choice(["us-east", "us-west", "eu", "apac"]),
            "user_tier": random.choice(["free", "premium", "enterprise"]),
        }

        try:
            result: SelectResult = self.client.select(
                experiment_id=self.experiment_id,
                context_id=context_id,
                context_vector=context_vector,
                context_metadata=context_metadata,
            )

            events.request.fire(
                request_type="grpc",
                name="Select",
                response_time=0,  # locust handles timing
                response_length=0,
                exception=None,
                context={},
            )

            # probabilistically queue feedback for later
            if random.random() < settings.feedback_probability:
                pending = PendingFeedback(request_id=result.request_id)
                self.pending_feedbacks.append(pending)

                # schedule async feedback with delay
                delay_ms = random.randint(
                    settings.feedback_delay_min_ms,
                    settings.feedback_delay_max_ms,
                )
                gevent.spawn_later(delay_ms / 1000.0, self._send_feedback, pending)

        except Exception as e:
            events.request.fire(
                request_type="grpc",
                name="Select",
                response_time=0,
                response_length=0,
                exception=e,
                context={},
            )

    @task(1)
    def health_check(self) -> None:
        """periodic health check."""
        if not self.client:
            return

        try:
            self.client.health_check()
            events.request.fire(
                request_type="grpc",
                name="Health",
                response_time=0,
                response_length=0,
                exception=None,
                context={},
            )
        except Exception as e:
            events.request.fire(
                request_type="grpc",
                name="Health",
                response_time=0,
                response_length=0,
                exception=e,
                context={},
            )

    def _send_feedback(self, pending: PendingFeedback) -> None:
        """send feedback asynchronously (called via gevent)."""
        if not self.client or not self.client.is_connected:
            return

        # determine reward based on probability
        reward = 1.0 if random.random() < settings.reward_success_probability else 0.0

        try:
            self.client.feedback(
                request_id=pending.request_id,
                reward=reward,
            )
            events.request.fire(
                request_type="grpc",
                name="Feedback",
                response_time=0,
                response_length=0,
                exception=None,
                context={},
            )
        except Exception as e:
            events.request.fire(
                request_type="grpc",
                name="Feedback",
                response_time=0,
                response_length=0,
                exception=e,
                context={},
            )


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """setup shared experiment before test starts."""
    if SingleExperimentUser._setup_done:
        return

    client = ProxyClient()
    client.connect()

    try:
        # create pool
        pool_id = client.create_pool(
            name=f"{settings.default_pool_name}-{uuid.uuid4().hex[:8]}",
            num_arms=settings.default_num_arms,
        )
        SingleExperimentUser.pool_id = pool_id

        # create experiment
        experiment_id = client.create_experiment(
            name=f"{settings.default_experiment_name}-{uuid.uuid4().hex[:8]}",
            pool_id=pool_id,
            protocol=settings.default_protocol,
        )
        SingleExperimentUser.experiment_id = experiment_id
        SingleExperimentUser._setup_done = True

        print(f"created pool: {pool_id}")
        print(f"created experiment: {experiment_id}")

    finally:
        client.close()


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """cleanup experiment after test ends."""
    if not SingleExperimentUser._setup_done:
        return

    client = ProxyClient()
    client.connect()

    try:
        if SingleExperimentUser.experiment_id:
            client.delete_experiment(SingleExperimentUser.experiment_id)
            print(f"deleted experiment: {SingleExperimentUser.experiment_id}")

        if SingleExperimentUser.pool_id:
            client.delete_pool(SingleExperimentUser.pool_id)
            print(f"deleted pool: {SingleExperimentUser.pool_id}")

    except Exception as e:
        print(f"cleanup error: {e}")

    finally:
        client.close()
        SingleExperimentUser._setup_done = False
        SingleExperimentUser.experiment_id = None
        SingleExperimentUser.pool_id = None