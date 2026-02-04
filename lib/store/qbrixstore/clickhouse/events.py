from __future__ import annotations

import json

from qbrixstore.redis.streams import SelectionEvent as BaseSelectionEvent
from qbrixstore.redis.streams import FeedbackEvent as BaseFeedbackEvent


class SelectionEvent(BaseSelectionEvent):
    """selection event with clickhouse row conversion."""

    def to_row(self) -> tuple:
        """convert to clickhouse row tuple."""
        return (
            self.tenant_id,
            self.experiment_id,
            self.request_id,
            self.arm_id,
            self.arm_name,
            self.arm_index,
            self.is_default,
            self.context_id,
            self.context_vector,
            json.dumps(self.context_metadata),
            self.timestamp_ms,
            self.policy,
        )

    @classmethod
    def from_base(cls, event: BaseSelectionEvent) -> SelectionEvent:
        """convert from base selection event."""
        return cls(
            tenant_id=event.tenant_id,
            experiment_id=event.experiment_id,
            request_id=event.request_id,
            arm_id=event.arm_id,
            arm_name=event.arm_name,
            arm_index=event.arm_index,
            is_default=event.is_default,
            context_id=event.context_id,
            context_vector=event.context_vector,
            context_metadata=event.context_metadata,
            timestamp_ms=event.timestamp_ms,
            policy=event.policy,
        )


class FeedbackEvent(BaseFeedbackEvent):
    """feedback event with clickhouse row conversion."""

    def to_row(self) -> tuple:
        """convert to clickhouse row tuple."""
        return (
            self.tenant_id,
            self.experiment_id,
            self.request_id,
            self.arm_index,
            self.reward,
            self.context_id,
            self.context_vector,
            json.dumps(self.context_metadata),
            self.timestamp_ms,
        )

    @classmethod
    def from_base(cls, event: BaseFeedbackEvent) -> FeedbackEvent:
        """convert from base feedback event."""
        return cls(
            tenant_id=event.tenant_id,
            experiment_id=event.experiment_id,
            request_id=event.request_id,
            arm_index=event.arm_index,
            reward=event.reward,
            context_id=event.context_id,
            context_vector=event.context_vector,
            context_metadata=event.context_metadata,
            timestamp_ms=event.timestamp_ms,
        )
