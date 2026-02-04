from __future__ import annotations

import json
from typing import Sequence

import clickhouse_connect
from clickhouse_connect.driver.client import Client

from qbrixstore.redis.streams import SelectionEvent
from qbrixstore.redis.streams import FeedbackEvent
from qbrixstore.config import ClickHouseSettings


class ClickHouseClient:
    """client for batch inserting events to clickhouse."""

    SELECTION_COLUMNS = [
        "tenant_id",
        "experiment_id",
        "request_id",
        "arm_id",
        "arm_name",
        "arm_index",
        "is_default",
        "context_id",
        "context_vector",
        "context_metadata",
        "timestamp_ms",
        "policy",
    ]

    FEEDBACK_COLUMNS = [
        "tenant_id",
        "experiment_id",
        "request_id",
        "arm_index",
        "reward",
        "context_id",
        "context_vector",
        "context_metadata",
        "timestamp_ms",
    ]

    def __init__(self, settings: ClickHouseSettings | None = None):
        if settings is None:
            settings = ClickHouseSettings()
        self._settings = settings
        self._client: Client | None = None

    def connect(self, create_database: bool = True) -> None:
        if create_database:
            init_client = clickhouse_connect.get_client(
                host=self._settings.host,
                port=self._settings.port,
                username=self._settings.user,
                password=self._settings.password,
            )
            init_client.command(f"CREATE DATABASE IF NOT EXISTS {self._settings.database}")
            init_client.close()

        self._client = clickhouse_connect.get_client(
            host=self._settings.host,
            port=self._settings.port,
            username=self._settings.user,
            password=self._settings.password,
            database=self._settings.database,
        )

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    @property
    def client(self) -> Client:
        if self._client is None:
            raise RuntimeError("ClickHouse client not connected. Call connect() first.")
        return self._client

    def insert_selection_events(self, events: Sequence[SelectionEvent]) -> None:
        """batch insert selection events."""
        if not events:
            return

        rows = [self._selection_to_row(event) for event in events]
        self.client.insert(
            table="selection_events",
            data=rows,
            column_names=self.SELECTION_COLUMNS,
        )

    def insert_feedback_events(self, events: Sequence[FeedbackEvent]) -> None:
        """batch insert feedback events."""
        if not events:
            return

        rows = [self._feedback_to_row(event) for event in events]
        self.client.insert(
            table="feedback_events",
            data=rows,
            column_names=self.FEEDBACK_COLUMNS,
        )

    @staticmethod
    def _selection_to_row(event: SelectionEvent) -> tuple:
        """convert selection event to clickhouse row tuple."""
        return (
            event.tenant_id,
            event.experiment_id,
            event.request_id,
            event.arm_id,
            event.arm_name,
            event.arm_index,
            event.is_default,
            event.context_id,
            event.context_vector,
            json.dumps(event.context_metadata),
            event.timestamp_ms,
            event.policy,
        )

    @staticmethod
    def _feedback_to_row(event: FeedbackEvent) -> tuple:
        """convert feedback event to clickhouse row tuple."""
        return (
            event.tenant_id,
            event.experiment_id,
            event.request_id,
            event.arm_index,
            event.reward,
            event.context_id,
            event.context_vector,
            json.dumps(event.context_metadata),
            event.timestamp_ms,
        )

    def query_selection_events(
        self,
        tenant_id: str,
        experiment_id: str | None = None,
        request_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """query selection events with optional filters."""
        conditions = ["tenant_id = {tenant_id:String}"]
        params = {"tenant_id": tenant_id, "limit": limit, "offset": offset}

        if experiment_id:
            conditions.append("experiment_id = {experiment_id:String}")
            params["experiment_id"] = experiment_id

        if request_id:
            conditions.append("request_id = {request_id:String}")
            params["request_id"] = request_id

        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT *
            FROM selection_events
            WHERE {where_clause}
            ORDER BY timestamp_ms DESC
            LIMIT {{limit:UInt32}} OFFSET {{offset:UInt32}}
        """

        result = self.client.query(query, parameters=params)
        return [dict(zip(result.column_names, row)) for row in result.result_rows]

    def query_feedback_events(
        self,
        tenant_id: str,
        experiment_id: str | None = None,
        request_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """query feedback events with optional filters."""
        conditions = ["tenant_id = {tenant_id:String}"]
        params = {"tenant_id": tenant_id, "limit": limit, "offset": offset}

        if experiment_id:
            conditions.append("experiment_id = {experiment_id:String}")
            params["experiment_id"] = experiment_id

        if request_id:
            conditions.append("request_id = {request_id:String}")
            params["request_id"] = request_id

        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT *
            FROM feedback_events
            WHERE {where_clause}
            ORDER BY timestamp_ms DESC
            LIMIT {{limit:UInt32}} OFFSET {{offset:UInt32}}
        """

        result = self.client.query(query, parameters=params)
        return [dict(zip(result.column_names, row)) for row in result.result_rows]

    def get_experiment_stats(
        self,
        tenant_id: str,
        experiment_id: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> dict:
        """get aggregated stats for an experiment."""
        conditions = [
            "tenant_id = {tenant_id:String}",
            "experiment_id = {experiment_id:String}",
        ]
        params = {"tenant_id": tenant_id, "experiment_id": experiment_id}

        if start_ms:
            conditions.append("timestamp_ms >= {start_ms:Int64}")
            params["start_ms"] = start_ms  # noqa

        if end_ms:
            conditions.append("timestamp_ms < {end_ms:Int64}")
            params["end_ms"] = end_ms  # noqa

        where_clause = " AND ".join(conditions)

        selection_query = f"""
            SELECT
                count() as total_selections,
                countIf(is_default = true) as default_selections,
                uniqExact(context_id) as unique_contexts,
                min(timestamp_ms) as first_selection_ms,
                max(timestamp_ms) as last_selection_ms
            FROM selection_events
            WHERE {where_clause}
        """

        feedback_query = f"""
            SELECT
                count() as total_feedback,
                avg(reward) as avg_reward,
                min(reward) as min_reward,
                max(reward) as max_reward
            FROM feedback_events
            WHERE {where_clause}
        """

        selection_result = self.client.query(selection_query, parameters=params)
        feedback_result = self.client.query(feedback_query, parameters=params)

        sel_row = selection_result.result_rows[0] if selection_result.result_rows else [0, 0, 0, 0, 0]
        fb_row = feedback_result.result_rows[0] if feedback_result.result_rows else [0, None, None, None]

        return {
            "total_selections": sel_row[0],
            "default_selections": sel_row[1],
            "unique_contexts": sel_row[2],
            "first_selection_ms": sel_row[3],
            "last_selection_ms": sel_row[4],
            "total_feedback": fb_row[0],
            "avg_reward": fb_row[1],
            "min_reward": fb_row[2],
            "max_reward": fb_row[3],
        }

    def get_experiment_timeseries(
        self,
        tenant_id: str,
        experiment_id: str,
        interval_ms: int = 3600000,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[dict]:
        """get time series data for an experiment."""
        conditions = [
            "tenant_id = {tenant_id:String}",
            "experiment_id = {experiment_id:String}",
        ]
        params = {
            "tenant_id": tenant_id,
            "experiment_id": experiment_id,
            "interval_ms": interval_ms,
        }

        if start_ms:
            conditions.append("timestamp_ms >= {start_ms:Int64}")
            params["start_ms"] = start_ms

        if end_ms:
            conditions.append("timestamp_ms < {end_ms:Int64}")
            params["end_ms"] = end_ms

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT
                intDiv(timestamp_ms, {{interval_ms:Int64}}) * {{interval_ms:Int64}} as bucket,
                count() as selections,
                countIf(is_default = true) as default_selections
            FROM selection_events
            WHERE {where_clause}
            GROUP BY bucket
            ORDER BY bucket ASC
        """

        result = self.client.query(query, parameters=params)
        return [
            {"timestamp_ms": row[0], "selections": row[1], "default_selections": row[2]}
            for row in result.result_rows
        ]

    def get_arm_stats(
        self,
        tenant_id: str,
        experiment_id: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[dict]:
        """get per-arm statistics for an experiment."""
        conditions = [
            "tenant_id = {tenant_id:String}",
            "experiment_id = {experiment_id:String}",
        ]
        params = {"tenant_id": tenant_id, "experiment_id": experiment_id}

        if start_ms:
            conditions.append("timestamp_ms >= {start_ms:Int64}")
            params["start_ms"] = start_ms  # noqa

        if end_ms:
            conditions.append("timestamp_ms < {end_ms:Int64}")
            params["end_ms"] = end_ms  # noqa

        where_clause = " AND ".join(conditions)

        selection_query = f"""
            SELECT
                arm_index,
                arm_name,
                count() as selections
            FROM selection_events
            WHERE {where_clause}
            GROUP BY arm_index, arm_name
            ORDER BY arm_index
        """

        feedback_query = f"""
            SELECT
                arm_index,
                count() as feedback_count,
                avg(reward) as avg_reward
            FROM feedback_events
            WHERE {where_clause}
            GROUP BY arm_index
        """

        selection_result = self.client.query(selection_query, parameters=params)
        feedback_result = self.client.query(feedback_query, parameters=params)

        feedback_map = {
            row[0]: {"feedback_count": row[1], "avg_reward": row[2]}
            for row in feedback_result.result_rows
        }

        return [
            {
                "arm_index": row[0],
                "arm_name": row[1],
                "selections": row[2],
                "feedback_count": feedback_map.get(row[0], {}).get("feedback_count", 0),
                "avg_reward": feedback_map.get(row[0], {}).get("avg_reward"),
            }
            for row in selection_result.result_rows
        ]

    def health_check(self) -> bool:
        """check if clickhouse is healthy."""
        try:
            result = self.client.query("SELECT 1")
            return result.result_rows == [(1,)]
        except Exception:  # noqa
            return False
