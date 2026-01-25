"""Unit tests for parameter backends."""

from qbrixcore.param.backend import InMemoryParamBackend
from qbrixcore.param.state import BaseParamState


class TestInMemoryParamBackend:
    def test_backend_creation(self):
        """test backend creation initializes empty store."""
        backend = InMemoryParamBackend()

        assert backend.store == {}

    def test_backend_get_nonexistent_experiment(self):
        """test getting params for non-existent experiment returns None."""
        backend = InMemoryParamBackend()

        result = backend.get("nonexistent-exp")

        assert result is None

    def test_backend_set_and_get_params(self):
        """test setting and retrieving params."""
        backend = InMemoryParamBackend()
        params = BaseParamState(num_arms=3)
        experiment_id = "exp-123"

        backend.set(experiment_id, params)
        result = backend.get(experiment_id)

        assert result is not None
        assert result == params
        assert result.num_arms == 3

    def test_backend_set_overwrites_existing_params(self):
        """test setting params overwrites existing params for same experiment."""
        backend = InMemoryParamBackend()
        params1 = BaseParamState(num_arms=3)
        params2 = BaseParamState(num_arms=5)
        experiment_id = "exp-123"

        backend.set(experiment_id, params1)
        backend.set(experiment_id, params2)
        result = backend.get(experiment_id)

        assert result == params2
        assert result.num_arms == 5

    def test_backend_multiple_experiments(self):
        """test storing params for multiple experiments."""
        backend = InMemoryParamBackend()
        params1 = BaseParamState(num_arms=3)
        params2 = BaseParamState(num_arms=4)
        params3 = BaseParamState(num_arms=5)

        backend.set("exp-1", params1)
        backend.set("exp-2", params2)
        backend.set("exp-3", params3)

        assert backend.get("exp-1") == params1
        assert backend.get("exp-2") == params2
        assert backend.get("exp-3") == params3

    def test_backend_isolation_between_experiments(self):
        """test params are isolated between experiments."""
        backend = InMemoryParamBackend()
        params1 = BaseParamState(num_arms=3)
        params2 = BaseParamState(num_arms=5)

        backend.set("exp-1", params1)
        backend.set("exp-2", params2)

        # modifying params1 should not affect stored params
        params1.num_arms = 10

        stored_params1 = backend.get("exp-1")
        assert stored_params1 is params1  # same object reference

    def test_backend_store_dictionary_access(self):
        """test direct store access contains correct data."""
        backend = InMemoryParamBackend()
        params = BaseParamState(num_arms=3)
        experiment_id = "exp-123"

        backend.set(experiment_id, params)

        assert experiment_id in backend.store
        assert backend.store[experiment_id] == params
        assert len(backend.store) == 1

    def test_backend_empty_experiment_id(self):
        """test storing params with empty string experiment id."""
        backend = InMemoryParamBackend()
        params = BaseParamState(num_arms=3)

        backend.set("", params)
        result = backend.get("")

        assert result == params
