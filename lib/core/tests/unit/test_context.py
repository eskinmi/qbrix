"""Unit tests for Context class."""

import numpy as np

from qbrixcore.context import Context


class TestContext:
    def test_context_creation_with_defaults(self):
        """test context creation with auto-generated id and default values."""
        ctx = Context()

        assert ctx.id is not None
        assert isinstance(ctx.id, str)
        assert ctx.vector == []
        assert ctx.metadata == {}

    def test_context_creation_with_custom_id(self):
        """test context creation with custom id."""
        ctx = Context(id="custom-ctx-123")

        assert ctx.id == "custom-ctx-123"

    def test_context_creation_with_list_vector(self):
        """test context creation with list vector."""
        vector = [1.0, 2.0, 3.0]
        ctx = Context(vector=vector)

        assert ctx.vector == vector
        assert isinstance(ctx.vector, list)

    def test_context_creation_with_numpy_vector(self):
        """test context creation with numpy array vector."""
        vector = np.array([1.0, 2.0, 3.0])
        ctx = Context(vector=vector)

        assert isinstance(ctx.vector, np.ndarray)
        assert np.array_equal(ctx.vector, vector)

    def test_context_creation_with_metadata(self):
        """test context creation with metadata."""
        metadata = {"user_id": "user-123", "session": "session-456"}
        ctx = Context(metadata=metadata)

        assert ctx.metadata == metadata
        assert ctx.metadata["user_id"] == "user-123"
        assert ctx.metadata["session"] == "session-456"

    def test_context_creation_with_all_fields(self):
        """test context creation with all fields specified."""
        ctx = Context(
            id="ctx-789",
            vector=[0.1, 0.2, 0.3],
            metadata={"key": "value"}
        )

        assert ctx.id == "ctx-789"
        assert ctx.vector == [0.1, 0.2, 0.3]
        assert ctx.metadata == {"key": "value"}

    def test_context_unique_ids(self):
        """test that different contexts get unique ids."""
        ctx1 = Context()
        ctx2 = Context()

        assert ctx1.id != ctx2.id

    def test_context_empty_vector(self):
        """test context with empty vector list."""
        ctx = Context(vector=[])

        assert ctx.vector == []
        assert len(ctx.vector) == 0

    def test_context_numpy_vector_shape(self):
        """test context numpy vector maintains shape."""
        vector = np.array([[1.0], [2.0], [3.0]])
        ctx = Context(vector=vector)

        assert ctx.vector.shape == (3, 1)

    def test_context_metadata_empty_dict(self):
        """test context with empty metadata."""
        ctx = Context(metadata={})

        assert ctx.metadata == {}
        assert len(ctx.metadata) == 0

    def test_context_vector_with_negative_values(self):
        """test context with negative values in vector."""
        vector = [-1.5, -0.3, 2.1, -4.8]
        ctx = Context(vector=vector)

        assert ctx.vector == vector

    def test_context_vector_with_zero_values(self):
        """test context with zero values in vector."""
        vector = [0.0, 0.0, 0.0]
        ctx = Context(vector=vector)

        assert ctx.vector == vector
