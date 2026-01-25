"""Unit tests for callback system."""

import pytest

from qbrixcore.callback import BaseCallback
from qbrixcore.callback import register


class TestBaseCallback:
    def test_base_callback_is_abstract(self):
        """test BaseCallback is abstract and requires scope property."""
        with pytest.raises(TypeError):
            BaseCallback()  # noqa

    def test_base_callback_methods_exist(self):
        """test BaseCallback defines expected methods."""
        assert hasattr(BaseCallback, 'on_select_start')
        assert hasattr(BaseCallback, 'on_select_end')
        assert hasattr(BaseCallback, 'on_feed_start')
        assert hasattr(BaseCallback, 'on_feed_end')

    def test_concrete_callback_implementation(self):
        """test concrete callback can be created."""
        class ConcreteCallback(BaseCallback):
            @property
            def scope(self):
                return "test"

        callback = ConcreteCallback()
        assert callback.scope == "test"

    def test_callback_default_methods_do_nothing(self):
        """test default callback methods do nothing."""
        class ConcreteCallback(BaseCallback):
            @property
            def scope(self):
                return "test"

        callback = ConcreteCallback()

        # should not raise
        callback.on_select_start()
        callback.on_select_end()
        callback.on_feed_start()
        callback.on_feed_end()

    def test_callback_can_override_methods(self):
        """test callback can override hook methods."""
        class TrackingCallback(BaseCallback):
            def __init__(self):
                self.select_started = False
                self.select_ended = False

            @property
            def scope(self):
                return "tracking"

            def on_select_start(self, *args, **kwargs):
                self.select_started = True

            def on_select_end(self, *args, **kwargs):
                self.select_ended = True

        callback = TrackingCallback()
        callback.on_select_start()
        callback.on_select_end()

        assert callback.select_started is True
        assert callback.select_ended is True


class TestRegisterDecorator:
    def test_register_decorator_basic(self):
        """test register decorator wraps method correctly."""
        class MockAgent:
            def __init__(self):
                self.callbacks = []

            @register()
            def select(self):
                return "selected"

        agent = MockAgent()
        result = agent.select()

        assert result == "selected"

    def test_register_decorator_invokes_start_callback(self):
        """test register decorator invokes on_start callback."""
        class MockCallback(BaseCallback):
            def __init__(self):
                self.start_called = False

            @property
            def scope(self):
                return "test"

            def on_select_start(self, agent):  # noqa
                self.start_called = True

        class MockAgent:
            def __init__(self):
                self.callbacks = []

            @register()
            def select(self):
                return "selected"

        callback = MockCallback()
        agent = MockAgent()
        agent.callbacks.append(callback)

        agent.select()

        assert callback.start_called is True

    def test_register_decorator_invokes_end_callback(self):
        """test register decorator invokes on_end callback."""
        class MockCallback(BaseCallback):
            def __init__(self):
                self.end_called = False

            @property
            def scope(self):
                return "test"

            def on_select_end(self, agent):  # noqa
                self.end_called = True

        class MockAgent:
            def __init__(self):
                self.callbacks = []

            @register()
            def select(self):
                return "selected"

        callback = MockCallback()
        agent = MockAgent()
        agent.callbacks.append(callback)

        agent.select()

        assert callback.end_called is True

    def test_register_decorator_with_custom_method_name(self):
        """test register decorator with custom method name."""
        class MockCallback(BaseCallback):
            def __init__(self):
                self.start_called = False

            @property
            def scope(self):
                return "test"

            def on_custom_start(self, agent):  # noqa
                self.start_called = True

        class MockAgent:
            def __init__(self):
                self.callbacks = []

            @register(method_name="custom")
            def do_something(self):
                return "done"

        callback = MockCallback()
        agent = MockAgent()
        agent.callbacks.append(callback)

        agent.do_something()

        assert callback.start_called is True

    def test_register_decorator_invokes_multiple_callbacks(self):
        """test register decorator invokes multiple callbacks."""
        class MockCallback(BaseCallback):
            def __init__(self, name):
                self.name = name
                self.start_called = False

            @property
            def scope(self):
                return self.name

            def on_select_start(self, agent):  # noqa
                self.start_called = True

        class MockAgent:
            def __init__(self):
                self.callbacks = []

            @register()
            def select(self):
                return "selected"

        callback1 = MockCallback("cb1")
        callback2 = MockCallback("cb2")
        agent = MockAgent()
        agent.callbacks.append(callback1)
        agent.callbacks.append(callback2)

        agent.select()

        assert callback1.start_called is True
        assert callback2.start_called is True

    def test_register_decorator_with_no_callbacks(self):
        """test register decorator works when no callbacks registered."""
        class MockAgent:
            def __init__(self):
                self.callbacks = []

            @register()
            def select(self):
                return "selected"

        agent = MockAgent()
        result = agent.select()

        assert result == "selected"

    def test_register_decorator_with_empty_callbacks_list(self):
        """test register decorator works with empty callbacks list."""
        class MockAgent:
            def __init__(self):
                self.callbacks = []

            @register()
            def select(self):
                return "selected"

        agent = MockAgent()
        result = agent.select()

        assert result == "selected"

    def test_register_decorator_preserves_method_signature(self):
        """test register decorator preserves method arguments."""
        class MockAgent:
            def __init__(self):
                self.callbacks = []

            @register()
            def select(self, context, pool):
                return f"{context}-{pool}"

        agent = MockAgent()
        result = agent.select("ctx1", "pool1")

        assert result == "ctx1-pool1"

    def test_register_decorator_callback_receives_agent(self):
        """test callback receives agent instance."""
        received_agent = None

        class MockCallback(BaseCallback):
            @property
            def scope(self):
                return "test"

            def on_select_start(self, agent):  # noqa
                nonlocal received_agent
                received_agent = agent

        class MockAgent:
            def __init__(self):
                self.callbacks = []
                self.name = "test-agent"

            @register()
            def select(self):
                return "selected"

        callback = MockCallback()
        agent = MockAgent()
        agent.callbacks.append(callback)

        agent.select()

        assert received_agent is agent
        assert received_agent.name == "test-agent"  # noqa

    def test_register_decorator_callback_without_hook_method(self):
        """test callback without hook method is skipped gracefully."""
        class MockCallback(BaseCallback):
            @property
            def scope(self):
                return "test"
            # no on_select_start method

        class MockAgent:
            def __init__(self):
                self.callbacks = []

            @register()
            def select(self):
                return "selected"

        callback = MockCallback()
        agent = MockAgent()
        agent.callbacks.append(callback)

        # should not raise
        result = agent.select()
        assert result == "selected"
