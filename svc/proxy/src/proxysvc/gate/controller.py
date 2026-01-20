from enum import Flag, auto

from .config import FeatureGateConfig, Rule, BaseArmModel


class ExperimentFlagState(Flag):
    """flags representing the state of an experiment."""

    ENABLED = auto()
    ACTIVE = auto()
    BLACKOUT = auto()
    DISABLED = auto()
    RPOS = auto()  # context is within rollout percentage
    RNEG = auto()  # context is outside rollout percentage


class FeatureGate:
    """feature gate controller for experiment-based decision-making."""

    negset = (
        ExperimentFlagState.DISABLED
        | ExperimentFlagState.BLACKOUT
        | ExperimentFlagState.RNEG
    )

    @classmethod
    def render_feature_flags(
        cls, config: FeatureGateConfig, context_id: str
    ) -> ExperimentFlagState:
        """render experiment state flags based on configuration and context."""
        experiment = config.experiment
        state = ExperimentFlagState(0)

        if experiment.enabled:
            state |= ExperimentFlagState.ENABLED
        else:
            state |= ExperimentFlagState.DISABLED

        if experiment.schedule.is_in_active_schedule():
            state |= ExperimentFlagState.ACTIVE
        else:
            state |= ExperimentFlagState.BLACKOUT

        if experiment.rollout.is_in_rollout(context_id):
            state |= ExperimentFlagState.RPOS
        else:
            state |= ExperimentFlagState.RNEG

        return state

    @classmethod
    def render_rules(cls, config: FeatureGateConfig, metadata: dict) -> Rule | None:
        """apply rules to metadata and return first matching rule."""
        if not (rules := config.rules):
            return None

        for rule in rules:
            if rule.eval(metadata):
                return rule

        return None

    @classmethod
    def control(
        cls, config: FeatureGateConfig, context_id: str, metadata: dict
    ) -> BaseArmModel | None:
        """determine which arm to return based on experiment state and rules.

        returns the committed arm if experiment is disabled, in blackout,
        or context is outside rollout. otherwise evaluates rules and returns
        matching rule's arm, or None if bandit selection should proceed.
        """
        fstate = cls.render_feature_flags(config, context_id)
        if cls.negset & fstate:
            return config.experiment.arm.committed

        if rule := cls.render_rules(config, metadata):
            return rule.arm.committed

        return None
