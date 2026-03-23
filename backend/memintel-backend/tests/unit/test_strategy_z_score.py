"""
tests/unit/test_strategy_z_score.py
Unit tests for ZScoreStrategy.
"""
import pytest

from app.models.condition import DecisionType, DecisionValue
from app.models.errors import ErrorType, MemintelError
from app.models.result import ConceptOutputType, ConceptResult
from app.strategies.z_score import ZScoreStrategy


@pytest.fixture
def strategy():
    return ZScoreStrategy()


def _result(value: float, result_type: str = "float") -> ConceptResult:
    return ConceptResult(
        value=value,
        type=ConceptOutputType(result_type),
        entity="entity_1",
        version="1.0",
        deterministic=True,
    )


def _history(values: list[float]) -> list[ConceptResult]:
    return [_result(v) for v in values]


# History with mean=0.5, std≈0.158
_HISTORY = _history([0.3, 0.4, 0.5, 0.6, 0.7])


class TestZScoreHappyPath:
    def test_above_fires(self, strategy):
        """Value far above the mean exceeds threshold → fires."""
        history = _history([0.0, 0.0, 0.0, 0.0])  # mean=0, std=0 → edge case
        # Use history with actual std
        r = strategy.evaluate(
            _result(10.0), _HISTORY,
            {"threshold": 2.0, "direction": "above"},
        )
        assert r.value is True

    def test_above_does_not_fire(self, strategy):
        """Value equal to mean → z=0 → does not exceed threshold."""
        r = strategy.evaluate(
            _result(0.5), _HISTORY,
            {"threshold": 2.0, "direction": "above"},
        )
        assert r.value is False

    def test_below_fires(self, strategy):
        """Value far below the mean → z strongly negative → fires for 'below'."""
        r = strategy.evaluate(
            _result(-10.0), _HISTORY,
            {"threshold": 2.0, "direction": "below"},
        )
        assert r.value is True

    def test_below_does_not_fire_for_positive_z(self, strategy):
        """High value → positive z → does not fire for direction='below'."""
        r = strategy.evaluate(
            _result(10.0), _HISTORY,
            {"threshold": 2.0, "direction": "below"},
        )
        assert r.value is False

    def test_any_fires_above(self, strategy):
        r = strategy.evaluate(
            _result(10.0), _HISTORY,
            {"threshold": 2.0, "direction": "any"},
        )
        assert r.value is True

    def test_any_fires_below(self, strategy):
        r = strategy.evaluate(
            _result(-10.0), _HISTORY,
            {"threshold": 2.0, "direction": "any"},
        )
        assert r.value is True

    def test_any_does_not_fire_at_mean(self, strategy):
        r = strategy.evaluate(
            _result(0.5), _HISTORY,
            {"threshold": 2.0, "direction": "any"},
        )
        assert r.value is False

    def test_zero_std_does_not_fire(self, strategy):
        """All history values identical → std=0 → undefined z → does not fire."""
        flat_history = _history([0.5, 0.5, 0.5, 0.5])
        r = strategy.evaluate(
            _result(0.5), flat_history,
            {"threshold": 1.0, "direction": "any"},
        )
        assert r.value is False

    def test_empty_history_does_not_fire(self, strategy):
        r = strategy.evaluate(
            _result(100.0), [],
            {"threshold": 2.0, "direction": "any"},
        )
        assert r.value is False


class TestZScoreDecisionValue:
    def test_returns_decision_value_not_bool(self, strategy):
        r = strategy.evaluate(_result(10.0), _HISTORY, {"threshold": 2.0, "direction": "any"})
        assert isinstance(r, DecisionValue)
        assert not isinstance(r, bool)

    def test_decision_type_is_boolean(self, strategy):
        r = strategy.evaluate(_result(10.0), _HISTORY, {"threshold": 2.0, "direction": "any"})
        assert r.decision_type == DecisionType.BOOLEAN

    def test_provenance(self, strategy):
        r = strategy.evaluate(
            _result(10.0), _HISTORY,
            {"threshold": 2.0, "direction": "any"},
            condition_id="dau_anomaly", condition_version="3.0",
        )
        assert r.condition_id == "dau_anomaly"
        assert r.condition_version == "3.0"


class TestZScoreTypeError:
    def test_categorical_raises_type_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(
                _result("high", "categorical"), _HISTORY,
                {"threshold": 2.0, "direction": "any"},
            )
        assert exc.value.error_type == ErrorType.TYPE_ERROR

    def test_boolean_raises_type_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(
                _result(True, "boolean"), _HISTORY,
                {"threshold": 2.0, "direction": "any"},
            )
        assert exc.value.error_type == ErrorType.TYPE_ERROR


class TestZScoreSemanticError:
    def test_missing_threshold(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(_result(1.0), _HISTORY, {"direction": "above"})
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR

    def test_missing_direction(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(_result(1.0), _HISTORY, {"threshold": 2.0})
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR

    def test_zero_threshold_raises_semantic_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(_result(1.0), _HISTORY, {"threshold": 0.0, "direction": "above"})
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR

    def test_negative_threshold_raises_semantic_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(_result(1.0), _HISTORY, {"threshold": -1.0, "direction": "above"})
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR

    def test_invalid_direction(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(_result(1.0), _HISTORY, {"threshold": 2.0, "direction": "top"})
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR
