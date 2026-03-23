"""
tests/unit/test_strategy_threshold.py
Unit tests for ThresholdStrategy.
"""
import pytest

from app.models.condition import DecisionType
from app.models.errors import ErrorType, MemintelError
from app.models.result import ConceptOutputType, ConceptResult
from app.strategies.threshold import ThresholdStrategy


@pytest.fixture
def strategy():
    return ThresholdStrategy()


def _result(value: float, result_type: str = "float") -> ConceptResult:
    return ConceptResult(
        value=value,
        type=ConceptOutputType(result_type),
        entity="entity_1",
        version="1.0",
        deterministic=True,
    )


class TestThresholdHappyPath:
    def test_above_fires(self, strategy):
        r = strategy.evaluate(
            _result(0.9), [],
            {"direction": "above", "value": 0.8},
            condition_id="c1", condition_version="1.0",
        )
        assert r.value is True

    def test_below_fires(self, strategy):
        r = strategy.evaluate(
            _result(0.3), [],
            {"direction": "below", "value": 0.5},
            condition_id="c1", condition_version="1.0",
        )
        assert r.value is True

    def test_exactly_at_threshold_does_not_fire(self, strategy):
        """Strict inequality — equal to threshold does not fire."""
        r = strategy.evaluate(
            _result(0.8), [],
            {"direction": "above", "value": 0.8},
        )
        assert r.value is False

    def test_above_does_not_fire(self, strategy):
        r = strategy.evaluate(
            _result(0.5), [],
            {"direction": "above", "value": 0.8},
        )
        assert r.value is False

    def test_below_does_not_fire(self, strategy):
        r = strategy.evaluate(
            _result(0.9), [],
            {"direction": "below", "value": 0.5},
        )
        assert r.value is False

    def test_negative_threshold(self, strategy):
        r = strategy.evaluate(
            _result(-10.0), [],
            {"direction": "below", "value": -5.0},
        )
        assert r.value is True

    def test_int_result_accepted(self, strategy):
        """int is a subtype of float — threshold accepts it."""
        r = strategy.evaluate(
            _result(15), [],
            {"direction": "above", "value": 14.0},
        )
        assert r.value is True

    def test_history_is_ignored(self, strategy):
        """Threshold does not use history."""
        history = [_result(v) for v in [0.1, 0.5, 0.9]]
        r = strategy.evaluate(
            _result(0.95), history,
            {"direction": "above", "value": 0.8},
        )
        assert r.value is True


class TestThresholdDecisionValue:
    def test_returns_decision_value_not_bool(self, strategy):
        from app.models.condition import DecisionValue
        r = strategy.evaluate(_result(0.9), [], {"direction": "above", "value": 0.5})
        assert isinstance(r, DecisionValue)
        assert not isinstance(r, bool)

    def test_decision_type_is_boolean(self, strategy):
        r = strategy.evaluate(_result(0.9), [], {"direction": "above", "value": 0.5})
        assert r.decision_type == DecisionType.BOOLEAN

    def test_provenance_condition_id(self, strategy):
        r = strategy.evaluate(
            _result(0.9), [],
            {"direction": "above", "value": 0.5},
            condition_id="churn_risk", condition_version="2.1",
        )
        assert r.condition_id == "churn_risk"
        assert r.condition_version == "2.1"

    def test_provenance_entity(self, strategy):
        r = strategy.evaluate(
            _result(0.9), [],
            {"direction": "above", "value": 0.5},
            condition_id="c", condition_version="1",
        )
        assert r.entity == "entity_1"


class TestThresholdTypeError:
    def test_categorical_input_raises_type_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(
                _result("high", "categorical"), [],
                {"direction": "above", "value": 0.5},
            )
        assert exc.value.error_type == ErrorType.TYPE_ERROR

    def test_boolean_input_raises_type_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(
                _result(True, "boolean"), [],
                {"direction": "above", "value": 0.5},
            )
        assert exc.value.error_type == ErrorType.TYPE_ERROR


class TestThresholdSemanticError:
    def test_missing_direction_raises_semantic_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(_result(0.9), [], {"value": 0.5})
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR

    def test_missing_value_raises_semantic_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(_result(0.9), [], {"direction": "above"})
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR

    def test_invalid_direction_raises_semantic_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(_result(0.9), [], {"direction": "top", "value": 0.5})
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR

    def test_empty_params_raises_semantic_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(_result(0.9), [], {})
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR
