"""
tests/unit/test_strategy_change.py
Unit tests for ChangeStrategy.
"""
import pytest

from app.models.condition import DecisionType, DecisionValue
from app.models.errors import ErrorType, MemintelError
from app.models.result import ConceptOutputType, ConceptResult
from app.strategies.change import ChangeStrategy


@pytest.fixture
def strategy():
    return ChangeStrategy()


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


class TestChangeHappyPath:
    def test_increase_fires(self, strategy):
        """previous=1.0 → current=2.0: 100% increase > 50% threshold → fires."""
        r = strategy.evaluate(
            _result(2.0), _history([1.0]),
            {"direction": "increase", "value": 0.5},
        )
        assert r.value is True

    def test_increase_does_not_fire(self, strategy):
        """previous=1.0 → current=1.1: 10% increase < 50% threshold → no fire."""
        r = strategy.evaluate(
            _result(1.1), _history([1.0]),
            {"direction": "increase", "value": 0.5},
        )
        assert r.value is False

    def test_decrease_fires(self, strategy):
        """previous=1.0 → current=0.3: 70% decrease > 50% threshold → fires."""
        r = strategy.evaluate(
            _result(0.3), _history([1.0]),
            {"direction": "decrease", "value": 0.5},
        )
        assert r.value is True

    def test_decrease_does_not_fire(self, strategy):
        """previous=1.0 → current=0.9: 10% decrease < 50% threshold → no fire."""
        r = strategy.evaluate(
            _result(0.9), _history([1.0]),
            {"direction": "decrease", "value": 0.5},
        )
        assert r.value is False

    def test_any_fires_on_increase(self, strategy):
        r = strategy.evaluate(
            _result(2.0), _history([1.0]),
            {"direction": "any", "value": 0.5},
        )
        assert r.value is True

    def test_any_fires_on_decrease(self, strategy):
        r = strategy.evaluate(
            _result(0.3), _history([1.0]),
            {"direction": "any", "value": 0.5},
        )
        assert r.value is True

    def test_any_does_not_fire_on_small_change(self, strategy):
        r = strategy.evaluate(
            _result(1.05), _history([1.0]),
            {"direction": "any", "value": 0.5},
        )
        assert r.value is False

    def test_uses_most_recent_history(self, strategy):
        """Most recent history value is used (history[-1])."""
        r = strategy.evaluate(
            _result(2.0), _history([0.5, 0.8, 1.0]),
            {"direction": "increase", "value": 0.5},
        )
        # previous=1.0 (last), change=100% → fires
        assert r.value is True

    def test_empty_history_does_not_fire(self, strategy):
        r = strategy.evaluate(
            _result(10.0), [],
            {"direction": "any", "value": 0.5},
        )
        assert r.value is False

    def test_zero_previous_nonzero_current_fires(self, strategy):
        """
        previous=0, current=1.0 — any non-zero movement from zero fires.
        reason='infinite_change'; no ZeroDivisionError raised.
        """
        r = strategy.evaluate(
            _result(1.0), _history([0.0]),
            {"direction": "increase", "value": 0.5},
        )
        assert r.value is True
        assert r.reason == "infinite_change"


class TestChangeDecisionValue:
    def test_returns_decision_value_not_bool(self, strategy):
        r = strategy.evaluate(
            _result(2.0), _history([1.0]),
            {"direction": "increase", "value": 0.5},
        )
        assert isinstance(r, DecisionValue)
        assert not isinstance(r, bool)

    def test_decision_type_is_boolean(self, strategy):
        r = strategy.evaluate(
            _result(2.0), _history([1.0]),
            {"direction": "increase", "value": 0.5},
        )
        assert r.decision_type == DecisionType.BOOLEAN

    def test_provenance(self, strategy):
        r = strategy.evaluate(
            _result(2.0), _history([1.0]),
            {"direction": "increase", "value": 0.5},
            condition_id="price_change", condition_version="1.0",
        )
        assert r.condition_id == "price_change"
        assert r.entity == "entity_1"


class TestChangeTypeError:
    def test_categorical_raises_type_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(
                _result("high", "categorical"), _history([1.0]),
                {"direction": "increase", "value": 0.5},
            )
        assert exc.value.error_type == ErrorType.TYPE_ERROR

    def test_boolean_raises_type_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(
                _result(True, "boolean"), _history([1.0]),
                {"direction": "increase", "value": 0.5},
            )
        assert exc.value.error_type == ErrorType.TYPE_ERROR


class TestChangeSemanticError:
    def test_missing_direction(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(_result(2.0), _history([1.0]), {"value": 0.5})
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR

    def test_missing_value(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(_result(2.0), _history([1.0]), {"direction": "increase"})
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR

    def test_invalid_direction(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(
                _result(2.0), _history([1.0]),
                {"direction": "above", "value": 0.5},
            )
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR

    def test_negative_value_raises_semantic_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(
                _result(2.0), _history([1.0]),
                {"direction": "increase", "value": -0.1},
            )
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR
