"""
tests/unit/test_strategy_percentile.py
Unit tests for PercentileStrategy.
"""
import pytest

from app.models.condition import DecisionType, DecisionValue
from app.models.errors import ErrorType, MemintelError
from app.models.result import ConceptOutputType, ConceptResult
from app.strategies.percentile import PercentileStrategy


@pytest.fixture
def strategy():
    return PercentileStrategy()


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


# 10 history values evenly spaced 0.1..1.0
_HISTORY_10 = _history([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


class TestPercentileHappyPath:
    def test_top_10_fires(self, strategy):
        """Value of 1.5 is above p90 of history → fires in top 10%."""
        r = strategy.evaluate(
            _result(1.5), _HISTORY_10,
            {"direction": "top", "value": 10},
        )
        assert r.value is True

    def test_top_10_does_not_fire(self, strategy):
        """Value of 0.5 is in the middle → does not fire in top 10%."""
        r = strategy.evaluate(
            _result(0.5), _HISTORY_10,
            {"direction": "top", "value": 10},
        )
        assert r.value is False

    def test_bottom_10_fires(self, strategy):
        """Value of -0.5 is below p10 of history → fires in bottom 10%."""
        r = strategy.evaluate(
            _result(-0.5), _HISTORY_10,
            {"direction": "bottom", "value": 10},
        )
        assert r.value is True

    def test_bottom_10_does_not_fire(self, strategy):
        """Value of 0.5 is in the middle → does not fire in bottom 10%."""
        r = strategy.evaluate(
            _result(0.5), _HISTORY_10,
            {"direction": "bottom", "value": 10},
        )
        assert r.value is False

    def test_top_50_fires(self, strategy):
        """Value above median fires in top 50%."""
        r = strategy.evaluate(
            _result(0.9), _HISTORY_10,
            {"direction": "top", "value": 50},
        )
        assert r.value is True

    def test_empty_history_does_not_fire(self, strategy):
        """Without history, cannot rank — does not fire."""
        r = strategy.evaluate(
            _result(0.99), [],
            {"direction": "top", "value": 10},
        )
        assert r.value is False


class TestPercentileDecisionValue:
    def test_returns_decision_value_not_bool(self, strategy):
        r = strategy.evaluate(_result(1.5), _HISTORY_10, {"direction": "top", "value": 10})
        assert isinstance(r, DecisionValue)
        assert not isinstance(r, bool)

    def test_decision_type_is_boolean(self, strategy):
        r = strategy.evaluate(_result(1.5), _HISTORY_10, {"direction": "top", "value": 10})
        assert r.decision_type == DecisionType.BOOLEAN

    def test_provenance(self, strategy):
        r = strategy.evaluate(
            _result(1.5), _HISTORY_10,
            {"direction": "top", "value": 10},
            condition_id="ltv_top", condition_version="1.1",
        )
        assert r.condition_id == "ltv_top"
        assert r.condition_version == "1.1"
        assert r.entity == "entity_1"


class TestPercentileTypeError:
    def test_categorical_raises_type_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(
                _result("high", "categorical"), _HISTORY_10,
                {"direction": "top", "value": 10},
            )
        assert exc.value.error_type == ErrorType.TYPE_ERROR

    def test_boolean_raises_type_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(
                _result(True, "boolean"), _HISTORY_10,
                {"direction": "top", "value": 10},
            )
        assert exc.value.error_type == ErrorType.TYPE_ERROR


class TestPercentileSemanticError:
    def test_missing_direction(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(_result(0.9), _HISTORY_10, {"value": 10})
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR

    def test_missing_value(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(_result(0.9), _HISTORY_10, {"direction": "top"})
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR

    def test_invalid_direction(self, strategy):
        """Percentile uses 'top'/'bottom', NOT 'above'/'below'."""
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(_result(0.9), _HISTORY_10, {"direction": "above", "value": 10})
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR

    def test_value_out_of_range(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(_result(0.9), _HISTORY_10, {"direction": "top", "value": 110})
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR
