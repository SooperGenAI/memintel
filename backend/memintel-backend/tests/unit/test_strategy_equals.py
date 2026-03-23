"""
tests/unit/test_strategy_equals.py
Unit tests for EqualsStrategy.
"""
import pytest

from app.models.condition import DecisionType, DecisionValue
from app.models.errors import ErrorType, MemintelError
from app.models.result import ConceptOutputType, ConceptResult
from app.strategies.equals import EqualsStrategy


@pytest.fixture
def strategy():
    return EqualsStrategy()


def _result(value: str | float | bool, result_type: str = "categorical") -> ConceptResult:
    return ConceptResult(
        value=value,
        type=ConceptOutputType(result_type),
        entity="entity_1",
        version="1.0",
        deterministic=True,
    )


class TestEqualsHappyPath:
    def test_fires_on_matching_label(self, strategy):
        r = strategy.evaluate(
            _result("high"), [],
            {"value": "high"},
            condition_id="segment_check", condition_version="1.0",
        )
        assert r.value == "high"

    def test_does_not_fire_on_non_matching_label(self, strategy):
        r = strategy.evaluate(
            _result("low"), [],
            {"value": "high"},
        )
        assert r.value == ""

    def test_fires_with_labels_set(self, strategy):
        """When labels is provided, result.value must be in labels AND match target."""
        r = strategy.evaluate(
            _result("high"), [],
            {"value": "high", "labels": ["low", "medium", "high"]},
        )
        assert r.value == "high"

    def test_does_not_fire_with_labels_non_match(self, strategy):
        r = strategy.evaluate(
            _result("medium"), [],
            {"value": "high", "labels": ["low", "medium", "high"]},
        )
        assert r.value == ""


class TestEqualsDecisionValue:
    def test_returns_decision_value_not_str(self, strategy):
        r = strategy.evaluate(_result("high"), [], {"value": "high"})
        assert isinstance(r, DecisionValue)
        assert not isinstance(r, str)

    def test_decision_type_is_categorical(self, strategy):
        """equals strategy produces decision<categorical>, not decision<boolean>."""
        r = strategy.evaluate(_result("high"), [], {"value": "high"})
        assert r.decision_type == DecisionType.CATEGORICAL

    def test_decision_type_is_categorical_on_no_fire(self, strategy):
        r = strategy.evaluate(_result("low"), [], {"value": "high"})
        assert r.decision_type == DecisionType.CATEGORICAL

    def test_provenance(self, strategy):
        r = strategy.evaluate(
            _result("at_risk"), [],
            {"value": "at_risk"},
            condition_id="segment_match", condition_version="2.0",
        )
        assert r.condition_id == "segment_match"
        assert r.condition_version == "2.0"
        assert r.entity == "entity_1"

    def test_fired_value_is_the_matched_label(self, strategy):
        """When fired, DecisionValue.value is the matched label string."""
        r = strategy.evaluate(_result("churned"), [], {"value": "churned"})
        assert r.value == "churned"
        assert isinstance(r.value, str)


class TestEqualsTypeError:
    def test_float_input_raises_type_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(
                _result(0.9, "float"), [],
                {"value": "high"},
            )
        assert exc.value.error_type == ErrorType.TYPE_ERROR

    def test_boolean_input_raises_type_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(
                _result(True, "boolean"), [],
                {"value": "high"},
            )
        assert exc.value.error_type == ErrorType.TYPE_ERROR

    def test_value_not_in_labels_raises_type_error(self, strategy):
        """result.value not in declared labels → type_error (label closure violation)."""
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(
                _result("unknown_label"), [],
                {"value": "unknown_label", "labels": ["low", "medium", "high"]},
            )
        assert exc.value.error_type == ErrorType.TYPE_ERROR


class TestEqualsSemanticError:
    def test_missing_value_raises_semantic_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(_result("high"), [], {})
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR

    def test_empty_labels_raises_semantic_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(
                _result("high"), [],
                {"value": "high", "labels": []},
            )
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR

    def test_non_string_value_raises_semantic_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(
                _result("high"), [],
                {"value": 0.9},  # numeric value param — not a string
            )
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR
