"""
tests/unit/test_strategy_composite.py
Unit tests for CompositeStrategy.
"""
import pytest

from app.models.condition import DecisionType, DecisionValue
from app.models.errors import ErrorType, MemintelError
from app.models.result import ConceptOutputType, ConceptResult
from app.strategies.composite import CompositeStrategy


@pytest.fixture
def strategy():
    return CompositeStrategy()


def _result(value: float = 0.5) -> ConceptResult:
    return ConceptResult(
        value=value,
        type=ConceptOutputType.FLOAT,
        entity="entity_1",
        version="1.0",
        deterministic=True,
    )


def _boolean_dv(fired: bool, condition_id: str = "c1") -> DecisionValue:
    return DecisionValue(
        value=fired,
        decision_type=DecisionType.BOOLEAN,
        condition_id=condition_id,
        condition_version="1.0",
        entity="entity_1",
    )


def _categorical_dv(label: str = "high", condition_id: str = "c_eq") -> DecisionValue:
    return DecisionValue(
        value=label,
        decision_type=DecisionType.CATEGORICAL,
        condition_id=condition_id,
        condition_version="1.0",
        entity="entity_1",
    )


class TestCompositeAND:
    def test_and_both_true_fires(self, strategy):
        r = strategy.evaluate(
            _result(), [],
            {
                "operator": "AND",
                "operand_results": [_boolean_dv(True), _boolean_dv(True)],
            },
        )
        assert r.value is True

    def test_and_one_false_does_not_fire(self, strategy):
        r = strategy.evaluate(
            _result(), [],
            {
                "operator": "AND",
                "operand_results": [_boolean_dv(True), _boolean_dv(False)],
            },
        )
        assert r.value is False

    def test_and_both_false_does_not_fire(self, strategy):
        r = strategy.evaluate(
            _result(), [],
            {
                "operator": "AND",
                "operand_results": [_boolean_dv(False), _boolean_dv(False)],
            },
        )
        assert r.value is False

    def test_and_three_operands_all_true(self, strategy):
        r = strategy.evaluate(
            _result(), [],
            {
                "operator": "AND",
                "operand_results": [
                    _boolean_dv(True), _boolean_dv(True), _boolean_dv(True)
                ],
            },
        )
        assert r.value is True

    def test_and_three_operands_one_false(self, strategy):
        r = strategy.evaluate(
            _result(), [],
            {
                "operator": "AND",
                "operand_results": [
                    _boolean_dv(True), _boolean_dv(False), _boolean_dv(True)
                ],
            },
        )
        assert r.value is False


class TestCompositeOR:
    def test_or_one_true_fires(self, strategy):
        r = strategy.evaluate(
            _result(), [],
            {
                "operator": "OR",
                "operand_results": [_boolean_dv(False), _boolean_dv(True)],
            },
        )
        assert r.value is True

    def test_or_both_false_does_not_fire(self, strategy):
        r = strategy.evaluate(
            _result(), [],
            {
                "operator": "OR",
                "operand_results": [_boolean_dv(False), _boolean_dv(False)],
            },
        )
        assert r.value is False

    def test_or_both_true_fires(self, strategy):
        r = strategy.evaluate(
            _result(), [],
            {
                "operator": "OR",
                "operand_results": [_boolean_dv(True), _boolean_dv(True)],
            },
        )
        assert r.value is True


class TestCompositeDecisionValue:
    def test_returns_decision_value_not_bool(self, strategy):
        r = strategy.evaluate(
            _result(), [],
            {"operator": "AND", "operand_results": [_boolean_dv(True), _boolean_dv(True)]},
        )
        assert isinstance(r, DecisionValue)
        assert not isinstance(r, bool)

    def test_decision_type_is_boolean(self, strategy):
        """Composite always produces decision<boolean>."""
        r = strategy.evaluate(
            _result(), [],
            {"operator": "AND", "operand_results": [_boolean_dv(True), _boolean_dv(True)]},
        )
        assert r.decision_type == DecisionType.BOOLEAN

    def test_provenance(self, strategy):
        r = strategy.evaluate(
            _result(), [],
            {"operator": "AND", "operand_results": [_boolean_dv(True), _boolean_dv(True)]},
            condition_id="composite_risk", condition_version="1.0",
        )
        assert r.condition_id == "composite_risk"
        assert r.condition_version == "1.0"
        assert r.entity == "entity_1"


class TestCompositeTypeError:
    def test_categorical_operand_raises_type_error(self, strategy):
        """
        decision<categorical> (from equals strategy) cannot be used as a composite
        operand — raises type_error.
        """
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(
                _result(), [],
                {
                    "operator": "AND",
                    "operand_results": [
                        _boolean_dv(True),
                        _categorical_dv("high"),   # decision<categorical> — invalid
                    ],
                },
            )
        assert exc.value.error_type == ErrorType.TYPE_ERROR

    def test_all_categorical_operands_raise_type_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(
                _result(), [],
                {
                    "operator": "OR",
                    "operand_results": [
                        _categorical_dv("low"),
                        _categorical_dv("high"),
                    ],
                },
            )
        assert exc.value.error_type == ErrorType.TYPE_ERROR

    def test_non_decision_value_operand_raises_semantic_error(self, strategy):
        """A raw bool in operand_results raises semantic_error (not a DecisionValue)."""
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(
                _result(), [],
                {
                    "operator": "AND",
                    "operand_results": [True, False],
                },
            )
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR


class TestCompositeSemanticError:
    def test_missing_operator_raises_semantic_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(
                _result(), [],
                {"operand_results": [_boolean_dv(True), _boolean_dv(True)]},
            )
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR

    def test_missing_operand_results_raises_semantic_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(
                _result(), [],
                {"operator": "AND"},
            )
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR

    def test_empty_operand_results_raises_semantic_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(
                _result(), [],
                {"operator": "AND", "operand_results": []},
            )
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR

    def test_invalid_operator_raises_semantic_error(self, strategy):
        with pytest.raises(MemintelError) as exc:
            strategy.evaluate(
                _result(), [],
                {
                    "operator": "XOR",   # not a valid operator
                    "operand_results": [_boolean_dv(True), _boolean_dv(True)],
                },
            )
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR
