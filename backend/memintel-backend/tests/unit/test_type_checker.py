"""
tests/unit/test_type_checker.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for app/compiler/type_checker.py.

Covers all subtype rules, null propagation, and condition strategy compatibility
as defined in memintel_type_system.md v1.1.
"""
import pytest

from app.compiler.type_checker import GraphNode, TypeChecker
from app.models.errors import ErrorType, MemintelError


@pytest.fixture
def checker() -> TypeChecker:
    return TypeChecker()


# ── is_assignable — subtype rules ─────────────────────────────────────────────

class TestIsAssignable:
    """Unit tests for TypeChecker.is_assignable."""

    # ── Identity ──────────────────────────────────────────────────────────────

    def test_identity_float(self, checker):
        assert checker.is_assignable('float', 'float') is True

    def test_identity_int(self, checker):
        assert checker.is_assignable('int', 'int') is True

    def test_identity_boolean(self, checker):
        assert checker.is_assignable('boolean', 'boolean') is True

    def test_identity_string(self, checker):
        assert checker.is_assignable('string', 'string') is True

    def test_identity_categorical(self, checker):
        assert checker.is_assignable('categorical', 'categorical') is True

    def test_identity_time_series_float(self, checker):
        assert checker.is_assignable('time_series<float>', 'time_series<float>') is True

    # ── int → float widening (§4.1) ───────────────────────────────────────────

    def test_int_to_float_widening(self, checker):
        """int is a strict subtype of float — widening is always valid."""
        assert checker.is_assignable('int', 'float') is True

    def test_float_to_int_not_assignable(self, checker):
        """Narrowing float → int requires explicit to_int(); implicit is rejected."""
        assert checker.is_assignable('float', 'int') is False

    # ── T → T? nullable widening (§4.2) ──────────────────────────────────────

    def test_float_to_float_nullable(self, checker):
        """Non-nullable is always assignable to its nullable variant."""
        assert checker.is_assignable('float', 'float?') is True

    def test_boolean_to_boolean_nullable(self, checker):
        assert checker.is_assignable('boolean', 'boolean?') is True

    def test_string_to_string_nullable(self, checker):
        assert checker.is_assignable('string', 'string?') is True

    def test_nullable_to_non_nullable_rejected(self, checker):
        """float? → float is not implicit; requires null-handling operator."""
        assert checker.is_assignable('float?', 'float') is False

    # ── Container covariance via int→float (§4.1) ─────────────────────────────

    def test_time_series_int_to_time_series_float(self, checker):
        assert checker.is_assignable('time_series<int>', 'time_series<float>') is True

    def test_time_series_float_to_time_series_int_rejected(self, checker):
        assert checker.is_assignable('time_series<float>', 'time_series<int>') is False

    def test_list_int_to_list_float(self, checker):
        assert checker.is_assignable('list<int>', 'list<float>') is True

    def test_list_float_to_list_int_rejected(self, checker):
        assert checker.is_assignable('list<float>', 'list<int>') is False

    # ── Unrelated types ───────────────────────────────────────────────────────

    def test_float_to_boolean_rejected(self, checker):
        assert checker.is_assignable('float', 'boolean') is False

    def test_int_to_boolean_rejected(self, checker):
        assert checker.is_assignable('int', 'boolean') is False

    def test_string_to_float_rejected(self, checker):
        assert checker.is_assignable('string', 'float') is False

    def test_decision_boolean_to_float_rejected(self, checker):
        assert checker.is_assignable('decision<boolean>', 'float') is False


# ── check_node — operator type validation ─────────────────────────────────────

class TestCheckNode:
    """Unit tests for TypeChecker.check_node."""

    # ── int → float widening through an operator ──────────────────────────────

    def test_int_input_accepted_for_float_operator(self, checker):
        """Operators declaring float inputs accept int via widening."""
        node = GraphNode(op='z_score_op', node_id='n1')
        result = checker.check_node(node, {'input': 'int'})
        assert result == 'float'

    def test_time_series_int_accepted_for_mean(self, checker):
        """mean declares time_series<float> input; time_series<int> is accepted."""
        node = GraphNode(op='mean', node_id='n2')
        result = checker.check_node(node, {'input': 'time_series<int>'})
        assert result == 'float'

    # ── float → int narrowing must be explicit ────────────────────────────────

    def test_float_to_int_requires_to_int_operator(self, checker):
        """
        count requires time_series<int>. Passing time_series<float> is a type_error
        because float→int narrowing is not implicit.
        """
        node = GraphNode(op='count', node_id='n3')
        with pytest.raises(MemintelError) as exc_info:
            checker.check_node(node, {'input': 'time_series<float>'})
        assert exc_info.value.error_type == ErrorType.TYPE_ERROR

    def test_to_int_operator_accepts_float(self, checker):
        """to_int is the explicit cast; it accepts float and produces int."""
        node = GraphNode(op='to_int', node_id='n4')
        result = checker.check_node(node, {'input': 'float'})
        assert result == 'int'

    # ── T → T? nullable widening through an operator ──────────────────────────

    def test_non_nullable_accepted_where_nullable_expected(self, checker):
        """
        coalesce declares input: float?.  Passing float (non-nullable) is valid
        via T → T? widening, and the output remains float (null_handler=True).
        """
        node = GraphNode(op='coalesce', node_id='n5')
        result = checker.check_node(node, {'input': 'float', 'default': 'float'})
        assert result == 'float'

    # ── time_series<int> → time_series<float> ─────────────────────────────────

    def test_time_series_int_to_time_series_float_via_pct_change(self, checker):
        """pct_change declares time_series<float>; accepts time_series<int> via widening."""
        node = GraphNode(op='pct_change', node_id='n6')
        result = checker.check_node(node, {'input': 'time_series<int>'})
        assert result == 'float'

    # ── Null propagation (§6.3) ───────────────────────────────────────────────

    def test_nullable_input_propagates_to_output(self, checker):
        """
        If an operator input receives T? (nullable) where T is expected,
        the output type becomes T? (null propagates through).
        """
        node = GraphNode(op='z_score_op', node_id='n7')
        result = checker.check_node(node, {'input': 'float?'})
        assert result == 'float?'

    def test_null_propagation_through_arithmetic(self, checker):
        """Nullable input to add propagates nullable to output."""
        node = GraphNode(op='add', node_id='n8')
        result = checker.check_node(node, {'a': 'float?', 'b': 'float'})
        assert result == 'float?'

    def test_null_propagation_chain(self, checker):
        """
        Simulates a two-node chain:
          float? → z_score_op → float?   (node 1 output)
          float? → normalize  → float?   (node 2 receives nullable output)
        """
        node1 = GraphNode(op='z_score_op', node_id='n9a')
        out1 = checker.check_node(node1, {'input': 'float?'})
        assert out1 == 'float?'

        node2 = GraphNode(op='normalize', node_id='n9b')
        out2 = checker.check_node(node2, {'input': out1})
        assert out2 == 'float?'

    def test_coalesce_strips_nullable(self, checker):
        """
        coalesce is a null-handling operator: it consumes float? and produces
        float (non-nullable), stopping null propagation.
        """
        node = GraphNode(op='coalesce', node_id='n10')
        result = checker.check_node(node, {'input': 'float?', 'default': 'float'})
        assert result == 'float'

    def test_fill_null_strips_nullable_from_time_series(self, checker):
        """fill_null consumes time_series<float?> and produces time_series<float>."""
        node = GraphNode(op='fill_null', node_id='n11')
        result = checker.check_node(node, {'input': 'time_series<float?>', 'value': 'float'})
        assert result == 'time_series<float>'

    def test_drop_null_strips_nullable_from_time_series(self, checker):
        """drop_null consumes time_series<float?> and produces time_series<float>."""
        node = GraphNode(op='drop_null', node_id='n12')
        result = checker.check_node(node, {'input': 'time_series<float?>'})
        assert result == 'time_series<float>'

    # ── decision<T> blocking (§10 rule 8) ─────────────────────────────────────

    def test_decision_boolean_cannot_flow_into_concept_operator(self, checker):
        """
        decision<boolean> may only flow to composite or unwrap_decision.
        Passing it to a concept operator (e.g. z_score_op) is a type_error.
        """
        node = GraphNode(op='z_score_op', node_id='n13')
        with pytest.raises(MemintelError) as exc_info:
            checker.check_node(node, {'input': 'decision<boolean>'})
        assert exc_info.value.error_type == ErrorType.TYPE_ERROR

    def test_decision_boolean_cannot_flow_into_add(self, checker):
        node = GraphNode(op='add', node_id='n14')
        with pytest.raises(MemintelError) as exc_info:
            checker.check_node(node, {'a': 'decision<boolean>', 'b': 'float'})
        assert exc_info.value.error_type == ErrorType.TYPE_ERROR

    def test_decision_boolean_accepted_by_composite(self, checker):
        """composite is the only condition strategy that takes decision<boolean>."""
        node = GraphNode(op='composite', node_id='n15')
        result = checker.check_node(node, {'input': 'decision<boolean>'})
        assert result == 'decision<boolean>'

    def test_decision_boolean_accepted_by_unwrap_decision(self, checker):
        """unwrap_decision strips decision<boolean> to boolean."""
        node = GraphNode(op='unwrap_decision', node_id='n16')
        result = checker.check_node(node, {'input': 'decision<boolean>'})
        assert result == 'boolean'

    def test_decision_categorical_unwraps_to_categorical(self, checker):
        """unwrap_decision(decision<categorical>) → categorical."""
        node = GraphNode(op='unwrap_decision', node_id='n17')
        result = checker.check_node(node, {'input': 'decision<categorical>'})
        assert result == 'categorical'

    # ── equals strategy: rejects float input (§9.1) ──────────────────────────

    def test_equals_strategy_on_float_input_raises_type_error(self, checker):
        """
        equals strategy accepts only categorical and string.
        Passing float is a type_error (§9.1, §15 type_strategy_map).
        """
        node = GraphNode(op='equals', node_id='n18')
        with pytest.raises(MemintelError) as exc_info:
            checker.check_node(node, {'input': 'float'})
        assert exc_info.value.error_type == ErrorType.TYPE_ERROR

    def test_equals_strategy_on_int_input_raises_type_error(self, checker):
        node = GraphNode(op='equals', node_id='n19')
        with pytest.raises(MemintelError) as exc_info:
            checker.check_node(node, {'input': 'int'})
        assert exc_info.value.error_type == ErrorType.TYPE_ERROR

    def test_equals_strategy_on_categorical_valid(self, checker):
        node = GraphNode(op='equals', node_id='n20')
        result = checker.check_node(node, {'input': 'categorical'})
        assert result == 'decision<categorical>'

    def test_equals_strategy_on_string_valid(self, checker):
        node = GraphNode(op='equals', node_id='n21')
        result = checker.check_node(node, {'input': 'string'})
        assert result == 'decision<categorical>'

    # ── composite strategy: rejects decision<categorical> (§9.1) ─────────────

    def test_composite_strategy_on_decision_categorical_raises_type_error(self, checker):
        """
        composite accepts only decision<boolean>.
        Passing decision<categorical> is a type_error.
        """
        node = GraphNode(op='composite', node_id='n22')
        with pytest.raises(MemintelError) as exc_info:
            checker.check_node(node, {'input': 'decision<categorical>'})
        assert exc_info.value.error_type == ErrorType.TYPE_ERROR

    def test_composite_strategy_on_float_raises_type_error(self, checker):
        """composite rejects raw scalars entirely."""
        node = GraphNode(op='composite', node_id='n23')
        with pytest.raises(MemintelError) as exc_info:
            checker.check_node(node, {'input': 'float'})
        assert exc_info.value.error_type == ErrorType.TYPE_ERROR

    # ── Threshold / percentile / z_score / change: float and int valid ─────────

    def test_threshold_on_float_valid(self, checker):
        node = GraphNode(op='threshold', node_id='n24')
        result = checker.check_node(node, {'input': 'float'})
        assert result == 'decision<boolean>'

    def test_threshold_on_int_valid_via_widening(self, checker):
        """int is accepted by threshold via int→float widening."""
        node = GraphNode(op='threshold', node_id='n25')
        result = checker.check_node(node, {'input': 'int'})
        assert result == 'decision<boolean>'

    def test_threshold_on_categorical_raises_type_error(self, checker):
        """Threshold does not accept categorical."""
        node = GraphNode(op='threshold', node_id='n26')
        with pytest.raises(MemintelError) as exc_info:
            checker.check_node(node, {'input': 'categorical'})
        assert exc_info.value.error_type == ErrorType.TYPE_ERROR

    def test_threshold_on_string_raises_type_error(self, checker):
        node = GraphNode(op='threshold', node_id='n27')
        with pytest.raises(MemintelError) as exc_info:
            checker.check_node(node, {'input': 'string'})
        assert exc_info.value.error_type == ErrorType.TYPE_ERROR

    # ── Unknown operator ──────────────────────────────────────────────────────

    def test_unknown_operator_raises_type_error(self, checker):
        node = GraphNode(op='does_not_exist', node_id='n28')
        with pytest.raises(MemintelError) as exc_info:
            checker.check_node(node, {})
        assert exc_info.value.error_type == ErrorType.TYPE_ERROR

    # ── Missing required input ────────────────────────────────────────────────

    def test_missing_required_input_raises_type_error(self, checker):
        """check_node raises type_error when a declared input is absent."""
        node = GraphNode(op='z_score_op', node_id='n29')
        with pytest.raises(MemintelError) as exc_info:
            checker.check_node(node, {})  # 'input' not provided
        assert exc_info.value.error_type == ErrorType.TYPE_ERROR
