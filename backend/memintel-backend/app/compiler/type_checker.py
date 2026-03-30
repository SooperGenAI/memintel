"""
app/compiler/type_checker.py
────────────────────────────────────────────────────────────────────────────────
Compile-time type checker for the Memintel execution graph.

Enforces all type rules from memintel_type_system.md v1.1:

  Subtype rules (§4):
    int → float                          widening, implicit
    T   → T?                             nullable widening, implicit
    time_series<int> → time_series<float>  via int→float
    list<int>        → list<float>         via int→float
    float → int                          REQUIRES explicit to_int() — type_error otherwise
    T?  → T                              REQUIRES null-handling operator — type_error otherwise

  Null propagation (§6.3):
    If any input to an operator is T?, the output is T? unless the operator
    explicitly handles null (coalesce, drop_null, fill_null).

  Decision types (§9.3, §10 rule 8):
    decision<T> does not flow into concept operator inputs.
    Valid destinations: action bindings, unwrap_decision, composite strategy.

  Condition strategy compatibility (§9.1):
    threshold / percentile / z_score / change → accept float, int (via widening)
    equals                                    → accept categorical, string ONLY
    composite                                 → accept decision<boolean> ONLY

  Categorical label sets (§5, Rules 12 & 13):
    Labeled categorical type string: categorical{label1,label2,...}  (sorted)
    Bare 'categorical' (no label set) as a DAG node output → type_error  (Rule 12)
    categorical{A,B} → categorical (operator wildcard)    → True
    categorical{A,B} → categorical{A,B}                   → True   (identity)
    categorical{A,B} → categorical{C,D}                   → False  (Rule 13: different sets)
    Operators that pass categorical through (passthrough, equals, unwrap_decision)
    preserve the label set in their resolved output type.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.models.errors import ErrorType, MemintelError


# ── Internal helpers ───────────────────────────────────────────────────────────

def _is_nullable(t: str) -> bool:
    return t.endswith('?')


def _strip_nullable(t: str) -> str:
    return t[:-1] if t.endswith('?') else t


def _make_nullable(t: str) -> str:
    return t if t.endswith('?') else t + '?'


def _is_decision(t: str) -> bool:
    return t.startswith('decision<')


def _type_error(message: str, location: str | None = None) -> MemintelError:
    return MemintelError(ErrorType.TYPE_ERROR, message, location=location)


# ── Categorical type helpers (§5, Rules 12 & 13) ───────────────────────────────

def _is_categorical(t: str) -> bool:
    """Returns True for bare 'categorical' or labeled 'categorical{...}'."""
    base = _strip_nullable(t)
    return base == 'categorical' or base.startswith('categorical{')


def _has_label_set(t: str) -> bool:
    """Returns True when the type has an explicit label set (categorical{...})."""
    base = _strip_nullable(t)
    return base.startswith('categorical{') and base.endswith('}')


def _categorical_labels(t: str) -> frozenset[str] | None:
    """
    Extracts the label set from a labeled categorical type string.

    Returns None for bare 'categorical' (no label set declared).
    Returns a frozenset of label strings for 'categorical{a,b,c}'.
    """
    base = _strip_nullable(t)
    if base.startswith('categorical{') and base.endswith('}'):
        inner = base[len('categorical{'):-1]
        return frozenset(inner.split(',')) if inner else frozenset()
    return None


def _make_categorical(labels: frozenset[str]) -> str:
    """Constructs a labeled categorical type string with alphabetically sorted labels."""
    return 'categorical{' + ','.join(sorted(labels)) + '}'


# ── Registry models ────────────────────────────────────────────────────────────

@dataclass
class InputSpec:
    """
    Declares a single named input for an operator.

    ``accepted_types`` lists every declared base type that the input accepts.
    Subtype widening (int→float, T→T?) is applied on top of this list at
    check time — it is NOT necessary to include 'int' when 'float' is listed.
    """
    name: str
    accepted_types: list[str]


@dataclass
class OutputSpec:
    type: str  # declared output type (before nullable propagation)


@dataclass
class OperatorSpec:
    inputs: list[InputSpec]
    output: OutputSpec
    # True for coalesce / drop_null / fill_null: they consume T? and produce T.
    null_handler: bool = False


# ── Graph node ─────────────────────────────────────────────────────────────────

@dataclass
class GraphNode:
    """Minimal graph node representation used by the type checker."""
    op: str
    node_id: str = ''


# ── Operator registry ──────────────────────────────────────────────────────────

OPERATOR_REGISTRY: dict[str, OperatorSpec] = {
    # ── Arithmetic ────────────────────────────────────────────────────────────
    'add':      OperatorSpec([InputSpec('a', ['float']), InputSpec('b', ['float'])],       OutputSpec('float')),
    'subtract': OperatorSpec([InputSpec('a', ['float']), InputSpec('b', ['float'])],       OutputSpec('float')),
    'multiply': OperatorSpec([InputSpec('a', ['float']), InputSpec('b', ['float'])],       OutputSpec('float')),
    'divide':   OperatorSpec([InputSpec('a', ['float']), InputSpec('b', ['float'])],       OutputSpec('float')),

    # ── Aggregation ───────────────────────────────────────────────────────────
    'mean':  OperatorSpec([InputSpec('input', ['time_series<float>'])], OutputSpec('float')),
    'sum':   OperatorSpec([InputSpec('input', ['time_series<float>'])], OutputSpec('float')),
    'min':   OperatorSpec([InputSpec('input', ['time_series<float>'])], OutputSpec('float')),
    'max':   OperatorSpec([InputSpec('input', ['time_series<float>'])], OutputSpec('float')),
    # count is int-specific; rejects time_series<float> (§7.4)
    'count': OperatorSpec([InputSpec('input', ['time_series<int>'])],   OutputSpec('int')),

    # ── Time-series ───────────────────────────────────────────────────────────
    'pct_change':     OperatorSpec([InputSpec('input', ['time_series<float>'])], OutputSpec('float')),
    'rate_of_change': OperatorSpec([InputSpec('input', ['time_series<float>'])], OutputSpec('float')),
    'moving_average': OperatorSpec([InputSpec('input', ['time_series<float>'])], OutputSpec('float')),

    # ── Statistical ───────────────────────────────────────────────────────────
    'z_score_op':    OperatorSpec([InputSpec('input', ['float'])], OutputSpec('float')),
    'percentile_op': OperatorSpec([InputSpec('input', ['float'])], OutputSpec('float')),

    # ── Transformation ────────────────────────────────────────────────────────
    'normalize':    OperatorSpec([InputSpec('input',  ['float'])],       OutputSpec('float')),
    'weighted_sum': OperatorSpec([InputSpec('values', ['list<float>'])], OutputSpec('float')),

    # ── Explicit cast: float → int (§4.3) ─────────────────────────────────────
    # Accepts float (and int via widening). Runtime raises type_error on NaN/Inf.
    'to_int': OperatorSpec([InputSpec('input', ['float'])], OutputSpec('int')),

    # ── Decision unwrapping (§9.3) ────────────────────────────────────────────
    # Polymorphic: decision<boolean> → boolean, decision<categorical> → categorical.
    # Handled as a special case in check_node.
    'unwrap_decision': OperatorSpec(
        [InputSpec('input', ['decision<boolean>', 'decision<categorical>'])],
        OutputSpec('boolean'),  # overridden in check_node for categorical variant
    ),

    # ── Null-handling operators (§6.4) ────────────────────────────────────────
    # These consume T? and produce T (null_handler=True suppresses propagation).
    'coalesce': OperatorSpec(
        [InputSpec('input', ['float?']), InputSpec('default', ['float'])],
        OutputSpec('float'),
        null_handler=True,
    ),
    'drop_null': OperatorSpec(
        [InputSpec('input', ['time_series<float?>'])],
        OutputSpec('time_series<float>'),
        null_handler=True,
    ),
    'fill_null': OperatorSpec(
        [InputSpec('input',  ['time_series<float?>']), InputSpec('value', ['float'])],
        OutputSpec('time_series<float>'),
        null_handler=True,
    ),

    # ── Passthrough — identity operator for categorical / string concepts ─────
    # Exposes a categorical or string primitive as a concept output without
    # transformation.  Used when the concept's value IS the primitive label
    # (e.g. an ML classifier whose output is already a categorical label).
    'passthrough': OperatorSpec([InputSpec('input', ['categorical', 'string'])], OutputSpec('categorical')),

    # ── Condition strategies (§9) ─────────────────────────────────────────────
    'threshold':  OperatorSpec([InputSpec('input', ['float'])],                           OutputSpec('decision<boolean>')),
    'percentile': OperatorSpec([InputSpec('input', ['float'])],                           OutputSpec('decision<boolean>')),
    'z_score':    OperatorSpec([InputSpec('input', ['float'])],                           OutputSpec('decision<boolean>')),
    'change':     OperatorSpec([InputSpec('input', ['float'])],                           OutputSpec('decision<boolean>')),
    'equals':     OperatorSpec([InputSpec('input', ['categorical', 'string'])],           OutputSpec('decision<categorical>')),
    'composite':  OperatorSpec([InputSpec('input', ['decision<boolean>'])],               OutputSpec('decision<boolean>')),
}

# Operators that are legitimate destinations for decision<T> values.
_DECISION_ACCEPTING_OPS: frozenset[str] = frozenset({'composite', 'unwrap_decision'})


# ── TypeChecker ────────────────────────────────────────────────────────────────

class TypeChecker:
    """
    Validates operator input/output type compatibility for a single graph node.

    Public API
    ----------
    is_assignable(actual, expected) -> bool
        Returns True when ``actual`` satisfies ``expected`` under the subtype
        rules defined in memintel_type_system.md v1.1.

    check_node(node, input_types) -> str
        Validates ``input_types`` against the declared operator signature,
        applies null propagation, and returns the resolved output type string.
        Raises ``MemintelError(type_error, ...)`` on any violation.
    """

    # ── Subtype rules ──────────────────────────────────────────────────────────

    def is_assignable(self, actual: str, expected: str) -> bool:
        """
        Returns True if ``actual`` is assignable to ``expected``.

        Rules (§4, §12 rule 3):
          actual == expected            → True  (identity)
          int → float                   → True  (widening, implicit)
          T → T?                        → True  (nullable widening)
          time_series<int> → time_series<float>  → True
          list<int>        → list<float>          → True
          float → int                   → False  (requires explicit to_int())
          T? → T                        → False  (requires null-handling operator)
        """
        if actual == expected:
            return True

        # int → float widening (§4.1)
        if actual == 'int' and expected == 'float':
            return True

        # T → T? nullable widening (§4.2)
        if expected.endswith('?') and actual == _strip_nullable(expected):
            return True

        # Container covariance via int→float (§4.1)
        if actual == 'time_series<int>' and expected == 'time_series<float>':
            return True
        if actual == 'list<int>' and expected == 'list<float>':
            return True

        # Categorical wildcard: categorical{...} → categorical (§5, Rule 12)
        # Operator specs use bare 'categorical' to accept any labeled categorical.
        if _has_label_set(actual) and expected == 'categorical':
            return True
        # Nullable categorical wildcard: categorical{...}? → categorical?
        if _has_label_set(actual) and actual.endswith('?') and expected == 'categorical?':
            return True

        # decision<categorical{...}> → decision<categorical>
        # Allows unwrap_decision to accept labeled decision types.
        if (actual.startswith('decision<categorical{') and actual.endswith('>')
                and expected == 'decision<categorical>'):
            return True

        return False

    # ── Node checking ──────────────────────────────────────────────────────────

    def check_node(self, node: GraphNode, input_types: dict[str, str]) -> str:
        """
        Validates ``input_types`` against the operator declared in ``node.op``
        and returns the resolved output type string.

        Raises
        ------
        MemintelError(type_error)  on any type violation.
        MemintelError(type_error)  if the operator is unknown.

        Null propagation (§6.3)
        ~~~~~~~~~~~~~~~~~~~~~~~
        If an input carries a nullable type (T?) where the operator declares T,
        the assignment is accepted and the output type becomes T? — unless the
        operator is a null-handling operator (coalesce / drop_null / fill_null),
        in which case the output remains non-nullable.

        Decision blocking (§10 rule 8)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        decision<T> may only flow into ``composite`` or ``unwrap_decision``.
        Any other operator receiving a decision<T> input raises type_error.
        """
        op_name = node.op
        if op_name not in OPERATOR_REGISTRY:
            raise _type_error(
                f"Unknown operator '{op_name}'",
                location=node.node_id or op_name,
            )

        spec = OPERATOR_REGISTRY[op_name]
        nullable_propagated = False

        for inp in spec.inputs:
            actual = input_types.get(inp.name)
            if actual is None:
                raise _type_error(
                    f"Operator '{op_name}' requires input '{inp.name}' but it was not provided",
                    location=f"{node.node_id or op_name}.{inp.name}",
                )

            # ── decision<T> blocking ──────────────────────────────────────────
            if _is_decision(actual) and op_name not in _DECISION_ACCEPTING_OPS:
                raise _type_error(
                    f"Operator '{op_name}': input '{inp.name}' received '{actual}'. "
                    f"decision<T> values may only flow into 'composite' or "
                    f"'unwrap_decision', not into concept operators.",
                    location=f"{node.node_id or op_name}.{inp.name}",
                )

            # ── Direct assignability check ────────────────────────────────────
            if any(self.is_assignable(actual, accepted) for accepted in inp.accepted_types):
                continue  # valid — no nullability issue

            # ── Nullable propagation check ────────────────────────────────────
            # If actual is T? and T is assignable to a declared accepted type,
            # accept the input and mark the output as nullable (§6.3).
            if _is_nullable(actual):
                stripped = _strip_nullable(actual)
                if any(self.is_assignable(stripped, accepted) for accepted in inp.accepted_types):
                    if not spec.null_handler:
                        nullable_propagated = True
                    continue

            # ── No match — raise type_error ───────────────────────────────────
            declared = ', '.join(f"'{t}'" for t in inp.accepted_types)
            raise _type_error(
                f"Operator '{op_name}': input '{inp.name}' expects {declared}, "
                f"got '{actual}'",
                location=f"{node.node_id or op_name}.{inp.name}",
            )

        # ── Resolve output type ───────────────────────────────────────────────
        output_type = self._resolve_output(op_name, spec, input_types)

        if nullable_propagated:
            output_type = _make_nullable(output_type)

        # ── Rule 12: categorical output must carry a declared label set ────────
        # Bare 'categorical' (no '{...}') is not a valid DAG node output type.
        # Operators that pass categorical through (passthrough, equals,
        # unwrap_decision) inherit the label set from their input; if the input
        # itself was bare categorical, this check fires there too.
        if _strip_nullable(output_type) == 'categorical':
            raise _type_error(
                f"Operator '{op_name}': output type 'categorical' must declare "
                f"a closed label set, e.g. categorical{{low,medium,high}}. "
                f"Bare 'categorical' is not a valid DAG node output type (Rule 12).",
                location=node.node_id or op_name,
            )

        return output_type

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _resolve_output(
        self,
        op_name: str,
        spec: OperatorSpec,
        input_types: dict[str, str],
    ) -> str:
        """
        Returns the declared output type, applying polymorphic overrides to
        preserve categorical label sets through the DAG (Rules 12 & 13).

        passthrough  — returns the input type verbatim (preserves label set).
        equals       — wraps the input categorical type in decision<> so the
                       label set survives into the decision value.
        unwrap_decision — strips the decision<> wrapper; returns the inner type
                          (which may be a labeled categorical{...}).
        """
        if op_name == 'passthrough':
            # Preserve the input type exactly, including any label set.
            return input_types.get('input', 'categorical')

        if op_name == 'equals':
            # Carry the input type (categorical label set or string) into the
            # decision type exactly — string input yields decision<string>.
            input_type = input_types.get('input', 'categorical')
            return f'decision<{_strip_nullable(input_type)}>'

        if op_name == 'unwrap_decision':
            actual_input = input_types.get('input', '')
            # Strip 'decision<' / '>' to recover the inner type (boolean or
            # labeled categorical{...}).
            if actual_input.startswith('decision<') and actual_input.endswith('>'):
                return actual_input[len('decision<'):-1]
            return 'boolean'

        return spec.output.type
