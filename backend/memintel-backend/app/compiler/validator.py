"""
app/compiler/validator.py
────────────────────────────────────────────────────────────────────────────────
Validation pipeline for ConceptDefinition.

Runs six phases in strict order:

  Phase 1 — validate_schema()     → syntax_error   (malformed / unresolvable refs)
  Phase 2 — validate_operators()  → reference_error (op not in OPERATOR_REGISTRY)
  Phase 3 — validate_types()      → type_error      (TypeChecker violations)
  Phase 4 — validate_strategies() → semantic_error  (output type / strategy compat)
  Phase 5 — validate_actions()    → semantic_error  (output bindability)
  Phase 6 — validate_graph()      → graph_error     (cycles, disconnected nodes)

Each phase may also surface errors from earlier phases (e.g. validate_types
calls _topo_sort_features which raises graph_error on a cycle). The validate()
driver collects the first error from each phase.

The public API is the Validator class.  Helper functions at module level are
reusable by the dag_builder and tests.
"""
from __future__ import annotations

import bisect

from app.compiler.type_checker import OPERATOR_REGISTRY
from app.compiler.type_checker import GraphNode as TCGraphNode
from app.compiler.type_checker import TypeChecker
from app.models.concept import ConceptDefinition, MemintelType
from app.models.condition import TYPE_STRATEGY_COMPATIBILITY
from app.models.errors import ErrorType, MemintelError, ValidationErrorItem
from app.models.result import MissingDataPolicy


# ── Module-level helpers (also imported by dag_builder) ────────────────────────

def _syntax_error(msg: str, location: str | None = None) -> MemintelError:
    return MemintelError(ErrorType.SYNTAX_ERROR, msg, location=location)


def _ref_error(msg: str, location: str | None = None) -> MemintelError:
    return MemintelError(ErrorType.REFERENCE_ERROR, msg, location=location)


def _type_error(msg: str, location: str | None = None) -> MemintelError:
    return MemintelError(ErrorType.TYPE_ERROR, msg, location=location)


def _semantic_error(msg: str, location: str | None = None) -> MemintelError:
    return MemintelError(ErrorType.SEMANTIC_ERROR, msg, location=location)


def _graph_error(msg: str, location: str | None = None) -> MemintelError:
    return MemintelError(ErrorType.GRAPH_ERROR, msg, location=location)


def resolve_primitive_type(prim_ref) -> str:
    """
    Apply missing_data_policy to determine the resolved output type of a primitive.

    Rules (§6.2 of memintel_type_system.md):
      None / 'null'   → T?   (null propagates through the graph)
      'zero'          → T    (null replaced with 0; non-nullable)
      'forward_fill'  → T    (null replaced with last known value; non-nullable)
      'backward_fill' → T    (null replaced with next known value; non-nullable)

    For categorical primitives with a declared label set (prim_ref.labels),
    the label set is encoded into the type string as categorical{a,b,c} so the
    type checker can enforce Rule 12 (no bare 'categorical' as a node output).
    """
    base = prim_ref.type
    if base == MemintelType.CATEGORICAL and getattr(prim_ref, 'labels', None):
        base = 'categorical{' + ','.join(sorted(prim_ref.labels)) + '}'
    if prim_ref.missing_data_policy in (None, MissingDataPolicy.NULL):
        return MemintelType.nullable(base) if not MemintelType.is_nullable(base) else base
    # zero / forward_fill / backward_fill → non-nullable
    return MemintelType.base_of(base)


def topo_sort_features(definition: ConceptDefinition) -> list[str]:
    """
    Return feature names in a deterministic topological execution order.

    Features with no inter-feature dependencies come first; later features
    depend only on earlier ones.  Alphabetical tie-breaking ensures the same
    definition always produces the same order.

    Raises MemintelError(graph_error) if a cycle is detected.
    """
    feature_names = set(definition.features.keys())

    # dependents[x] = set of features that depend on x
    # in_degree[x]  = count of features x depends on
    dependents: dict[str, set[str]] = {n: set() for n in feature_names}
    in_degree:  dict[str, int]      = {n: 0      for n in feature_names}

    for feat_name, feat_node in definition.features.items():
        for source_ref in feat_node.inputs.values():
            if isinstance(source_ref, str) and source_ref in feature_names:
                dependents[source_ref].add(feat_name)
                in_degree[feat_name] += 1

    ready: list[str] = sorted(n for n, d in in_degree.items() if d == 0)
    order: list[str] = []

    while ready:
        node = ready.pop(0)
        order.append(node)
        for dep in sorted(dependents[node]):
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                bisect.insort(ready, dep)

    if len(order) != len(feature_names):
        raise _graph_error(
            "Circular dependency detected in feature graph — "
            "the following features form a cycle: "
            f"{sorted(feature_names - set(order))}",
        )

    return order


# ── Validator ──────────────────────────────────────────────────────────────────

class Validator:
    """
    Compile-time validator for ConceptDefinition.

    Public API
    ----------
    validate(definition) → list[ValidationErrorItem]
        Runs all six phases, returns every error found.
        An empty list means the definition is valid and safe to compile.

    Individual phase methods may also be called directly (e.g. from tests or
    from a partial-validation route that only checks types).  Each raises
    MemintelError on the first violation it finds.
    """

    def validate(self, definition: ConceptDefinition) -> list[ValidationErrorItem]:
        """
        Run all six validation phases in strict order.

        If validate_schema fails, the remaining phases are skipped because later
        phases may crash on malformed input (unresolved references, etc.).
        All other phases run independently; their errors are accumulated.

        Returns an empty list when the definition is fully valid.
        """
        errors: list[ValidationErrorItem] = []

        phases = [
            self.validate_schema,
            self.validate_operators,
            self.validate_types,
            self.validate_strategies,
            self.validate_actions,
            self.validate_graph,
        ]

        for phase_fn in phases:
            try:
                phase_fn(definition)
            except MemintelError as exc:
                errors.append(ValidationErrorItem(
                    type=exc.error_type,
                    message=exc.message,
                    location=exc.location,
                ))
                # Schema errors make subsequent phases unreliable — stop early.
                if exc.error_type == ErrorType.SYNTAX_ERROR:
                    break

        return errors

    # ── Phase 1 ────────────────────────────────────────────────────────────────

    def validate_schema(self, definition: ConceptDefinition) -> None:
        """
        Phase 1 — Structural schema validation.

        Checks that every feature input reference resolves to a declared
        primitive name or feature name within the same definition.

        Raises MemintelError(syntax_error) on first unresolved reference.
        """
        all_names = set(definition.primitives.keys()) | set(definition.features.keys())

        for feat_name, feat_node in sorted(definition.features.items()):
            for slot_name, source_ref in sorted(feat_node.inputs.items()):
                if not isinstance(source_ref, str):
                    # Numeric or boolean literals are valid param-like inputs; skip.
                    continue
                if source_ref not in all_names:
                    raise _syntax_error(
                        f"Feature '{feat_name}' input slot '{slot_name}' references "
                        f"'{source_ref}' which is not a declared primitive or feature name.",
                        location=f"features.{feat_name}.inputs.{slot_name}",
                    )

    # ── Phase 2 ────────────────────────────────────────────────────────────────

    def validate_operators(self, definition: ConceptDefinition) -> None:
        """
        Phase 2 — Operator registry membership.

        Every feature op must exist in OPERATOR_REGISTRY.

        Raises MemintelError(reference_error) on first unknown operator.
        """
        for feat_name, feat_node in sorted(definition.features.items()):
            if feat_node.op not in OPERATOR_REGISTRY:
                raise _ref_error(
                    f"Feature '{feat_name}' uses operator '{feat_node.op}' which is not "
                    f"registered in OPERATOR_REGISTRY.",
                    location=f"features.{feat_name}.op",
                )

    # ── Phase 3 ────────────────────────────────────────────────────────────────

    def validate_types(self, definition: ConceptDefinition) -> None:
        """
        Phase 3 — Type compatibility across the DAG.

        Traces Memintel types from primitives through features in topological
        order using TypeChecker.check_node().  Verifies that the terminal
        feature's inferred type is assignable to the declared output_type.

        Null propagation and subtype widening rules are enforced by TypeChecker.

        Raises MemintelError(type_error) on any type mismatch.
        Silently returns if a cycle is detected (validate_graph handles that).
        """
        type_env: dict[str, str] = {}

        # Resolve primitive types (apply missing_data_policy)
        for prim_name, prim_ref in definition.primitives.items():
            type_env[prim_name] = resolve_primitive_type(prim_ref)

        # Topological sort — if this raises graph_error, skip (phase 6 will catch it)
        try:
            ordered = topo_sort_features(definition)
        except MemintelError:
            return

        checker = TypeChecker()

        for feat_name in ordered:
            feat_node = definition.features[feat_name]
            input_types: dict[str, str] = {}

            for slot_name, source_ref in feat_node.inputs.items():
                if not isinstance(source_ref, str):
                    continue
                if source_ref in type_env:
                    input_types[slot_name] = type_env[source_ref]
                # Unresolved refs are caught by phase 1.

            tc_node = TCGraphNode(op=feat_node.op, node_id=feat_name)
            # check_node raises type_error / reference_error on violations.
            inferred_output = checker.check_node(tc_node, input_types)
            type_env[feat_name] = inferred_output

        # Verify the terminal node's type matches the declared output_type.
        inferred_final = type_env.get(definition.output_feature)
        if inferred_final is None:
            return

        if inferred_final != definition.output_type:
            if not MemintelType.is_assignable(inferred_final, definition.output_type):
                raise _type_error(
                    f"Output feature '{definition.output_feature}' inferred type "
                    f"'{inferred_final}' is not assignable to declared "
                    f"output_type='{definition.output_type}'.",
                    location="output_feature",
                )

    # ── Phase 4 ────────────────────────────────────────────────────────────────

    def validate_strategies(self, definition: ConceptDefinition) -> None:
        """
        Phase 4 — Output type / condition strategy compatibility.

        The concept output_type must be usable by at least one condition
        strategy so the φ layer can evaluate it.  Also validates categorical
        label declarations and rejects decision/duration output types.

        Raises MemintelError(semantic_error) on violation.
        """
        output_type = definition.output_type
        base_type = MemintelType.base_of(output_type)

        # decision<T> and duration must never be concept outputs.
        if output_type == MemintelType.DURATION:
            raise _semantic_error(
                "output_type 'duration' is a compile-time parameter type — "
                "it cannot be the output type of a concept.",
                location="output_type",
            )
        if base_type in MemintelType.DECISION_TYPES:
            raise _semantic_error(
                f"output_type '{output_type}' is a decision type produced by conditions, "
                "not by concepts.",
                location="output_type",
            )

        # Categorical output requires labels.
        if base_type == MemintelType.CATEGORICAL and not definition.labels:
            raise _semantic_error(
                "Concept with output_type='categorical' must declare a non-empty labels list.",
                location="labels",
            )

        # Output type must have at least one applicable condition strategy.
        if base_type not in TYPE_STRATEGY_COMPATIBILITY:
            raise _semantic_error(
                f"Concept output_type '{output_type}' (base: '{base_type}') has no "
                "applicable condition strategies.  Concepts must produce a type "
                "that the φ layer can evaluate.",
                location="output_type",
            )
        if not TYPE_STRATEGY_COMPATIBILITY[base_type]:
            raise _semantic_error(
                f"Concept output_type '{output_type}' (base: '{base_type}') has no "
                "applicable condition strategies.",
                location="output_type",
            )

    # ── Phase 5 ────────────────────────────────────────────────────────────────

    def validate_actions(self, definition: ConceptDefinition) -> None:
        """
        Phase 5 — Output bindability for action triggering.

        The concept must have at least one feature.  Container outputs
        (time_series<*>, list<*>) cannot flow into the φ layer directly —
        an aggregation step is required first.

        Raises MemintelError(semantic_error) on violation.
        """
        if not definition.features:
            raise _semantic_error(
                "Concept must define at least one feature node.",
                location="features",
            )

        output_type = definition.output_type
        base_type = MemintelType.base_of(output_type)

        if base_type in MemintelType.CONTAINER_TYPES:
            raise _semantic_error(
                f"Concept output_type '{output_type}' is a container type.  "
                "Container outputs cannot be evaluated by condition strategies or "
                "trigger actions.  Apply an aggregation operator (mean, sum, "
                "pct_change, …) to reduce to a scalar first.",
                location="output_type",
            )

    # ── Phase 6 ────────────────────────────────────────────────────────────────

    def validate_graph(self, definition: ConceptDefinition) -> None:
        """
        Phase 6 — DAG topology validation.

        Two checks:
        1. Cycle detection via iterative DFS with tri-colour marking.
           A cycle raises graph_error.
        2. Disconnected-node detection: any feature not on the path from any
           primitive to output_feature is unreachable and raises graph_error.

        Raises MemintelError(graph_error) on first violation.
        """
        feature_names = set(definition.features.keys())

        # Build: feat → sorted list of feature deps (primitives are leaves; skip)
        deps: dict[str, list[str]] = {}
        for feat_name, feat_node in definition.features.items():
            feat_deps = sorted(
                ref for ref in feat_node.inputs.values()
                if isinstance(ref, str) and ref in feature_names
            )
            deps[feat_name] = feat_deps

        # ── Cycle detection (iterative DFS, tri-colour) ────────────────────────
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {n: WHITE for n in feature_names}

        for start in sorted(feature_names):
            if color[start] != WHITE:
                continue
            color[start] = GRAY
            # Stack entries: (node, iterator over its deps)
            stack: list[tuple[str, object]] = [(start, iter(deps[start]))]

            while stack:
                node, it = stack[-1]
                try:
                    dep = next(it)  # type: ignore[call-overload]
                    if color[dep] == GRAY:
                        raise _graph_error(
                            f"Circular dependency detected: '{dep}' is an ancestor "
                            f"of '{node}' and also depends on it.",
                            location=f"features.{node}",
                        )
                    if color[dep] == WHITE:
                        color[dep] = GRAY
                        stack.append((dep, iter(deps[dep])))
                except StopIteration:
                    color[node] = BLACK
                    stack.pop()

        # ── Disconnected-node detection ────────────────────────────────────────
        # Walk backwards from output_feature; any unreachable feature is dead.
        reachable: set[str] = set()
        stack_bfs: list[str] = [definition.output_feature]
        while stack_bfs:
            node = stack_bfs.pop()
            if node in reachable:
                continue
            reachable.add(node)
            stack_bfs.extend(deps.get(node, []))

        disconnected = feature_names - reachable
        if disconnected:
            raise _graph_error(
                f"Feature(s) {sorted(disconnected)} are not on any execution path "
                "to the output node.  Remove them or connect them to the output.",
                location="features",
            )
