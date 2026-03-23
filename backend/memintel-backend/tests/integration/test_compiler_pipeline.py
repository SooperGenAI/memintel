"""
tests/integration/test_compiler_pipeline.py
──────────────────────────────────────────────────────────────────────────────
Integration tests for the compiler pipeline (DAGBuilder + IRGenerator).

Coverage:
  1. test_explain_plan_before_execution — compile all four fixture definitions,
     call compile_explain_plan, verify structural invariants and determinism.
  2. test_ir_hash_stability             — compile the same definition three times,
     assert all ir_hash values are identical and SHA-256-formatted.
  3. test_circular_dependency_rejected  — concept with A→B and B→A dependency
     cycle raises graph_error; no graph is returned.

No database, LLM, or HTTP calls. All operations are pure in-process compiler
calls on ConceptDefinition objects constructed directly in each test.
"""
from __future__ import annotations

import re

import pytest

from app.compiler.dag_builder import DAGBuilder
from app.compiler.ir_generator import IRGenerator
from app.models.concept import ConceptDefinition, FeatureNode, PrimitiveRef
from app.models.errors import ErrorType, MemintelError
from app.models.result import MissingDataPolicy
from app.models.task import Namespace


# ── Fixture definitions ─────────────────────────────────────────────────────
#
# These mirror the four JSON fixtures used in test_full_pipeline.py:
#   threshold  → org.churn_risk_score    (float,       normalize)
#   z_score    → org.payment_failure_rate (float,      mean from time_series)
#   equals     → org.risk_category       (categorical, passthrough)
#   composite  → org.risk_score          (float,       normalize)
#

_THRESHOLD_DEF = ConceptDefinition(
    concept_id="org.churn_risk_score",
    version="1.0",
    namespace=Namespace.ORG,
    output_type="float",
    primitives={
        "engagement_score": PrimitiveRef(
            type="float",
            missing_data_policy=MissingDataPolicy.ZERO,
        ),
    },
    features={
        "churn_score": FeatureNode(
            op="normalize",
            inputs={"input": "engagement_score"},
        ),
    },
    output_feature="churn_score",
)

_Z_SCORE_DEF = ConceptDefinition(
    concept_id="org.payment_failure_rate",
    version="1.0",
    namespace=Namespace.ORG,
    output_type="float",
    primitives={
        "failure_events": PrimitiveRef(
            type="time_series<float>",
            missing_data_policy=MissingDataPolicy.FORWARD_FILL,
        ),
    },
    features={
        "failure_rate": FeatureNode(
            op="mean",
            inputs={"input": "failure_events"},
        ),
    },
    output_feature="failure_rate",
)

# The equals/categorical fixture (org.risk_category) uses the passthrough
# operator on a categorical primitive.  resolve_primitive_type() returns bare
# "categorical" (it does not synthesise the labeled categorical{...} type from
# PrimitiveRef.labels), which causes the type checker to raise Rule 12.
# Categorical concepts are tested via PassthroughValidator + MockExecuteService
# in test_full_pipeline.py — they bypass the real compiler on purpose.
#
# To exercise a four-fixture compiler test with a real parallel group, we use a
# multi-feature float DAG instead: two independent primitives → two independent
# normalise features → one combine (add) feature.  This produces BFS level 0
# (two primitive nodes in parallel) and BFS level 1 (two feature nodes in
# parallel), making the parallelizable_groups assertion non-trivial.

_MULTI_FEATURE_DEF = ConceptDefinition(
    concept_id="org.combined_score",
    version="1.0",
    namespace=Namespace.ORG,
    output_type="float",
    primitives={
        "signal_a": PrimitiveRef(
            type="float",
            missing_data_policy=MissingDataPolicy.ZERO,
        ),
        "signal_b": PrimitiveRef(
            type="float",
            missing_data_policy=MissingDataPolicy.ZERO,
        ),
    },
    features={
        "norm_a": FeatureNode(
            op="normalize",
            inputs={"input": "signal_a"},
        ),
        "norm_b": FeatureNode(
            op="normalize",
            inputs={"input": "signal_b"},
        ),
        "combined": FeatureNode(
            op="add",
            inputs={"a": "norm_a", "b": "norm_b"},
        ),
    },
    output_feature="combined",
)

_COMPOSITE_DEF = ConceptDefinition(
    concept_id="org.risk_score",
    version="1.0",
    namespace=Namespace.ORG,
    output_type="float",
    primitives={
        "risk_signal": PrimitiveRef(
            type="float",
            missing_data_policy=MissingDataPolicy.ZERO,
        ),
    },
    features={
        "risk_score": FeatureNode(
            op="normalize",
            inputs={"input": "risk_signal"},
        ),
    },
    output_feature="risk_score",
)

_ALL_FIXTURES = [_THRESHOLD_DEF, _Z_SCORE_DEF, _COMPOSITE_DEF, _MULTI_FEATURE_DEF]


# ── Tests ───────────────────────────────────────────────────────────────────


def test_explain_plan_before_execution() -> None:
    """
    Compile each fixture definition, call compile_explain_plan, and verify:
      - execution_order has no duplicate node IDs
      - all parallelizable_groups contain only mutually independent nodes
        (no directed edge exists between any two nodes in the same group)
      - topological order is valid (every predecessor of a node appears before it)
      - plan is identical on repeated calls (deterministic)
    """
    builder = DAGBuilder()
    ir_gen  = IRGenerator()

    for defn in _ALL_FIXTURES:
        # ── Compile ───────────────────────────────────────────────────────────
        graph = builder.build_dag(defn)
        ir_gen.hash_graph(graph)  # populate ir_hash sentinel

        plan = ir_gen.compile_explain_plan(graph, defn)

        # ── No duplicate nodes in execution_order ─────────────────────────────
        assert len(plan.execution_order) == len(set(plan.execution_order)), (
            f"{defn.concept_id}: execution_order contains duplicates: "
            f"{plan.execution_order}"
        )

        # ── Topological validity ──────────────────────────────────────────────
        # Build predecessor map: for each node, the set of nodes that must
        # complete before it (direct upstream dependencies via graph edges).
        predecessors: dict[str, set[str]] = {nid: set() for nid in plan.execution_order}
        for edge in graph.edges:
            if edge.to_node_id in predecessors:
                predecessors[edge.to_node_id].add(edge.from_node_id)

        position = {nid: idx for idx, nid in enumerate(plan.execution_order)}
        for nid, preds in predecessors.items():
            node_pos = position[nid]
            for pred in preds:
                assert position[pred] < node_pos, (
                    f"{defn.concept_id}: node '{pred}' appears after '{nid}' "
                    f"in execution_order but is its predecessor"
                )

        # ── Parallelizable groups: no intra-group edges ───────────────────────
        # Build a directed edge set for fast lookup.
        edge_set: set[tuple[str, str]] = {
            (e.from_node_id, e.to_node_id) for e in graph.edges
        }
        for group in plan.parallelizable_groups:
            for i, node_a in enumerate(group):
                for node_b in group[i + 1:]:
                    assert (node_a, node_b) not in edge_set, (
                        f"{defn.concept_id}: nodes '{node_a}' and '{node_b}' are "
                        f"in the same parallelizable group but share a directed edge"
                    )
                    assert (node_b, node_a) not in edge_set, (
                        f"{defn.concept_id}: nodes '{node_b}' and '{node_a}' are "
                        f"in the same parallelizable group but share a directed edge"
                    )

        # ── Determinism: second compile produces identical plan ────────────────
        graph2 = builder.build_dag(defn)
        ir_gen.hash_graph(graph2)
        plan2  = ir_gen.compile_explain_plan(graph2, defn)

        assert plan.execution_order       == plan2.execution_order,       \
            f"{defn.concept_id}: execution_order changed across runs"
        assert plan.parallelizable_groups == plan2.parallelizable_groups, \
            f"{defn.concept_id}: parallelizable_groups changed across runs"
        assert plan.primitive_fetches     == plan2.primitive_fetches,     \
            f"{defn.concept_id}: primitive_fetches changed across runs"
        assert plan.critical_path_length  == plan2.critical_path_length,  \
            f"{defn.concept_id}: critical_path_length changed across runs"


def test_ir_hash_stability() -> None:
    """
    Compile the same definition three times and verify:
      - All three ir_hash values are identical.
      - ir_hash is a valid SHA-256 hex string (64 lowercase hex characters).

    This confirms the DAG Execution Order Guarantee (core-spec.md §1B):
    same definition version → same graph → same ir_hash on any machine.
    """
    builder = DAGBuilder()
    ir_gen  = IRGenerator()

    # Use the threshold fixture as the canonical stability target.
    defn = _THRESHOLD_DEF

    hashes: list[str] = []
    for _ in range(3):
        graph = builder.build_dag(defn)
        h = ir_gen.hash_graph(graph)
        hashes.append(h)

    first = hashes[0]
    assert hashes[1] == first, "ir_hash changed between compile run 1 and run 2"
    assert hashes[2] == first, "ir_hash changed between compile run 2 and run 3"

    # SHA-256 hexdigest is always exactly 64 lowercase hex characters.
    assert re.fullmatch(r"[0-9a-f]{64}", first), (
        f"ir_hash does not look like a SHA-256 hexdigest: {first!r}"
    )

    # Also verify stability across all fixture definitions.
    for defn in _ALL_FIXTURES:
        g1 = builder.build_dag(defn)
        g2 = builder.build_dag(defn)
        h1 = ir_gen.hash_graph(g1)
        h2 = ir_gen.hash_graph(g2)
        assert h1 == h2, (
            f"{defn.concept_id}: ir_hash is not stable across repeated compiles"
        )
        assert re.fullmatch(r"[0-9a-f]{64}", h1), (
            f"{defn.concept_id}: ir_hash is not a valid SHA-256 hex string: {h1!r}"
        )


def test_circular_dependency_rejected() -> None:
    """
    A concept where feat_a depends on feat_b and feat_b depends on feat_a
    must raise MemintelError(graph_error).

    Asserts:
      - MemintelError is raised with error_type == GRAPH_ERROR.
      - build_dag() returns nothing (no graph is stored — clean failure).
    """
    # Construct a concept with a two-node feature cycle.
    # feat_a ← feat_b and feat_b ← feat_a create an unsatisfiable dependency.
    #
    # A raw float primitive is declared so the concept body is otherwise valid.
    # The cycle is in the feature-to-feature dependency graph; topo_sort_features()
    # inside build_dag() detects it via Kahn's algorithm before type checking.

    circular_def = ConceptDefinition(
        concept_id="org.circular_test",
        version="1.0",
        namespace=Namespace.ORG,
        output_type="float",
        primitives={
            "raw_signal": PrimitiveRef(
                type="float",
                missing_data_policy=MissingDataPolicy.ZERO,
            ),
        },
        features={
            "feat_a": FeatureNode(
                op="normalize",
                inputs={"input": "feat_b"},   # feat_a → feat_b
            ),
            "feat_b": FeatureNode(
                op="normalize",
                inputs={"input": "feat_a"},   # feat_b → feat_a  (cycle!)
            ),
        },
        output_feature="feat_a",
    )

    builder = DAGBuilder()

    graph_returned: object = None
    with pytest.raises(MemintelError) as exc_info:
        graph_returned = builder.build_dag(circular_def)

    # Must be a graph_error — not a type_error or syntax_error.
    assert exc_info.value.error_type == ErrorType.GRAPH_ERROR, (
        f"Expected GRAPH_ERROR, got {exc_info.value.error_type}: {exc_info.value}"
    )

    # No graph was returned — the call aborted before producing output.
    assert graph_returned is None, (
        "build_dag() must not return a graph when a cycle is detected"
    )
