"""
tests/unit/test_compiler.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for the compiler pipeline:

  app/compiler/validator.py    — six-phase validation
  app/compiler/dag_builder.py  — DAGBuilder (build_dag + optimisation passes)
  app/compiler/ir_generator.py — IRGenerator (hash_graph + compile_explain_plan)

Test coverage:
  - Same definition always produces the same ir_hash (determinism invariant)
  - Circular dependency raises graph_error
  - Dead node elimination removes nodes not on the output path
  - Topological order is correct for chained dependencies
  - Parallelizable groups are correct for independent branches
  - Validator phases raise the correct error types
  - compile_explain_plan returns accurate plan metadata
"""
import pytest

from app.compiler.dag_builder import DAGBuilder, _feat_node_id, _prim_node_id
from app.compiler.ir_generator import IRGenerator
from app.compiler.validator import Validator
from app.models.concept import ConceptDefinition, FeatureNode, PrimitiveRef
from app.models.errors import ErrorType, MemintelError
from app.models.result import MissingDataPolicy
from app.models.task import Namespace


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def builder() -> DAGBuilder:
    return DAGBuilder()


@pytest.fixture
def generator() -> IRGenerator:
    return IRGenerator()


@pytest.fixture
def validator() -> Validator:
    return Validator()


def _linear_chain_definition() -> ConceptDefinition:
    """
    A simple three-node chain:
      prim:user.score (float, zero-fill)
        → feat:step1  (normalize)
        → feat:step2  (z_score_op)   ← output
    """
    return ConceptDefinition(
        concept_id="test.chain",
        version="1.0",
        namespace=Namespace.PERSONAL,
        output_type="float",
        primitives={
            "user.score": PrimitiveRef(
                type="float",
                missing_data_policy=MissingDataPolicy.ZERO,
            ),
        },
        features={
            "step1": FeatureNode(op="normalize", inputs={"input": "user.score"}),
            "step2": FeatureNode(op="z_score_op", inputs={"input": "step1"}),
        },
        output_feature="step2",
    )


def _parallel_branches_definition() -> ConceptDefinition:
    """
    Two independent branches that merge:
      prim:user.a → feat:branch_a (normalize)  ──┐
                                                  └→ feat:output (add)
      prim:user.b → feat:branch_b (normalize)  ──┘
    branch_a and branch_b are at the same topological level (parallelizable).
    """
    return ConceptDefinition(
        concept_id="test.parallel",
        version="1.0",
        namespace=Namespace.PERSONAL,
        output_type="float",
        primitives={
            "user.a": PrimitiveRef(type="float", missing_data_policy=MissingDataPolicy.ZERO),
            "user.b": PrimitiveRef(type="float", missing_data_policy=MissingDataPolicy.ZERO),
        },
        features={
            "branch_a": FeatureNode(op="normalize", inputs={"input": "user.a"}),
            "branch_b": FeatureNode(op="normalize", inputs={"input": "user.b"}),
            "output":   FeatureNode(op="add",       inputs={"a": "branch_a", "b": "branch_b"}),
        },
        output_feature="output",
    )


def _dead_node_definition() -> ConceptDefinition:
    """
    Definition with one connected feature and one dead feature:
      prim:user.score → feat:useful   ← output
      prim:user.age   → feat:dead     (not connected to output path)
    """
    return ConceptDefinition(
        concept_id="test.dead",
        version="1.0",
        namespace=Namespace.PERSONAL,
        output_type="float",
        primitives={
            "user.score": PrimitiveRef(type="float", missing_data_policy=MissingDataPolicy.ZERO),
            "user.age":   PrimitiveRef(type="int",   missing_data_policy=MissingDataPolicy.ZERO),
        },
        features={
            "useful": FeatureNode(op="normalize", inputs={"input": "user.score"}),
            "dead":   FeatureNode(op="to_int",    inputs={"input": "user.age"}),
        },
        output_feature="useful",
    )


# ── ir_hash determinism ────────────────────────────────────────────────────────

class TestIRHashDeterminism:
    """Same definition version always produces the same ir_hash."""

    def test_same_definition_produces_same_hash(self, builder, generator):
        """
        Core invariant: compile the same definition twice → identical ir_hash.
        This verifies canonical serialisation and SHA-256 stability.
        """
        definition = _linear_chain_definition()

        graph1 = builder.build_dag(definition)
        hash1  = generator.hash_graph(graph1)

        graph2 = builder.build_dag(definition)
        hash2  = generator.hash_graph(graph2)

        assert hash1 == hash2

    def test_hash_is_nonempty_hex_string(self, builder, generator):
        """hash_graph returns a 64-character SHA-256 hex string."""
        graph = builder.build_dag(_linear_chain_definition())
        h = generator.hash_graph(graph)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_stored_on_graph(self, builder, generator):
        """hash_graph sets graph.ir_hash in-place."""
        graph = builder.build_dag(_linear_chain_definition())
        assert graph.ir_hash == ""          # sentinel before hashing
        h = generator.hash_graph(graph)
        assert graph.ir_hash == h

    def test_different_definitions_produce_different_hashes(self, builder, generator):
        """Two semantically different definitions produce different ir_hashes."""
        defn_a = _linear_chain_definition()
        defn_b = ConceptDefinition(
            concept_id="test.chain",
            version="2.0",         # different version
            namespace=Namespace.PERSONAL,
            output_type="float",
            primitives={
                "user.score": PrimitiveRef(type="float", missing_data_policy=MissingDataPolicy.ZERO),
            },
            features={
                "step1": FeatureNode(op="normalize", inputs={"input": "user.score"}),
                "step2": FeatureNode(op="z_score_op", inputs={"input": "step1"}),
            },
            output_feature="step2",
        )

        graph_a = builder.build_dag(defn_a)
        graph_b = builder.build_dag(defn_b)
        assert generator.hash_graph(graph_a) != generator.hash_graph(graph_b)


# ── Circular dependency detection ─────────────────────────────────────────────

class TestCircularDependency:
    """Circular dependencies must raise graph_error."""

    def test_two_node_cycle_raises_graph_error(self, builder):
        """
        feat_a → feat_b → feat_a forms a direct two-node cycle.
        build_dag must raise MemintelError(graph_error).
        """
        definition = ConceptDefinition(
            concept_id="test.cycle",
            version="1.0",
            namespace=Namespace.PERSONAL,
            output_type="float",
            primitives={
                "user.x": PrimitiveRef(
                    type="float",
                    missing_data_policy=MissingDataPolicy.ZERO,
                ),
            },
            features={
                "feat_a": FeatureNode(op="normalize",  inputs={"input": "feat_b"}),
                "feat_b": FeatureNode(op="z_score_op", inputs={"input": "feat_a"}),
            },
            output_feature="feat_a",
        )

        with pytest.raises(MemintelError) as exc_info:
            builder.build_dag(definition)
        assert exc_info.value.error_type == ErrorType.GRAPH_ERROR

    def test_three_node_cycle_raises_graph_error(self, builder):
        """
        feat_a → feat_b → feat_c → feat_a.
        All three nodes form a cycle; none is reachable in topological order.
        """
        definition = ConceptDefinition(
            concept_id="test.cycle3",
            version="1.0",
            namespace=Namespace.PERSONAL,
            output_type="float",
            primitives={
                "user.x": PrimitiveRef(
                    type="float",
                    missing_data_policy=MissingDataPolicy.ZERO,
                ),
            },
            features={
                "feat_a": FeatureNode(op="normalize",    inputs={"input": "feat_c"}),
                "feat_b": FeatureNode(op="z_score_op",   inputs={"input": "feat_a"}),
                "feat_c": FeatureNode(op="normalize",    inputs={"input": "feat_b"}),
            },
            output_feature="feat_a",
        )

        with pytest.raises(MemintelError) as exc_info:
            builder.build_dag(definition)
        assert exc_info.value.error_type == ErrorType.GRAPH_ERROR

    def test_validator_detects_cycle(self, validator):
        """Validator.validate_graph also raises graph_error on a cycle."""
        definition = ConceptDefinition(
            concept_id="test.cycle.val",
            version="1.0",
            namespace=Namespace.PERSONAL,
            output_type="float",
            primitives={
                "user.x": PrimitiveRef(
                    type="float",
                    missing_data_policy=MissingDataPolicy.ZERO,
                ),
            },
            features={
                "feat_a": FeatureNode(op="normalize",  inputs={"input": "feat_b"}),
                "feat_b": FeatureNode(op="z_score_op", inputs={"input": "feat_a"}),
            },
            output_feature="feat_a",
        )

        with pytest.raises(MemintelError) as exc_info:
            validator.validate_graph(definition)
        assert exc_info.value.error_type == ErrorType.GRAPH_ERROR


# ── Dead node elimination ──────────────────────────────────────────────────────

class TestDeadNodeElimination:
    """Dead nodes (not on the output path) are removed during optimisation."""

    def test_dead_feature_node_removed(self, builder):
        """
        'feat:dead' is not reachable from the output node → eliminated.
        The output graph must not contain it.
        """
        graph = builder.build_dag(_dead_node_definition())
        node_ids = {n.node_id for n in graph.nodes}

        assert _feat_node_id("useful") in node_ids, "live node must be present"
        assert _feat_node_id("dead")   not in node_ids, "dead node must be eliminated"

    def test_dead_primitive_node_removed(self, builder):
        """
        'prim:user.age' feeds only into the dead feature → also eliminated.
        """
        graph = builder.build_dag(_dead_node_definition())
        node_ids = {n.node_id for n in graph.nodes}

        assert _prim_node_id("user.score") in node_ids,   "live primitive must be present"
        assert _prim_node_id("user.age")   not in node_ids, "dead primitive must be eliminated"

    def test_dead_edges_removed(self, builder):
        """Edges that reference eliminated nodes are also removed."""
        graph = builder.build_dag(_dead_node_definition())
        dead_id = _feat_node_id("dead")
        for edge in graph.edges:
            assert edge.from_node_id != dead_id
            assert edge.to_node_id   != dead_id

    def test_live_node_count_after_elimination(self, builder):
        """Only the live nodes remain: prim:user.score + feat:useful = 2 nodes."""
        graph = builder.build_dag(_dead_node_definition())
        assert len(graph.nodes) == 2

    def test_output_node_preserved(self, builder):
        """The output node is always reachable — it must never be eliminated."""
        graph = builder.build_dag(_dead_node_definition())
        node_ids = {n.node_id for n in graph.nodes}
        assert graph.output_node_id in node_ids


# ── Topological order ─────────────────────────────────────────────────────────

class TestTopologicalOrder:
    """Topological order is deterministic and respects all dependency edges."""

    def test_linear_chain_order(self, builder):
        """
        user.score → step1 → step2.
        Expected order: [prim:user.score, feat:step1, feat:step2].
        """
        graph = builder.build_dag(_linear_chain_definition())
        order = graph.topological_order

        prim_idx  = order.index(_prim_node_id("user.score"))
        step1_idx = order.index(_feat_node_id("step1"))
        step2_idx = order.index(_feat_node_id("step2"))

        assert prim_idx  < step1_idx, "primitive must come before step1"
        assert step1_idx < step2_idx, "step1 must come before step2"

    def test_every_dependency_precedes_dependent(self, builder):
        """For every edge (A → B) in the graph, A appears before B in the order."""
        graph = builder.build_dag(_parallel_branches_definition())
        order = graph.topological_order
        pos = {nid: idx for idx, nid in enumerate(order)}

        for edge in graph.edges:
            src_pos = pos.get(edge.from_node_id)
            dst_pos = pos.get(edge.to_node_id)
            if src_pos is not None and dst_pos is not None:
                assert src_pos < dst_pos, (
                    f"Edge {edge.from_node_id} → {edge.to_node_id}: "
                    f"source at position {src_pos} must precede destination at {dst_pos}"
                )

    def test_order_includes_all_nodes(self, builder):
        """Every node in the graph appears exactly once in topological_order."""
        graph = builder.build_dag(_linear_chain_definition())
        node_ids = {n.node_id for n in graph.nodes}
        assert set(graph.topological_order) == node_ids
        assert len(graph.topological_order) == len(node_ids)

    def test_same_definition_same_order(self, builder):
        """Repeated compilation of the same definition produces identical order."""
        defn = _linear_chain_definition()
        order_a = builder.build_dag(defn).topological_order
        order_b = builder.build_dag(defn).topological_order
        assert order_a == order_b


# ── Parallelizable groups ─────────────────────────────────────────────────────

class TestParallelizableGroups:
    """Parallelizable groups are computed correctly and fixed at compile time."""

    def test_independent_branches_in_same_group(self, builder):
        """
        branch_a and branch_b are independent → they appear in the same group.
        """
        graph = builder.build_dag(_parallel_branches_definition())
        # Find the group that contains branch_a
        branch_a_id = _feat_node_id("branch_a")
        branch_b_id = _feat_node_id("branch_b")

        group_containing_a = next(
            (g for g in graph.parallelizable_groups if branch_a_id in g), None
        )
        assert group_containing_a is not None, "branch_a must appear in some group"
        assert branch_b_id in group_containing_a, (
            "branch_b must be in the same group as branch_a (they are independent)"
        )

    def test_dependent_nodes_in_different_groups(self, builder):
        """
        feat:step1 depends on prim:user.score → they must be in different groups.
        """
        graph = builder.build_dag(_linear_chain_definition())
        prim_id  = _prim_node_id("user.score")
        step1_id = _feat_node_id("step1")

        groups = graph.parallelizable_groups
        group_of_prim  = next((i for i, g in enumerate(groups) if prim_id  in g), None)
        group_of_step1 = next((i for i, g in enumerate(groups) if step1_id in g), None)

        assert group_of_prim  is not None
        assert group_of_step1 is not None
        assert group_of_prim < group_of_step1, (
            "primitive must be in an earlier group than a feature that depends on it"
        )

    def test_output_node_in_last_group(self, builder):
        """The output node (deepest in the DAG) is always in the last group."""
        graph = builder.build_dag(_linear_chain_definition())
        last_group = graph.parallelizable_groups[-1]
        assert graph.output_node_id in last_group

    def test_groups_cover_all_nodes(self, builder):
        """Every node is assigned to exactly one parallelizable group."""
        graph = builder.build_dag(_parallel_branches_definition())
        all_in_groups = {nid for group in graph.parallelizable_groups for nid in group}
        all_node_ids  = {n.node_id for n in graph.nodes}
        assert all_in_groups == all_node_ids


# ── Node deduplication ─────────────────────────────────────────────────────────

class TestNodeDeduplication:
    """Identical subgraph nodes are merged."""

    def test_duplicate_identical_features_merged(self, builder):
        """
        Two features with the same op + inputs + params are deduplicated.
        The output graph has fewer nodes than the input definition.
        """
        definition = ConceptDefinition(
            concept_id="test.dedup",
            version="1.0",
            namespace=Namespace.PERSONAL,
            output_type="float",
            primitives={
                "user.score": PrimitiveRef(
                    type="float",
                    missing_data_policy=MissingDataPolicy.ZERO,
                ),
            },
            features={
                # dup_a and dup_b are identical: same op, same input, same params
                "dup_a":  FeatureNode(op="normalize", inputs={"input": "user.score"}),
                "dup_b":  FeatureNode(op="normalize", inputs={"input": "user.score"}),
                # output uses dup_a; dup_b will be eliminated by dead-node + dedup
                "output": FeatureNode(op="z_score_op", inputs={"input": "dup_a"}),
            },
            output_feature="output",
        )

        graph = builder.build_dag(definition)
        ops = [n.op for n in graph.nodes]
        # There should be exactly one 'normalize' node after deduplication.
        assert ops.count("normalize") == 1

    def test_no_duplicate_nodes_in_simple_chain(self, builder):
        """A chain with no identical nodes is unchanged by deduplication."""
        graph = builder.build_dag(_linear_chain_definition())
        # All ops are distinct: primitive_fetch, normalize, z_score_op
        ops = [n.op for n in graph.nodes]
        assert len(ops) == len(set(ops)), "all ops in a simple chain should be unique"


# ── Validator phase tests ──────────────────────────────────────────────────────

class TestValidatorPhases:
    """Each validation phase raises the correct error type."""

    def test_validate_schema_unknown_reference(self, validator):
        """
        A feature input referencing an undeclared name raises syntax_error.
        """
        definition = ConceptDefinition(
            concept_id="test.schema",
            version="1.0",
            namespace=Namespace.PERSONAL,
            output_type="float",
            primitives={
                "user.score": PrimitiveRef(
                    type="float",
                    missing_data_policy=MissingDataPolicy.ZERO,
                ),
            },
            features={
                "step1": FeatureNode(op="normalize", inputs={"input": "does_not_exist"}),
            },
            output_feature="step1",
        )

        with pytest.raises(MemintelError) as exc_info:
            validator.validate_schema(definition)
        assert exc_info.value.error_type == ErrorType.SYNTAX_ERROR

    def test_validate_operators_unknown_op(self, validator):
        """An unregistered operator name raises reference_error."""
        definition = ConceptDefinition(
            concept_id="test.ops",
            version="1.0",
            namespace=Namespace.PERSONAL,
            output_type="float",
            primitives={
                "user.score": PrimitiveRef(
                    type="float",
                    missing_data_policy=MissingDataPolicy.ZERO,
                ),
            },
            features={
                "step1": FeatureNode(op="not_a_real_op", inputs={"input": "user.score"}),
            },
            output_feature="step1",
        )

        with pytest.raises(MemintelError) as exc_info:
            validator.validate_operators(definition)
        assert exc_info.value.error_type == ErrorType.REFERENCE_ERROR

    def test_validate_types_type_mismatch(self, validator):
        """Passing a time_series<float> where float is expected raises type_error."""
        definition = ConceptDefinition(
            concept_id="test.types",
            version="1.0",
            namespace=Namespace.PERSONAL,
            output_type="float",
            primitives={
                "user.ts": PrimitiveRef(
                    type="time_series<float>",
                    missing_data_policy=MissingDataPolicy.ZERO,
                ),
            },
            features={
                # z_score_op expects float, but user.ts is time_series<float>
                "step1": FeatureNode(op="z_score_op", inputs={"input": "user.ts"}),
            },
            output_feature="step1",
        )

        with pytest.raises(MemintelError) as exc_info:
            validator.validate_types(definition)
        assert exc_info.value.error_type == ErrorType.TYPE_ERROR

    def test_validate_graph_disconnected_node(self, validator):
        """A feature not on any path to output_feature raises graph_error."""
        definition = ConceptDefinition(
            concept_id="test.graph",
            version="1.0",
            namespace=Namespace.PERSONAL,
            output_type="float",
            primitives={
                "user.score": PrimitiveRef(
                    type="float",
                    missing_data_policy=MissingDataPolicy.ZERO,
                ),
                "user.age": PrimitiveRef(
                    type="int",
                    missing_data_policy=MissingDataPolicy.ZERO,
                ),
            },
            features={
                "useful":      FeatureNode(op="normalize", inputs={"input": "user.score"}),
                "disconnected": FeatureNode(op="to_int",   inputs={"input": "user.age"}),
            },
            output_feature="useful",
        )

        with pytest.raises(MemintelError) as exc_info:
            validator.validate_graph(definition)
        assert exc_info.value.error_type == ErrorType.GRAPH_ERROR

    def test_validate_full_valid_definition_no_errors(self, validator):
        """A valid definition passes all six phases with no errors."""
        errors = validator.validate(_linear_chain_definition())
        assert errors == []

    def test_validate_full_returns_error_list(self, validator):
        """validate() accumulates errors across all phases into a list."""
        definition = ConceptDefinition(
            concept_id="test.multi_error",
            version="1.0",
            namespace=Namespace.PERSONAL,
            output_type="float",
            primitives={
                "user.score": PrimitiveRef(
                    type="float",
                    missing_data_policy=MissingDataPolicy.ZERO,
                ),
            },
            features={
                "step1": FeatureNode(op="bad_op", inputs={"input": "user.score"}),
            },
            output_feature="step1",
        )

        errors = validator.validate(definition)
        assert len(errors) >= 1
        assert any(e.type == ErrorType.REFERENCE_ERROR for e in errors)


# ── compile_explain_plan ───────────────────────────────────────────────────────

class TestCompileExplainPlan:
    """compile_explain_plan returns accurate metadata without executing."""

    def test_node_count(self, builder, generator):
        """node_count equals the number of nodes in the compiled graph."""
        graph = builder.build_dag(_linear_chain_definition())
        generator.hash_graph(graph)
        plan = generator.compile_explain_plan(graph, _linear_chain_definition())
        assert plan.node_count == len(graph.nodes)

    def test_execution_order_matches_graph(self, builder, generator):
        """execution_order in the plan mirrors graph.topological_order."""
        defn = _linear_chain_definition()
        graph = builder.build_dag(defn)
        generator.hash_graph(graph)
        plan = generator.compile_explain_plan(graph, defn)
        assert plan.execution_order == graph.topological_order

    def test_primitive_fetches_listed(self, builder, generator):
        """primitive_fetches contains the name of each primitive in the graph."""
        defn = _linear_chain_definition()
        graph = builder.build_dag(defn)
        generator.hash_graph(graph)
        plan = generator.compile_explain_plan(graph, defn)
        assert "user.score" in plan.primitive_fetches

    def test_critical_path_length_chain(self, builder, generator):
        """
        Linear chain: prim → step1 → step2.
        Critical path is 3 nodes long (after dead-node elimination).
        """
        defn = _linear_chain_definition()
        graph = builder.build_dag(defn)
        generator.hash_graph(graph)
        plan = generator.compile_explain_plan(graph, defn)
        # prim:user.score (1) → feat:step1 (2) → feat:step2 (3)
        assert plan.critical_path_length == 3

    def test_plan_does_not_mutate_graph(self, builder, generator):
        """compile_explain_plan must not change the graph's ir_hash."""
        defn  = _linear_chain_definition()
        graph = builder.build_dag(defn)
        h     = generator.hash_graph(graph)
        generator.compile_explain_plan(graph, defn)
        assert graph.ir_hash == h
