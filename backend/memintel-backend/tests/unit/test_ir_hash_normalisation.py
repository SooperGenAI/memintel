"""
tests/unit/test_ir_hash_normalisation.py
────────────────────────────────────────────────────────────────────────────────
Verifies the three normalisation fixes in app/compiler/ir_generator.py.

Each test builds two ExecutionGraph objects that are semantically identical but
differ in representation, then asserts that hash_graph() produces the same
ir_hash for both.

Gaps under test:
  1. Decimal params   — Decimal("1.0") and Decimal("1.00") must hash identically.
  2. List-valued params — ["high", "low"] and ["low", "high"] must hash identically.
  3. input_slot type  — int 0 and str "0" on an edge must hash identically.
"""
from decimal import Decimal

import pytest

from app.compiler.ir_generator import IRGenerator
from app.models.concept import ExecutionGraph, GraphEdge, GraphNode


# ── shared helpers ────────────────────────────────────────────────────────────

def _minimal_graph(nodes: list[GraphNode], edges: list[GraphEdge]) -> ExecutionGraph:
    """Return a minimal ExecutionGraph wrapping the supplied nodes and edges."""
    return ExecutionGraph(
        graph_id="test-graph-id",
        concept_id="test.concept",
        version="1.0",
        ir_hash="0" * 64,           # placeholder — overwritten by hash_graph()
        nodes=nodes,
        edges=edges,
        topological_order=[n.node_id for n in nodes],
        parallelizable_groups=[[n.node_id] for n in nodes],
        output_node_id=nodes[-1].node_id,
        output_type="float",
    )


@pytest.fixture
def generator() -> IRGenerator:
    return IRGenerator()


# ── Test 1: Decimal normalisation ─────────────────────────────────────────────

def test_decimal_param_variants_produce_same_hash(generator: IRGenerator) -> None:
    """
    Decimal("1.0") and Decimal("1.00") represent the same value.
    They must produce the same ir_hash after normalisation to float.
    """
    node_a = GraphNode(
        node_id="feat:score",
        op="threshold",
        inputs={},
        params={"threshold": Decimal("1.0")},
        output_type="bool",
    )
    node_b = GraphNode(
        node_id="feat:score",
        op="threshold",
        inputs={},
        params={"threshold": Decimal("1.00")},
        output_type="bool",
    )

    hash_a = generator.hash_graph(_minimal_graph([node_a], []))
    hash_b = generator.hash_graph(_minimal_graph([node_b], []))

    assert hash_a == hash_b, (
        f"Decimal('1.0') and Decimal('1.00') produced different hashes:\n"
        f"  Decimal('1.0')  → {hash_a}\n"
        f"  Decimal('1.00') → {hash_b}"
    )


# ── Test 2: List-valued params normalisation ──────────────────────────────────

def test_list_param_order_variants_produce_same_hash(generator: IRGenerator) -> None:
    """
    A list-valued param in different insertion orders must hash identically.
    ["high", "low"] and ["low", "high"] are semantically the same set.
    """
    node_a = GraphNode(
        node_id="feat:category_check",
        op="in_set",
        inputs={},
        params={"allowed": ["high", "low"]},
        output_type="bool",
    )
    node_b = GraphNode(
        node_id="feat:category_check",
        op="in_set",
        inputs={},
        params={"allowed": ["low", "high"]},
        output_type="bool",
    )

    hash_a = generator.hash_graph(_minimal_graph([node_a], []))
    hash_b = generator.hash_graph(_minimal_graph([node_b], []))

    assert hash_a == hash_b, (
        f'["high", "low"] and ["low", "high"] produced different hashes:\n'
        f'  ["high", "low"] → {hash_a}\n'
        f'  ["low", "high"] → {hash_b}'
    )


# ── Test 3: input_slot int vs str normalisation ───────────────────────────────

def test_input_slot_int_and_str_produce_same_hash(generator: IRGenerator) -> None:
    """
    An edge whose input_slot is the integer 0 and one whose input_slot is the
    string "0" are semantically identical. hash_graph() must produce the same
    ir_hash for both via the str(e.input_slot) cast.

    GraphEdge declares input_slot: str, so we bypass Pydantic validation by
    using model_construct() to inject an int directly.
    """
    prim = GraphNode(
        node_id="prim:x",
        op="primitive_fetch",
        inputs={},
        params={"source_name": "x", "declared_type": "float", "missing_data_policy": "zero"},
        output_type="float",
    )
    feat = GraphNode(
        node_id="feat:out",
        op="normalize",
        inputs={"input": "prim:x"},
        params={},
        output_type="float",
    )

    edge_str = GraphEdge(from_node_id="prim:x", to_node_id="feat:out", input_slot="0")
    # Bypass Pydantic coercion so we can inject a raw int into the str field.
    edge_int = GraphEdge.model_construct(from_node_id="prim:x", to_node_id="feat:out", input_slot=0)

    hash_str = generator.hash_graph(_minimal_graph([prim, feat], [edge_str]))
    hash_int = generator.hash_graph(_minimal_graph([prim, feat], [edge_int]))

    assert hash_str == hash_int, (
        f'input_slot "0" (str) and 0 (int) produced different hashes:\n'
        f'  str "0" → {hash_str}\n'
        f'  int  0  → {hash_int}'
    )
