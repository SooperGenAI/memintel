"""
app/compiler/dag_builder.py
────────────────────────────────────────────────────────────────────────────────
Builds ExecutionGraph from a validated ConceptDefinition.

Pipeline (all steps deterministic):

  1. Create primitive_fetch nodes (op='primitive_fetch') — one per primitive.
  2. Topological sort of feature nodes using Kahn's algorithm with alphabetical
     tie-breaking.  Raises graph_error on a cycle (mandatory per spec).
  3. Create feature nodes via TypeChecker.check_node() to infer output types.
  4. Build directed edge list.
  5. Compute full topological order (primitives then features).
  6. Compute parallelizable_groups (BFS levels — fixed at compile time).
  7. Apply dead_node_elimination optimisation pass.
  8. Apply node_deduplication optimisation pass.
  9. Recompute topological order and groups after optimisations.

graph_id is a deterministic SHA-256 of "concept_id:version" — stable across
machines without needing to inspect graph content.

ir_hash is computed separately by IRGenerator.hash_graph() after build_dag()
returns.  The graph is returned with ir_hash='' as a sentinel.
"""
from __future__ import annotations

import bisect
import hashlib
import json

from app.compiler.type_checker import OPERATOR_REGISTRY
from app.compiler.type_checker import GraphNode as TCGraphNode
from app.compiler.type_checker import TypeChecker
from app.compiler.validator import resolve_primitive_type, topo_sort_features
from app.models.concept import ConceptDefinition, ExecutionGraph, GraphEdge, GraphNode
from app.models.errors import ErrorType, MemintelError


def _graph_error(msg: str, location: str | None = None) -> MemintelError:
    return MemintelError(ErrorType.GRAPH_ERROR, msg, location=location)


def _prim_node_id(name: str) -> str:
    return f"prim:{name}"


def _feat_node_id(name: str) -> str:
    return f"feat:{name}"


class DAGBuilder:
    """
    Compiles a ConceptDefinition into an ExecutionGraph.

    Public API
    ----------
    build_dag(definition) → ExecutionGraph
        Returns the compiled graph with deterministic topological order,
        parallelizable groups, and optimisation passes applied.
        ir_hash is set to '' — call IRGenerator.hash_graph() to populate it.

    Raises
    ------
    MemintelError(graph_error)     on circular dependency.
    MemintelError(type_error)      on type mismatch in any feature node.
    MemintelError(reference_error) on unknown operator.
    """

    def __init__(self) -> None:
        self._checker = TypeChecker()

    # ── Public entry point ─────────────────────────────────────────────────────

    def build_dag(self, definition: ConceptDefinition) -> ExecutionGraph:
        """
        Build and return the ExecutionGraph for ``definition``.

        The returned graph has ir_hash='' as a sentinel.
        Populate it with ``IRGenerator.hash_graph(graph)`` before persisting.
        """
        # ── Step 1: Primitive fetch nodes ──────────────────────────────────────
        prim_nodes: list[GraphNode] = []
        prim_type_env: dict[str, str] = {}

        for prim_name in sorted(definition.primitives.keys()):
            prim_ref = definition.primitives[prim_name]
            resolved_type = resolve_primitive_type(prim_ref)
            prim_type_env[prim_name] = resolved_type
            policy_val = (
                prim_ref.missing_data_policy.value
                if prim_ref.missing_data_policy is not None
                else None
            )
            prim_nodes.append(GraphNode(
                node_id=_prim_node_id(prim_name),
                op="primitive_fetch",
                inputs={},
                params={
                    "source_name": prim_name,
                    "declared_type": prim_ref.type,
                    "missing_data_policy": policy_val,
                },
                output_type=resolved_type,
            ))

        # ── Step 2: Topological sort of features ───────────────────────────────
        feature_order = topo_sort_features(definition)  # raises graph_error on cycle

        # ── Step 3: Feature nodes — infer types via TypeChecker ────────────────
        type_env: dict[str, str] = dict(prim_type_env)
        feature_nodes: list[GraphNode] = []

        for feat_name in feature_order:
            feat_def = definition.features[feat_name]
            node_id = _feat_node_id(feat_name)

            # Resolve symbolic input references → node_ids + collect types
            resolved_inputs: dict[str, str] = {}
            input_types:     dict[str, str] = {}

            for slot_name, source_ref in feat_def.inputs.items():
                if not isinstance(source_ref, str):
                    continue  # literal scalar — not a DAG edge
                if source_ref in definition.primitives:
                    ref_node_id = _prim_node_id(source_ref)
                elif source_ref in definition.features:
                    ref_node_id = _feat_node_id(source_ref)
                else:
                    ref_node_id = source_ref  # unresolved (validator should have caught this)

                resolved_inputs[slot_name] = ref_node_id
                if source_ref in type_env:
                    input_types[slot_name] = type_env[source_ref]

            tc_node = TCGraphNode(op=feat_def.op, node_id=feat_name)
            output_type = self._checker.check_node(tc_node, input_types)
            type_env[feat_name] = output_type

            feature_nodes.append(GraphNode(
                node_id=node_id,
                op=feat_def.op,
                inputs=resolved_inputs,
                params=feat_def.params,
                output_type=output_type,
            ))

        all_nodes: list[GraphNode] = prim_nodes + feature_nodes

        # ── Step 4: Edge list ──────────────────────────────────────────────────
        edges: list[GraphEdge] = []
        for fn in feature_nodes:
            for slot_name, upstream_id in fn.inputs.items():
                edges.append(GraphEdge(
                    from_node_id=upstream_id,
                    to_node_id=fn.node_id,
                    input_slot=slot_name,
                ))

        # ── Step 5–6: Initial topo order + parallel groups ─────────────────────
        output_node_id = _feat_node_id(definition.output_feature)

        # ── Step 7: dead_node_elimination ─────────────────────────────────────
        all_nodes, edges = self._dead_node_elimination(all_nodes, edges, output_node_id)

        # ── Step 8: node_deduplication ────────────────────────────────────────
        all_nodes, edges = self._node_deduplication(all_nodes, edges)

        # ── Step 9: Recompute after optimisations ──────────────────────────────
        topo_order     = self._full_topo_sort(all_nodes, edges)
        parallel_grps  = self._parallelizable_groups(all_nodes, edges)
        output_type    = type_env[definition.output_feature]

        # graph_id: deterministic SHA-256 of "concept_id:version"
        graph_id = hashlib.sha256(
            f"{definition.concept_id}:{definition.version}".encode("utf-8")
        ).hexdigest()[:32]

        return ExecutionGraph(
            graph_id=graph_id,
            concept_id=definition.concept_id,
            version=definition.version,
            ir_hash="",          # caller must populate via IRGenerator.hash_graph()
            nodes=all_nodes,
            edges=edges,
            topological_order=topo_order,
            parallelizable_groups=parallel_grps,
            output_node_id=output_node_id,
            output_type=output_type,
        )

    # ── Topological sort (full graph) ──────────────────────────────────────────

    def _full_topo_sort(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> list[str]:
        """
        Deterministic topological sort of all graph nodes.

        Uses Kahn's algorithm with alphabetical tie-breaking.
        Raises graph_error if a cycle is detected (should not happen after
        dag_builder's cycle check, but included as a safety net).
        """
        node_ids = {n.node_id for n in nodes}

        # dependents[x] = set of nodes that depend on x
        dependents: dict[str, set[str]] = {nid: set() for nid in node_ids}
        in_degree:  dict[str, int]      = {nid: 0     for nid in node_ids}

        for edge in edges:
            src, dst = edge.from_node_id, edge.to_node_id
            if src in node_ids and dst in node_ids:
                dependents[src].add(dst)
                in_degree[dst] += 1

        ready: list[str] = sorted(nid for nid, d in in_degree.items() if d == 0)
        order: list[str] = []

        while ready:
            node = ready.pop(0)
            order.append(node)
            for dep in sorted(dependents[node]):
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    bisect.insort(ready, dep)

        if len(order) != len(node_ids):
            raise _graph_error(
                "Cycle detected in compiled execution graph — this should have "
                "been caught by validate_graph().",
            )

        return order

    # ── Parallelizable groups ──────────────────────────────────────────────────

    def _parallelizable_groups(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> list[list[str]]:
        """
        Assign each node to a BFS depth level.

        Nodes at the same level have no inter-dependencies and may execute
        concurrently.  Groups are returned in execution order (level 0 first).

        This computation is fixed at compile time and must not change between
        executions of the same graph (core-spec.md §1B DAG Execution Order).
        """
        node_ids = {n.node_id for n in nodes}

        # in_deps[x] = set of nodes that must complete before x can run
        in_deps: dict[str, set[str]] = {nid: set() for nid in node_ids}
        for edge in edges:
            if edge.from_node_id in node_ids and edge.to_node_id in node_ids:
                in_deps[edge.to_node_id].add(edge.from_node_id)

        # BFS-level assignment
        level: dict[str, int] = {}
        for nid in node_ids:
            if not in_deps[nid]:
                level[nid] = 0

        changed = True
        while changed:
            changed = False
            for nid in sorted(node_ids):      # sorted for determinism
                if nid in level:
                    continue
                if all(dep in level for dep in in_deps[nid]):
                    level[nid] = max(level[dep] for dep in in_deps[nid]) + 1
                    changed = True

        if not level:
            return []

        max_level = max(level.values())
        return [
            sorted(nid for nid, lvl in level.items() if lvl == lv)
            for lv in range(max_level + 1)
            if any(lvl == lv for lvl in level.values())
        ]

    # ── Optimisation: dead node elimination ────────────────────────────────────

    def _dead_node_elimination(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        output_node_id: str,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """
        Remove nodes that are not on any path to ``output_node_id``.

        Walks backwards from the output node through the reverse edge set.
        Any node not reached is dead (it can never affect the concept result).
        Its edges are also removed.
        """
        # upstream[x] = set of node_ids that feed into x
        upstream: dict[str, set[str]] = {n.node_id: set() for n in nodes}
        for edge in edges:
            if edge.to_node_id in upstream:
                upstream[edge.to_node_id].add(edge.from_node_id)

        # BFS backwards from output
        reachable: set[str] = set()
        stack: list[str] = [output_node_id]
        while stack:
            nid = stack.pop()
            if nid in reachable:
                continue
            reachable.add(nid)
            stack.extend(upstream.get(nid, set()))

        live_nodes = [n for n in nodes if n.node_id in reachable]
        live_edges = [
            e for e in edges
            if e.from_node_id in reachable and e.to_node_id in reachable
        ]
        return live_nodes, live_edges

    # ── Optimisation: node deduplication ──────────────────────────────────────

    def _node_deduplication(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """
        Deduplicate nodes with identical (op, inputs, params).

        When two nodes compute exactly the same thing, all downstream edges
        pointing to the duplicate are redirected to the canonical (first-seen)
        node, and the duplicate is removed.

        Identity key: canonical JSON of {op, sorted(inputs), sorted(params)}.
        """
        # canonical_key → first node_id seen with that key
        canonical: dict[str, str] = {}
        # duplicate node_id → canonical node_id it should be replaced by
        replacement: dict[str, str] = {}

        for node in nodes:
            key = json.dumps(
                {
                    "op":     node.op,
                    "inputs": dict(sorted(node.inputs.items())),
                    "params": dict(sorted(node.params.items())),
                },
                sort_keys=True,
                default=str,
            )
            if key in canonical:
                replacement[node.node_id] = canonical[key]
            else:
                canonical[key] = node.node_id

        if not replacement:
            return nodes, edges

        def _remap(nid: str) -> str:
            return replacement.get(nid, nid)

        deduped_nodes = [n for n in nodes if n.node_id not in replacement]

        seen_edges: set[tuple[str, str, str]] = set()
        deduped_edges: list[GraphEdge] = []
        for edge in edges:
            new_from = _remap(edge.from_node_id)
            new_to   = _remap(edge.to_node_id)
            if new_from == new_to:
                continue   # self-loop created by dedup — discard
            key = (new_from, new_to, edge.input_slot)
            if key not in seen_edges:
                seen_edges.add(key)
                deduped_edges.append(GraphEdge(
                    from_node_id=new_from,
                    to_node_id=new_to,
                    input_slot=edge.input_slot,
                ))

        return deduped_nodes, deduped_edges
