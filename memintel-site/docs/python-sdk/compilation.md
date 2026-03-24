---
id: compilation
title: Compilation
sidebar_label: Compilation
---

# Compilation

Compilation transforms a semantic Concept definition into a deterministic `ExecutionGraph` (IR). The same definition version always produces the same `ir_hash` — this is the foundation of the system's reproducibility guarantees.

:::note Invariant
`ir_hash` is deterministic. Same definition version → same `ir_hash` on any machine. Use canonical serialisation (sorted keys, stable field order) before hashing.
:::

---

## Compile Definition

```
POST /compile
```

Compiles a Concept definition into a fully resolved, typed, immutable `ExecutionGraph`. The returned `graph_id` is stable for a given definition version. Cache it at service startup and use `POST /execute/graph` in the hot path.

### Request — CompileRequest

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | str | **Required** | Fully qualified concept id. |
| `version` | str | **Required** | Explicit version. |

### Response — ExecutionGraph

| Field | Type | Required | Description |
|---|---|---|---|
| `graph_id` | str | Always | Pass to `POST /execute/graph`. |
| `ir_hash` | str | Always | Deterministic hash. Same definition → same hash. |
| `nodes` | list | Always | `GraphNode[]`. Each: `id`, `op`, `inputs`, `params`, `output_type`. |
| `edges` | list | Always | `GraphEdge[]`. Each: `from`, `to`, `data_type`. |

### Response Codes

| Status | Description |
|---|---|
| **200** | ExecutionGraph. |
| **400** | `graph_error` (circular dependency), `type_error`, or `semantic_error`. |
| **401** | Unauthorised. |
| **404** | Definition not found. |

### Python Startup Pattern

```python
GRAPH_CACHE: dict[str, str] = {}  # 'namespace.id:version' → graph_id

async def startup():
    for concept_id, version in ACTIVE_DEFINITIONS:
        graph = await client.compile(id=concept_id, version=version)
        GRAPH_CACHE[f'{concept_id}:{version}'] = graph.graph_id

# Then in the hot path:
graph_id = GRAPH_CACHE['org.churn_risk:1.2']
result = await client.execute_graph(graph_id=graph_id, entity=entity, timestamp=ts)
```

---

## Compile Semantic View

```
POST /compile/semantic
```

Returns the semantic view of a compiled concept — the meaning layer, not the execution layer. Exposes `semantic_hash`, resolved features, and the normalised canonical form. Use for deduplication checks and as input to the LLM semantic-refine loop.

### Response — SemanticGraph

| Field | Type | Required | Description |
|---|---|---|---|
| `semantic_hash` | str | Always | Identical for semantically equivalent concepts regardless of structure. |
| `features` | list | Always | `SemanticFeature[]`. Each: `name`, `layer`, `meaning`, `meaning_hash`. |
| `normalized_form` | dict | Always | Canonical compiler-normalised form. |
| `dependencies` | list | Always | `SemanticDependency[]`. Each: `id`, `version`, `type`, `depth`. |
| `equivalences` | list | Optional | Concepts with the same `semantic_hash`. |

### Response Codes

| Status | Description |
|---|---|
| **200** | SemanticGraph. |
| **400** | Validation error. |
| **401** | Unauthorised. |
| **404** | Not found. |

---

## Compile Explain Plan

```
POST /compile/explain-plan
```

Returns the planned execution order and parallelizable node groups without executing. The SQL `EXPLAIN` equivalent. Use to understand graph topology before running and identify concurrent node groups.

### Response — ExecutionPlan

| Field | Type | Required | Description |
|---|---|---|---|
| `steps` | list | Always | `WorkflowStep[]` in execution sequence. |
| `parallelizable_groups` | list | Optional | Groups of node IDs that can execute concurrently. |
| `optimizations_applied` | list | Optional | Passes applied: `node_deduplication`, `dead_node_elimination`. |

### Response Codes

| Status | Description |
|---|---|
| **200** | ExecutionPlan. |
| **400** | Validation error. |
| **401** | Unauthorised. |
| **404** | Not found. |
