"""
app/models/concept.py
──────────────────────────────────────────────────────────────────────────────
Concept domain models.

Covers three distinct layers of the concept lifecycle:

  1. Authoring     — ConceptDefinition, PrimitiveRef, FeatureNode
                     The authored YAML / JSON body stored as JSONB in the
                     definitions table (definition_type='concept').

  2. Compilation   — GraphNode, GraphEdge, ExecutionGraph
                     The compiled IR stored in the execution_graphs table.
                     SemanticGraph and ExecutionPlan are compiler output views
                     not persisted to the DB (returned from compile endpoints).

  3. Registry      — DefinitionResponse, VersionSummary, SearchResult,
                     SemanticDiffResult, LineageResult
                     Generic response shapes used by all definition write and
                     read operations regardless of definition_type.

Also contains MemintelType — the single authoritative source for the type
vocabulary used across the compiler, runtime, and type-checker.

Dependency direction
────────────────────
  concept.py  →  result.py (MissingDataPolicy, for PrimitiveRef)
  concept.py  →  task.py   (Namespace, for ConceptDefinition)
  concept.py  →  errors.py (none; errors are raised by callers, not here)

No other models module imports from concept.py, so there are no cycles.

Design notes
────────────
MemintelType is a plain class with class-level string constants, not an enum.
  Nullable variants (float?) and future parameterised types are constructed
  dynamically. An enum cannot represent an open-ended set of nullable strings,
  so we use constants + helper methods instead.

Two distinct node models for authoring vs compilation.
  FeatureNode (authored) uses symbolic references — input names point to
  primitive names or feature names as strings, mirroring the YAML format.
  GraphNode (compiled) uses resolved node IDs — inputs are fully resolved to
  upstream node_ids, output_type is inferred by the type checker, and the
  ordering is set by the topological sort. Conflating these into one model
  would force the compiler to mutate authored inputs in place.

ExecutionGraph carries its own graph_id, concept_id, version, and ir_hash.
  The store decomposes these into separate DB columns (for indexing) alongside
  the graph_body JSONB (the full model serialisation). The model does not have
  a separate graph_body field — it IS the body.

DefinitionResponse is generic — all four definition types (concept, condition,
  action, primitive) return the same response shape from write operations.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.result import MissingDataPolicy
from app.models.task import Namespace


# ── Memintel type vocabulary ──────────────────────────────────────────────────

# ── V7 vocabulary context ─────────────────────────────────────────────────────

#: Maximum number of IDs allowed in either list of VocabularyContext.
#: The cap is per-list, NOT combined (499 concept IDs + 499 condition IDs = valid;
#: 501 concept IDs + 0 condition IDs = invalid).
MAX_VOCABULARY_IDS: int = 500


class VocabularyContext(BaseModel):
    """
    Scopes Memintel CoR Step 2 (Concept Selection) to a bounded vocabulary.

    Canvas assembles this from the concept_ids and condition_ids belonging to
    the org's allowed modules and passes it in POST /tasks. Memintel never
    receives Canvas identity fields (user_id, org_id, module_id) — only the
    opaque Memintel-registered identifiers that form the vocabulary.

    Three distinct states (enforced at the service layer, NOT here):
      Field absent (None)      → global fallback, existing behaviour unchanged.
      Both lists empty         → vocabulary_mismatch error (before LLM call).
      Non-empty list(s)        → CoR Step 2 restricted to provided IDs.

    Validator: each list is independently capped at MAX_VOCABULARY_IDS (500).
    The cap applies per list, not to the combined total.
      499 concept IDs + 499 condition IDs = valid (998 total).
      501 concept IDs +   0 condition IDs = invalid (per-list cap exceeded).
    """
    available_concept_ids:   list[str]
    available_condition_ids: list[str]

    @field_validator("available_concept_ids")
    @classmethod
    def _cap_concept_ids(cls, v: list[str]) -> list[str]:
        if len(v) > MAX_VOCABULARY_IDS:
            raise ValueError(
                f"available_concept_ids exceeds the maximum of {MAX_VOCABULARY_IDS} entries. "
                f"Got {len(v)}. Reduce the number of installed modules or filter the vocabulary."
            )
        return v

    @field_validator("available_condition_ids")
    @classmethod
    def _cap_condition_ids(cls, v: list[str]) -> list[str]:
        if len(v) > MAX_VOCABULARY_IDS:
            raise ValueError(
                f"available_condition_ids exceeds the maximum of {MAX_VOCABULARY_IDS} entries. "
                f"Got {len(v)}. Reduce the number of installed modules or filter the vocabulary."
            )
        return v


# ── Memintel type vocabulary ──────────────────────────────────────────────────

class MemintelType:
    """
    Authoritative type string constants for the Memintel type system v1.1.

    Use these constants everywhere a Memintel type string is required —
    in PrimitiveRef.type, GraphNode.output_type, TYPE_STRATEGY_COMPATIBILITY,
    TypeChecker method signatures, etc.

    Why a class, not an enum
    ────────────────────────
    Nullable variants are formed by appending '?' to any base type (float?).
    An enum cannot represent this open-ended set. The class provides named
    constants for all base types plus helper methods to construct and test
    nullable and container variants.

    Type hierarchy summary (from memintel_type_system.md v1.1)
    ──────────────────────────────────────────────────────────
    Scalar:    float, int, boolean, string, categorical
    Container: time_series<float>, time_series<int>, list<float>, list<int>
    Modifier:  T?  (any of the above wrapped as nullable)
    Decision:  decision<boolean>, decision<categorical>  (condition output only)
    Parameter: duration  (compile-time only — NOT a valid node output type)

    Subtype rules:
      int               → float              (implicit widening)
      T                 → T?                 (implicit nullable widening)
      time_series<int>  → time_series<float> (via int→float)
      list<int>         → list<float>        (via int→float)
      float             → int               REQUIRES explicit to_int()
      T?                → T                 REQUIRES null-handling operator
    """

    # ── Scalar types ──────────────────────────────────────────────────────────
    FLOAT       = "float"
    INT         = "int"
    BOOLEAN     = "boolean"
    STRING      = "string"
    CATEGORICAL = "categorical"

    # ── Container types ───────────────────────────────────────────────────────
    TIME_SERIES_FLOAT = "time_series<float>"
    TIME_SERIES_INT   = "time_series<int>"
    LIST_FLOAT        = "list<float>"
    LIST_INT          = "list<int>"

    # ── Decision types (condition output only — not valid operator inputs) ────
    DECISION_BOOLEAN     = "decision<boolean>"
    DECISION_CATEGORICAL = "decision<categorical>"

    # ── Parameter-only type (MUST NOT appear as a node output type) ──────────
    DURATION = "duration"

    # ── Type classification sets ──────────────────────────────────────────────

    #: All scalar base types.
    SCALAR_TYPES: frozenset[str] = frozenset({
        FLOAT, INT, BOOLEAN, STRING, CATEGORICAL,
    })

    #: All container base types.
    CONTAINER_TYPES: frozenset[str] = frozenset({
        TIME_SERIES_FLOAT, TIME_SERIES_INT, LIST_FLOAT, LIST_INT,
    })

    #: Decision types — produced by conditions, not by concept operators.
    DECISION_TYPES: frozenset[str] = frozenset({
        DECISION_BOOLEAN, DECISION_CATEGORICAL,
    })

    #: All types that are valid as DAG node output types.
    #: DURATION and nullable variants are excluded from this base set.
    #: Call MemintelType.nullable(t) to check nullable variants.
    NODE_OUTPUT_TYPES: frozenset[str] = SCALAR_TYPES | CONTAINER_TYPES | DECISION_TYPES

    #: Types that numeric condition strategies (threshold, percentile, z_score,
    #: change) accept as input. Note: after concept execution, the concept
    #: output type (float or int) is what flows into the condition — not the
    #: primitive's time_series type.
    NUMERIC_TYPES: frozenset[str] = frozenset({FLOAT, INT})

    # ── Helpers ───────────────────────────────────────────────────────────────

    @classmethod
    def nullable(cls, base_type: str) -> str:
        """Return the nullable variant: float → float?, time_series<float> → time_series<float>?"""
        return f"{base_type}?"

    @classmethod
    def is_nullable(cls, type_str: str) -> bool:
        """Return True if the type string is a nullable variant (ends with '?')."""
        return type_str.endswith("?")

    @classmethod
    def base_of(cls, type_str: str) -> str:
        """Strip the nullable modifier: float? → float. No-op for non-nullable."""
        return type_str[:-1] if type_str.endswith("?") else type_str

    @classmethod
    def is_valid_node_output(cls, type_str: str) -> bool:
        """
        Return True if type_str can be the declared output type of a graph node.

        Accepts both non-nullable and nullable variants of NODE_OUTPUT_TYPES.
        Rejects DURATION (parameter-only) and any unrecognised string.
        """
        base = cls.base_of(type_str)
        return base in cls.NODE_OUTPUT_TYPES

    @classmethod
    def is_assignable(cls, actual: str, expected: str) -> bool:
        """
        Return True if `actual` satisfies `expected` under the subtype rules.

        Mirrors TypeChecker.is_assignable() at the model layer so that the
        compiler's type checker and any model-level pre-validation agree.
        This is the single implementation — TypeChecker delegates to this.
        """
        if actual == expected:
            return True
        # int → float widening
        if expected == cls.FLOAT and actual == cls.INT:
            return True
        # T → T? nullable widening
        if cls.is_nullable(expected) and actual == cls.base_of(expected):
            return True
        # time_series<int> → time_series<float>
        if expected == cls.TIME_SERIES_FLOAT and actual == cls.TIME_SERIES_INT:
            return True
        # list<int> → list<float>
        if expected == cls.LIST_FLOAT and actual == cls.LIST_INT:
            return True
        # categorical{...} → categorical  (labeled is a subtype of bare)
        # categorical{...} → categorical{...}  (same label set, already caught by ==)
        actual_base = cls.base_of(actual)
        expected_base = cls.base_of(expected)
        actual_nullable = cls.is_nullable(actual)
        expected_nullable = cls.is_nullable(expected)
        if (actual_base.startswith("categorical{") and actual_base.endswith("}")
                and expected_base == cls.CATEGORICAL
                and actual_nullable == expected_nullable):
            return True
        return False


# ── Authoring models ──────────────────────────────────────────────────────────

class PrimitiveRef(BaseModel):
    """
    A primitive data source referenced within a concept definition.

    Specifies the expected Memintel type of the primitive and how to handle
    missing data. The actual source configuration (connector, SQL query, etc.)
    lives in the registered primitive config (memintel.config.md / PrimitiveConfig)
    — this ref only declares the contract the concept author expects.

    missing_data_policy resolves output nullability at compile time:
      None / 'null'   → T?  (null propagates)
      'zero'          → T   (null replaced with 0)
      'forward_fill'  → T   (null replaced with last known value)
      'backward_fill' → T   (null replaced with next known value)

    labels must be declared when type is 'categorical'. The compiler validates
    that all execution paths produce only the declared labels (§5 of type spec).
    """
    type: str
    missing_data_policy: MissingDataPolicy | None = None
    labels: list[str] | None = None  # required when type='categorical'

    @model_validator(mode="after")
    def _validate_categorical_labels(self) -> PrimitiveRef:
        if self.type == MemintelType.CATEGORICAL and not self.labels:
            raise ValueError(
                "labels must be declared for categorical primitives"
            )
        if self.type != MemintelType.CATEGORICAL and self.labels is not None:
            raise ValueError(
                "labels may only be declared on categorical primitives"
            )
        return self

    @field_validator("type")
    @classmethod
    def _valid_primitive_type(cls, v: str) -> str:
        # Primitives may be nullable (T?) but not decision or duration types.
        base = MemintelType.base_of(v)
        valid_bases = MemintelType.SCALAR_TYPES | MemintelType.CONTAINER_TYPES
        if base not in valid_bases:
            raise ValueError(
                f"'{v}' is not a valid primitive type. "
                "Primitives must be scalar or container types (nullable variants allowed)."
            )
        return v


class FeatureNode(BaseModel):
    """
    A single authored operator node in a concept's feature DAG.

    This is the *authored* representation — it mirrors the YAML format used
    in concept definitions. The `inputs` dict maps each operator input slot
    name to either a primitive name (e.g. 'user.activity_count') or another
    feature name in this concept (e.g. 'activity_drop').

    The compiler resolves symbolic names to upstream node_ids when producing
    GraphNode. output_type is NOT set by the author — it is inferred by the
    type checker during compilation.

    params carries operator-specific parameters (e.g. {'window': '30d'} for
    pct_change or moving_average). Duration values must follow the format:
    positive integer + suffix (h/d/w/m/y). E.g. '30d', '4w'.
    """
    op: str
    inputs: dict[str, Any]           # slot_name → primitive_name | feature_name | literal
    params: dict[str, Any] = Field(default_factory=dict)


class ConceptDefinition(BaseModel):
    """
    The authored concept body. Stored as JSONB in the definitions table with
    definition_type='concept'.

    Primitives declare the data sources the concept depends on (names and
    expected types). Features define the DAG of operators that transform
    primitive data into the concept's output value.

    output_feature names the terminal FeatureNode in the DAG — the node whose
    inferred output type must match output_type. The compiler validates this.

    For categorical concepts, labels declares the closed label set. The type
    checker enforces that all execution paths produce only declared labels.

    Immutability: once registered under (concept_id, version), the body is
    frozen. Semantic changes require a new version.
    """
    concept_id: str
    version: str
    namespace: Namespace
    output_type: str                  # declared concept output type
    labels: list[str] | None = None   # required when output_type='categorical'
    description: str | None = None
    primitives: dict[str, PrimitiveRef]  # primitive_name → PrimitiveRef
    features: dict[str, FeatureNode]     # feature_name   → FeatureNode
    output_feature: str                  # name of the terminal feature node
    created_at: datetime | None = None
    deprecated: bool = False

    @model_validator(mode="after")
    def _validate_output_feature_exists(self) -> ConceptDefinition:
        if self.output_feature not in self.features:
            raise ValueError(
                f"output_feature '{self.output_feature}' is not defined in features. "
                f"Available features: {list(self.features)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_categorical_labels(self) -> ConceptDefinition:
        if self.output_type == MemintelType.CATEGORICAL and not self.labels:
            raise ValueError(
                "labels must be declared for concepts with output_type='categorical'"
            )
        if self.output_type != MemintelType.CATEGORICAL and self.labels is not None:
            raise ValueError(
                "labels may only be declared on categorical concepts"
            )
        return self

    @field_validator("output_type")
    @classmethod
    def _valid_output_type(cls, v: str) -> str:
        # Concepts must produce scalar types (not containers, decisions, or duration).
        # After concept execution, the runtime returns a scalar ConceptResult.value.
        valid = MemintelType.SCALAR_TYPES
        if v not in valid:
            raise ValueError(
                f"'{v}' is not a valid concept output type. "
                f"Concepts must produce one of: {sorted(valid)}"
            )
        return v


# ── Compiled graph models ─────────────────────────────────────────────────────

class GraphNode(BaseModel):
    """
    A node in the compiled execution graph (DAG).

    Produced by the compiler from FeatureNode. Unlike FeatureNode (which uses
    symbolic names), GraphNode uses resolved node_ids for all inputs.

    output_type is set by the TypeChecker — it is the inferred type of this
    node's output, validated against the downstream node's input requirements.

    The node_id is a stable, deterministic identifier derived from the node's
    position in the DAG (e.g. SHA-256 of the canonical serialisation of op +
    inputs + params). Stable node_ids ensure ir_hash stability across machines.
    """
    node_id: str
    op: str
    inputs: dict[str, str | list[str]]   # slot_name → upstream node_id(s)
    params: dict[str, Any] = Field(default_factory=dict)
    output_type: str                      # inferred by TypeChecker


class GraphEdge(BaseModel):
    """
    A directed edge in the compiled execution DAG.

    from_node_id → to_node_id, feeding the named input slot.
    The edge list fully describes the DAG topology and is the input to the
    topological sort that produces the deterministic execution order.
    """
    from_node_id: str
    to_node_id: str
    input_slot: str    # which input of to_node this edge feeds


class ExecutionGraph(BaseModel):
    """
    The compiled execution graph for a concept. Stored in the execution_graphs
    table (graph_body JSONB column).

    graph_id — UUID generated at compile time. Stable: recompiling the same
      (concept_id, version) always produces the same graph_id because it is
      derived from the concept_id + version (not from graph content).

    ir_hash — SHA-256 of the canonical graph JSON (sorted keys, stable field
      order). Must be deterministic: same definition → same hash, always, on
      any machine. Used for audit verification at execution time.

    Graph replacement invariant (from persistence-schema.md §3.5):
      Recompiling an unchanged definition MUST produce the same ir_hash.
      If the existing graph has a different ir_hash → CompilerInvariantError.
      GraphStore.store() enforces this check before any overwrite.

    topological_order — node_ids in the deterministic execution order produced
      by the topological sort. Fixed at compile time; the runtime must follow
      this order and must not reorder nodes based on data availability.

    parallelizable_groups — groups of node_ids that can execute concurrently
      (they have no dependency relationship within their group). Also fixed at
      compile time. The runtime may parallelize within a group.
    """
    graph_id: str
    concept_id: str
    version: str
    ir_hash: str                              # SHA-256 of canonical graph JSON
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    topological_order: list[str]              # node_ids in execution order
    parallelizable_groups: list[list[str]]    # groups of concurrent node_ids
    output_node_id: str                       # node whose output is the concept result
    output_type: str                          # the concept's resolved output type
    created_at: datetime | None = None


# ── Compiler output views (not persisted) ─────────────────────────────────────

class SemanticGraph(BaseModel):
    """
    The semantic view of a concept, produced by POST /compile/semantic.

    semantic_hash is stable for semantically equivalent concepts: the same
    underlying computation always produces the same hash regardless of the
    authoring structure (canonical reduction is applied before hashing).

    Two concepts with the same semantic_hash are semantically equivalent —
    they compute the same function, even if their DAG structures differ.

    Used for:
      - Deduplication in the feature registry
      - Equivalence detection before promotion
      - Detecting breaking vs non-breaking version changes
    """
    concept_id: str
    version: str
    semantic_hash: str
    features: list[str]               # canonical feature ids after reduction
    input_primitives: list[str]       # primitive names the concept depends on
    equivalences: list[str] = Field(default_factory=list)
    # Each entry is "concept_id:version" of an equivalent concept


class ExecutionPlan(BaseModel):
    """
    The explain plan produced by POST /compile/explain-plan.

    The SQL EXPLAIN equivalent for a concept execution graph. Returns the
    execution order and parallelizable groups without running the graph.

    Use before executing a concept to understand:
      - Which primitives will be fetched
      - Which nodes can run concurrently
      - The total node count and critical path
    """
    concept_id: str
    version: str
    node_count: int
    execution_order: list[str]              # node_ids in topological order
    parallelizable_groups: list[list[str]]  # groups of concurrent node_ids
    primitive_fetches: list[str]            # primitive names fetched at runtime
    critical_path_length: int               # longest dependency chain (node count)


# ── Registry response models ───────────────────────────────────────────────────

class DefinitionResponse(BaseModel):
    """
    Response returned by all definition write operations:
      POST /registry/definitions  (register)
      POST /registry/definitions/{id}/deprecate
      POST /registry/definitions/{id}/promote
      POST /definitions/concepts
      POST /definitions/conditions
      POST /definitions/primitives

    Used for all definition_type values: concept, condition, action, primitive.

    meaning_hash — semantic hash (concepts only). Stable for semantically
      equivalent concepts regardless of authoring structure.
    ir_hash — execution graph hash (concepts only). Same definition version
      always produces the same ir_hash (compilation invariant).
    """
    definition_id: str
    version: str
    definition_type: str       # concept | condition | action | primitive
    namespace: Namespace
    meaning_hash: str | None = None
    ir_hash: str | None = None
    deprecated: bool = False
    deprecated_at: datetime | None = None
    replacement_version: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class VersionSummary(BaseModel):
    """
    Summary of a single version of a definition, returned by
    GET /registry/definitions/{id}/versions.

    Versions are returned newest-first (by created_at DESC). The created_at
    field is the ordering key — version strings are NOT lexicographically
    sorted (they may be semver, date-stamped, or arbitrary strings).
    """
    version: str
    created_at: datetime | None = None
    deprecated: bool = False
    meaning_hash: str | None = None
    ir_hash: str | None = None


class SearchResult(BaseModel):
    """
    Paginated list of definitions, returned by GET /registry/definitions
    and GET /registry/search.

    Cursor-based pagination: next_cursor is the last definition_id seen.
    Pass as ?cursor= on the next request.
    """
    items: list[DefinitionResponse]
    has_more: bool
    next_cursor: str | None = None
    total_count: int


class SemanticDiffResult(BaseModel):
    """
    Result of comparing two versions of a definition, returned by
    GET /registry/definitions/{id}/semantic-diff.

    equivalence_status values:
      equivalent — same semantic_hash; meaning is identical; safe to promote.
      compatible — meaning changed but backward-compatible; review recommended.
      breaking   — downstream conditions/actions may be invalidated; governance
                   required before promotion.
      unknown    — could not determine; treat as breaking.

    The caller should block on 'breaking' status before promoting to a higher
    namespace. See py-instructions.md for the governance pattern.
    """
    definition_id: str
    version_from: str
    version_to: str
    equivalence_status: str    # equivalent | compatible | breaking | unknown
    summary: str
    changes: list[dict[str, Any]] = Field(default_factory=list)


class LineageResult(BaseModel):
    """
    Definition lineage, returned by GET /registry/definitions/{id}/lineage.

    Traces the version history and namespace promotion path of a definition.
    Each entry in the chain is a (version, namespace) pair in creation order.
    """
    definition_id: str
    chain: list[VersionSummary]
    promoted_to: dict[str, str] = Field(default_factory=dict)
    # Maps version → namespace it was promoted to (e.g. {"1.2": "org"})
