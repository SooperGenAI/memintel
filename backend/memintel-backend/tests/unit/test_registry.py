"""
tests/unit/test_registry.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for DefinitionRegistry and FeatureRegistry.

Coverage:
  1. 409 on duplicate id+version
  2. Immutability: cannot update a registered definition
  3. versions() returns newest-first
  4. promote() to global fails without elevated_key
  5. semantic_diff returns 'equivalent' for identical definitions
  6. register() rejects unvalidated definitions (freezing check)

Also covers:
  7. get() returns correct body; get() raises 404 for missing definition
  8. deprecate() marks version as deprecated; existing refs still resolve
  9. Feature registry: on_duplicate='warn'   → registers + returns duplicates
 10. Feature registry: on_duplicate='reject' → ConflictError (HTTP 409)
 11. Feature registry: on_duplicate='merge'  → returns existing feature

Test isolation: every test creates its own MockDefinitionStore, DefinitionRegistry,
and FeatureRegistry.  No shared mutable state between tests.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import pytest

from app.models.concept import (
    ConceptDefinition,
    DefinitionResponse,
    SearchResult,
    VersionSummary,
)
from app.models.errors import (
    ConflictError,
    MemintelError,
    NotFoundError,
    ValidationError,
)
from app.models.task import Namespace
from app.registry.definitions import DefinitionRegistry, _compute_meaning_hash
from app.registry.features import (
    FeatureExecution,
    FeatureMeaning,
    FeatureRegistry,
    RegisteredFeature,
)


# ── Mock store ────────────────────────────────────────────────────────────────

class MockDefinitionStore:
    """
    In-memory DefinitionStore for unit testing.

    Simulates the persistence contract without a database:
      - register() raises ConflictError on duplicate (definition_id, version).
      - versions() returns summaries newest-first by insertion order (LIFO).
      - promote() raises MemintelError(AUTH_ERROR) for global without elevated_key.
      - get_metadata() returns DefinitionResponse or None.
      - deprecate() marks deprecated=True on the stored response.
    """

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], DefinitionResponse] = {}
        self._bodies: dict[tuple[str, str], dict] = {}
        self._insert_order: list[tuple[str, str]] = []  # for newest-first ordering

    async def register(
        self,
        definition_id: str,
        version: str,
        definition_type: str,
        namespace: str,
        body: dict[str, Any],
        meaning_hash: str | None = None,
        ir_hash: str | None = None,
    ) -> DefinitionResponse:
        key = (definition_id, version)
        if key in self._rows:
            raise ConflictError(
                f"Definition '{definition_id}' version '{version}' is already registered. "
                "Definitions are immutable — create a new version instead.",
                location=f"{definition_id}:{version}",
            )
        ts = datetime.now(timezone.utc)
        response = DefinitionResponse(
            definition_id=definition_id,
            version=version,
            definition_type=definition_type,
            namespace=Namespace(namespace),
            meaning_hash=meaning_hash,
            ir_hash=ir_hash,
            deprecated=False,
            created_at=ts,
            updated_at=ts,
        )
        self._rows[key] = response
        self._bodies[key] = body
        self._insert_order.append(key)
        return response

    async def get(self, definition_id: str, version: str) -> dict[str, Any] | None:
        return self._bodies.get((definition_id, version))

    async def get_metadata(
        self, definition_id: str, version: str
    ) -> DefinitionResponse | None:
        return self._rows.get((definition_id, version))

    async def versions(self, definition_id: str) -> list[VersionSummary]:
        # Newest-first: reverse the insertion order, filter by definition_id.
        ordered = [
            key for key in reversed(self._insert_order)
            if key[0] == definition_id
        ]
        return [
            VersionSummary(
                version=key[1],
                created_at=self._rows[key].created_at,
                deprecated=self._rows[key].deprecated,
                meaning_hash=self._rows[key].meaning_hash,
                ir_hash=self._rows[key].ir_hash,
            )
            for key in ordered
        ]

    async def list(
        self,
        definition_type: str | None = None,
        namespace: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> SearchResult:
        items = list(self._rows.values())
        if definition_type:
            items = [i for i in items if i.definition_type == definition_type]
        if namespace:
            items = [i for i in items if i.namespace.value == namespace]
        return SearchResult(items=items[:limit], has_more=False, total_count=len(items))

    async def deprecate(
        self,
        definition_id: str,
        version: str,
        replacement_version: str | None,
        reason: str,
    ) -> DefinitionResponse:
        key = (definition_id, version)
        if key not in self._rows:
            raise NotFoundError(
                f"Definition '{definition_id}' version '{version}' not found.",
                location=f"{definition_id}:{version}",
            )
        old = self._rows[key]
        updated = old.model_copy(update={
            "deprecated": True,
            "replacement_version": replacement_version,
        })
        self._rows[key] = updated
        return updated

    async def promote(
        self,
        definition_id: str,
        version: str,
        from_namespace: str,
        to_namespace: str,
        elevated_key: bool = False,
    ) -> DefinitionResponse:
        from app.models.errors import ErrorType
        if to_namespace == "global" and not elevated_key:
            raise MemintelError(
                ErrorType.AUTH_ERROR,
                f"Promoting to namespace '{to_namespace}' requires elevated privileges.",
                suggestion="Pass elevated_key=True with a valid admin API key.",
            )
        key = (definition_id, version)
        if key not in self._rows:
            raise NotFoundError(
                f"Definition '{definition_id}' version '{version}' not found.",
                location=f"{definition_id}:{version}",
            )
        old = self._rows[key]
        updated = old.model_copy(update={"namespace": Namespace(to_namespace)})
        self._rows[key] = updated
        return updated


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_registry(store: MockDefinitionStore | None = None) -> DefinitionRegistry:
    return DefinitionRegistry(store=store or MockDefinitionStore())


def _minimal_concept(
    concept_id: str = "org.churn_risk",
    version: str = "1.0",
    namespace: str = "org",
) -> ConceptDefinition:
    """
    Minimal valid ConceptDefinition used across tests.

    One primitive (float, zero policy → non-nullable), one normalize feature.
    """
    from app.models.concept import PrimitiveRef, FeatureNode
    from app.models.result import MissingDataPolicy
    return ConceptDefinition(
        concept_id=concept_id,
        version=version,
        namespace=Namespace(namespace),
        output_type="float",
        primitives={
            "churn_prob": PrimitiveRef(
                type="float",
                missing_data_policy=MissingDataPolicy.ZERO,
            ),
        },
        features={
            "score": FeatureNode(op="normalize", inputs={"input": "churn_prob"}),
        },
        output_feature="score",
    )


def run(coro):
    """Convenience helper to run a coroutine in tests."""
    return asyncio.run(coro)


# ── 1. HTTP 409 on duplicate id+version ──────────────────────────────────────

class TestDuplicateRegistration:
    def test_409_on_duplicate_id_version(self):
        store    = MockDefinitionStore()
        registry = _make_registry(store)
        concept  = _minimal_concept()

        # First registration succeeds.
        run(registry.register(concept, "org"))

        # Second registration of the same (id, version) → ConflictError.
        with pytest.raises(ConflictError) as exc_info:
            run(registry.register(concept, "org"))

        assert "already registered" in str(exc_info.value).lower() or \
               exc_info.value.error_type.value == "conflict"

    def test_different_versions_do_not_conflict(self):
        store    = MockDefinitionStore()
        registry = _make_registry(store)

        v1 = _minimal_concept(version="1.0")
        v2 = _minimal_concept(version="2.0")

        r1 = run(registry.register(v1, "org"))
        r2 = run(registry.register(v2, "org"))

        assert r1.version == "1.0"
        assert r2.version == "2.0"

    def test_different_ids_do_not_conflict(self):
        store    = MockDefinitionStore()
        registry = _make_registry(store)

        a = _minimal_concept(concept_id="org.concept_a")
        b = _minimal_concept(concept_id="org.concept_b")

        r_a = run(registry.register(a, "org"))
        r_b = run(registry.register(b, "org"))

        assert r_a.definition_id == "org.concept_a"
        assert r_b.definition_id == "org.concept_b"


# ── 2. Immutability — cannot update a registered definition ──────────────────

class TestImmutability:
    def test_registered_definition_body_unchanged_on_conflict(self):
        store    = MockDefinitionStore()
        registry = _make_registry(store)
        concept  = _minimal_concept()

        run(registry.register(concept, "org"))
        original_body = run(registry.get("org.churn_risk", "1.0"))

        # Attempt to re-register (same id+version but would-be different body).
        with pytest.raises(ConflictError):
            run(registry.register(concept, "personal"))  # even different namespace

        # Body in store is unchanged — no mutation happened.
        body_after = run(registry.get("org.churn_risk", "1.0"))
        assert body_after == original_body

    def test_registered_definition_metadata_unchanged_on_conflict(self):
        store    = MockDefinitionStore()
        registry = _make_registry(store)
        concept  = _minimal_concept()

        r1 = run(registry.register(concept, "org"))
        original_namespace = r1.namespace

        with pytest.raises(ConflictError):
            run(registry.register(concept, "global"))

        # The stored namespace was not mutated.
        meta = run(store.get_metadata("org.churn_risk", "1.0"))
        assert meta.namespace == original_namespace


# ── 3. versions() returns newest-first ───────────────────────────────────────

class TestVersionsOrder:
    def test_versions_newest_first(self):
        store    = MockDefinitionStore()
        registry = _make_registry(store)

        v1 = _minimal_concept(version="1.0")
        v2 = _minimal_concept(version="2.0")
        v3 = _minimal_concept(version="3.0")

        run(registry.register(v1, "org"))
        run(registry.register(v2, "org"))
        run(registry.register(v3, "org"))

        result = run(registry.versions("org.churn_risk"))
        assert [s.version for s in result.versions] == ["3.0", "2.0", "1.0"]

    def test_versions_result_has_correct_definition_id(self):
        store    = MockDefinitionStore()
        registry = _make_registry(store)

        run(registry.register(_minimal_concept(version="1.0"), "org"))

        result = run(registry.versions("org.churn_risk"))
        assert result.definition_id == "org.churn_risk"

    def test_versions_raises_404_for_unknown_id(self):
        registry = _make_registry()
        with pytest.raises(NotFoundError):
            run(registry.versions("org.does_not_exist"))

    def test_versions_single_entry_list(self):
        store    = MockDefinitionStore()
        registry = _make_registry(store)

        run(registry.register(_minimal_concept(version="1.0"), "org"))
        result = run(registry.versions("org.churn_risk"))

        assert len(result.versions) == 1
        assert result.versions[0].version == "1.0"


# ── 4. promote() to global fails without elevated_key ────────────────────────

class TestPromoteToGlobal:
    def test_promote_to_global_without_key_raises_auth_error(self):
        from app.models.errors import ErrorType

        store    = MockDefinitionStore()
        registry = _make_registry(store)
        concept  = _minimal_concept(concept_id="org.churn_risk", version="1.0")

        run(registry.register(concept, "org"))

        with pytest.raises(MemintelError) as exc_info:
            run(registry.promote(
                "org.churn_risk", "1.0",
                from_namespace="org",
                to_namespace="global",
                elevated_key=False,
            ))

        assert exc_info.value.error_type == ErrorType.AUTH_ERROR

    def test_promote_to_global_with_key_succeeds(self):
        store    = MockDefinitionStore()
        registry = _make_registry(store)
        concept  = _minimal_concept(concept_id="org.churn_risk", version="1.0")

        run(registry.register(concept, "org"))
        result = run(registry.promote(
            "org.churn_risk", "1.0",
            from_namespace="org",
            to_namespace="global",
            elevated_key=True,
        ))

        assert result.namespace == Namespace.GLOBAL

    def test_promote_downgrade_rejected(self):
        from app.models.errors import ErrorType

        store    = MockDefinitionStore()
        registry = _make_registry(store)
        concept  = _minimal_concept(concept_id="org.churn_risk", version="1.0")

        run(registry.register(concept, "org"))

        with pytest.raises(MemintelError) as exc_info:
            run(registry.promote(
                "org.churn_risk", "1.0",
                from_namespace="org",
                to_namespace="team",  # downgrade: org → team not allowed
                elevated_key=True,
            ))

        assert exc_info.value.error_type == ErrorType.SEMANTIC_ERROR


# ── 5. semantic_diff returns 'equivalent' for identical definitions ───────────

class TestSemanticDiff:
    def test_equivalent_for_identical_meaning_hash(self):
        store    = MockDefinitionStore()
        registry = _make_registry(store)
        concept  = _minimal_concept()

        # Register two versions with the same meaning content.
        # _compute_meaning_hash ignores concept_id and version.
        shared_hash = _compute_meaning_hash(concept)

        run(store.register(
            "org.churn_risk", "1.0", "concept", "org",
            body=concept.model_dump(), meaning_hash=shared_hash,
        ))
        body_v2 = concept.model_copy(update={"version": "2.0"}).model_dump()
        run(store.register(
            "org.churn_risk", "2.0", "concept", "org",
            body=body_v2,
            meaning_hash=shared_hash,
        ))

        diff = run(registry.semantic_diff("org.churn_risk", "1.0", "2.0"))

        assert diff.equivalence_status == "equivalent"
        assert diff.definition_id == "org.churn_risk"
        assert diff.version_from == "1.0"
        assert diff.version_to   == "2.0"

    def test_compatible_for_different_meaning_hash(self):
        store    = MockDefinitionStore()
        registry = _make_registry(store)

        concept_v1 = _minimal_concept(version="1.0")
        concept_v2 = _minimal_concept(version="2.0")

        h1 = _compute_meaning_hash(concept_v1)
        h2 = "different_hash_" + "a" * 32   # force a different hash

        run(store.register("org.churn_risk", "1.0", "concept", "org",
                           body=concept_v1.model_dump(), meaning_hash=h1))
        run(store.register("org.churn_risk", "2.0", "concept", "org",
                           body=concept_v2.model_dump(), meaning_hash=h2))

        diff = run(registry.semantic_diff("org.churn_risk", "1.0", "2.0"))

        assert diff.equivalence_status == "compatible"

    def test_unknown_when_no_meaning_hash(self):
        store    = MockDefinitionStore()
        registry = _make_registry(store)
        concept  = _minimal_concept()

        # Register without meaning_hash.
        run(store.register("org.churn_risk", "1.0", "concept", "org",
                           body=concept.model_dump(), meaning_hash=None))
        run(store.register("org.churn_risk", "2.0", "concept", "org",
                           body=concept.model_dump(), meaning_hash=None))

        diff = run(registry.semantic_diff("org.churn_risk", "1.0", "2.0"))

        assert diff.equivalence_status == "unknown"

    def test_diff_raises_404_for_missing_version(self):
        store    = MockDefinitionStore()
        registry = _make_registry(store)
        concept  = _minimal_concept()

        run(store.register("org.churn_risk", "1.0", "concept", "org",
                           body=concept.model_dump()))

        with pytest.raises(NotFoundError):
            run(registry.semantic_diff("org.churn_risk", "1.0", "9.9"))

    def test_auto_computed_hash_identical_for_same_content(self):
        store    = MockDefinitionStore()
        registry = _make_registry(store)
        concept  = _minimal_concept()

        # register() auto-computes meaning_hash — both versions use same body.
        run(registry.register(_minimal_concept(version="1.0"), "org"))
        run(registry.register(_minimal_concept(version="2.0"), "org"))

        diff = run(registry.semantic_diff("org.churn_risk", "1.0", "2.0"))

        assert diff.equivalence_status == "equivalent"

    def test_breaking_for_divergent_ir_hash(self):
        """
        semantic_diff() returns 'breaking' when both meaning_hash and ir_hash
        differ between two versions.
        """
        store    = MockDefinitionStore()
        registry = _make_registry(store)
        concept  = _minimal_concept()

        h1 = _compute_meaning_hash(concept)
        h2 = "different_meaning_" + "b" * 28   # different meaning_hash

        run(store.register(
            "org.churn_risk", "1.0", "concept", "org",
            body=concept.model_dump(),
            meaning_hash=h1,
            ir_hash="ir_hash_v1_" + "a" * 53,
        ))
        run(store.register(
            "org.churn_risk", "2.0", "concept", "org",
            body=concept.model_copy(update={"version": "2.0"}).model_dump(),
            meaning_hash=h2,
            ir_hash="ir_hash_v2_" + "b" * 53,   # different ir_hash
        ))

        diff = run(registry.semantic_diff("org.churn_risk", "1.0", "2.0"))

        assert diff.equivalence_status == "breaking"
        assert "breaking" in diff.summary.lower()

    def test_compatible_when_ir_hash_same_despite_different_meaning_hash(self):
        """
        semantic_diff() returns 'compatible' (not 'breaking') when meaning_hash
        differs but ir_hash is the same — execution graph unchanged.
        """
        store    = MockDefinitionStore()
        registry = _make_registry(store)
        concept  = _minimal_concept()

        h1 = _compute_meaning_hash(concept)
        h2 = "different_meaning_" + "c" * 28
        shared_ir = "shared_ir_hash_" + "x" * 49

        run(store.register(
            "org.churn_risk", "1.0", "concept", "org",
            body=concept.model_dump(),
            meaning_hash=h1,
            ir_hash=shared_ir,
        ))
        run(store.register(
            "org.churn_risk", "2.0", "concept", "org",
            body=concept.model_copy(update={"version": "2.0"}).model_dump(),
            meaning_hash=h2,
            ir_hash=shared_ir,   # same ir_hash
        ))

        diff = run(registry.semantic_diff("org.churn_risk", "1.0", "2.0"))

        assert diff.equivalence_status == "compatible"


# ── Promote blocked by 'breaking' semantic_diff ───────────────────────────────

class TestPromoteBreaking:
    """
    Tests that promote() is blocked when semantic_diff() returns 'breaking',
    and succeeds when the diff is 'compatible'.
    """

    def test_promote_raises_on_breaking_diff(self):
        """
        promote() must raise MemintelError(SEMANTIC_ERROR) when the semantic diff
        between the two registered versions is 'breaking'.
        """
        from app.models.errors import ErrorType

        store    = MockDefinitionStore()
        registry = _make_registry(store)
        concept  = _minimal_concept()

        h1 = _compute_meaning_hash(concept)
        h2 = "different_meaning_" + "d" * 28

        # v1 — baseline
        run(store.register(
            "org.churn_risk", "1.0", "concept", "org",
            body=concept.model_dump(),
            meaning_hash=h1,
            ir_hash="ir_v1_" + "a" * 58,
        ))
        # v2 — divergent meaning_hash AND ir_hash → 'breaking'
        run(store.register(
            "org.churn_risk", "2.0", "concept", "org",
            body=concept.model_copy(update={"version": "2.0"}).model_dump(),
            meaning_hash=h2,
            ir_hash="ir_v2_" + "b" * 58,
        ))

        with pytest.raises(MemintelError) as exc_info:
            run(registry.promote(
                "org.churn_risk", "2.0",
                from_namespace="org",
                to_namespace="global",
                elevated_key=True,
            ))

        assert exc_info.value.error_type == ErrorType.SEMANTIC_ERROR
        assert "breaking" in str(exc_info.value).lower()

    def test_promote_succeeds_on_compatible_diff(self):
        """
        promote() succeeds when semantic_diff() returns 'compatible' — meaning
        changed but the compiled execution graph (ir_hash) is the same.
        """
        store    = MockDefinitionStore()
        registry = _make_registry(store)
        concept  = _minimal_concept()

        h1 = _compute_meaning_hash(concept)
        h2 = "different_meaning_" + "e" * 28
        shared_ir = "shared_ir_" + "y" * 54

        # v1 — baseline
        run(store.register(
            "org.churn_risk", "1.0", "concept", "org",
            body=concept.model_dump(),
            meaning_hash=h1,
            ir_hash=shared_ir,
        ))
        # v2 — different meaning_hash but same ir_hash → 'compatible'
        run(store.register(
            "org.churn_risk", "2.0", "concept", "org",
            body=concept.model_copy(update={"version": "2.0"}).model_dump(),
            meaning_hash=h2,
            ir_hash=shared_ir,
        ))

        result = run(registry.promote(
            "org.churn_risk", "2.0",
            from_namespace="org",
            to_namespace="global",
            elevated_key=True,
        ))

        assert result.namespace == Namespace.GLOBAL


# ── 6. register() rejects unvalidated definitions (freezing check) ───────────

class TestFreezingCheck:
    def test_register_rejects_definition_with_unresolved_reference(self):
        """
        A concept feature that references a primitive not declared in `primitives`
        should fail validate_schema() and be rejected by register() before the store
        is touched.
        """
        from app.models.concept import FeatureNode, PrimitiveRef
        from app.models.result import MissingDataPolicy

        store    = MockDefinitionStore()
        registry = _make_registry(store)

        bad_concept = ConceptDefinition(
            concept_id="org.bad_concept",
            version="1.0",
            namespace=Namespace.ORG,
            output_type="float",
            primitives={
                "real_prim": PrimitiveRef(
                    type="float",
                    missing_data_policy=MissingDataPolicy.ZERO,
                ),
            },
            features={
                "score": FeatureNode(
                    op="normalize",
                    # References 'ghost_prim' which is NOT in primitives — schema error.
                    inputs={"input": "ghost_prim"},
                ),
            },
            output_feature="score",
        )

        with pytest.raises(ValidationError):
            run(registry.register(bad_concept, "org"))

        # Store must NOT have been touched.
        stored = run(store.get("org.bad_concept", "1.0"))
        assert stored is None

    def test_register_rejects_definition_with_type_error(self):
        """
        A concept with a type mismatch in the DAG (e.g. passing a time_series
        to an operator that expects a scalar) should fail validate_types().
        """
        from app.models.concept import FeatureNode, PrimitiveRef
        from app.models.result import MissingDataPolicy

        store    = MockDefinitionStore()
        registry = _make_registry(store)

        # 'normalize' expects float input, but 'add' requires 'a' and 'b'.
        # Here we deliberately create a valid-looking concept with a missing input slot.
        bad_concept = ConceptDefinition(
            concept_id="org.type_error_concept",
            version="1.0",
            namespace=Namespace.ORG,
            output_type="float",
            primitives={
                "prim_a": PrimitiveRef(type="float",
                                       missing_data_policy=MissingDataPolicy.ZERO),
                "prim_b": PrimitiveRef(type="float",
                                       missing_data_policy=MissingDataPolicy.ZERO),
            },
            features={
                "result": FeatureNode(
                    op="add",
                    # Missing 'b' input — validate_types will raise.
                    inputs={"a": "prim_a"},
                ),
            },
            output_feature="result",
        )

        with pytest.raises(ValidationError):
            run(registry.register(bad_concept, "org"))

        stored = run(store.get("org.type_error_concept", "1.0"))
        assert stored is None

    def test_valid_definition_passes_freezing_and_is_stored(self):
        store    = MockDefinitionStore()
        registry = _make_registry(store)
        concept  = _minimal_concept()

        result = run(registry.register(concept, "org"))

        assert result.definition_id == "org.churn_risk"
        stored = run(store.get("org.churn_risk", "1.0"))
        assert stored is not None


# ── 7. get() ──────────────────────────────────────────────────────────────────

class TestGet:
    def test_get_returns_body(self):
        store    = MockDefinitionStore()
        registry = _make_registry(store)
        concept  = _minimal_concept()

        run(registry.register(concept, "org"))
        body = run(registry.get("org.churn_risk", "1.0"))

        assert body is not None
        assert body["concept_id"] == "org.churn_risk"

    def test_get_raises_404_for_missing(self):
        registry = _make_registry()
        with pytest.raises(NotFoundError):
            run(registry.get("org.does_not_exist", "1.0"))

    def test_get_requires_exact_version(self):
        store    = MockDefinitionStore()
        registry = _make_registry(store)

        run(registry.register(_minimal_concept(version="1.0"), "org"))

        # Version "2.0" was never registered.
        with pytest.raises(NotFoundError):
            run(registry.get("org.churn_risk", "2.0"))


# ── 8. deprecate() ───────────────────────────────────────────────────────────

class TestDeprecate:
    def test_deprecate_marks_version_deprecated(self):
        store    = MockDefinitionStore()
        registry = _make_registry(store)
        concept  = _minimal_concept()

        run(registry.register(concept, "org"))
        result = run(registry.deprecate(
            "org.churn_risk", "1.0",
            replacement_version="2.0",
            reason="Superseded by v2.",
        ))

        assert result.deprecated is True
        assert result.replacement_version == "2.0"

    def test_deprecate_does_not_delete_body(self):
        store    = MockDefinitionStore()
        registry = _make_registry(store)
        concept  = _minimal_concept()

        run(registry.register(concept, "org"))
        run(registry.deprecate("org.churn_risk", "1.0",
                               replacement_version=None, reason="old"))

        # Body still resolvable after deprecation.
        body = run(registry.get("org.churn_risk", "1.0"))
        assert body is not None

    def test_deprecate_raises_404_for_missing(self):
        registry = _make_registry()
        with pytest.raises(NotFoundError):
            run(registry.deprecate("org.ghost", "1.0",
                                   replacement_version=None, reason=""))


# ── 9–11. Feature registry ────────────────────────────────────────────────────

def _make_feature(
    feature_id: str = "org.activity_rate",
    description: str = "User activity rate",
    semantic_type: str = "rate",
) -> RegisteredFeature:
    return RegisteredFeature(
        id=feature_id,
        namespace=Namespace.ORG,
        version="1.0",
        meaning=FeatureMeaning(
            description=description,
            semantic_type=semantic_type,
            unit="per_day",
        ),
        execution=FeatureExecution(
            op="rate_of_change",
            input="raw_activity",
            window="7d",
        ),
    )


class TestFeatureRegistryWarn:
    def test_warn_registers_and_reports_duplicate(self):
        store    = MockDefinitionStore()
        registry = FeatureRegistry(store)

        feat_a = _make_feature("org.activity_rate_a")
        feat_b = _make_feature("org.activity_rate_b")  # same meaning → same hash

        run(registry.register_feature(feat_a, on_duplicate="warn"))
        result = run(registry.register_feature(feat_b, on_duplicate="warn"))

        assert result.status == "warn"
        assert "org.activity_rate_a" in result.duplicates

    def test_no_duplicate_status_is_registered(self):
        store    = MockDefinitionStore()
        registry = FeatureRegistry(store)

        feat = _make_feature()
        result = run(registry.register_feature(feat, on_duplicate="warn"))

        assert result.status == "registered"
        assert result.duplicates == []

    def test_meaning_hash_is_deterministic(self):
        store    = MockDefinitionStore()
        registry = FeatureRegistry(store)

        feat = _make_feature()
        r1 = run(registry.register_feature(feat, on_duplicate="warn"))

        # Same meaning content → same hash.
        from app.registry.features import _compute_meaning_hash as fhash, FeatureMeaning
        expected = fhash(FeatureMeaning(
            description="User activity rate", semantic_type="rate", unit="per_day"
        ))
        assert r1.meaning_hash == expected


class TestFeatureRegistryReject:
    def test_reject_raises_409_on_duplicate(self):
        store    = MockDefinitionStore()
        registry = FeatureRegistry(store)

        feat_a = _make_feature("org.feat_a")
        feat_b = _make_feature("org.feat_b")  # same meaning

        run(registry.register_feature(feat_a, on_duplicate="reject"))

        with pytest.raises(ConflictError):
            run(registry.register_feature(feat_b, on_duplicate="reject"))

    def test_reject_no_duplicate_succeeds(self):
        store    = MockDefinitionStore()
        registry = FeatureRegistry(store)

        feat = _make_feature()
        result = run(registry.register_feature(feat, on_duplicate="reject"))

        assert result.status == "registered"


class TestFeatureRegistryMerge:
    def test_merge_returns_existing_feature_id(self):
        store    = MockDefinitionStore()
        registry = FeatureRegistry(store)

        feat_a = _make_feature("org.canonical")
        feat_b = _make_feature("org.duplicate")  # same meaning

        run(registry.register_feature(feat_a, on_duplicate="merge"))
        result = run(registry.register_feature(feat_b, on_duplicate="merge"))

        assert result.status == "merged"
        assert result.id == "org.canonical"

    def test_merge_does_not_register_new_feature(self):
        store    = MockDefinitionStore()
        registry = FeatureRegistry(store)

        feat_a = _make_feature("org.canonical")
        feat_b = _make_feature("org.duplicate")

        run(registry.register_feature(feat_a, on_duplicate="merge"))
        run(registry.register_feature(feat_b, on_duplicate="merge"))

        # Only one feature should be in the store.
        listing = run(store.list(definition_type="feature"))
        assert len(listing.items) == 1
        assert listing.items[0].definition_id == "org.canonical"


# ── FIX 4: list_features returns non-empty body ───────────────────────────────

class TestListFeaturesReturnsBody:
    """
    RegistryService.list_features() must return non-empty body dicts for
    registered features — not hardcoded `{}`.

    Uses a minimal fake pool that returns a pre-seeded row for the direct
    pool query introduced by FIX 4.
    """

    def _make_pool(self, rows: list[dict]) -> Any:
        """Return a fake pool whose fetch() returns the given rows."""
        import datetime

        class _FakePool:
            async def fetchrow(self, *a, **kw): return None
            async def fetch(self, query: str, *args: Any) -> list[dict]:
                return rows
            async def fetchval(self, *a, **kw): return None
            async def execute(self, *a, **kw): pass

        return _FakePool()

    def test_list_features_returns_body_from_db(self):
        """
        list_features() must include the feature body fetched from the DB.
        """
        import asyncio, json, datetime
        from app.services.registry import RegistryService

        feature_body = {
            "feature_id": "org.my_feature",
            "version": "1.0",
            "description": "A test feature",
        }
        fake_row = {
            "definition_id": "org.my_feature",
            "version": "1.0",
            "meaning_hash": "abc123",
            "body": json.dumps(feature_body),
            "created_at": datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
        }

        pool = self._make_pool([fake_row])
        svc = RegistryService(pool=pool)

        result = asyncio.run(svc.list_features())

        assert len(result["items"]) == 1
        item = result["items"][0]
        assert item["feature_id"] == "org.my_feature"
        assert item["body"] != {}, (
            "list_features() must return the real body, not hardcoded {}"
        )
        assert item["body"] == feature_body

    def test_list_features_empty_when_no_features(self):
        """list_features() returns empty items list when no features exist."""
        import asyncio
        from app.services.registry import RegistryService

        pool = self._make_pool([])
        svc = RegistryService(pool=pool)

        result = asyncio.run(svc.list_features())

        assert result["items"] == []
        assert result["has_more"] is False
