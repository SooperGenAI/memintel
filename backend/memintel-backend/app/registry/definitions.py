"""
app/registry/definitions.py
──────────────────────────────────────────────────────────────────────────────
DefinitionRegistry — service layer for the Internal Platform registry.

Wraps DefinitionStore (persistence) and Validator (compiler) to enforce
the full governance contract defined in the Internal Platform API Reference
(Section 2) and py-instructions.md "Registry Governance".

Key invariants
──────────────
Immutability:
  Definitions are frozen on registration.  The (id, version) pair is
  permanent.  Updates always require a new version.  register() raises
  ConflictError (HTTP 409) if the pair already exists.

Definition freezing:
  Before persisting, register() runs validate_schema() and validate_types()
  on the definition body.  This prevents garbage entering the registry —
  the compiler must sign off before the store is touched.

Namespace promotion path:
  personal → team → org → global.
  Promoting to 'global' requires elevated_key=True on the call.
  promote() additionally runs semantic_diff and blocks on 'breaking' status.

Semantic diff:
  semantic_diff() compares two versions at the meaning layer by comparing
  their stored meaning_hash fields.  Same hash → 'equivalent'.
  Different hash → 'compatible' (deeper analysis is a compile-service concern).
  Missing hash → 'unknown' (treat as breaking).

No implicit latest:
  get() requires an explicit version.  There is no latest resolution.

Public API
──────────
register(body, namespace)                → DefinitionResponse
get(definition_id, version)              → dict
list(type, namespace, limit, cursor)     → SearchResult
versions(definition_id)                  → VersionListResult
deprecate(id, version, replacement, why) → DefinitionResponse
promote(id, version, from_ns, to_ns, elevated_key)  → DefinitionResponse
semantic_diff(id, v_from, v_to)         → SemanticDiffResult
"""
from __future__ import annotations

import hashlib
import json
import structlog
from typing import Any

from app.compiler.validator import Validator
from app.models.concept import (
    ConceptDefinition,
    DefinitionResponse,
    SearchResult,
    SemanticDiffResult,
    VersionSummary,
)
from app.models.errors import (
    ErrorType,
    MemintelError,
    NotFoundError,
    ValidationError,
    ValidationErrorItem,
)
from app.models.task import Namespace

log = structlog.get_logger(__name__)

# ── Namespace promotion order ─────────────────────────────────────────────────

_NAMESPACE_RANK: dict[str, int] = {
    "personal": 0,
    "team":     1,
    "org":      2,
    "global":   3,
}


# ── VersionListResult ─────────────────────────────────────────────────────────

class VersionListResult:
    """
    All registered versions of a definition, newest-first.

    definition_id — fully qualified identifier.
    versions      — list[VersionSummary] ordered newest-first by created_at.
    """

    __slots__ = ("definition_id", "versions")

    def __init__(self, definition_id: str, versions: list[VersionSummary]) -> None:
        self.definition_id = definition_id
        self.versions = versions


# ── DefinitionRegistry ────────────────────────────────────────────────────────

class DefinitionRegistry:
    """
    Registry governance service for Memintel definitions.

    Parameters
    ──────────
    store     — DefinitionStore (or duck-typed mock for tests).
    validator — Validator for the definition-freezing check.
                Defaults to a plain Validator() if not injected.
    """

    def __init__(
        self,
        store: Any,
        validator: Validator | None = None,
    ) -> None:
        self._store = store
        self._validator = validator or Validator()

    # ── register ─────────────────────────────────────────────────────────────

    async def register(
        self,
        body: ConceptDefinition | dict[str, Any],
        namespace: str | Namespace,
        definition_type: str = "concept",
        meaning_hash: str | None = None,
        ir_hash: str | None = None,
    ) -> DefinitionResponse:
        """
        Validate and persist a definition.

        Definition-freezing check:
          For concepts, runs validate_schema() then validate_types() before
          touching the store.  Any compiler violation raises ValidationError
          (HTTP 400) and the definition is NOT stored.

        Immutability:
          The store raises ConflictError (HTTP 409) if (definition_id, version)
          already exists.  This method never overwrites an existing definition.

        Parameters
        ──────────
        body            — ConceptDefinition model or raw dict.
        namespace       — Target namespace string or Namespace enum.
        definition_type — 'concept' | 'condition' | 'primitive' | 'action'.
        meaning_hash    — Semantic hash.  Computed from body when not provided.
        ir_hash         — IR hash for compiled concepts (optional).
        """
        ns_str = namespace.value if isinstance(namespace, Namespace) else namespace

        if isinstance(body, ConceptDefinition):
            definition_type = "concept"
            definition_body = body.model_dump()
            definition_id   = body.concept_id
            version         = body.version
            self._freeze_check(body)
            if meaning_hash is None:
                meaning_hash = _compute_meaning_hash(body)
        else:
            definition_body = body
            definition_id = (
                body.get("condition_id")
                or body.get("action_id")
                or body.get("concept_id")
                or body.get("id")
                or ""
            )
            version = body.get("version", "")
            if not definition_id or not version:
                raise MemintelError(
                    ErrorType.SYNTAX_ERROR,
                    "definition body must contain an id field "
                    "(concept_id / condition_id / action_id / id) and 'version'.",
                )
            if definition_type == "concept":
                self._freeze_check_dict(body)
                if meaning_hash is None:
                    try:
                        defn = ConceptDefinition(**body)
                        meaning_hash = _compute_meaning_hash(defn)
                    except Exception:
                        pass

        log.info(
            "registry_register",
            extra={
                "definition_id": definition_id,
                "version": version,
                "namespace": ns_str,
                "definition_type": definition_type,
            },
        )

        return await self._store.register(
            definition_id=definition_id,
            version=version,
            definition_type=definition_type,
            namespace=ns_str,
            body=definition_body,
            meaning_hash=meaning_hash,
            ir_hash=ir_hash,
        )

    # ── get ──────────────────────────────────────────────────────────────────

    async def get(
        self,
        definition_id: str,
        version: str,
    ) -> dict[str, Any]:
        """
        Return the raw body dict for a definition.

        Version is always required — there is no implicit latest resolution.
        Raises NotFoundError (HTTP 404) when the definition does not exist.
        """
        body = await self._store.get(definition_id, version)
        if body is None:
            raise NotFoundError(
                f"Definition '{definition_id}' version '{version}' not found.",
                location=f"{definition_id}:{version}",
                suggestion="Check the id and version, or call versions() to list available versions.",
            )
        return body

    # ── list ─────────────────────────────────────────────────────────────────

    async def list(
        self,
        definition_type: str | None = None,
        namespace: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> SearchResult:
        """
        Return a cursor-paginated list of definitions.

        tags is advisory: the store does not index tags, so filtering by tags
        requires a full body scan not implemented at this layer.
        """
        result = await self._store.list(
            definition_type=definition_type,
            namespace=namespace,
            limit=limit,
            cursor=cursor,
        )
        return result

    # ── versions ─────────────────────────────────────────────────────────────

    async def versions(self, definition_id: str) -> VersionListResult:
        """
        Return all versions of a definition, newest-first (by created_at DESC).

        Raises NotFoundError if the definition_id has no registered versions.
        """
        summaries = await self._store.versions(definition_id)
        if not summaries:
            raise NotFoundError(
                f"No versions found for definition '{definition_id}'.",
                location=definition_id,
            )
        return VersionListResult(definition_id=definition_id, versions=summaries)

    # ── deprecate ────────────────────────────────────────────────────────────

    async def deprecate(
        self,
        definition_id: str,
        version: str,
        replacement_version: str | None = None,
        reason: str = "",
    ) -> DefinitionResponse:
        """
        Mark a version as deprecated.

        Existing references continue to resolve.  New registrations referencing
        this version receive a compiler warning.  Deprecation is idempotent.

        Raises NotFoundError if the definition does not exist.
        """
        result = await self._store.deprecate(
            definition_id=definition_id,
            version=version,
            replacement_version=replacement_version,
            reason=reason,
        )
        log.info(
            "registry_deprecate",
            extra={
                "definition_id": definition_id,
                "version": version,
                "replacement_version": replacement_version,
                "reason": reason,
            },
        )
        return result

    # ── promote ──────────────────────────────────────────────────────────────

    async def promote(
        self,
        definition_id: str,
        version: str,
        from_namespace: str,
        to_namespace: str,
        elevated_key: bool = False,
    ) -> DefinitionResponse:
        """
        Promote a definition to a higher namespace.

        Promotion path: personal → team → org → global.
        Skipping levels is permitted (e.g. personal → org).
        Downgrading (org → team) is rejected.

        Promoting to 'global' requires elevated_key=True.

        Runs semantic_diff before promotion and blocks on 'breaking' status.

        Raises:
          MemintelError(SEMANTIC_ERROR)  — downgrade attempt or breaking diff.
          MemintelError(AUTH_ERROR)      — promoting to 'global' without key.
          NotFoundError                  — source definition not found.
        """
        from_rank = _NAMESPACE_RANK.get(from_namespace, -1)
        to_rank   = _NAMESPACE_RANK.get(to_namespace, -1)

        if to_rank <= from_rank:
            raise MemintelError(
                ErrorType.SEMANTIC_ERROR,
                f"Cannot promote from '{from_namespace}' to '{to_namespace}': "
                "target namespace must be a higher level than source. "
                "Promotion path: personal → team → org → global.",
                suggestion="Choose a higher-level target namespace.",
            )

        await self._check_diff_before_promote(definition_id, version)

        result = await self._store.promote(
            definition_id=definition_id,
            version=version,
            from_namespace=from_namespace,
            to_namespace=to_namespace,
            elevated_key=elevated_key,
        )
        log.info(
            "registry_promote",
            extra={
                "definition_id": definition_id,
                "version": version,
                "from_namespace": from_namespace,
                "to_namespace": to_namespace,
            },
        )
        return result

    # ── semantic_diff ────────────────────────────────────────────────────────

    async def semantic_diff(
        self,
        definition_id: str,
        version_from: str,
        version_to: str,
    ) -> SemanticDiffResult:
        """
        Compare two versions at the meaning layer.

        equivalence_status:
          equivalent — same meaning_hash; meaning unchanged; safe to promote.
          breaking   — meaning_hash differs AND ir_hash diverges; downstream
                       conditions/actions may be invalidated; governance required.
          compatible — meaning_hash differs but ir_hash is the same or absent;
                       meaning changed but backward-compatible; review recommended.
          unknown    — one or both lack a meaning_hash; treat as breaking.

        Raises NotFoundError if either version does not exist.
        """
        meta_from = await self._store.get_metadata(definition_id, version_from)
        meta_to   = await self._store.get_metadata(definition_id, version_to)

        if meta_from is None:
            raise NotFoundError(
                f"Definition '{definition_id}' version '{version_from}' not found.",
                location=f"{definition_id}:{version_from}",
            )
        if meta_to is None:
            raise NotFoundError(
                f"Definition '{definition_id}' version '{version_to}' not found.",
                location=f"{definition_id}:{version_to}",
            )

        h_from = meta_from.meaning_hash
        h_to   = meta_to.meaning_hash

        if h_from is not None and h_to is not None:
            if h_from == h_to:
                status  = "equivalent"
                summary = (
                    f"Versions '{version_from}' and '{version_to}' share the same "
                    f"meaning_hash ({h_from[:12]}...). Meaning is unchanged — safe to promote."
                )
                changes: list[dict[str, Any]] = []
            else:
                ir_from = meta_from.ir_hash
                ir_to   = meta_to.ir_hash
                if ir_from is not None and ir_to is not None and ir_from != ir_to:
                    status  = "breaking"
                    summary = (
                        f"Breaking change detected: both meaning_hash and ir_hash differ "
                        f"between '{version_from}' and '{version_to}'. "
                        f"Downstream conditions/actions may be invalidated — "
                        "governance review required before promoting."
                    )
                    changes = [
                        {"hash_from": h_from, "hash_to": h_to},
                        {"ir_hash_from": ir_from, "ir_hash_to": ir_to},
                    ]
                else:
                    status  = "compatible"
                    summary = (
                        f"Semantic hashes differ: '{version_from}' ({h_from[:12]}...) vs "
                        f"'{version_to}' ({h_to[:12]}...). Meaning has changed — review recommended."
                    )
                    changes = [{"hash_from": h_from, "hash_to": h_to}]
        else:
            status  = "unknown"
            summary = (
                "Could not determine equivalence — one or both versions lack a "
                "meaning_hash. Treat as breaking and review before promoting."
            )
            changes = []

        return SemanticDiffResult(
            definition_id=definition_id,
            version_from=version_from,
            version_to=version_to,
            equivalence_status=status,
            summary=summary,
            changes=changes,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _freeze_check(self, definition: ConceptDefinition) -> None:
        """
        Run validate_schema() then validate_types() — the definition-freezing check.

        Raises ValidationError if either phase fails.  The definition is never
        stored when this raises.
        """
        errors: list[ValidationErrorItem] = []

        try:
            self._validator.validate_schema(definition)
        except MemintelError as exc:
            errors.append(ValidationErrorItem(
                type=exc.error_type,
                message=exc.message,
                location=exc.location,
            ))
            raise ValidationError(errors)  # schema failure stops further checking

        try:
            self._validator.validate_types(definition)
        except MemintelError as exc:
            errors.append(ValidationErrorItem(
                type=exc.error_type,
                message=exc.message,
                location=exc.location,
            ))

        if errors:
            raise ValidationError(errors)

    def _freeze_check_dict(self, body: dict[str, Any]) -> None:
        """
        Parse a raw dict as ConceptDefinition and run _freeze_check().

        Raises ValidationError on parse failure or validator violations.
        """
        import pydantic

        try:
            definition = ConceptDefinition(**body)
        except (pydantic.ValidationError, Exception) as exc:
            raise ValidationError(
                [ValidationErrorItem(
                    type=ErrorType.SYNTAX_ERROR,
                    message=f"Definition body failed schema parsing: {exc}",
                    location="body",
                )],
                message="Definition body could not be parsed as a ConceptDefinition.",
            )

        self._freeze_check(definition)

    async def _check_diff_before_promote(
        self,
        definition_id: str,
        version: str,
    ) -> None:
        """
        Run semantic_diff against the most recent other version.

        Blocks if equivalence_status is 'breaking'.
        No-op when there is nothing to compare or when diff is unavailable.
        """
        try:
            all_versions = await self._store.versions(definition_id)
        except Exception:
            return

        other = [v for v in all_versions if v.version != version]
        if not other:
            return

        other_version = other[0].version
        try:
            diff = await self.semantic_diff(definition_id, other_version, version)
        except (NotFoundError, MemintelError):
            return

        if diff.equivalence_status == "breaking":
            raise MemintelError(
                ErrorType.SEMANTIC_ERROR,
                f"Promotion blocked: semantic diff between '{other_version}' and "
                f"'{version}' is 'breaking'. {diff.summary}",
                suggestion=(
                    "Review semantic changes, obtain governance approval, and "
                    "re-run promote() after resolving downstream impacts."
                ),
            )


# ── Meaning hash ──────────────────────────────────────────────────────────────

def _compute_meaning_hash(definition: ConceptDefinition) -> str:
    """
    Deterministic semantic hash of a concept's meaning content.

    Identity fields (concept_id, version, namespace, description, created_at)
    are excluded so that two concepts with identical computation produce the
    same hash regardless of their id or version strings.
    """
    semantic_content = {
        "output_type":    definition.output_type,
        "labels":         sorted(definition.labels) if definition.labels else None,
        "primitives":     {k: v.model_dump() for k, v in sorted(definition.primitives.items())},
        "features":       {k: v.model_dump() for k, v in sorted(definition.features.items())},
        "output_feature": definition.output_feature,
    }
    canonical = json.dumps(semantic_content, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()
