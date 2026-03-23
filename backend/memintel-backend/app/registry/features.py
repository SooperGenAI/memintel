"""
app/registry/features.py
──────────────────────────────────────────────────────────────────────────────
FeatureRegistry — first-class feature registration with duplicate detection.

Features are semantic primitives registered independently of any Concept and
reusable across many Concepts.  The feature registry is the foundational layer
of the Memintel meaning graph (Internal Platform API Reference §Feature Registry).

On registration the compiler computes meaning_hash from the feature's `meaning`
block (description, semantic_type, unit).  Before persisting, the registry scans
for hash collisions.  The on_duplicate policy controls the outcome:

  'warn'   — register the new feature; return duplicates[] in the response.
  'reject' — raise ConflictError (HTTP 409) listing conflicting features.
  'merge'  — return the existing feature with the matching meaning_hash
             (no new id is created; the caller gets the canonical feature).

Public API
──────────
FeatureRegistry.register_feature(feature, on_duplicate) → FeatureRegistrationResult

Models
──────
RegisteredFeature        — the feature to register.
FeatureRegistrationResult — the response.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from app.models.errors import ConflictError, MemintelError, ErrorType
from app.models.task import Namespace

log = logging.getLogger(__name__)


# ── Feature models ────────────────────────────────────────────────────────────

class FeatureMeaning(BaseModel):
    """
    The meaning block of a feature — what it represents semantically.

    meaning_hash is computed from this block: two features with the same
    description, semantic_type, and unit have the same meaning, regardless
    of their id or execution implementation.
    """
    description: str
    semantic_type: str          # e.g. 'probability', 'rate', 'count', 'score'
    unit: str | None = None     # e.g. 'per_day', 'percentage', None


class FeatureExecution(BaseModel):
    """
    The execution block of a feature — how it is computed.

    This is the implementation detail; it does not affect the meaning_hash.
    Two features may have the same meaning but different execution strategies.
    """
    op: str                                      # operator name
    input: str | list[str]                       # primitive name(s) or feature name(s)
    window: str | None = None                    # duration parameter e.g. '30d'
    params: dict[str, Any] = Field(default_factory=dict)


class RegisteredFeature(BaseModel):
    """
    A feature to register in the feature registry.

    id        — fully qualified feature identifier (namespace.id).
    namespace — personal | team | org | global.  Must match the prefix in id.
    version   — explicit semantic version.
    meaning   — the semantic content; drives meaning_hash computation.
    execution — the computation implementation.
    tags      — optional list of tags for discovery.
    """
    id: str
    namespace: Namespace
    version: str = "1.0"
    meaning: FeatureMeaning
    execution: FeatureExecution
    tags: list[str] = Field(default_factory=list)
    description: str | None = None


class FeatureRegistrationResult(BaseModel):
    """
    Response from register_feature().

    status:
      registered   — new feature stored; meaning_hash is unique in the registry.
      warn         — registered with duplicates[] populated (on_duplicate='warn').
      merged       — existing feature returned; no new id created (on_duplicate='merge').

    duplicates — list of feature ids that share the same meaning_hash.
                 Empty when status='registered' and no collisions were found.
    """
    id: str
    version: str
    meaning_hash: str
    status: str                          # registered | warn | merged
    duplicates: list[str] = Field(default_factory=list)


# ── FeatureRegistry ───────────────────────────────────────────────────────────

class FeatureRegistry:
    """
    Feature registration service.

    Parameters
    ──────────
    store — DefinitionStore (or duck-typed mock for tests).
            Features are stored as definition_type='feature'.
    """

    def __init__(self, store: Any) -> None:
        self._store = store

    async def register_feature(
        self,
        feature: RegisteredFeature,
        on_duplicate: str = "warn",
    ) -> FeatureRegistrationResult:
        """
        Register a feature with duplicate detection.

        1. Compute meaning_hash from feature.meaning.
        2. Scan registry for features with the same meaning_hash.
        3. Apply on_duplicate policy.
        4. Persist and return FeatureRegistrationResult.

        on_duplicate values:
          'warn'   — register; include duplicate ids in response.
          'reject' — raise ConflictError listing conflicting feature ids.
          'merge'  — return the existing feature (no new row written).

        Raises ConflictError (HTTP 409) when on_duplicate='reject' and a
        collision exists, or when (id, version) is already registered.
        """
        if on_duplicate not in ("warn", "reject", "merge"):
            raise MemintelError(
                ErrorType.PARAMETER_ERROR,
                f"on_duplicate must be 'warn', 'reject', or 'merge'; got '{on_duplicate}'.",
                location="on_duplicate",
            )

        meaning_hash = _compute_meaning_hash(feature.meaning)

        # Scan for hash collisions in the feature registry.
        existing = await self._store.list(definition_type="feature")
        duplicates = [
            item.definition_id
            for item in existing.items
            if item.meaning_hash == meaning_hash
            and item.definition_id != feature.id
        ]

        # ── on_duplicate: reject ──────────────────────────────────────────────
        if duplicates and on_duplicate == "reject":
            raise ConflictError(
                f"Feature with meaning_hash '{meaning_hash}' already exists. "
                f"Conflicting features: {duplicates}. "
                "Use on_duplicate='warn' to register anyway, or "
                "'merge' to reuse the existing feature.",
                location=feature.id,
                suggestion="Inspect the conflicting features and consolidate if appropriate.",
            )

        # ── on_duplicate: merge ───────────────────────────────────────────────
        if duplicates and on_duplicate == "merge":
            # Return the first existing feature — no new row.
            existing_id = duplicates[0]
            existing_meta = next(
                (item for item in existing.items if item.definition_id == existing_id),
                None,
            )
            merged_version = existing_meta.version if existing_meta else feature.version
            log.info(
                "feature_merged",
                extra={
                    "new_id": feature.id,
                    "existing_id": existing_id,
                    "meaning_hash": meaning_hash,
                },
            )
            return FeatureRegistrationResult(
                id=existing_id,
                version=merged_version,
                meaning_hash=meaning_hash,
                status="merged",
                duplicates=duplicates,
            )

        # ── Register (warn or no collision) ───────────────────────────────────
        body = feature.model_dump()
        await self._store.register(
            definition_id=feature.id,
            version=feature.version,
            definition_type="feature",
            namespace=feature.namespace.value,
            body=body,
            meaning_hash=meaning_hash,
        )

        status = "warn" if duplicates else "registered"

        log.info(
            "feature_registered",
            extra={
                "id": feature.id,
                "version": feature.version,
                "meaning_hash": meaning_hash,
                "status": status,
                "duplicates": duplicates,
            },
        )

        return FeatureRegistrationResult(
            id=feature.id,
            version=feature.version,
            meaning_hash=meaning_hash,
            status=status,
            duplicates=duplicates,
        )


# ── Meaning hash ──────────────────────────────────────────────────────────────

def _compute_meaning_hash(meaning: FeatureMeaning) -> str:
    """
    Deterministic SHA-256 hash of a feature's meaning block.

    Hashes description, semantic_type, and unit (sorted JSON).
    Two features with identical meaning blocks produce the same hash regardless
    of their id, namespace, or execution implementation.
    """
    content = {
        "description":   meaning.description,
        "semantic_type": meaning.semantic_type,
        "unit":          meaning.unit,
    }
    canonical = json.dumps(content, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()
