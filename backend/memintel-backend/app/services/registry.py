"""
app/services/registry.py
──────────────────────────────────────────────────────────────────────────────
RegistryService — definition lifecycle management.

Manages the full lifecycle of definitions (concepts, conditions, actions,
primitives) in the definitions table:
  - Register (create or update) definitions
  - List and search with cursor-based pagination
  - Version history and lineage traversal
  - Semantic diff between versions
  - Deprecation and namespace promotion

Also manages the feature registry (registered_features table):
  - Register features with semantic deduplication
  - Get feature by id
  - List usages (definitions that depend on a feature)

TODO: full implementation in a future session.
"""
from __future__ import annotations

import asyncpg


class RegistryService:
    """
    Manages definition registration, versioning, and lifecycle.

    register()         — creates or updates a definition; returns DefinitionResponse.
    list_definitions() — paginated list with optional type/namespace filter.
    search()           — semantic search over definition names and metadata.
    get_versions()     — returns VersionSummary list for a definition.
    get_lineage()      — returns LineageResult (version chain + promotions).
    get_semantic_diff() — compares two versions; returns SemanticDiffResult.
    deprecate()        — marks a version deprecated; returns DefinitionResponse.
    promote()          — promotes to a higher namespace; returns DefinitionResponse.
    find_similar()     — semantic similarity search; returns SearchResult.

    register_feature() — registers a feature; returns FeatureRegistrationResult.
    get_feature()      — returns RegisteredFeature by id.
    get_feature_usages() — returns UsageResult for a feature.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
