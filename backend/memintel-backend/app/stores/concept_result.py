"""
app/stores/concept_result.py
──────────────────────────────────────────────────────────────────────────────
ConceptResultStore — asyncpg-backed persistence for the `concept_results` table.

Responsibilities
────────────────
1. store()         — insert one concept execution result for (concept_id, entity).
2. fetch_history() — return the last N results for (concept_id, entity), oldest
                     first, as raw asyncpg Records. Used by z_score, percentile,
                     and change strategies to build their history reference frame.

Table: concept_results
──────────────────────
Column        Type                 Notes
──────────    ───────────────      ─────────────────────────────────────────────
concept_id    TEXT  NOT NULL       matches ConceptDefinition.concept_id
version       TEXT  NOT NULL       matches ConceptDefinition.version
entity        TEXT  NOT NULL       the evaluated entity identifier
value         DOUBLE PRECISION     numeric concept output; NULL for non-numeric
output_type   TEXT  NOT NULL       ConceptOutputType string ('float', 'boolean', …)
evaluated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()

Index recommended in production:
  CREATE INDEX ON concept_results (concept_id, entity, evaluated_at DESC);

History semantics
─────────────────
fetch_history() returns rows ordered oldest-first so that `history[-1]` is
the most recent prior result, matching the expectation of ChangeStrategy
(which reads history[-1].value as the previous value) and consistent with
ZScoreStrategy and PercentileStrategy which iterate all values.

Only numeric results (output_type='float') should be written by
ExecuteService._store_concept_result() — this ensures history rows can
always be safely cast to float for strategy math.
"""
from __future__ import annotations

import structlog
from typing import Any

import asyncpg

log = structlog.get_logger(__name__)


class ConceptResultStore:
    """Async store for the `concept_results` table."""

    _TABLE = "concept_results"

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # ── store ─────────────────────────────────────────────────────────────────

    async def store(
        self,
        concept_id: str,
        version: str,
        entity: str,
        value: float | None,
        output_type: str,
        output_text: str | None = None,
    ) -> None:
        """
        Persist one concept execution result.

        value is None for categorical results (text stored in output_text).
        output_text is the string value for categorical results.

        Raises on any DB error — callers (ExecuteService._store_concept_result)
        must catch and log rather than propagate, since history storage is
        best-effort and must not block the primary evaluation path.
        """
        float_value = float(value) if value is not None else None
        await self._pool.execute(
            f"""
            INSERT INTO {self._TABLE} (concept_id, version, entity, value, output_type, output_text)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            concept_id,
            version,
            entity,
            float_value,
            output_type,
            output_text,
        )

    # ── fetch_history ─────────────────────────────────────────────────────────

    async def fetch_history(
        self,
        concept_id: str,
        entity: str,
        limit: int = 30,
    ) -> list[Any]:
        """
        Return the last ``limit`` concept results for (concept_id, entity).

        Results are returned oldest-first (the DB query orders DESC to pick
        the most recent, then Python reverses for oldest-first delivery).

        Each row has columns: value, output_type, entity, version, evaluated_at.
        The caller (ExecuteService._evaluate_strategy) converts rows to
        ConceptResult objects for the strategy's history parameter.

        Returns an empty list when no results are stored yet.
        Raises on DB errors — caller catches and falls back to empty history.
        """
        rows = await self._pool.fetch(
            f"""
            SELECT value, output_type, entity, version, evaluated_at
            FROM {self._TABLE}
            WHERE concept_id = $1 AND entity = $2
              AND value IS NOT NULL
            ORDER BY evaluated_at DESC
            LIMIT $3
            """,
            concept_id,
            entity,
            limit,
        )
        # Reverse so that index 0 is oldest, index -1 is most recent.
        return list(reversed(rows))
