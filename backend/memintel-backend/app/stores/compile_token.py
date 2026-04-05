"""
app/stores/compile_token.py
──────────────────────────────────────────────────────────────────────────────
CompileTokenStore — asyncpg-backed persistence for `compile_tokens`.

Atomicity guarantee
───────────────────
consume() uses a single conditional UPDATE...RETURNING:

    UPDATE compile_tokens
    SET used = TRUE
    WHERE token_string = $1
      AND used = FALSE
      AND expires_at > NOW()
    RETURNING *

This guarantees exactly-once redemption under concurrent requests:
  - Two callers presenting the same token race on the UPDATE.
  - Exactly one gets RETURNING rows; the other gets zero rows.
  - The loser then does a follow-up SELECT to determine the precise
    failure reason (consumed vs expired vs not found).
  - No application-level lock is needed — Postgres row-level locking
    inside the UPDATE provides the isolation.

Token lifecycle
───────────────
  create(token)  → INSERT with caller-supplied expires_at and ir_hash.
                   Raises ConflictError if token_string already exists.

  get(token_string) → SELECT by token_string.
                      Returns None if not found.

  consume(token_string) → Atomically marks used=TRUE.
                          Returns the updated CompileToken on success.
                          Raises on any failure (see consume docstring).

Column ↔ field mapping
──────────────────────
DB column     Python field (CompileToken)
────────────  ────────────────────────────────────────
token_id      token_id      (UUID PK — internal)
token_string  token_string  (opaque token returned to caller)
identifier    identifier    (locked at compile time)
ir_hash       ir_hash       (SHA-256 of compiled concept IR)
output_type   output_type   (declared at compile time; carried to Phase 2)
expires_at    expires_at
used          used
created_at    created_at
"""
from __future__ import annotations

from datetime import datetime, timezone

import asyncpg
import structlog

from app.models.concept_compile import CompileToken
from app.models.errors import (
    CompileTokenConsumedError,
    CompileTokenExpiredError,
    CompileTokenNotFoundError,
    ConflictError,
)

log = structlog.get_logger(__name__)


class CompileTokenStore:
    """Async store for the `compile_tokens` table."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # ── create ────────────────────────────────────────────────────────────────

    async def create(self, token: CompileToken) -> CompileToken:
        """
        Persist a compile token and return it.

        The token_id, token_string, ir_hash, and expires_at must all be
        populated on the incoming CompileToken — the caller is responsible
        for generating them (see ConceptCompilerService in Session M-3).

        Raises ConflictError if token_string already exists (UNIQUE constraint).
        """
        try:
            row = await self._pool.fetchrow(
                """
                INSERT INTO compile_tokens (
                    token_id, token_string, identifier, ir_hash, output_type, expires_at
                )
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING *
                """,
                token.token_id,
                token.token_string,
                token.identifier,
                token.ir_hash,
                token.output_type,
                token.expires_at,
            )
        except asyncpg.UniqueViolationError as exc:
            raise ConflictError(
                f"compile_token with token_string already exists.",
                location="token_string",
            ) from exc

        log.info(
            "compile_token_created",
            identifier=token.identifier,
            expires_at=token.expires_at.isoformat(),
            # token_string intentionally omitted — treat as a secret
        )
        return _row_to_token(row)

    # ── get ───────────────────────────────────────────────────────────────────

    async def get(self, token_string: str) -> CompileToken | None:
        """
        Fetch a compile token by its opaque token_string.

        Returns None if not found. Does not check expiry or used status —
        the caller must inspect those fields if needed.
        """
        row = await self._pool.fetchrow(
            "SELECT * FROM compile_tokens WHERE token_string = $1",
            token_string,
        )
        if row is None:
            return None
        return _row_to_token(row)

    # ── consume ───────────────────────────────────────────────────────────────

    async def consume(self, token_string: str) -> CompileToken:
        """
        Atomically mark a compile token as used and return it.

        The conditional UPDATE guarantees exactly-once redemption:
          UPDATE compile_tokens SET used = TRUE
          WHERE token_string = $1 AND used = FALSE AND expires_at > NOW()
          RETURNING *

        If the UPDATE returns a row, the token was valid and is now consumed.
        If zero rows are updated, a follow-up SELECT determines the reason:

          Token not found              → CompileTokenNotFoundError  (HTTP 404)
          Token found, used=True       → CompileTokenConsumedError  (HTTP 409)
          Token found, expires_at past → CompileTokenExpiredError   (HTTP 400)

        NOTE: CompileTokenExpiredError maps to HTTP 400, not 410.
        This is a deliberate cross-team contract — the Canvas consumer is
        written against 400 for expired tokens. Do not change it.
        """
        # Attempt atomic redemption
        row = await self._pool.fetchrow(
            """
            UPDATE compile_tokens
            SET used = TRUE
            WHERE token_string = $1
              AND used = FALSE
              AND expires_at > NOW()
            RETURNING *
            """,
            token_string,
        )

        if row is not None:
            token = _row_to_token(row)
            log.info(
                "compile_token_consumed",
                identifier=token.identifier,
            )
            return token

        # Zero rows updated — determine the precise failure reason
        lookup = await self._pool.fetchrow(
            "SELECT * FROM compile_tokens WHERE token_string = $1",
            token_string,
        )

        if lookup is None:
            raise CompileTokenNotFoundError(
                "compile_token not found.",
            )

        if lookup["used"]:
            raise CompileTokenConsumedError(
                "compile_token has already been consumed.",
                suggestion="Call POST /concepts/compile to obtain a new token.",
            )

        # used=FALSE but expires_at <= NOW() → expired
        # HTTP 400, not 410 — cross-team contract with Canvas consumer.
        raise CompileTokenExpiredError(
            "compile_token has expired. Call POST /concepts/compile again.",
        )


# ── Row mapping helper ────────────────────────────────────────────────────────

def _row_to_token(row: asyncpg.Record) -> CompileToken:
    """
    Convert an asyncpg Record from compile_tokens into a CompileToken.

    asyncpg returns TIMESTAMPTZ columns as timezone-aware datetime objects,
    so no manual UTC coercion is needed.
    """
    return CompileToken(
        token_id=str(row["token_id"]),
        token_string=row["token_string"],
        identifier=row["identifier"],
        ir_hash=row["ir_hash"],
        output_type=row["output_type"],
        expires_at=row["expires_at"],
        used=row["used"],
        created_at=row["created_at"],
    )
