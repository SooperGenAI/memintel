"""
app/stores/calibration_token.py
──────────────────────────────────────────────────────────────────────────────
CalibrationTokenStore — asyncpg-backed persistence for `calibration_tokens`.

Atomicity guarantee
───────────────────
resolve_and_invalidate() uses a single UPDATE...RETURNING inside an explicit
transaction. The WHERE clause:
    used_at IS NULL AND expires_at > NOW()
ensures that exactly one of any number of concurrent callers succeeds. All
others receive None — the at-most-once property is enforced by Postgres, not
by application-level locking.

Token lifecycle
───────────────
  create()                 → INSERT with expires_at = NOW() + 24h.
                             Returns the opaque token_string to the caller.
                             The CalibrationToken.id (BIGSERIAL) and
                             CalibrationToken.created_at are never exposed
                             to API callers (excluded from model serialisation).

  resolve_and_invalidate() → Atomically sets used_at = NOW() and returns
                             the full CalibrationToken. Returns None if the
                             token is expired, already used, or not found.
                             Caller maps None → HTTP 400.

Token generation
────────────────
token_string is generated with secrets.token_urlsafe(32), producing a
43-character URL-safe base64 string with 256 bits of entropy. This is
sufficient for a single-use, 24-hour token.

Column ↔ field mapping
──────────────────────
DB column           Python field (CalibrationToken)
──────────────────  ────────────────────────────────────────────
id                  id               (excluded from API — internal BIGSERIAL)
token_string        token_string
condition_id        condition_id
condition_version   condition_version
recommended_params  recommended_params (JSONB)
expires_at          expires_at
used_at             used_at          (excluded from API)
created_at          created_at       (excluded from API)
"""
from __future__ import annotations

import json
import structlog
import secrets
from datetime import datetime, timedelta, timezone

import asyncpg

from app.models.calibration import CalibrationToken

log = structlog.get_logger(__name__)

#: Token validity window.
TOKEN_TTL_HOURS: int = 24


class CalibrationTokenStore:
    """Async store for the `calibration_tokens` table."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # ── create ────────────────────────────────────────────────────────────────

    async def create(self, token: CalibrationToken) -> str:
        """
        Persist a calibration token and return the opaque token_string.

        A cryptographically random token_string is generated here — any
        token_string already set on the incoming CalibrationToken is ignored.
        expires_at is always computed as NOW() + 24h server-side.

        Returns the token_string. The caller embeds this in CalibrationResult
        for the API response; the full token object is never returned from here.
        """
        token_string = secrets.token_urlsafe(32)
        expires_at = datetime.now(tz=timezone.utc) + timedelta(hours=TOKEN_TTL_HOURS)

        await self._pool.execute(
            """
            INSERT INTO calibration_tokens (
                token_string, condition_id, condition_version,
                recommended_params, expires_at
            )
            VALUES ($1, $2, $3, $4, $5)
            """,
            token_string,
            token.condition_id,
            token.condition_version,
            json.dumps(token.recommended_params),
            expires_at,
        )

        log.info(
            "calibration_token_created",
            extra={
                "condition_id": token.condition_id,
                "condition_version": token.condition_version,
                "expires_at": expires_at.isoformat(),
                # token_string intentionally omitted — treat as a secret
            },
        )
        return token_string

    # ── resolve_and_invalidate ────────────────────────────────────────────────

    async def resolve_and_invalidate(
        self, token_string: str
    ) -> CalibrationToken | None:
        """
        Atomically redeem a token and return its payload.

        Sets used_at = NOW() in a single UPDATE...RETURNING inside a
        transaction. The WHERE clause enforces all three validity conditions:
          1. token_string matches
          2. used_at IS NULL  (not already redeemed)
          3. expires_at > NOW()  (not expired)

        Returns None if the token is not found, already used, or expired.
        A None return must be mapped to HTTP 400 by the caller.

        Concurrent calls for the same token_string: exactly one receives the
        CalibrationToken; all others receive None. This is guaranteed by
        Postgres row-level locking inside the transaction.
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    UPDATE calibration_tokens
                    SET used_at = NOW()
                    WHERE token_string = $1
                      AND used_at IS NULL
                      AND expires_at > NOW()
                    RETURNING
                        id, token_string, condition_id, condition_version,
                        recommended_params, expires_at, used_at, created_at
                    """,
                    token_string,
                )

        if row is None:
            log.info(
                "calibration_token_invalid",
                extra={"reason": "expired_used_or_not_found"},
            )
            return None

        log.info(
            "calibration_token_redeemed",
            extra={
                "condition_id": row["condition_id"],
                "condition_version": row["condition_version"],
            },
        )
        return _row_to_token(row)


# ── Row mapping helper ────────────────────────────────────────────────────────

def _row_to_token(row: asyncpg.Record) -> CalibrationToken:
    """
    Convert an asyncpg Record from calibration_tokens into a CalibrationToken.

    recommended_params is stored as JSONB; asyncpg may return it as a dict
    or a string depending on codec registration — both cases are handled.
    The internal DB fields (id, used_at, created_at) are populated so the
    store layer can inspect them, but they are excluded from API serialisation
    by CalibrationToken's Field(exclude=True) declarations.
    """
    params_raw = row["recommended_params"]
    if isinstance(params_raw, str):
        params_raw = json.loads(params_raw)

    return CalibrationToken(
        token_string=row["token_string"],
        condition_id=row["condition_id"],
        condition_version=row["condition_version"],
        recommended_params=params_raw,
        expires_at=row["expires_at"],
        id=row["id"],
        used_at=row["used_at"],
        created_at=row["created_at"],
    )
