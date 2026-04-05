"""
app/services/concept_registration.py
──────────────────────────────────────────────────────────────────────────────
ConceptRegistrationService — Phase 2 of the two-phase concept registration flow.

Pipeline (strictly in this order)
──────────────────────────────────
  1. consume(compile_token)         — atomic exactly-once redemption.
                                      Raises on any token failure BEFORE any
                                      write to the definitions table.
  2. Verify token.identifier == request.identifier.
                                      Raises IdentifierMismatchError (422) on
                                      mismatch — no DB write is attempted.
  3. Register concept in DefinitionStore using token.ir_hash.
                                      On ConflictError: check ir_hash of the
                                      existing definition:
                                        same ir_hash  → idempotent; return 201
                                        diff ir_hash  → IdentifierConflictError (409)
  4. Return RegisterConceptResponse (201 Created).

Idempotency rule
────────────────
Two separate compile→register flows for the same (identifier, ir_hash) both
succeed with HTTP 201 and return the SAME concept_id. No duplicate row is
created. This is idempotent, not a conflict.

Identifier conflict rule
────────────────────────
Same identifier + different ir_hash → HTTP 409 IDENTIFIER_CONFLICT.
The existing definition is NOT overwritten under any circumstances.

Token failure codes (locked, cross-team contract)
──────────────────────────────────────────────────
  COMPILE_TOKEN_NOT_FOUND  → 404  (token string unknown)
  COMPILE_TOKEN_CONSUMED   → 409  (token already used)
  COMPILE_TOKEN_EXPIRED    → 400  (TTL exceeded — 400, NOT 410)

HARD RULE: consume() is called FIRST. No write to definitions may occur
before the token is successfully consumed. If consume() raises, the method
returns immediately — no definition lookup or write is attempted.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import asyncpg
import structlog

from app.models.concept_compile import RegisterConceptRequest, RegisterConceptResponse
from app.models.errors import (
    ConflictError,
    IdentifierConflictError,
    IdentifierMismatchError,
)
from app.stores.compile_token import CompileTokenStore
from app.stores.definition import DefinitionStore

log = structlog.get_logger(__name__)

_DEFAULT_VERSION = "1.0.0"


class ConceptRegistrationService:
    """
    Phase 2 concept registration service.

    Consumes a compile_token, verifies the identifier, and registers the
    concept in the DefinitionStore. Returns RegisterConceptResponse on success.

    Parameters
    ──────────
    token_store      — CompileTokenStore (or duck-typed mock for tests).
                       When None, a CompileTokenStore(pool) is created inside
                       register().
    definition_store — DefinitionStore (or duck-typed mock for tests).
                       When None, a DefinitionStore(pool) is created inside
                       register().
    """

    def __init__(
        self,
        token_store: Any = None,
        definition_store: Any = None,
    ) -> None:
        self._token_store = token_store
        self._definition_store = definition_store

    async def register(
        self,
        request: RegisterConceptRequest,
        pool: asyncpg.Pool | None,
    ) -> RegisterConceptResponse:
        """
        Run Phase 2 of two-phase concept registration.

        Pipeline order is strictly enforced — see module docstring.
        """
        token_store = (
            self._token_store if self._token_store is not None
            else CompileTokenStore(pool)
        )
        definition_store = (
            self._definition_store if self._definition_store is not None
            else DefinitionStore(pool)
        )

        # ── Step 1: Consume token (atomic, must be first) ──────────────────────
        token = await token_store.consume(request.compile_token)

        # ── Step 2: Verify identifier match ────────────────────────────────────
        if token.identifier != request.identifier:
            raise IdentifierMismatchError(
                f"compile_token was issued for identifier '{token.identifier}', "
                f"but request identifier is '{request.identifier}'.",
                suggestion="Use the identifier that was specified at compile time.",
            )

        # ── Step 3: Register in DefinitionStore ────────────────────────────────
        version = _DEFAULT_VERSION
        body: dict = {
            "concept_id": request.identifier,
            "version": version,
            "output_type": token.output_type,
        }

        try:
            defn = await definition_store.register(
                definition_id=request.identifier,
                version=version,
                definition_type="concept",
                namespace="personal",
                body=body,
                ir_hash=token.ir_hash,
            )
        except ConflictError:
            # (identifier, version) already exists — check ir_hash for idempotency
            existing = await definition_store.get_metadata(request.identifier, version)
            if existing is not None and existing.ir_hash == token.ir_hash:
                # Same formula compiled twice → idempotent success
                log.info(
                    "concept_registration_idempotent",
                    identifier=request.identifier,
                    ir_hash=token.ir_hash,
                )
                defn = existing
            else:
                raise IdentifierConflictError(
                    f"'{request.identifier}' is already registered with a different formula.",
                    suggestion=(
                        "Use a different identifier, or re-compile with the same definition body."
                    ),
                )

        log.info(
            "concept_registered",
            identifier=request.identifier,
            version=version,
            ir_hash=token.ir_hash,
        )

        # ── Step 4: Return response ────────────────────────────────────────────
        registered_at = defn.created_at or datetime.now(tz=timezone.utc)
        return RegisterConceptResponse(
            concept_id=request.identifier,
            identifier=request.identifier,
            version=version,
            output_type=token.output_type,
            registered_at=registered_at,
        )
