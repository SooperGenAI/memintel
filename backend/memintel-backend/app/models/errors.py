"""
app/models/errors.py
──────────────────────────────────────────────────────────────────────────────
Error taxonomy for the Memintel backend.

Three distinct layers are represented here:

  1. ErrorType          — canonical machine-readable string enum.
                          Matches the API spec verbatim. Callers MUST branch on
                          this field, never on .message.

  2. Wire models        — Pydantic models that serialise directly to the
                          ErrorResponse JSON shape required by the API spec:
                            { "error": { "type": "...", "message": "..." } }

  3. Python exceptions  — MemintelError and its typed subclasses.
                          Raised anywhere in the application; caught by the
                          FastAPI exception handler in main.py and converted to
                          the wire shape with the appropriate HTTP status code.

No imports from other app modules — this file is imported by every layer.
"""
from __future__ import annotations

from enum import Enum
from http import HTTPStatus
from typing import Any

import structlog
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

_log = structlog.get_logger(__name__)


# ── 1. Error type enum ────────────────────────────────────────────────────────

class ErrorType(str, Enum):
    """
    Canonical error type identifiers. The string value is the wire value.

    Rules (from API spec):
      - Callers must branch on .type, never on .message.
      - Message text is for humans; type is for machines.
      - The enum spelling is the ground truth for YAML, HTTP, and Python alike.
        In particular: z_score (not zscore), not_found (not 404), conflict (not 409).
    """
    # ── Compiler / static analysis ────────────────────────────────────────────
    SYNTAX_ERROR    = "syntax_error"     # Malformed YAML / JSON input
    TYPE_ERROR      = "type_error"       # Operator input/output type mismatch
    SEMANTIC_ERROR  = "semantic_error"   # Strategy missing, composite nesting, etc.
    REFERENCE_ERROR = "reference_error"  # Referenced definition not found in registry
    PARAMETER_ERROR = "parameter_error"  # Invalid or missing strategy parameter
    GRAPH_ERROR     = "graph_error"      # DAG cycle, missing output node, etc.

    # ── Runtime ───────────────────────────────────────────────────────────────
    EXECUTION_ERROR   = "execution_error"    # Unexpected runtime failure
    EXECUTION_TIMEOUT = "execution_timeout"  # Concept execution exceeded time limit

    # ── HTTP / access control ─────────────────────────────────────────────────
    AUTH_ERROR          = "auth_error"           # Missing or invalid API key
    NOT_FOUND           = "not_found"            # Resource does not exist
    CONFLICT            = "conflict"             # Unique constraint / state violation
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"  # Too many requests

    # ── Domain ────────────────────────────────────────────────────────────────
    BOUNDS_EXCEEDED      = "bounds_exceeded"       # Param outside guardrail bounds
    ACTION_BINDING_FAILED = "action_binding_failed" # Action could not be resolved/fired

    # ── V7 — Vocabulary and compile token ─────────────────────────────────────
    VOCABULARY_MISMATCH          = "vocabulary_mismatch"           # No concept in vocab matched
    VOCABULARY_CONTEXT_TOO_LARGE = "vocabulary_context_too_large"  # Either list exceeds 500 IDs
    COMPILE_TOKEN_EXPIRED        = "compile_token_expired"         # Token TTL exceeded
    COMPILE_TOKEN_NOT_FOUND      = "compile_token_not_found"       # Token not found or malformed
    COMPILE_TOKEN_CONSUMED       = "compile_token_consumed"        # Token already used

    # ── V7 — Concept compilation pipeline ─────────────────────────────────────
    COMPILATION_ERROR   = "compilation_error"    # CoR pipeline step failed → HTTP 422
    TYPE_MISMATCH       = "type_mismatch"        # output_type incompatible with formula → HTTP 422
    IDENTIFIER_MISMATCH = "identifier_mismatch"  # token identifier ≠ request identifier → HTTP 422
    IDENTIFIER_CONFLICT = "identifier_conflict"  # identifier exists with different formula → HTTP 409

    # ── V7 — Task authoring with pre-compiled concept ──────────────────────
    CONCEPT_NOT_FOUND = "concept_not_found"      # concept_id not in registry → HTTP 404


# ── HTTP status code mapping ──────────────────────────────────────────────────

_HTTP_STATUS: dict[ErrorType, int] = {
    ErrorType.SYNTAX_ERROR:          400,
    ErrorType.TYPE_ERROR:            400,
    ErrorType.SEMANTIC_ERROR:        400,
    ErrorType.REFERENCE_ERROR:       400,
    ErrorType.PARAMETER_ERROR:       400,
    ErrorType.GRAPH_ERROR:           400,
    ErrorType.EXECUTION_ERROR:       500,
    ErrorType.EXECUTION_TIMEOUT:     504,
    ErrorType.AUTH_ERROR:            401,
    ErrorType.NOT_FOUND:             404,
    ErrorType.CONFLICT:              409,
    ErrorType.RATE_LIMIT_EXCEEDED:   429,
    ErrorType.BOUNDS_EXCEEDED:       400,
    ErrorType.ACTION_BINDING_FAILED: 400,

    # V7 — Vocabulary and compile token
    ErrorType.VOCABULARY_MISMATCH:          422,
    ErrorType.VOCABULARY_CONTEXT_TOO_LARGE: 422,
    # COMPILE_TOKEN_EXPIRED → 400, NOT 410.
    # Cross-team contract: Canvas consumer is written against 400.
    # Do not change to 410 on the basis of HTTP semantics.
    ErrorType.COMPILE_TOKEN_EXPIRED:        400,
    ErrorType.COMPILE_TOKEN_NOT_FOUND:      404,
    ErrorType.COMPILE_TOKEN_CONSUMED:       409,

    # V7 — Concept compilation pipeline
    ErrorType.COMPILATION_ERROR:   422,
    ErrorType.TYPE_MISMATCH:       422,
    ErrorType.IDENTIFIER_MISMATCH: 422,
    ErrorType.IDENTIFIER_CONFLICT: 409,

    # V7 — Task authoring with pre-compiled concept
    ErrorType.CONCEPT_NOT_FOUND: 404,
}


def http_status_for(error_type: ErrorType) -> int:
    """Return the HTTP status code for an ErrorType."""
    return _HTTP_STATUS[error_type]


# ── 2. Wire models ────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    """
    The inner ``error`` object serialised inside ErrorResponse.

    This is the exact shape the API spec requires:
      { "type": "...", "message": "...", "location": null, "suggestion": null }

    Callers must branch on ``type``, never on ``message``.
    """
    type: ErrorType
    message: str
    location: str | None = None
    suggestion: str | None = None


class ErrorResponse(BaseModel):
    """
    Top-level API error envelope.

    Wire shape:
      { "error": { "type": "...", "message": "..." } }
    """
    error: ErrorDetail

    @classmethod
    def from_exc(cls, exc: MemintelError) -> ErrorResponse:
        return cls(
            error=ErrorDetail(
                type=exc.error_type,
                message=exc.message,
                location=exc.location,
                suggestion=exc.suggestion,
            )
        )


class ValidationErrorItem(BaseModel):
    """
    A single validation error produced by the compiler or validator.

    ``type`` is restricted to the subset of error types that the compiler
    can produce (matches the ValidationError schema in developer_api.yaml).
    """
    type: ErrorType
    message: str
    location: str | None = None
    suggestion: str | None = None

    model_config = {"use_enum_values": True}


# ── 3. Python exceptions ──────────────────────────────────────────────────────

class MemintelError(Exception):
    """
    Base exception for all domain errors in the Memintel backend.

    Raised anywhere in the application. The FastAPI exception handler
    (registered in main.py) catches this and converts it to an ErrorResponse
    JSON body with the correct HTTP status code.

    Usage:
        raise MemintelError(ErrorType.NOT_FOUND, "Task not found", location="task_id")
        raise MemintelError(ErrorType.CONFLICT, "Version already registered")

    Subclasses exist for the most common cases — prefer them over constructing
    MemintelError directly for clarity at the call site.
    """

    def __init__(
        self,
        error_type: ErrorType,
        message: str,
        *,
        location: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.location = location
        self.suggestion = suggestion
        _log_fn = (
            _log.debug
            if error_type in (ErrorType.NOT_FOUND, ErrorType.CONFLICT)
            else _log.warning
        )
        _log_fn(
            "memintel_error",
            error_type=error_type.value,
            location=location,
        )

    @property
    def http_status(self) -> int:
        return http_status_for(self.error_type)

    def to_response(self) -> ErrorResponse:
        return ErrorResponse.from_exc(self)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"type={self.error_type.value!r}, "
            f"message={self.message!r}"
            f"{f', location={self.location!r}' if self.location else ''}"
            ")"
        )


# ── Typed subclasses ──────────────────────────────────────────────────────────
# Prefer these at call sites — they make intent clear and reduce the chance of
# passing the wrong ErrorType.

class NotFoundError(MemintelError):
    """Resource does not exist → HTTP 404."""
    def __init__(
        self,
        message: str = "Resource not found",
        *,
        location: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(
            ErrorType.NOT_FOUND,
            message,
            location=location,
            suggestion=suggestion,
        )


class ConflictError(MemintelError):
    """
    Unique constraint or state machine violation → HTTP 409.

    Used for:
      - Registering a definition with an existing (id, version) pair
      - Submitting duplicate feedback for the same decision
      - Transitioning a job from a terminal state
      - Updating a deleted task
    """
    def __init__(
        self,
        message: str = "Conflict",
        *,
        location: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(
            ErrorType.CONFLICT,
            message,
            location=location,
            suggestion=suggestion,
        )


class ValidationError(MemintelError):
    """
    One or more compiler / validator errors.

    Carries the full list of ``ValidationErrorItem`` objects so that callers
    can report all problems in a single response rather than failing on the
    first error found.
    """
    def __init__(
        self,
        errors: list[ValidationErrorItem],
        *,
        message: str | None = None,
    ) -> None:
        summary = message or f"{len(errors)} validation error(s)"
        super().__init__(ErrorType.TYPE_ERROR, summary)
        self.errors = errors

    @classmethod
    def single(
        cls,
        error_type: ErrorType,
        message: str,
        *,
        location: str | None = None,
        suggestion: str | None = None,
    ) -> ValidationError:
        """Convenience constructor for a single-error ValidationError."""
        return cls(
            [ValidationErrorItem(
                type=error_type,
                message=message,
                location=location,
                suggestion=suggestion,
            )],
            message=message,
        )


class CompilerInvariantError(MemintelError):
    """
    A determinism invariant was violated during compilation.

    This represents a critical compiler bug:
      - Same (concept_id, version) produced a different ir_hash on recompilation.
      - The existing graph MUST NOT be overwritten when this is raised.
      - Always logged at ERROR level with full context.

    See persistence-schema.md §3.5 for the graph replacement invariant.
    """
    def __init__(
        self,
        concept_id: str,
        version: str,
        existing_hash: str,
        new_hash: str,
    ) -> None:
        message = (
            f"ir_hash mismatch for ({concept_id!r}, {version!r}): "
            f"existing={existing_hash!r}, new={new_hash!r}. "
            "Compiler is non-deterministic or definition was mutated — "
            "do NOT overwrite the existing graph. Investigate immediately."
        )
        super().__init__(
            ErrorType.GRAPH_ERROR,
            message,
            location=f"{concept_id}:{version}",
            suggestion="Re-examine the definition body and compiler hash logic.",
        )
        self.concept_id = concept_id
        self.version = version
        self.existing_hash = existing_hash
        self.new_hash = new_hash

    @property
    def http_status(self) -> int:
        # Invariant violations are server-side bugs, not client errors.
        return 500


class AuthError(MemintelError):
    """Missing or invalid API key → HTTP 401."""
    def __init__(self, message: str = "Unauthorized") -> None:
        super().__init__(ErrorType.AUTH_ERROR, message)


class RateLimitError(MemintelError):
    """Rate limit exceeded → HTTP 429."""
    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(ErrorType.RATE_LIMIT_EXCEEDED, message)


class BoundsExceededError(MemintelError):
    """
    A parameter value lies outside the guardrail bounds → HTTP 400.

    Raised by CalibrationService when a recommended parameter adjustment
    would breach the min/max bounds declared in memintel.guardrails.md.
    """
    def __init__(
        self,
        message: str = "Parameter outside allowed bounds",
        *,
        location: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(
            ErrorType.BOUNDS_EXCEEDED,
            message,
            location=location,
            suggestion=suggestion,
        )


class ExecutionTimeoutError(MemintelError):
    """Concept execution exceeded its time limit → HTTP 504."""
    def __init__(self, message: str = "Execution timed out") -> None:
        super().__init__(ErrorType.EXECUTION_TIMEOUT, message)


# ── V7 typed subclasses ───────────────────────────────────────────────────────

class VocabularyMismatchError(MemintelError):
    """
    No concept in vocabulary_context matched the intent → HTTP 422.

    Raised before the LLM when vocabulary_context is present but
    both available_concept_ids and available_condition_ids are empty,
    or after Step 2 when no concept in the vocabulary matched.
    """
    def __init__(
        self,
        message: str = "None of the available concepts match this intent.",
        *,
        suggestion: str | None = "Review the module vocabulary or rephrase the agent intent.",
    ) -> None:
        super().__init__(
            ErrorType.VOCABULARY_MISMATCH,
            message,
            suggestion=suggestion,
        )


class VocabularyContextTooLargeError(MemintelError):
    """
    Either available_concept_ids or available_condition_ids exceeds
    MAX_VOCABULARY_IDS (500 per list) → HTTP 422.

    The cap is per-list, not combined.
    """
    def __init__(
        self,
        message: str = "vocabulary_context exceeds the maximum of 500 IDs per list.",
        *,
        suggestion: str | None = (
            "Reduce the number of installed modules or filter the vocabulary before sending."
        ),
    ) -> None:
        super().__init__(
            ErrorType.VOCABULARY_CONTEXT_TOO_LARGE,
            message,
            suggestion=suggestion,
        )


class CompileTokenExpiredError(MemintelError):
    """
    compile_token TTL exceeded → HTTP 400.

    NOTE: HTTP 400, not 410. This is a deliberate cross-team contract
    decision — the Canvas consumer is written against 400 for expired
    tokens. Do not change to 410 on the basis of HTTP semantics.
    """
    def __init__(
        self,
        message: str = "compile_token has expired. Call POST /concepts/compile again.",
    ) -> None:
        super().__init__(ErrorType.COMPILE_TOKEN_EXPIRED, message)


class CompileTokenNotFoundError(MemintelError):
    """compile_token not found or malformed → HTTP 404."""
    def __init__(
        self,
        message: str = "compile_token not found.",
    ) -> None:
        super().__init__(ErrorType.COMPILE_TOKEN_NOT_FOUND, message)


class CompileTokenConsumedError(MemintelError):
    """compile_token has already been used → HTTP 409."""
    def __init__(
        self,
        message: str = "compile_token has already been consumed.",
        *,
        suggestion: str | None = "Call POST /concepts/compile to obtain a new token.",
    ) -> None:
        super().__init__(
            ErrorType.COMPILE_TOKEN_CONSUMED,
            message,
            suggestion=suggestion,
        )


class CompilationError(MemintelError):
    """
    CoR pipeline step failed → HTTP 422.

    Raised when a CoR step (Intent Parsing, Signal Identification,
    DAG Construction, or Type Validation) cannot be completed.
    failed_at_step is the 1-based step index where failure occurred.
    """
    def __init__(
        self,
        message: str,
        *,
        failed_at_step: int | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(
            ErrorType.COMPILATION_ERROR,
            message,
            suggestion=suggestion,
        )
        self.failed_at_step = failed_at_step


class TypeMismatchError(MemintelError):
    """
    output_type is incompatible with the compiled formula → HTTP 422.

    Raised at Step 4 (Type Validation) when the requested output_type
    cannot be produced by the formula strategy selected in Step 3.
    """
    def __init__(
        self,
        message: str = "output_type is incompatible with the compiled formula.",
        *,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(
            ErrorType.TYPE_MISMATCH,
            message,
            suggestion=suggestion,
        )


class IdentifierMismatchError(MemintelError):
    """
    Request identifier differs from the compile_token's locked identifier → HTTP 422.

    The identifier is locked at compile time (POST /concepts/compile).
    POST /concepts/register MUST supply the same identifier — any mismatch
    is rejected immediately, before any DB write occurs.
    """
    def __init__(
        self,
        message: str = "identifier does not match the compile_token.",
        *,
        suggestion: str | None = "Use the identifier that was specified at compile time.",
    ) -> None:
        super().__init__(
            ErrorType.IDENTIFIER_MISMATCH,
            message,
            suggestion=suggestion,
        )


class IdentifierConflictError(MemintelError):
    """
    Identifier already registered with a different compiled formula → HTTP 409.

    Raised when POST /concepts/register encounters an existing definition
    for the same identifier but with a different ir_hash (different formula).
    Idempotent re-registration (same identifier + same ir_hash) is NOT an
    error — it returns the existing concept_id with HTTP 201.
    """
    def __init__(
        self,
        message: str = "identifier is already registered with a different formula.",
        *,
        suggestion: str | None = (
            "Use a different identifier, or re-compile with the same definition body."
        ),
    ) -> None:
        super().__init__(
            ErrorType.IDENTIFIER_CONFLICT,
            message,
            suggestion=suggestion,
        )


class ConceptNotFoundError(MemintelError):
    """
    concept_id provided in CreateTaskRequest does not exist in the registry → HTTP 404.

    Raised by TaskAuthoringService when concept_id is supplied but the registry
    has no versions for that identifier. The caller must register the concept
    first via POST /concepts/compile + POST /concepts/register.
    """
    def __init__(
        self,
        message: str = "concept not found in registry.",
        *,
        suggestion: str | None = (
            "Register the concept via POST /concepts/compile and POST /concepts/register first."
        ),
    ) -> None:
        super().__init__(
            ErrorType.CONCEPT_NOT_FOUND,
            message,
            suggestion=suggestion,
        )


# ── 4. FastAPI exception handler ──────────────────────────────────────────────

async def memintel_error_handler(request: Request, exc: MemintelError) -> JSONResponse:
    """
    FastAPI exception handler for MemintelError.

    Register in main.py:
        app.add_exception_handler(MemintelError, memintel_error_handler)

    Converts any MemintelError (and all subclasses) into the canonical
    ErrorResponse wire shape with the correct HTTP status code.

    Does NOT log the error — callers that need logging should do so before
    raising, or a middleware layer can log request/response pairs.
    """
    return JSONResponse(
        status_code=exc.http_status,
        content=exc.to_response().model_dump(mode="json"),
    )
