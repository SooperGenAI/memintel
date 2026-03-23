"""
app/api/deps.py
──────────────────────────────────────────────────────────────────────────────
Shared FastAPI dependency functions.

require_elevated_key
────────────────────
Guards internal platform endpoints (POST /compile, POST /execute/graph,
registry write operations, POST /agents/semantic-refine,
POST /definitions/batch).

The caller must supply the X-Elevated-Key header whose value matches the
MEMINTEL_ELEVATED_KEY environment variable loaded at startup and stored on
app.state.elevated_key.

Returns None on success; raises HTTPException 403 on failure.
The 403 body uses the ErrorResponse wire shape so callers receive a consistent
error envelope regardless of whether the error is auth or domain.
"""
from __future__ import annotations

import os

from fastapi import Header, HTTPException, Request

from app.models.errors import ErrorDetail, ErrorResponse, ErrorType


async def require_elevated_key(
    request: Request,
    x_elevated_key: str | None = Header(default=None),
) -> None:
    """
    FastAPI dependency — enforces elevated key for internal platform endpoints.

    Reads the expected key from app.state.elevated_key (loaded at startup
    from the MEMINTEL_ELEVATED_KEY environment variable).  If the header is
    absent or does not match, returns HTTP 403 with an ErrorResponse body.
    """
    configured: str | None = getattr(request.app.state, "elevated_key", None)
    if not x_elevated_key or not configured or x_elevated_key != configured:
        raise HTTPException(
            status_code=403,
            detail=ErrorResponse(
                error=ErrorDetail(
                    type=ErrorType.AUTH_ERROR,
                    message="This endpoint requires an elevated API key. "
                            "Supply a valid X-Elevated-Key header.",
                )
            ).model_dump(mode="json"),
        )
