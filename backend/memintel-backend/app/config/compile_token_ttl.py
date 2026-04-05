"""
app/config/compile_token_ttl.py
──────────────────────────────────────────────────────────────────────────────
TTL configuration for compile tokens (V7).

Environment variable
────────────────────
  MEMINTEL_COMPILE_TOKEN_TTL_SECONDS
    Integer seconds. Default: 1800 (30 minutes).
    Minimum: 300 (5 minutes) — values below this are clamped, not rejected.
    Rationale: the SI reviews the compiled concept before clicking Register.
    30 minutes gives adequate review time; 5-minute minimum prevents
    misconfiguration from creating dangerously short-lived tokens.

Usage
─────
    from app.config.compile_token_ttl import get_compile_token_ttl

    ttl = get_compile_token_ttl()
    expires_at = datetime.now(tz=timezone.utc) + timedelta(seconds=ttl)
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

#: Default token TTL in seconds (30 minutes).
_DEFAULT_TTL_SECONDS: int = 1800

#: Minimum enforced TTL in seconds (5 minutes).
#: Values below this are clamped to this minimum, not rejected.
_MIN_TTL_SECONDS: int = 300


def get_compile_token_ttl() -> int:
    """
    Return the compile token TTL in seconds.

    Reads MEMINTEL_COMPILE_TOKEN_TTL_SECONDS from the environment.
    Falls back to 1800 (30 min) if the variable is absent or empty.
    Values below 300 (5 min) are clamped to 300 with a warning log.

    Returns an integer >= 300.
    """
    raw = os.getenv("MEMINTEL_COMPILE_TOKEN_TTL_SECONDS", "").strip()

    if not raw:
        return _DEFAULT_TTL_SECONDS

    try:
        ttl = int(raw)
    except ValueError:
        log.warning(
            "compile_token_ttl_invalid",
            extra={
                "raw_value": raw,
                "fallback": _DEFAULT_TTL_SECONDS,
            },
        )
        return _DEFAULT_TTL_SECONDS

    if ttl < _MIN_TTL_SECONDS:
        log.warning(
            "compile_token_ttl_clamped",
            extra={
                "configured_seconds": ttl,
                "clamped_to_seconds": _MIN_TTL_SECONDS,
                "reason": (
                    f"MEMINTEL_COMPILE_TOKEN_TTL_SECONDS={ttl} is below the "
                    f"minimum of {_MIN_TTL_SECONDS}s. Clamped to {_MIN_TTL_SECONDS}s."
                ),
            },
        )
        return _MIN_TTL_SECONDS

    return ttl
