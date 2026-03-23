"""
app/services/action.py
──────────────────────────────────────────────────────────────────────────────
ActionService — action dispatch and execution.

Dispatches webhook, notification, workflow, and register actions.
Evaluates fire_on conditions, handles dry_run mode, and wraps delivery
failures in ActionResult rather than raising exceptions.

Actions are best-effort — trigger() always returns ActionResult and never
raises, even on delivery failure.

TODO: full implementation in a future session.
"""
from __future__ import annotations

import asyncpg


class ActionService:
    """
    Dispatches registered actions for a given entity.

    trigger() evaluates fire_on, handles dry_run simulation, and executes
    the delivery mechanism (webhook POST, notification, workflow enqueue,
    or entity registration). Failures are captured in ActionResult.status
    and ActionResult.error — never raised as exceptions.

    Actions are best-effort — trigger() always returns an ActionResult.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
