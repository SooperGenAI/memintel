"""
app/observability.py
──────────────────────────────────────────────────────────────────────────────
Structlog configuration for the Memintel backend.

Call configure_structlog() exactly once at application startup (main.py).
All modules use structlog.get_logger(__name__) — the processor pipeline is
applied globally after configure_structlog() is called.

Output format: JSON (one object per line).  Suitable for log aggregation
pipelines (Datadog, CloudWatch, ELK, etc.).

Required log events and their fields (from py-instructions.md):
  concept_executed    — concept_id, version, entity, timestamp, deterministic,
                        cache_hit, compute_time_ms, result_type
  condition_evaluated — condition_id, condition_version, entity, timestamp,
                        decision_value, decision_type, strategy_type,
                        params_applied, actions_triggered_count
  calibration_recommended — condition_id, condition_version, strategy_type,
                        old_params, recommended_params, delta_alerts,
                        feedback_direction
  calibration_applied — condition_id, previous_version, new_version,
                        params_applied, tasks_pending_rebind (count)
  memintel_error      — error_type, location

NEVER logged (PII / security):
  - Raw primitive data or entity attributes
  - Feedback note fields
  - Credential or token values
"""
from __future__ import annotations

import logging

import structlog


def configure_structlog() -> None:
    """
    Configure structlog for JSON output.

    Must be called once at startup before any log events are emitted.
    After this call, structlog.get_logger(__name__).info("event", ...)
    writes a JSON line to stdout.
    """
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
