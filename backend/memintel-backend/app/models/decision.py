"""
app/models/decision.py
──────────────────────────────────────────────────────────────────────────────
DecisionRecord — the audit record written after every evaluate_full() call.

Stored in the `decisions` table. Every field maps 1-to-1 to a DB column.
Decision recording is fire-and-forget — never fails the primary evaluation.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DecisionRecord(BaseModel):
    """
    Immutable audit record of a single evaluate_full() pipeline execution.

    Written to the decisions table after every evaluation, regardless of
    whether the condition fired. The record carries full provenance:
      - which concept+version produced the value
      - which condition+version evaluated it
      - for which entity
      - what the strategy parameters were at evaluation time
      - what primitive values were fetched (input_primitives)
      - which primitives had fetch errors (signal_errors)
      - whether the condition fired and which actions were triggered
      - whether the evaluation was a dry run
    """
    decision_id: str | None = None  # assigned by DB on INSERT; None before persisting
    concept_id: str
    concept_version: str
    condition_id: str
    condition_version: str
    entity_id: str
    evaluated_at: datetime | None = None  # set by DB DEFAULT NOW()
    fired: bool
    concept_value: float | int | bool | str | None = None
    threshold_applied: dict[str, Any] | None = None
    ir_hash: str | None = None
    input_primitives: dict[str, Any] | None = None
    signal_errors: dict[str, str] | None = None
    reason: str | None = None
    action_ids_fired: list[str] = Field(default_factory=list)
    dry_run: bool = False
