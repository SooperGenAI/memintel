"""
app/llm/fixtures.py
──────────────────────────────────────────────────────────────────────────────
LLMFixtureClient — a development stub that returns pre-authored task fixtures
instead of calling a live LLM.

Activated when USE_LLM_FIXTURES=True in the environment.  TaskAuthoringService
injects this client in place of the real LLM client during local development
and testing, removing the LLM dependency from the inner development loop.

Each fixture is a JSON file under app/llm/fixtures/ representing the output
the LLM would produce for a POST /tasks request: a concept definition, a
condition definition with a fully specified strategy, and a resolved action.

Routing strategy
────────────────
generate_task() selects a fixture by scanning ``intent`` for keywords.
The keyword lists are ordered from most specific to least specific so that
compound intents (e.g. "composite AND churn and high-value") resolve
deterministically to the most relevant fixture.

Priority order:
  1. composite  — "composite", "and of", "or of", "multi-factor"
  2. z_score    — "z_score", "z-score", "anomaly", "spike", "deviation", "payment"
  3. equals     — "equals", "categorical", "category", "label", "classify", "segment"
  4. threshold  — default (covers "churn", "threshold", "above", "below", and any
                  intent that doesn't match a more specific keyword)
"""
from __future__ import annotations

import json
from pathlib import Path

from app.llm.base import LLMClientBase

_FIXTURES_DIR = Path(__file__).parent / "fixtures"

_FIXTURE_FILES = {
    "threshold":              _FIXTURES_DIR / "threshold_task.json",
    "z_score":                _FIXTURES_DIR / "z_score_task.json",
    "composite":              _FIXTURES_DIR / "composite_task.json",
    "equals":                 _FIXTURES_DIR / "equals_task.json",
    "agent_query":            _FIXTURES_DIR / "agent_query.json",
    "agent_define":           _FIXTURES_DIR / "agent_define.json",
    "agent_define_condition": _FIXTURES_DIR / "agent_define_condition.json",
    "agent_semantic_refine":  _FIXTURES_DIR / "agent_semantic_refine.json",
    "agent_compile_workflow": _FIXTURES_DIR / "agent_compile_workflow.json",
}

# Keywords that override the default (threshold) fixture.
# Checked in priority order — first match wins.
_ROUTING_RULES: list[tuple[str, list[str]]] = [
    ("composite", ["composite", "and of", "or of", "multi-factor", "multifactor"]),
    ("z_score",   ["z_score", "z-score", "anomaly", "spike", "deviation",
                   "payment failure", "payment_failure"]),
    ("equals",    ["equals", "categorical", "category", "label", "classify",
                   "classification", "segment"]),
]


class LLMFixtureClient(LLMClientBase):
    """
    Development stub for the LLM client used by TaskAuthoringService.

    Returns pre-authored fixture dicts instead of calling a real LLM.
    Used when USE_LLM_FIXTURES=True is set in the environment.

    Interface mirrors the real LLM client so TaskAuthoringService can
    swap clients without branching.
    """

    # ── Agent methods ──────────────────────────────────────────────────────────

    def generate_query(self, query: str, context: dict) -> dict:
        """Return fixture results for a natural language registry query."""
        return self._load("agent_query")

    def generate_define(self, description: str, context: dict) -> dict:
        """Return a fixture concept draft for a natural language description."""
        return self._load("agent_define")

    def generate_define_condition(
        self, description: str, concept_body: dict, context: dict
    ) -> dict:
        """Return a fixture condition draft for a natural language description."""
        return self._load("agent_define_condition")

    def generate_semantic_refine(
        self, definition_body: dict, instruction: str, context: dict
    ) -> dict:
        """Return a fixture refined definition for a natural language instruction."""
        return self._load("agent_semantic_refine")

    def generate_workflow(self, description: str, context: dict) -> dict:
        """Return a fixture ExecutionPlan for a natural language workflow description."""
        return self._load("agent_compile_workflow")

    # ── Task method ────────────────────────────────────────────────────────────

    def generate_task(self, intent: str, context: dict) -> dict:
        """
        Route ``intent`` to the most relevant fixture and return it as a dict.

        Parameters
        ----------
        intent:
            Natural language description of the task to create.  Keyword
            scanning is case-insensitive.
        context:
            LLM prompt context (guardrails, app context, primitives, …).
            Accepted but unused by this stub — fixtures are static.

        Returns
        -------
        dict
            Parsed fixture with keys ``concept``, ``condition``, ``action``.
            Structure matches TaskAuthoringService's expected LLM output.
        """
        fixture_key = self._route(intent)
        return self._load(fixture_key)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _route(self, intent: str) -> str:
        """Return the fixture key for ``intent`` using keyword routing."""
        normalised = intent.lower()
        for fixture_key, keywords in _ROUTING_RULES:
            if any(kw in normalised for kw in keywords):
                return fixture_key
        return "threshold"

    def _load(self, fixture_key: str) -> dict:
        """Load and parse the JSON fixture for ``fixture_key``."""
        path = _FIXTURE_FILES[fixture_key]
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
