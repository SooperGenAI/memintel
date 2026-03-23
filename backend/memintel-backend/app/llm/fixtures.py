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

_FIXTURES_DIR = Path(__file__).parent / "fixtures"

_FIXTURE_FILES = {
    "threshold":  _FIXTURES_DIR / "threshold_task.json",
    "z_score":    _FIXTURES_DIR / "z_score_task.json",
    "composite":  _FIXTURES_DIR / "composite_task.json",
    "equals":     _FIXTURES_DIR / "equals_task.json",
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


class LLMFixtureClient:
    """
    Development stub for the LLM client used by TaskAuthoringService.

    Returns pre-authored fixture dicts instead of calling a real LLM.
    Used when USE_LLM_FIXTURES=True is set in the environment.

    Interface mirrors the real LLM client so TaskAuthoringService can
    swap clients without branching.
    """

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
