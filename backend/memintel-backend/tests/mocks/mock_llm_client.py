"""
tests/mocks/mock_llm_client.py
──────────────────────────────────────────────────────────────────────────────
MockLLMClient — deterministic test double for the LLM client.

Implements the same interface as the real LLM client (generate_task) without
making any network calls.  Returns pre-programmed responses based on keyword
matching in the intent string.

Interface contract
──────────────────
Extends LLMClientBase (same ABC as AnthropicClient, LLMFixtureClient, etc.).
generate_task(intent, context) → dict with keys: concept, condition, action.
The returned dict structure must be parseable by TaskAuthoringService:
  concept   → ConceptDefinition(**concept)
  condition → ConditionDefinition(**condition)
  action    → ActionDefinition(**action)

Routing rules (priority order, first match wins)
─────────────────────────────────────────────────
1. "error rate" | "deviation" | "baseline" | "anomaly" | "spike"
   → z_score on service.error_rate_5m (threshold=2.0, direction=above)

2. "renewal" | "days to renewal"
   → threshold on account.days_to_renewal (direction=below, value=60.0)

3. "engagement" | "session"
   → change on user.session_frequency_trend_8w (direction=decrease, value=0.2)

4. "tier" | "plan" | "segment"
   → equals on account.plan_tier (value="enterprise")

5. "churn" | "active user" | "user rate"
   → threshold on account.active_user_rate_30d (direction=below, value=0.35)

6. default (no keyword match)
   → threshold on account.active_user_rate_30d (direction=below, value=0.5)

Unique IDs
──────────
Each call increments an instance counter so that multiple POST /tasks calls
within one test do not produce conflicting definition IDs (which would cause
ConflictError on the second registration).

Concept structure
─────────────────
All float scenarios use z_score_op on a float primitive — the same op used
in the existing e2e pipeline tests (test_pipeline_e2e.py).  The static
executor returns the raw input value when no historical baseline exists
(missing_data_policy="zero"), making threshold assertions deterministic.

The equals scenario uses a categorical primitive with a passthrough op, which
is the only op that accepts categorical input.
"""
from __future__ import annotations

from app.llm.base import LLMClientBase


class MockLLMClient(LLMClientBase):
    """
    Deterministic LLM test double for task authoring tests.

    Each instance maintains its own call counter so IDs are unique per call
    even when the same client instance is shared across multiple POST /tasks
    calls within one test.
    """

    def __init__(self) -> None:
        self._counter: int = 0
        self.last_intent: str = ""
        self.last_context: dict = {}
        self.call_count: int = 0

    # ── generate_task ─────────────────────────────────────────────────────────

    def generate_task(self, intent: str, context: dict) -> dict:
        """
        Return a deterministic task definition (concept + condition + action)
        based on keyword matching in ``intent``.

        Records the most recent call in self.last_intent and self.last_context
        so tests can verify the service passed the intent correctly.
        """
        self._counter += 1
        self.call_count += 1
        self.last_intent = intent
        self.last_context = context

        scenario = self._route(intent)
        suffix = f"{self._counter:04d}"
        return self._build(scenario, suffix)

    # ── generate_semantic_refine ──────────────────────────────────────────────

    def generate_semantic_refine(
        self, definition_body: dict, instruction: str, context: dict
    ) -> dict:
        """
        Return a minimal valid refinement response.

        The agents route uses its own LLM client selection path (also
        controlled by USE_LLM_FIXTURES), so this method is a safety net
        in case the mock is injected into the agent service as well.
        """
        import copy
        proposed = copy.deepcopy(definition_body)
        if "description" in proposed:
            proposed["description"] = f"{proposed['description']} (refined: {instruction})"
        return {
            "proposed": proposed,
            "changes": [
                {
                    "field": "description",
                    "from": definition_body.get("description", ""),
                    "to": proposed.get("description", ""),
                    "reason": instruction,
                }
            ],
            "breaking": False,
        }

    # ── Internal routing ──────────────────────────────────────────────────────

    def _route(self, intent: str) -> str:
        """Return scenario key for the intent using keyword priority matching."""
        low = intent.lower()
        if any(kw in low for kw in ["error rate", "deviation", "baseline", "anomaly", "spike"]):
            return "z_score"
        if any(kw in low for kw in ["renewal", "days to renewal"]):
            return "renewal"
        if any(kw in low for kw in ["engagement", "session"]):
            return "change"
        if any(kw in low for kw in ["tier", "plan", "segment"]):
            return "equals"
        if any(kw in low for kw in ["churn", "active user", "user rate"]):
            return "churn"
        return "default"

    def _build(self, scenario: str, suffix: str) -> dict:
        """Dispatch to the correct scenario builder."""
        builders = {
            "churn":   self._churn,
            "renewal": self._renewal,
            "z_score": self._z_score,
            "change":  self._change,
            "equals":  self._equals,
            "default": self._default,
        }
        return builders[scenario](suffix)

    # ── Scenario builders ─────────────────────────────────────────────────────

    def _churn(self, suffix: str) -> dict:
        """Threshold on account.active_user_rate_30d — below 0.35."""
        return self._float_scenario(
            suffix=suffix,
            name="churn",
            primitive="account.active_user_rate_30d",
            description="Active user rate for churn risk detection",
            strategy={
                "type": "threshold",
                "params": {"direction": "below", "value": 0.35},
            },
        )

    def _renewal(self, suffix: str) -> dict:
        """Threshold on account.days_to_renewal — below 60."""
        return self._float_scenario(
            suffix=suffix,
            name="renewal",
            primitive="account.days_to_renewal",
            description="Days to renewal for early churn warning",
            strategy={
                "type": "threshold",
                "params": {"direction": "below", "value": 60.0},
            },
        )

    def _z_score(self, suffix: str) -> dict:
        """Z-score on service.error_rate_5m — threshold 2.0 above."""
        return self._float_scenario(
            suffix=suffix,
            name="zscore",
            primitive="service.error_rate_5m",
            description="Error rate anomaly detection via z-score",
            strategy={
                "type": "z_score",
                "params": {"threshold": 2.0, "direction": "above"},
            },
        )

    def _change(self, suffix: str) -> dict:
        """Change on user.session_frequency_trend_8w — decrease 0.2."""
        return self._float_scenario(
            suffix=suffix,
            name="change",
            primitive="user.session_frequency_trend_8w",
            description="Session frequency trend decline detection",
            strategy={
                "type": "change",
                "params": {"direction": "decrease", "value": 0.2},
            },
        )

    def _equals(self, suffix: str) -> dict:
        """Equals on account.plan_tier — target 'enterprise'."""
        labels = ["free", "professional", "enterprise"]
        cid = f"mock.concept_equals_{suffix}"
        cond_id = f"mock.cond_equals_{suffix}"
        action_id = f"mock.action_equals_{suffix}"
        return {
            "concept": {
                "concept_id": cid,
                "version": "v1",
                "namespace": "org",
                "output_type": "categorical",
                "labels": labels,
                "description": "Plan tier for segment targeting",
                "primitives": {
                    "account.plan_tier": {
                        "type": "categorical",
                        "missing_data_policy": "zero",
                        "labels": labels,
                    }
                },
                "features": {
                    "tier": {
                        "op": "passthrough",
                        "inputs": {"input": "account.plan_tier"},
                        "params": {},
                    }
                },
                "output_feature": "tier",
            },
            "condition": {
                "condition_id": cond_id,
                "version": "v1",
                "concept_id": cid,
                "concept_version": "v1",
                "namespace": "org",
                "strategy": {
                    "type": "equals",
                    "params": {"value": "enterprise", "labels": labels},
                },
            },
            "action": self._action(action_id, cond_id),
        }

    def _default(self, suffix: str) -> dict:
        """Default: threshold on account.active_user_rate_30d — below 0.5."""
        return self._float_scenario(
            suffix=suffix,
            name="default",
            primitive="account.active_user_rate_30d",
            description="Active user rate — default threshold detection",
            strategy={
                "type": "threshold",
                "params": {"direction": "below", "value": 0.5},
            },
        )

    # ── Shared builders ───────────────────────────────────────────────────────

    def _float_scenario(
        self,
        *,
        suffix: str,
        name: str,
        primitive: str,
        description: str,
        strategy: dict,
    ) -> dict:
        """
        Build a complete task output dict for a float-primitive scenario.

        Uses z_score_op on a float primitive — the same operator used in the
        existing e2e pipeline test (test_pipeline_e2e.py), which is known to
        produce concept_value == raw_input_value under StaticDataConnector.
        """
        cid = f"mock.concept_{name}_{suffix}"
        cond_id = f"mock.cond_{name}_{suffix}"
        action_id = f"mock.action_{name}_{suffix}"
        return {
            "concept": {
                "concept_id": cid,
                "version": "v1",
                "namespace": "org",
                "output_type": "float",
                "description": description,
                "primitives": {
                    primitive: {
                        "type": "float",
                        "missing_data_policy": "zero",
                    }
                },
                "features": {
                    "output": {
                        "op": "z_score_op",
                        "inputs": {"input": primitive},
                        "params": {},
                    }
                },
                "output_feature": "output",
            },
            "condition": {
                "condition_id": cond_id,
                "version": "v1",
                "concept_id": cid,
                "concept_version": "v1",
                "namespace": "org",
                "strategy": strategy,
            },
            "action": self._action(action_id, cond_id),
        }

    @staticmethod
    def _action(action_id: str, condition_id: str) -> dict:
        """Build a complete, resolvable action dict."""
        return {
            "action_id": action_id,
            "version": "v1",
            "namespace": "org",
            "config": {
                "type": "webhook",
                "endpoint": "https://mock.example.com/alert",
            },
            "trigger": {
                "fire_on": "true",
                "condition_id": condition_id,
                "condition_version": "v1",
            },
        }
