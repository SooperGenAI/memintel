"""
tests/integration/conftest_v7.py
──────────────────────────────────────────────────────────────────────────────
Shared mock infrastructure for Memintel V7 cross-module integration tests.

Registered as a pytest plugin via:
  pytest_plugins = ["tests.integration.conftest_v7"]
in tests/integration/conftest.py.

Provides
────────
LLMMockClient
    Deterministic LLM stub implementing LLMClientBase.
    Routes by context["step"] for ConceptCompilerService (steps 1-4).
    Routes by intent substring for TaskAuthoringService (no step key).
    Tracks call_count for assertion in pre-LLM validation tests.

LOAN_PRIMITIVES
    Mock primitive values for the loan domain, keyed by signal name.

Fixtures
────────
llm_mock         — fresh LLMMockClient (call_count=0)
loan_compile_request — CompileConceptRequest for loan.repayment_ratio
loan_task_request    — CreateTaskRequest for an overdue loan alert
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import httpx

from app.llm.base import LLMClientBase
from app.models.concept_compile import CompileConceptRequest
from app.models.task import CreateTaskRequest, DeliveryConfig, DeliveryType


# ── Step-3 responses for the concept compiler ─────────────────────────────────
# Step 3 (DAG Construction) requires formula_summary + signal_bindings.
# Steps 1, 2, 4 only need summary + outcome.

_STEP3_REPAYMENT = {
    "summary": "Repayment ratio = payments_on_time / payments_due",
    "outcome": "accepted",
    "formula_summary": "payments_on_time / payments_due",
    "signal_bindings": [
        {"signal_name": "payments_on_time", "role": "numerator"},
        {"signal_name": "payments_due", "role": "denominator"},
    ],
}

_STEP3_OVERDUE = {
    "summary": "Days overdue = loan.days_overdue",
    "outcome": "accepted",
    "formula_summary": "loan.days_overdue",
    "signal_bindings": [
        {"signal_name": "loan.days_overdue", "role": "input"},
    ],
}

_STEP3_CREDIT = {
    "summary": "Credit score passthrough",
    "outcome": "accepted",
    "formula_summary": "loan.credit_score",
    "signal_bindings": [
        {"signal_name": "loan.credit_score", "role": "input"},
    ],
}

_STEP3_GENERIC = {
    "summary": "Generic formula: signal_a",
    "outcome": "accepted",
    "formula_summary": "signal_a",
    "signal_bindings": [
        {"signal_name": "signal_a", "role": "input"},
    ],
}

# keyword → step 3 response (searched by substring in lower-case intent)
_STEP3_MAP: dict[str, dict] = {
    "repayment": _STEP3_REPAYMENT,
    "overdue":   _STEP3_OVERDUE,
    "credit":    _STEP3_CREDIT,
}


# ── Full task-authoring responses ─────────────────────────────────────────────
# These are returned when generate_task() is called by TaskAuthoringService
# (no "step" key in context).  Must include concept, condition, action keys.

_TASK_REPAYMENT = {
    "concept": {
        "concept_id": "loan.repayment_ratio",
        "version": "v1",
        "namespace": "org",
        "output_type": "float",
        "description": "Loan repayment ratio",
        "primitives": {
            "loan.payments_on_time": {"type": "float", "missing_data_policy": "zero"},
            "loan.payments_due":     {"type": "float", "missing_data_policy": "zero"},
        },
        "features": {
            "output": {
                "op": "divide",
                "inputs": {
                    "a": "loan.payments_on_time",
                    "b": "loan.payments_due",
                },
                "params": {},
            }
        },
        "output_feature": "output",
    },
    "condition": {
        "condition_id":      "loan.repayment_below_threshold",
        "version":           "v1",
        "concept_id":        "loan.repayment_ratio",
        "concept_version":   "v1",
        "namespace":         "org",
        "strategy": {
            "type":   "threshold",
            "params": {"direction": "below", "value": 0.8},
        },
    },
    "action": {
        "action_id": "loan.alert_action",
        "version":   "v1",
        "namespace": "org",
        "config": {"type": "webhook", "endpoint": "https://loan.example.com/hook"},
        "trigger": {
            "fire_on":            "true",
            "condition_id":       "loan.repayment_below_threshold",
            "condition_version":  "v1",
        },
    },
}

_TASK_OVERDUE = {
    "concept": {
        "concept_id":    "loan.days_overdue_concept",
        "version":       "v1",
        "namespace":     "org",
        "output_type":   "float",
        "description":   "Number of days overdue",
        "primitives": {
            "loan.days_overdue": {"type": "float", "missing_data_policy": "zero"},
        },
        "features": {
            "output": {
                "op":     "z_score_op",
                "inputs": {"input": "loan.days_overdue"},
                "params": {},
            }
        },
        "output_feature": "output",
    },
    "condition": {
        "condition_id":    "loan.overdue_condition",
        "version":         "v1",
        "concept_id":      "loan.days_overdue_concept",
        "concept_version": "v1",
        "namespace":       "org",
        "strategy": {
            "type":   "threshold",
            "params": {"direction": "above", "value": 30},
        },
    },
    "action": {
        "action_id": "loan.overdue_alert",
        "version":   "v1",
        "namespace": "org",
        "config": {"type": "webhook", "endpoint": "https://loan.example.com/overdue"},
        "trigger": {
            "fire_on":           "true",
            "condition_id":      "loan.overdue_condition",
            "condition_version": "v1",
        },
    },
}

_TASK_GENERIC = {
    "concept": {
        "concept_id":  "mock.generic_concept",
        "version":     "v1",
        "namespace":   "org",
        "output_type": "float",
        "description": "Generic mock concept",
        "primitives": {
            "mock.signal_a": {"type": "float", "missing_data_policy": "zero"},
        },
        "features": {
            "output": {
                "op":     "z_score_op",
                "inputs": {"input": "mock.signal_a"},
                "params": {},
            }
        },
        "output_feature": "output",
    },
    "condition": {
        "condition_id":    "mock.generic_condition",
        "version":         "v1",
        "concept_id":      "mock.generic_concept",
        "concept_version": "v1",
        "namespace":       "org",
        "strategy": {
            "type":   "threshold",
            "params": {"direction": "above", "value": 0.5},
        },
    },
    "action": {
        "action_id": "mock.generic_action",
        "version":   "v1",
        "namespace": "org",
        "config": {"type": "webhook", "endpoint": "https://mock.example.com/hook"},
        "trigger": {
            "fire_on":           "true",
            "condition_id":      "mock.generic_condition",
            "condition_version": "v1",
        },
    },
}

# keyword → task response (searched by substring in lower-case intent)
_TASK_MAP: dict[str, dict] = {
    "repayment": _TASK_REPAYMENT,
    "overdue":   _TASK_OVERDUE,
}


# ── LLMMockClient ─────────────────────────────────────────────────────────────

class LLMMockClient(LLMClientBase):
    """
    Deterministic LLM stub for V7 cross-module integration tests.

    Routing rules
    ─────────────
    Concept compiler (context has a "step" key):
      step 1, 2, 4 → minimal accepted response (no extra keys required).
      step 3       → formula_summary + signal_bindings, routed by intent keyword.

    Task authoring (no "step" key in context):
      Routes by intent substring (lower-case).  Falls back to _TASK_GENERIC.

    call_count is incremented on every call, allowing tests to assert that the
    LLM was (or was not) called before a validation error is raised.
    """

    def __init__(self) -> None:
        self.call_count: int = 0

    def reset_call_count(self) -> None:
        self.call_count = 0

    def generate_task(self, intent: str, context: dict) -> dict:
        self.call_count += 1
        step = context.get("step")

        # ── Concept compiler path ──────────────────────────────────────────────
        if step in (1, 2):
            return {"summary": f"Step {step} completed.", "outcome": "accepted"}

        if step == 3:
            intent_lower = intent.lower()
            for keyword, resp in _STEP3_MAP.items():
                if keyword in intent_lower:
                    return resp
            return _STEP3_GENERIC

        if step == 4:
            return {"summary": "Type validation passed.", "outcome": "accepted"}

        # ── Task authoring path ────────────────────────────────────────────────
        intent_lower = intent.lower()
        for keyword, resp in _TASK_MAP.items():
            if keyword in intent_lower:
                return resp
        return _TASK_GENERIC


# ── Loan domain primitive values ──────────────────────────────────────────────
# Mock values that MockConnector / ExecutionEngine would return at runtime.
# Not used directly in service tests but available for execution-layer tests.

LOAN_PRIMITIVES: dict[str, float] = {
    "loan.repayment_ratio":     0.65,
    "loan.days_overdue":        45.0,
    "loan.outstanding_balance": 12_500.0,
    "loan.credit_score":        620.0,
}


# ── Shared async helpers (used by test_v7_cross_functional.py) ────────────────

async def compile_and_register(
    client: "httpx.AsyncClient",
    identifier: str,
    description: str,
    output_type: str,
    signal_names: list[str],
) -> tuple[str, str]:
    """
    Run POST /concepts/compile then POST /concepts/register.

    Returns (concept_id, compile_token).  Asserts HTTP 201 at each step so
    a failing assertion fails fast at the call site in the test.
    """
    compile_resp = await client.post("/concepts/compile", json={
        "identifier": identifier,
        "description": description,
        "output_type": output_type,
        "signal_names": signal_names,
        "return_reasoning": False,
        "stream": False,
    })
    assert compile_resp.status_code == 201, (
        f"compile failed ({compile_resp.status_code}): {compile_resp.text}"
    )
    token = compile_resp.json()["compile_token"]

    register_resp = await client.post("/concepts/register", json={
        "compile_token": token,
        "identifier": identifier,
    })
    assert register_resp.status_code == 201, (
        f"register failed ({register_resp.status_code}): {register_resp.text}"
    )
    concept_id = register_resp.json()["concept_id"]
    return concept_id, token


# ── Pytest fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def llm_mock() -> LLMMockClient:
    """Fresh LLMMockClient with call_count reset to 0."""
    return LLMMockClient()


@pytest.fixture
def loan_compile_request() -> CompileConceptRequest:
    """CompileConceptRequest for the loan repayment ratio concept (non-streaming)."""
    return CompileConceptRequest(
        identifier="loan.repayment_ratio",
        description="Ratio of on-time payments to total payments due over 90 days",
        output_type="float",
        signal_names=["payments_on_time", "payments_due"],
        stream=False,
        return_reasoning=False,
    )


@pytest.fixture
def loan_task_request() -> CreateTaskRequest:
    """CreateTaskRequest for an overdue loan alert task (non-streaming)."""
    return CreateTaskRequest(
        intent="alert when loan is overdue by more than 30 days",
        entity_scope="loan",
        delivery=DeliveryConfig(
            type=DeliveryType.WEBHOOK,
            endpoint="https://loan.example.com/hook",
        ),
        stream=False,
        return_reasoning=False,
    )
