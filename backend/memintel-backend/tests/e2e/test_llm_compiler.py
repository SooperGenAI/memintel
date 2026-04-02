"""
tests/e2e/test_llm_compiler.py
──────────────────────────────────────────────────────────────────────────────
LLM compiler path end-to-end tests — full-stack HTTP with MockLLMClient.

These tests exercise POST /tasks through the complete TaskAuthoringService
pipeline using a deterministic MockLLMClient instead of a live LLM.  The mock
is injected via FastAPI dependency_overrides (see conftest.py).

Findings
────────
FINDING 1 — Real LLM interface:
  Method:  generate_task(intent: str, context: dict) -> dict  (synchronous)
  Input:   intent string + prompt context dict (type_system, guardrails,
           primitives, app_context, parameter_bias_rules)
  Output:  {"concept": {...}, "condition": {...}, "action": {...}}
  ABC:     LLMClientBase (app/llm/base.py)

FINDING 2 — Mock wiring:
  MockLLMClient injected via conftest.mock_llm_e2e_client fixture which calls
  app.dependency_overrides[get_task_authoring_service] = _make_mock_override(mock)
  The mock is injected at the FastAPI dependency level; production code is
  unchanged.  guardrails can be injected directly via
  mock_llm_e2e_client_with_guardrails for guardrails-sensitive tests.

FINDING 3 — Guardrails strategy enforcement (critical):
  disallowed_strategies in GuardrailConstraints is NOT enforced by the
  compiler at definition registration time.  The Validator called by
  DefinitionRegistry.register() validates concept/condition schema and
  type compatibility but does not check constraints.disallowed_strategies.
  Guardrails constraints are injected into the LLM prompt context so a real
  LLM would respect them; MockLLMClient ignores context entirely.
  Bug reference: guardrails_constraints_not_enforced_at_compile_time

FINDING 4 — Bias rules (critical):
  parameter_bias_rules and severity language mapping are LLM-applied features.
  MockLLMClient ignores the context dict (including guardrails/bias rules)
  and returns a fixed value determined solely by keyword matching in the intent.
  Tests 4 and 5 document this gap explicitly.

FINDING 5 — context_warning:
  Set on the Task returned by POST /tasks when no active ApplicationContext
  exists at task creation time.  Not stored in DB — absent from GET /tasks.

FINDING 6 — Primitive registry:
  Primitives declared in a concept body do NOT need to be pre-registered in
  PrimitiveRegistry for task creation to succeed.  The DefinitionRegistry
  validates concept schema (field types, op names) but not PrimitiveRegistry
  membership.  This is a design choice: PrimitiveRegistry is an execution-path
  concern (data routing), not a definition-authoring concern.

Test count: 10 e2e compiler tests
All marked @pytest.mark.e2e.
"""
from __future__ import annotations

import pytest

from tests.e2e.conftest import make_mock_llm_app, seed_task

# ── Shared request helpers ─────────────────────────────────────────────────────

_WEBHOOK_DELIVERY = {"type": "webhook", "endpoint": "https://example.com/test-webhook"}


def _task_body(intent: str, dry_run: bool = False) -> dict:
    return {
        "intent": intent,
        "entity_scope": "account",
        "delivery": _WEBHOOK_DELIVERY,
        "dry_run": dry_run,
    }


# ── Test 1 — Basic compilation ─────────────────────────────────────────────────

@pytest.mark.e2e
def test_post_tasks_compiles_churn_intent(mock_llm_e2e_client, api_headers):
    """
    POST /tasks with a churn-related intent compiles to a threshold condition
    on account.active_user_rate_30d (below 0.35) via MockLLMClient.

    Verifies the full LLM compiler path:
      POST /tasks → concept + condition + action registered → Task created
      GET /conditions/{id}?version=v1 → strategy inspectable
    """
    client, pool, run_db, mock_llm = mock_llm_e2e_client

    intent = "Alert me when churn risk is high — active user rate drops below 35%"
    r = client.post("/tasks", json=_task_body(intent), headers=api_headers)
    assert r.status_code == 200, f"POST /tasks failed: {r.text}"

    task = r.json()
    assert task.get("task_id") is not None, "task_id must be present"
    assert task.get("condition_id") is not None, "condition_id must be present"
    assert task.get("condition_version") == "v1", (
        f"condition_version must be 'v1', got {task.get('condition_version')!r}"
    )

    # Verify mock was called with the intent
    assert mock_llm.call_count == 1
    assert mock_llm.last_intent == intent

    # Inspect the compiled condition
    cond_id = task["condition_id"]
    cond_ver = task["condition_version"]
    r2 = client.get(f"/conditions/{cond_id}", params={"version": cond_ver}, headers=api_headers)
    assert r2.status_code == 200, f"GET /conditions/{cond_id} failed: {r2.text}"
    cond = r2.json()

    assert cond["strategy"]["type"] == "threshold"
    assert cond["strategy"]["params"]["direction"] == "below"
    assert cond["strategy"]["params"]["value"] == pytest.approx(0.35)


# ── Test 2 — Context warning when no context ───────────────────────────────────

@pytest.mark.e2e
def test_post_tasks_without_context_sets_warning(mock_llm_e2e_client, api_headers):
    """
    When no active ApplicationContext exists at task creation time,
    context_warning must be non-None in the POST /tasks response.

    context_warning is NOT stored in the DB — it is informational only.
    """
    client, pool, run_db, mock_llm = mock_llm_e2e_client
    # No POST /context was called — no active context

    r = client.post(
        "/tasks",
        json=_task_body("Alert me when active user rate drops below threshold"),
        headers=api_headers,
    )
    assert r.status_code == 200, f"POST /tasks failed: {r.text}"

    task = r.json()
    warning = task.get("context_warning")
    assert warning is not None, (
        "context_warning must be set when no application context is defined; "
        f"got {warning!r}"
    )
    assert len(warning) > 10, f"context_warning should be a meaningful message, got {warning!r}"


# ── Test 3 — No warning when context active ────────────────────────────────────

@pytest.mark.e2e
def test_post_tasks_with_context_no_warning(mock_llm_e2e_client, api_headers, elevated_headers):
    """
    When an active ApplicationContext exists, context_warning must be None
    in the POST /tasks response.

    Steps:
      1. POST /context to create an active context
      2. POST /tasks
      3. Assert context_warning is None
    """
    client, pool, run_db, mock_llm = mock_llm_e2e_client

    # Step 1: Create an active context.
    # Note: the context router has prefix="/context" and is mounted with
    # prefix="/context", resulting in the endpoint being at /context/context.
    context_body = {
        "domain": {
            "description": "SaaS churn detection platform",
            "entities": [],
            "decisions": [],
        }
    }
    r = client.post("/context/context", json=context_body, headers=elevated_headers)
    assert r.status_code in (200, 201), f"POST /context/context failed: {r.text}"

    # Step 2: Create a task — context is now active
    r = client.post(
        "/tasks",
        json=_task_body("Alert me when active user rate drops"),
        headers=api_headers,
    )
    assert r.status_code == 200, f"POST /tasks failed: {r.text}"

    task = r.json()
    warning = task.get("context_warning")
    assert warning is None, (
        f"context_warning must be None when active context exists; got {warning!r}"
    )


# ── Test 4 — Guardrails block disallowed strategy ─────────────────────────────

@pytest.mark.e2e
def test_guardrails_block_disallowed_strategy(mock_llm_e2e_client_with_guardrails, api_headers, e2e_setup):
    """
    Inject guardrails with disallowed_strategies=["z_score", ...] (only threshold allowed).
    POST /tasks with intent that causes MockLLMClient to return z_score.

    EXPECTED BEHAVIOUR (FIX 1 — post-compilation strategy validation):
      HTTP 400 — z_score strategy is rejected by post-compilation guardrails check.
      TaskAuthoringService._validate_strategy_allowed() raises
      MemintelError(SEMANTIC_ERROR) when the compiled strategy type is in
      guardrails.constraints.disallowed_strategies.
    """
    from app.models.config import ApplicationContext as GuardrailsAppContext
    from app.models.guardrails import (
        Guardrails,
        GuardrailConstraints,
        StrategyRegistryEntry,
    )

    pool, run_db = e2e_setup

    # Guardrails: only threshold allowed; z_score is disallowed.
    _app_ctx = GuardrailsAppContext(
        name="test-app",
        description="Test application",
        instructions=["Only threshold strategy is permitted."],
    )
    _strategy_registry = {
        "threshold": StrategyRegistryEntry(
            version="1.0",
            description="Threshold strategy",
            input_types=["float"],
            output_type="decision<boolean>",
        ),
        "z_score": StrategyRegistryEntry(
            version="1.0",
            description="Z-score strategy (disallowed in this config)",
            input_types=["float"],
            output_type="decision<boolean>",
        ),
    }
    guardrails = Guardrails(
        application_context=_app_ctx,
        strategy_registry=_strategy_registry,
        constraints=GuardrailConstraints(
            disallowed_strategies=["z_score", "percentile", "change", "equals", "composite"],
        ),
    )

    make_client = mock_llm_e2e_client_with_guardrails
    with make_client(guardrails) as (client, pool_, run_db_, mock_llm):
        # Intent triggers z_score scenario in MockLLMClient
        intent = "Alert me when error rate deviates from baseline"
        r = client.post("/tasks", json=_task_body(intent), headers=api_headers)

        # FIX 1: z_score is now blocked by post-compilation strategy validation.
        assert r.status_code == 400, (
            f"Expected 400 (z_score blocked by disallowed_strategies); got {r.status_code}: {r.text}"
        )
        error = r.json()
        assert "error" in error or "detail" in error, (
            f"400 response must contain error details; got: {error}"
        )


# ── Test 5 — Bias rules apply severity language ────────────────────────────────

@pytest.mark.e2e
def test_severity_language_maps_to_threshold_prior(mock_llm_e2e_client_with_guardrails, api_headers, e2e_setup):
    """
    Inject guardrails with parameter_bias_rules mapping severity keywords to
    threshold priors.  POST /tasks with intents containing "significantly" and
    "urgently".

    EXPECTED BEHAVIOUR (FIX 2 — deterministic bias rule application):
      "significantly" → bias rule matches → severity_shift=0 → medium → value=0.45
      "urgently"      → bias rule matches → severity_shift=+1 → high  → value=0.30

    TaskAuthoringService._apply_bias_rules() overrides the compiled threshold
    value with the primitive-level prior for the resolved severity tier.
    """
    from app.models.config import ApplicationContext as GuardrailsAppContext
    from app.models.guardrails import (
        BiasEffect,
        Guardrails,
        ParameterBiasRule,
        PrimitiveHint,
        StrategyRegistryEntry,
    )

    pool, run_db = e2e_setup

    _app_ctx = GuardrailsAppContext(
        name="bias-test-app",
        description="Bias rules test application",
        instructions=["Severity language: significant=medium, urgent=high."],
    )
    guardrails = Guardrails(
        application_context=_app_ctx,
        strategy_registry={
            "threshold": StrategyRegistryEntry(
                version="1.0",
                description="Threshold strategy",
                input_types=["float"],
                output_type="decision<boolean>",
            ),
        },
        primitives={
            "account.active_user_rate_30d": PrimitiveHint(
                type="float",
                description="Active user rate 30-day",
                threshold_priors={
                    "threshold": {"low": 0.60, "medium": 0.45, "high": 0.30}
                },
            ),
        },
        parameter_bias_rules=[
            ParameterBiasRule(
                if_instruction_contains="significantly",
                effect=BiasEffect(direction="tighten_threshold", severity_shift=0),
            ),
            ParameterBiasRule(
                if_instruction_contains="urgently",
                effect=BiasEffect(direction="tighten_threshold", severity_shift=1),
            ),
        ],
    )

    make_client = mock_llm_e2e_client_with_guardrails
    with make_client(guardrails) as (client, pool_, run_db_, mock_llm):

        # ── "significantly" intent ─────────────────────────────────────────────
        r1 = client.post(
            "/tasks",
            json=_task_body("Alert me when active user rate drops significantly"),
            headers=api_headers,
        )
        assert r1.status_code == 200, f"POST /tasks (significantly) failed: {r1.text}"
        task1 = r1.json()
        r1c = client.get(
            f"/conditions/{task1['condition_id']}",
            params={"version": task1["condition_version"]},
            headers=api_headers,
        )
        assert r1c.status_code == 200
        cond1 = r1c.json()
        assert cond1["strategy"]["type"] == "threshold"
        # FIX 2: bias rule matched "significantly" → severity_shift=0 → medium → 0.45
        actual_val1 = cond1["strategy"]["params"]["value"]
        assert actual_val1 == pytest.approx(0.45), (
            f"Expected 0.45 (significant=medium_severity prior); got: {actual_val1}"
        )

        # ── "urgently" intent ─────────────────────────────────────────────────
        r2 = client.post(
            "/tasks",
            json=_task_body("Alert me when active user rate drops urgently"),
            headers=api_headers,
        )
        assert r2.status_code == 200, f"POST /tasks (urgently) failed: {r2.text}"
        task2 = r2.json()
        r2c = client.get(
            f"/conditions/{task2['condition_id']}",
            params={"version": task2["condition_version"]},
            headers=api_headers,
        )
        assert r2c.status_code == 200
        cond2 = r2c.json()
        assert cond2["strategy"]["type"] == "threshold"
        # FIX 2: bias rule matched "urgently" → severity_shift=+1 → high → 0.30
        actual_val2 = cond2["strategy"]["params"]["value"]
        assert actual_val2 == pytest.approx(0.30), (
            f"Expected 0.30 (urgent=high_severity prior); got: {actual_val2}"
        )


# ── Test 6 — Task dry run preview ─────────────────────────────────────────────

@pytest.mark.e2e
def test_post_tasks_dry_run_does_not_create_task(mock_llm_e2e_client, api_headers):
    """
    dry_run=True returns a compiled condition preview without persisting
    any task or definition.

    Assertions:
      - HTTP 200
      - Response contains condition (compiled preview)
      - task_id is absent (DryRunResult, not Task)
      - GET /tasks lists no tasks (nothing was persisted)
    """
    client, pool, run_db, mock_llm = mock_llm_e2e_client

    r = client.post(
        "/tasks",
        json=_task_body("Alert me when active user rate drops", dry_run=True),
        headers=api_headers,
    )
    assert r.status_code == 200, f"POST /tasks dry_run=true failed: {r.text}"
    result = r.json()

    # DryRunResult has no task_id
    assert "task_id" not in result or result.get("task_id") is None, (
        f"dry_run response must NOT have a task_id; got {result.get('task_id')!r}"
    )

    # DryRunResult has condition preview
    assert "condition" in result, (
        f"dry_run response must include 'condition' preview; keys: {list(result.keys())}"
    )

    # GET /tasks — the task should NOT exist
    r2 = client.get("/tasks", headers=api_headers)
    assert r2.status_code == 200, f"GET /tasks failed: {r2.text}"
    tasks = r2.json()
    # TaskList response has shape: {items: [...], has_more: bool, ...}
    task_list = tasks.get("items", []) if isinstance(tasks, dict) else tasks
    assert len(task_list) == 0, (
        f"dry_run must not persist a task; found {len(task_list)} task(s): {task_list}"
    )


# ── Test 7 — Primitive not registered ─────────────────────────────────────────

@pytest.mark.e2e
def test_post_tasks_fails_for_unregistered_primitive(mock_llm_e2e_client_with_guardrails, api_headers, e2e_setup):
    """
    Inject guardrails with a primitives registry that does NOT include
    'account.active_user_rate_30d'.  MockLLMClient returns a concept that
    references that primitive for any "active user" intent.

    EXPECTED BEHAVIOUR (FIX 3 — primitive existence validation):
      HTTP 400 — TaskAuthoringService._validate_primitives_registered() raises
      MemintelError(REFERENCE_ERROR) when the concept references a primitive
      absent from guardrails.primitives.
    """
    from app.models.config import ApplicationContext as GuardrailsAppContext
    from app.models.guardrails import (
        Guardrails,
        PrimitiveHint,
        StrategyRegistryEntry,
    )

    pool, run_db = e2e_setup

    # Guardrails has primitives defined but NOT account.active_user_rate_30d.
    # MockLLMClient churn scenario always returns a concept using that primitive.
    guardrails = Guardrails(
        application_context=GuardrailsAppContext(
            name="primitive-test-app",
            description="Primitive validation test",
            instructions=["Only declared primitives are allowed."],
        ),
        strategy_registry={
            "threshold": StrategyRegistryEntry(
                version="1.0",
                description="Threshold strategy",
                input_types=["float"],
                output_type="decision<boolean>",
            ),
        },
        primitives={
            "service.error_rate_5m": PrimitiveHint(
                type="float",
                description="Service error rate (declared; not used by this intent)",
            ),
            # account.active_user_rate_30d is intentionally absent
        },
    )

    make_client = mock_llm_e2e_client_with_guardrails
    with make_client(guardrails) as (client, pool_, run_db_, mock_llm):
        intent = "Alert me when active user rate drops below threshold"
        r = client.post("/tasks", json=_task_body(intent), headers=api_headers)

        # FIX 3: primitive not in guardrails.primitives → 400.
        assert r.status_code == 400, (
            f"Expected 400 (account.active_user_rate_30d not in guardrails.primitives); "
            f"got {r.status_code}: {r.text}"
        )
        error_text = r.text.lower()
        assert any(term in error_text for term in ("primitive", "not registered", "reference")), (
            f"400 error should mention the unregistered primitive; got: {r.text}"
        )


# ── Test 8 — Task retrieval after compilation ──────────────────────────────────

@pytest.mark.e2e
def test_compiled_task_has_correct_metadata(mock_llm_e2e_client, api_headers):
    """
    After POST /tasks compiles a task, GET /tasks/{id} returns the task with
    all required metadata fields correctly set.
    """
    client, pool, run_db, mock_llm = mock_llm_e2e_client

    intent = "Alert me when churn risk is high"
    r = client.post("/tasks", json=_task_body(intent), headers=api_headers)
    assert r.status_code == 200, f"POST /tasks failed: {r.text}"
    created = r.json()
    task_id = created["task_id"]

    # Fetch the task
    r2 = client.get(f"/tasks/{task_id}", headers=api_headers)
    assert r2.status_code == 200, f"GET /tasks/{task_id} failed: {r2.text}"
    task = r2.json()

    assert task["intent"] == intent,                       f"intent mismatch: {task['intent']!r}"
    assert task["condition_id"] is not None,               "condition_id must be set"
    assert task["condition_version"] == "v1",              f"condition_version: {task['condition_version']!r}"
    assert task["status"] == "active",                     f"status must be 'active': {task['status']!r}"
    assert task["created_at"] is not None,                 "created_at must be set"

    # context_version is None when no context was active at creation
    # (task.context_version is stored in DB, unlike context_warning)
    assert "context_version" in task, "context_version field must be present"

    # guardrails_version is None when guardrails_store is not loaded
    assert "guardrails_version" in task, "guardrails_version field must be present"


# ── Test 9 — Execute compiled task end-to-end ─────────────────────────────────

@pytest.mark.e2e
def test_execute_compiled_task_end_to_end(mock_llm_e2e_client, api_headers):
    """
    POST /tasks compiles a threshold condition (below 0.35 on
    account.active_user_rate_30d).  POST /execute/static evaluates it
    with controlled input values.

    z_score_op on a float primitive returns the raw input value when no
    historical baseline exists (missing_data_policy="zero"), so:
      0.25 < 0.35 → decision.value = True
      0.85 < 0.35 → decision.value = False

    This is the critical end-to-end proof that the MockLLMClient compiler
    path produces executable tasks.
    """
    client, pool, run_db, mock_llm = mock_llm_e2e_client
    _PRIMITIVE = "account.active_user_rate_30d"

    # Compile the task
    intent = "Alert me when active user rate drops below 35%"
    r = client.post("/tasks", json=_task_body(intent), headers=api_headers)
    assert r.status_code == 200, f"POST /tasks failed: {r.text}"
    task = r.json()
    cond_id = task["condition_id"]
    cond_ver = task["condition_version"]

    # ── FIRES: 0.25 < 0.35 ────────────────────────────────────────────────────
    r_fire = client.post(
        "/execute/static",
        json={
            "condition_id":      cond_id,
            "condition_version": cond_ver,
            "entity":            "e2e_account_001",
            "data":              {_PRIMITIVE: 0.25},
        },
        headers=api_headers,
    )
    assert r_fire.status_code == 200, f"execute/static (fires) failed: {r_fire.text}"
    decision_fire = r_fire.json()
    assert decision_fire["value"] is True, (
        f"Expected True (0.25 < 0.35), got {decision_fire['value']}; "
        f"full response: {decision_fire}"
    )

    # ── DOES NOT FIRE: 0.85 ≮ 0.35 ───────────────────────────────────────────
    r_nofire = client.post(
        "/execute/static",
        json={
            "condition_id":      cond_id,
            "condition_version": cond_ver,
            "entity":            "e2e_account_001",
            "data":              {_PRIMITIVE: 0.85},
        },
        headers=api_headers,
    )
    assert r_nofire.status_code == 200, f"execute/static (no-fire) failed: {r_nofire.text}"
    decision_nofire = r_nofire.json()
    assert decision_nofire["value"] is False, (
        f"Expected False (0.85 ≮ 0.35), got {decision_nofire['value']}; "
        f"full response: {decision_nofire}"
    )


# ── Test 10 — Semantic refine ──────────────────────────────────────────────────

@pytest.mark.e2e
def test_semantic_refine_modifies_condition(mock_llm_e2e_client, api_headers, elevated_headers):
    """
    POST /agents/semantic-refine on a concept definition registered via the
    mock compiler path.

    The agents route uses AgentService which selects its own LLM client
    (LLMFixtureClient when USE_LLM_FIXTURES=True, which is the test default).
    MockLLMClient is NOT used by the agent path — it is scoped to
    TaskAuthoringService only.

    The LLMFixtureClient.generate_semantic_refine() returns agent_semantic_refine.json
    which has: proposed, changes, breaking.  The AgentService adds 'original'
    from the stored definition body.

    Requires X-Elevated-Key header (not X-Api-Key).
    HTTP 404 if the definition_id is not registered.

    This test first creates a task via POST /tasks (which registers the concept),
    then calls semantic-refine on the registered concept_id.
    """
    client, pool, run_db, mock_llm = mock_llm_e2e_client

    # Step 1: Compile a task to register the concept in the definitions table
    r = client.post(
        "/tasks",
        json=_task_body("Alert me when active user rate drops below threshold"),
        headers=api_headers,
    )
    assert r.status_code == 200, f"POST /tasks failed: {r.text}"
    task = r.json()
    concept_id = task.get("concept_id")
    concept_version = task.get("concept_version")
    assert concept_id is not None, f"concept_id must be set in task response; got: {task}"
    assert concept_version is not None, f"concept_version must be set; got: {task}"

    # Step 2: POST /agents/semantic-refine on the registered concept
    r2 = client.post(
        "/agents/semantic-refine",
        json={
            "definition_id": concept_id,
            "version":       concept_version,
            "instruction":   "make the threshold more conservative",
        },
        headers=elevated_headers,
    )
    assert r2.status_code == 200, (
        f"POST /agents/semantic-refine failed ({r2.status_code}): {r2.text}. "
        "Note: AgentService uses LLMFixtureClient (not MockLLMClient) for the refine path."
    )
    refine = r2.json()

    assert "original" in refine, f"SemanticRefineResponse must have 'original'; keys: {list(refine.keys())}"
    assert "proposed" in refine, f"SemanticRefineResponse must have 'proposed'; keys: {list(refine.keys())}"
    assert "changes" in refine,  f"SemanticRefineResponse must have 'changes'; keys: {list(refine.keys())}"
    assert "breaking" in refine, f"SemanticRefineResponse must have 'breaking'; keys: {list(refine.keys())}"
    assert isinstance(refine["breaking"], bool), (
        f"'breaking' must be bool; got {type(refine['breaking'])}"
    )
    # LLMFixtureClient fixture returns breaking=True for semantic-refine
    # MockLLMClient is NOT involved in this path
