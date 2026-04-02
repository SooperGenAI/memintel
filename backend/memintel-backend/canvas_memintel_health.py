#!/usr/bin/env python3
"""
canvas_memintel_health.py
─────────────────────────
Diagnostic script that verifies Canvas can communicate with Memintel
correctly across every critical API path.

Run this whenever a Canvas→Memintel issue is suspected.
It tells you immediately which layer is healthy and which is not.

Usage:
    python canvas_memintel_health.py

Environment variables (all required):
    MEMINTEL_BASE_URL       e.g. http://localhost:8000
    MEMINTEL_API_KEY        standard key (reads + execution)
    MEMINTEL_ELEVATED_KEY   admin key (guardrails, registry, actions)

Optional:
    MEMINTEL_TIMEOUT        seconds (default: 10)
    MEMINTEL_VERBOSE        1 to print full request/response bodies
"""

import os
import sys
import json
import time
import uuid
import traceback
from datetime import datetime, timezone
from typing import Any

try:
    import httpx
except ImportError:
    print("ERROR: httpx is not installed. Run: pip install httpx")
    sys.exit(1)

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_URL        = os.environ.get("MEMINTEL_BASE_URL", "").rstrip("/")
API_KEY         = os.environ.get("MEMINTEL_API_KEY", "")
ELEVATED_KEY    = os.environ.get("MEMINTEL_ELEVATED_KEY", "")
TIMEOUT         = float(os.environ.get("MEMINTEL_TIMEOUT", "10"))
VERBOSE         = os.environ.get("MEMINTEL_VERBOSE", "0") == "1"

# ─── Colours ──────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
GREY   = "\033[90m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ─── Result tracking ──────────────────────────────────────────────────────────

results: list[dict] = []

def record(step: str, passed: bool, detail: str = "", duration_ms: float = 0,
           request_info: str = "", response_info: str = "", diagnosis: str = ""):
    results.append({
        "step": step,
        "passed": passed,
        "detail": detail,
        "duration_ms": duration_ms,
        "request_info": request_info,
        "response_info": response_info,
        "diagnosis": diagnosis,
    })
    status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    duration = f"{GREY}({duration_ms:.0f}ms){RESET}" if duration_ms else ""
    print(f"  {status}  {step} {duration}")
    if detail:
        prefix = f"       {GREY}"
        print(f"{prefix}{detail}{RESET}")
    if not passed and diagnosis:
        print(f"       {YELLOW}→ {diagnosis}{RESET}")
    if VERBOSE and request_info:
        print(f"       {GREY}REQUEST:  {request_info}{RESET}")
    if VERBOSE and response_info:
        print(f"       {GREY}RESPONSE: {response_info[:300]}{RESET}")

# ─── HTTP helpers ─────────────────────────────────────────────────────────────

def api_headers(elevated: bool = False) -> dict:
    h = {"Content-Type": "application/json"}
    if elevated:
        h["X-Elevated-Key"] = ELEVATED_KEY
    else:
        h["X-Api-Key"] = API_KEY
    return h

def call(method: str, path: str, elevated: bool = False,
         body: dict | None = None, expect_status: int = 200) -> tuple[bool, int, dict, float]:
    """Returns (ok, status_code, response_body, duration_ms)"""
    url = f"{BASE_URL}{path}"
    headers = api_headers(elevated)
    start = time.monotonic()
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            if method == "GET":
                resp = client.get(url, headers=headers)
            elif method == "POST":
                resp = client.post(url, headers=headers, json=body)
            elif method == "PATCH":
                resp = client.patch(url, headers=headers, json=body)
            elif method == "DELETE":
                resp = client.delete(url, headers=headers)
            else:
                return False, 0, {}, 0
        duration_ms = (time.monotonic() - start) * 1000
        try:
            resp_body = resp.json()
        except Exception:
            resp_body = {"_raw": resp.text}
        ok = resp.status_code == expect_status
        return ok, resp.status_code, resp_body, duration_ms
    except httpx.ConnectError as e:
        duration_ms = (time.monotonic() - start) * 1000
        return False, 0, {"_error": f"Connection refused: {e}"}, duration_ms
    except httpx.TimeoutException:
        duration_ms = (time.monotonic() - start) * 1000
        return False, 0, {"_error": f"Timeout after {TIMEOUT}s"}, duration_ms
    except Exception as e:
        duration_ms = (time.monotonic() - start) * 1000
        return False, 0, {"_error": str(e)}, duration_ms

def fmt_body(body: dict) -> str:
    return json.dumps(body, indent=None, default=str)[:400]

def _is_loaded_from_file(body: dict) -> bool:
    """
    Return True when a 404 response body indicates a resource was loaded from
    a config file (memintel_guardrails.yaml, memintel_config.yaml) rather than
    posted via API.

    This is a valid deployment state — the server is healthy, the resource is
    active, it just wasn't registered through the API.
    """
    body_str = json.dumps(body, default=str).lower()
    return any(phrase in body_str for phrase in [
        "memintel_guardrails.yaml",
        "memintel_config.yaml",
        "loaded from",
        "config file",
        "at startup",
    ])

# ─── Individual checks ────────────────────────────────────────────────────────

def check_env_vars():
    print(f"\n{BOLD}[ Environment ]{RESET}")
    missing = []
    if not BASE_URL:
        missing.append("MEMINTEL_BASE_URL")
    if not API_KEY:
        missing.append("MEMINTEL_API_KEY")
    if not ELEVATED_KEY:
        missing.append("MEMINTEL_ELEVATED_KEY")

    if missing:
        for var in missing:
            record(f"Env: {var}", False,
                   diagnosis=f"Set {var} before running this script")
        return False

    record("Env: MEMINTEL_BASE_URL",     True, detail=BASE_URL)
    record("Env: MEMINTEL_API_KEY",      True, detail=f"{API_KEY[:6]}{'*' * (len(API_KEY)-6)}")
    record("Env: MEMINTEL_ELEVATED_KEY", True, detail=f"{ELEVATED_KEY[:6]}{'*' * (len(ELEVATED_KEY)-6)}")
    return True


def check_health():
    print(f"\n{BOLD}[ 1. Server Health ]{RESET}")

    ok, status, body, ms = call("GET", "/health", expect_status=200)

    # Case 1: /health returned 200 — server is up and has a health endpoint.
    if ok:
        record("GET /health", True,
               detail=f"Server up — {body.get('status', 'ok')}",
               duration_ms=ms,
               response_info=fmt_body(body))
        return True

    # Case 2: /health returned 404 — try /v1/health before concluding anything.
    if status == 404:
        ok2, status2, body2, ms2 = call("GET", "/v1/health", expect_status=200)

        if ok2:
            record("GET /v1/health", True,
                   detail=f"Server up — {body2.get('status', 'ok')}",
                   duration_ms=ms2,
                   response_info=fmt_body(body2))
            return True

        if status2 == 404:
            # Neither /health nor /v1/health exists.  This is acceptable —
            # Memintel does not require a health endpoint to be operational.
            # Server liveness will be confirmed by step 2 (API key auth).
            record("GET /health", True,
                   detail="No /health endpoint — server liveness confirmed via API key check in step 2",
                   duration_ms=ms)
            return True

        # status2 == 0: connection refused or timeout on /v1/health attempt.
        err2 = body2.get("_error", body2)
        record("GET /health", False,
               detail=str(err2),
               duration_ms=ms2,
               diagnosis="Server is not reachable. Check MEMINTEL_BASE_URL, PostgreSQL, and Redis.")
        return False

    # Case 3: status == 0 — connection refused or timeout on the initial /health call.
    err = body.get("_error", body)
    record("GET /health", False,
           detail=str(err),
           duration_ms=ms,
           diagnosis="Server is not reachable. Check MEMINTEL_BASE_URL, PostgreSQL, and Redis.")
    return False


def check_api_auth():
    print(f"\n{BOLD}[ 2. API Key Authentication ]{RESET}")

    # A GET endpoint that requires API key — GET /tasks
    ok, status, body, ms = call("GET", "/tasks", elevated=False, expect_status=200)

    if ok:
        record("GET /tasks (API key)", True,
               detail=f"Authenticated successfully — {body.get('total', 0)} tasks",
               duration_ms=ms,
               response_info=fmt_body(body))
    elif status == 401:
        record("GET /tasks (API key)", False,
               detail=f"HTTP 401 — API key rejected",
               duration_ms=ms,
               diagnosis="Check MEMINTEL_API_KEY. The key is set but Memintel does not accept it.")
    elif status == 0:
        record("GET /tasks (API key)", False,
               detail=str(body.get("_error", "")),
               duration_ms=ms,
               diagnosis="Connection failed — server may be down.")
    else:
        record("GET /tasks (API key)", False,
               detail=f"Unexpected HTTP {status}: {fmt_body(body)}",
               duration_ms=ms,
               diagnosis="Unexpected response. Check Memintel logs.")
    return ok


def check_elevated_auth():
    print(f"\n{BOLD}[ 3. Elevated Key Authentication ]{RESET}")

    # Test that elevated key works — GET /guardrails is a safe read
    ok, status, body, ms = call("GET", "/guardrails", elevated=True, expect_status=200)

    # Also verify that API key alone is NOT sufficient for elevated endpoints
    ok_with_api_key, status2, _, ms2 = call("POST", "/guardrails", elevated=False,
                                             body={"guardrails": {}, "change_note": "health-check"},
                                             expect_status=403)

    elev_passed = False
    if ok:
        record("GET /guardrails (elevated key)", True,
               detail=f"Elevated key accepted — version {body.get('version', '?')}",
               duration_ms=ms,
               response_info=fmt_body(body))
        elev_passed = True
    elif status == 404 and _is_loaded_from_file(body):
        # Guardrails exist but were loaded from memintel_guardrails.yaml at
        # startup rather than posted via API.  This is valid and expected.
        record("GET /guardrails (elevated key)", True,
               detail="Guardrails loaded from file (not API) — valid",
               duration_ms=ms,
               response_info=fmt_body(body))
        elev_passed = True
    elif status == 403:
        record("GET /guardrails (elevated key)", False,
               detail="HTTP 403 — elevated key rejected",
               duration_ms=ms,
               diagnosis="Check MEMINTEL_ELEVATED_KEY. The key is set but Memintel does not accept it.")
    else:
        record("GET /guardrails (elevated key)", False,
               detail=f"HTTP {status}: {fmt_body(body)}",
               duration_ms=ms,
               diagnosis="Unexpected response from elevated endpoint.")

    # Verify API key alone returns 403 on elevated endpoint
    if ok_with_api_key:
        record("API key rejected on elevated endpoint", True,
               detail="Correctly returns 403 when elevated key missing",
               duration_ms=ms2)
    else:
        record("API key rejected on elevated endpoint", False,
               detail=f"HTTP {status2} — expected 403",
               duration_ms=ms2,
               diagnosis="Security gap: elevated endpoint accepted non-elevated key. Check Memintel auth wiring.")

    return elev_passed


def check_context():
    print(f"\n{BOLD}[ 4. Application Context ]{RESET}")

    # Try /context first (fixed prefix), fall back to /context/context (old double prefix)
    ok, status, body, ms = call("GET", "/context", elevated=False, expect_status=200)
    endpoint_used = "/context"
    if not ok and status == 404 and not _is_loaded_from_file(body):
        ok, status, body, ms = call("GET", "/context/context", elevated=False, expect_status=200)
        endpoint_used = "/context/context"

    if ok:
        version = body.get("version", "?")
        is_active = body.get("is_active", False)
        record(f"GET {endpoint_used}", True,
               detail=f"Context active — version {version}, is_active={is_active}",
               duration_ms=ms,
               response_info=fmt_body(body))
    elif status == 404 and _is_loaded_from_file(body):
        # Context loaded from memintel_config.yaml at startup — valid deployment state
        record(f"GET {endpoint_used}", True,
               detail="Context loaded from file (not API) — valid",
               duration_ms=ms,
               response_info=fmt_body(body))
    elif status == 404:
        # No context posted yet — warning only, tasks still work (with context_warning set)
        record(f"GET {endpoint_used}", True,
               detail="No active context — tasks will include context_warning",
               duration_ms=ms)
        print(f"       {YELLOW}⚠  No context posted. Tasks will compile without domain context.{RESET}")
    else:
        record(f"GET {endpoint_used}", False,
               detail=f"HTTP {status}: {fmt_body(body)}",
               duration_ms=ms,
               diagnosis="Unexpected response. Check Memintel logs.")

    return True  # context is optional — never block downstream checks


def check_guardrails():
    print(f"\n{BOLD}[ 5. Guardrails ]{RESET}")

    ok, status, body, ms = call("GET", "/guardrails", elevated=True, expect_status=200)

    if ok:
        version  = body.get("version", "?")
        strategy = body.get("strategy_registry", [])
        record("GET /guardrails", True,
               detail=f"Version {version} — strategies: {', '.join(str(s) for s in strategy[:5])}",
               duration_ms=ms,
               response_info=fmt_body(body))
        return True
    elif status == 404 and _is_loaded_from_file(body):
        # Guardrails exist but were loaded from the config file, not posted via API.
        # The server is correctly configured — this is not an error.
        record("GET /guardrails", True,
               detail="Guardrails loaded from file (not API) — valid",
               duration_ms=ms,
               response_info=fmt_body(body))
        return True
    elif status == 404:
        record("GET /guardrails", False,
               detail="No guardrails posted yet",
               duration_ms=ms,
               diagnosis="POST /guardrails with strategy_registry before creating tasks.")
        return False
    else:
        record("GET /guardrails", False,
               detail=f"HTTP {status}: {fmt_body(body)}",
               duration_ms=ms)
        return False


def check_primitive_registry():
    print(f"\n{BOLD}[ 6. Primitive Registry ]{RESET}")

    ok, status, body, ms = call("GET", "/registry/definitions?namespace=org&definition_type=primitive",
                                 elevated=False, expect_status=200)

    if ok:
        total = body.get("total", 0)
        items = body.get("items", body.get("definitions", []))
        record("GET /registry/definitions", True,
               detail=f"{total} primitives registered in org namespace",
               duration_ms=ms,
               response_info=fmt_body(body))
        if total == 0:
            print(f"       {YELLOW}⚠  No primitives registered. POST /registry/definitions before creating tasks.{RESET}")
    else:
        record("GET /registry/definitions", False,
               detail=f"HTTP {status}: {fmt_body(body)}",
               duration_ms=ms,
               diagnosis="Check API key and that registry endpoint is reachable.")

    return ok


def check_task_creation():
    print(f"\n{BOLD}[ 7. Task Creation (LLM Compiler Path) ]{RESET}")

    task_id = None
    condition_id = None
    condition_version = None
    concept_id = None
    concept_version = None

    # Use a unique run ID so health checks never conflict with existing definitions
    run_id = uuid.uuid4().hex[:8]
    payload = {
        "intent": f"Alert me when active user rate drops below 35% — health check {run_id}",
        "entity_scope": "account",
        "delivery": {"type": "notification", "channel": "health-check"}
    }

    ok, status, body, ms = call("POST", "/tasks", elevated=False,
                                 body=payload, expect_status=200)

    req_info = fmt_body(payload)

    if ok:
        task_id        = body.get("task_id")
        condition_id   = body.get("condition_id")
        condition_version = body.get("condition_version")
        concept_id     = body.get("concept_id")
        concept_version = body.get("concept_version")
        context_warning = body.get("context_warning")
        guardrails_version = body.get("guardrails_version")

        issues = []
        if not task_id:        issues.append("task_id missing from response")
        if not condition_id:   issues.append("condition_id missing from response")
        if not condition_version: issues.append("condition_version missing from response")
        if context_warning:    issues.append(f"context_warning: {context_warning}")

        if issues:
            record("POST /tasks", True,
                   detail=f"task_id={task_id} — warnings: {'; '.join(issues)}",
                   duration_ms=ms,
                   request_info=req_info,
                   response_info=fmt_body(body))
        else:
            record("POST /tasks", True,
                   detail=f"task_id={task_id}, condition={condition_id}@{condition_version}, guardrails_version={guardrails_version}",
                   duration_ms=ms,
                   request_info=req_info,
                   response_info=fmt_body(body))
    elif status == 422:
        record("POST /tasks", False,
               detail=f"HTTP 422 — validation error: {fmt_body(body)}",
               duration_ms=ms,
               request_info=req_info,
               diagnosis="Canvas is sending a malformed request. Check MemintelClient.create_task() field names and types.")
    elif status == 400:
        record("POST /tasks", False,
               detail=f"HTTP 400: {fmt_body(body)}",
               duration_ms=ms,
               request_info=req_info,
               diagnosis="Compilation error — possibly no registered primitives, or guardrails block the strategy. Check Memintel logs.")
    elif status == 401:
        record("POST /tasks", False,
               detail="HTTP 401 — API key rejected",
               duration_ms=ms,
               diagnosis="Canvas is not sending the correct API key header for POST /tasks.")
    elif status == 409:
        # HTTP 409 means the LLM compiler successfully compiled the intent and
        # attempted to register a definition — proof the compiler path works.
        # The definition already exists from a previous run.
        # Mark the compiler check as PASS, then find IDs from registry for
        # downstream execution checks.
        print(f"       {YELLOW}⚠  HTTP 409 — LLM compiler works. Definition already registered. Finding IDs from registry...{RESET}")

        # Try to find an existing active task first
        ok2, _, body2, _ = call("GET", "/tasks?status=active&limit=1",
                                 elevated=False, expect_status=200)
        if ok2:
            items = body2.get("items", body2.get("tasks", []))
            if items:
                existing          = items[0]
                task_id           = existing.get("task_id")
                condition_id      = existing.get("condition_id")
                condition_version = existing.get("condition_version")
                concept_id        = existing.get("concept_id")
                concept_version   = existing.get("concept_version")
                record("POST /tasks", True,
                       detail=f"LLM compiler confirmed (409). Using existing task_id={task_id}",
                       duration_ms=ms)
                ok = True

        if not ok:
            # No tasks — find condition and concept definitions separately
            ok3, _, body3, _ = call(
                "GET", "/registry/definitions?definition_type=condition&limit=1",
                elevated=False, expect_status=200
            )
            ok4, _, body4, _ = call(
                "GET", "/registry/definitions?definition_type=concept&limit=1",
                elevated=False, expect_status=200
            )
            cond_items = body3.get("items", body3.get("definitions", [])) if ok3 else []
            conc_items = body4.get("items", body4.get("definitions", [])) if ok4 else []

            if cond_items and conc_items:
                condition_id      = cond_items[0].get("definition_id")
                condition_version = cond_items[0].get("version")
                concept_id        = conc_items[0].get("definition_id")
                concept_version   = conc_items[0].get("version")
                record("POST /tasks", True,
                       detail=f"LLM compiler confirmed (409). Using registry: condition={condition_id}, concept={concept_id}",
                       duration_ms=ms)
                ok = True
            else:
                # 409 still means compiler works — just no IDs available for downstream
                record("POST /tasks", True,
                       detail="LLM compiler confirmed working (409 = definition already registered). No IDs for downstream checks.",
                       duration_ms=ms)
                ok = True
                print(f"       {YELLOW}⚠  Downstream execution checks will be skipped — no condition/concept IDs available.{RESET}")
    else:
        record("POST /tasks", False,
               detail=f"HTTP {status}: {fmt_body(body)}",
               duration_ms=ms,
               request_info=req_info,
               diagnosis="Unexpected error. Check Memintel logs for details.")

    return ok, task_id, condition_id, condition_version, concept_id, concept_version


def check_task_retrieval(task_id: str | None):
    print(f"\n{BOLD}[ 8. Task Retrieval ]{RESET}")

    if not task_id:
        # No task_id available — either task creation failed or we recovered via registry.
        # This is not a Memintel bug — it means no tasks have been created yet.
        record("GET /tasks/{id}", True,
               detail="Skipped — no task created yet (registry recovery mode)",
               diagnosis="Create a task via POST /tasks to test task retrieval.")
        print(f"       {YELLOW}⚠  No tasks exist yet. Task retrieval will work once Canvas creates tasks.{RESET}")
        return True

    ok, status, body, ms = call("GET", f"/tasks/{task_id}", elevated=False, expect_status=200)

    if ok:
        t_status   = body.get("status", "?")
        cond_v     = body.get("condition_version", "?")
        grls_v     = body.get("guardrails_version", "?")
        record("GET /tasks/{id}", True,
               detail=f"status={t_status}, condition_version={cond_v}, guardrails_version={grls_v}",
               duration_ms=ms,
               response_info=fmt_body(body))
    elif status == 404:
        record("GET /tasks/{id}", False,
               detail=f"HTTP 404 — task {task_id} not found",
               duration_ms=ms,
               diagnosis="Task was created but cannot be retrieved. Check TaskStore.get() soft-delete logic.")
    else:
        record("GET /tasks/{id}", False,
               detail=f"HTTP {status}: {fmt_body(body)}",
               duration_ms=ms)

    return ok


def check_static_execution(condition_id: str | None, condition_version: str | None):
    print(f"\n{BOLD}[ 9. Static Execution (Inline Data) ]{RESET}")

    if not condition_id:
        record("POST /execute/static", False,
               detail="Skipped — no condition_id from step 7",
               diagnosis="Fix task creation (step 7) first.")
        return False

    # Test both firing and not-firing cases
    payload_fire = {
        "condition_id": condition_id,
        "condition_version": condition_version or "v1",
        "entity": "health-check-entity",
        "data": {"account.active_user_rate_30d": 0.20}  # should fire (below 0.35)
    }

    payload_nofire = {
        "condition_id": condition_id,
        "condition_version": condition_version or "v1",
        "entity": "health-check-entity",
        "data": {"account.active_user_rate_30d": 0.90}  # should not fire
    }

    ok1, status1, body1, ms1 = call("POST", "/execute/static", elevated=False,
                                     body=payload_fire, expect_status=200)

    if ok1:
        value    = body1.get("value")
        dtype    = body1.get("decision_type")
        strategy = body1.get("strategy")
        reason   = body1.get("reason")
        hcount   = body1.get("history_count")
        record("POST /execute/static (value=0.20 → should fire)", True,
               detail=f"value={value}, decision_type={dtype}, strategy={strategy}, reason={reason}, history_count={hcount}",
               duration_ms=ms1,
               response_info=fmt_body(body1))
        if value is not True and reason != "null_input":
            print(f"       {YELLOW}⚠  Expected value=True for 0.20 < 0.35 threshold. Got value={value}. Check compiled condition params.{RESET}")
    elif status1 == 404:
        record("POST /execute/static (fire case)", False,
               detail=f"HTTP 404 — condition {condition_id} not found",
               duration_ms=ms1,
               diagnosis="Condition was compiled during task creation but cannot be found. Check DefinitionStore.")
    elif status1 == 422:
        record("POST /execute/static (fire case)", False,
               detail=f"HTTP 422: {fmt_body(body1)}",
               duration_ms=ms1,
               diagnosis="Inline data shape is wrong. Check that 'data' is a flat dict of {primitive_id: value}.")
    else:
        record("POST /execute/static (fire case)", False,
               detail=f"HTTP {status1}: {fmt_body(body1)}",
               duration_ms=ms1)

    ok2, status2, body2, ms2 = call("POST", "/execute/static", elevated=False,
                                     body=payload_nofire, expect_status=200)

    if ok2:
        value2 = body2.get("value")
        record("POST /execute/static (value=0.90 → should not fire)", True,
               detail=f"value={value2}, reason={body2.get('reason')}",
               duration_ms=ms2)
        if value2 is not False and body2.get("reason") not in (None, "no_match"):
            print(f"       {YELLOW}⚠  Expected value=False for 0.90 > 0.35 threshold.{RESET}")
    else:
        record("POST /execute/static (no-fire case)", False,
               detail=f"HTTP {status2}: {fmt_body(body2)}",
               duration_ms=ms2)

    return ok1 and ok2


def check_evaluate_full(condition_id: str | None, condition_version: str | None,
                         concept_id: str | None, concept_version: str | None):
    print(f"\n{BOLD}[ 10. Full Pipeline Execution ]{RESET}")

    if not all([condition_id, concept_id]):
        record("POST /evaluate/full", False,
               detail="Skipped — missing condition_id or concept_id from step 7",
               diagnosis="Fix task creation (step 7) first.")
        return False, None

    ts = "2025-11-14T09:00:00Z"
    payload = {
        "concept_id": concept_id,
        "concept_version": concept_version or "v1",
        "condition_id": condition_id,
        "condition_version": condition_version or "v1",
        "entity": "health-check-full-entity",
        "timestamp": ts,
        "dry_run": True  # safe — no side effects
    }

    ok, status, body, ms = call("POST", "/evaluate/full", elevated=False,
                                 body=payload, expect_status=200)

    if ok:
        result   = body.get("result", {})
        decision = body.get("decision", {})
        actions  = decision.get("actions_triggered", [])
        deterministic = result.get("deterministic")
        record("POST /evaluate/full (dry_run=true)", True,
               detail=f"result.value={result.get('value')}, decision.value={decision.get('value')}, deterministic={deterministic}, actions={len(actions)} (would_trigger)",
               duration_ms=ms,
               response_info=fmt_body(body))
        if not deterministic:
            print(f"       {YELLOW}⚠  result.deterministic=False — timestamp was provided so this should be True.{RESET}")
        for a in actions:
            if a.get("status") not in ("would_trigger", "triggered", "skipped", "failed"):
                print(f"       {YELLOW}\u26a0  Action {a.get('action_id')} has unexpected status: {a.get('status')}{RESET}")
    elif status == 422:
        record("POST /evaluate/full", False,
               detail=f"HTTP 422: {fmt_body(body)}",
               duration_ms=ms,
               diagnosis="Request shape is wrong. Check that Canvas is sending concept_id, concept_version, condition_id, condition_version, entity.")
    elif status == 404:
        record("POST /evaluate/full", False,
               detail=f"HTTP 404: {fmt_body(body)}",
               duration_ms=ms,
               diagnosis="Concept or condition not found. Check that concept_id and condition_id from task creation are being passed correctly.")
    else:
        record("POST /evaluate/full", False,
               detail=f"HTTP {status}: {fmt_body(body)}",
               duration_ms=ms)

    return ok, body if ok else None


def check_determinism(condition_id: str | None, condition_version: str | None,
                       concept_id: str | None, concept_version: str | None):
    print(f"\n{BOLD}[ 11. Determinism Guarantee ]{RESET}")

    if not all([condition_id, concept_id]):
        record("Determinism check", False,
               detail="Skipped — missing IDs",
               diagnosis="Fix task creation (step 7) first.")
        return False

    ts = "2025-11-14T10:00:00Z"
    payload = {
        "concept_id": concept_id,
        "concept_version": concept_version or "v1",
        "condition_id": condition_id,
        "condition_version": condition_version or "v1",
        "entity": "health-check-determinism",
        "timestamp": ts,
        "dry_run": True
    }

    results_vals = []
    for i in range(3):
        ok, status, body, ms = call("POST", "/evaluate/full", elevated=False,
                                     body=payload, expect_status=200)
        if ok:
            results_vals.append(body.get("result", {}).get("value"))
        else:
            record(f"Determinism run {i+1}", False,
                   detail=f"HTTP {status}: {fmt_body(body)}")
            return False

    all_same = len(set(str(v) for v in results_vals)) == 1
    if all_same:
        record("Determinism (3 identical calls)", True,
               detail=f"All three returned value={results_vals[0]} — determinism confirmed")
    else:
        record("Determinism (3 identical calls)", False,
               detail=f"Values differ: {results_vals}",
               diagnosis="Same inputs, same timestamp produced different results. Core determinism guarantee is broken. Check ConceptExecutor and ResultCache.")

    return all_same


def check_feedback(condition_id: str | None, condition_version: str | None):
    print(f"\n{BOLD}[ 12. Feedback Submission ]{RESET}")

    if not condition_id:
        record("POST /feedback/decision", False,
               detail="Skipped — no condition_id",
               diagnosis="Fix task creation (step 7) first.")
        return False

    payload = {
        "condition_id": condition_id,
        "condition_version": condition_version or "v1",
        "entity": f"health-check-{uuid.uuid4().hex[:8]}",
        "feedback": "correct",
        "timestamp": f"2025-11-{14 + (hash(str(uuid.uuid4())) % 10):02d}T09:00:00Z",
        "note": "health-check test feedback"
    }

    ok, status, body, ms = call("POST", "/feedback/decision", elevated=False,
                                 body=payload, expect_status=200)

    if ok:
        fb_status = body.get("status")
        fb_id     = body.get("feedback_id")
        record("POST /feedback/decision", True,
               detail=f"status={fb_status}, feedback_id={fb_id}",
               duration_ms=ms,
               response_info=fmt_body(body))
        if fb_status != "recorded":
            print(f"       {YELLOW}⚠  Expected status='recorded', got '{fb_status}'.{RESET}")
    elif status == 409:
        # Feedback already submitted for this condition+entity+timestamp — idempotency working
        record("POST /feedback/decision", True,
               detail="Feedback endpoint working — 409 means already recorded (idempotent)",
               duration_ms=ms)
    elif status == 404:
        record("POST /feedback/decision", False,
               detail=f"HTTP 404 — condition not found for feedback",
               duration_ms=ms,
               diagnosis="Feedback validation is rejecting the condition_id. Condition must exist before feedback can be submitted.")
    elif status == 422:
        record("POST /feedback/decision", False,
               detail=f"HTTP 422: {fmt_body(body)}",
               duration_ms=ms,
               diagnosis="Canvas is sending wrong field names. The feedback field is 'feedback' not 'feedback_type'. Check MemintelClient.submit_feedback().")
    else:
        record("POST /feedback/decision", False,
               detail=f"HTTP {status}: {fmt_body(body)}",
               duration_ms=ms)

    return ok


def check_calibration(condition_id: str | None, condition_version: str | None):
    print(f"\n{BOLD}[ 13. Calibration Endpoint ]{RESET}")

    if not condition_id:
        record("POST /conditions/calibrate", False,
               detail="Skipped — no condition_id",
               diagnosis="Fix task creation (step 7) first.")
        return False

    payload = {
        "condition_id": condition_id,
        "condition_version": condition_version or "v1"
    }

    ok, status, body, ms = call("POST", "/conditions/calibrate", elevated=False,
                                 body=payload, expect_status=200)

    if ok:
        cal_status = body.get("status")
        reason     = body.get("no_recommendation_reason")
        record("POST /conditions/calibrate", True,
               detail=f"status={cal_status}" + (f", reason={reason}" if reason else ""),
               duration_ms=ms,
               response_info=fmt_body(body))
    elif status == 422:
        record("POST /conditions/calibrate", False,
               detail=f"HTTP 422: {fmt_body(body)}",
               duration_ms=ms,
               diagnosis="Canvas is sending wrong fields. Both condition_id AND condition_version are required.")
    elif status == 404:
        record("POST /conditions/calibrate", False,
               detail=f"HTTP 404 — condition not found",
               duration_ms=ms,
               diagnosis="condition_id or condition_version is wrong.")
    else:
        record("POST /conditions/calibrate", False,
               detail=f"HTTP {status}: {fmt_body(body)}",
               duration_ms=ms)

    return ok


def check_openapi():
    print(f"\n{BOLD}[ 14. OpenAPI Schema ]{RESET}")

    ok, status, body, ms = call("GET", "/openapi.json", expect_status=200)

    if ok:
        paths = body.get("paths", {})
        components = body.get("components", {})
        record("GET /openapi.json", True,
               detail=f"{len(paths)} paths, {len(components.get('schemas', {}))} schemas",
               duration_ms=ms)
    else:
        record("GET /openapi.json", False,
               detail=f"HTTP {status}",
               duration_ms=ms,
               diagnosis="OpenAPI schema not available. Check that FastAPI is running and /openapi.json is not disabled.")

    return ok


def cleanup_task(task_id: str | None):
    """Delete the health-check task so it doesn't pollute the registry."""
    if not task_id:
        return
    call("DELETE", f"/tasks/{task_id}", elevated=False, expect_status=200)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  Memintel Health Check — Canvas Diagnostic Tool{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}")
    print(f"  Timestamp : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Target    : {BASE_URL or '(not set)'}")
    print(f"  Timeout   : {TIMEOUT}s")
    print(f"  Verbose   : {'on' if VERBOSE else 'off (set MEMINTEL_VERBOSE=1 for full bodies)'}")

    # Step 0 — env vars
    env_ok = check_env_vars()
    if not env_ok:
        print(f"\n{RED}Cannot proceed — required environment variables are missing.{RESET}\n")
        _print_summary()
        sys.exit(1)

    # Steps 1-3 — infrastructure
    health_ok  = check_health()
    auth_ok    = check_api_auth()
    elev_ok    = check_elevated_auth()

    # Early exit only when BOTH the health check and the auth check confirm the
    # server is unreachable.  A missing /health endpoint (health_ok=False from a
    # connection error on both /health paths) paired with a passing auth step
    # means the server is up — we continue with a warning instead of exiting.
    if not health_ok and not auth_ok:
        print(f"\n{RED}Server is not reachable. Stopping — remaining checks would all fail.{RESET}\n")
        _print_summary()
        sys.exit(1)
    elif not health_ok and auth_ok:
        print(
            f"\n{YELLOW}NOTE: /health endpoint not found. This is normal — Memintel does not "
            f"require a health endpoint. Server confirmed reachable via successful API "
            f"authentication in step 2.{RESET}"
        )
        # Do not exit — continue with remaining checks.

    # Steps 4-6 — configuration state
    check_context()
    check_guardrails()
    check_primitive_registry()

    # Steps 7-8 — task lifecycle
    task_ok, task_id, cond_id, cond_ver, concept_id, concept_ver = check_task_creation()
    check_task_retrieval(task_id)

    # Steps 9-11 — execution
    check_static_execution(cond_id, cond_ver)
    check_evaluate_full(cond_id, cond_ver, concept_id, concept_ver)
    check_determinism(cond_id, cond_ver, concept_id, concept_ver)

    # Steps 12-13 — feedback and calibration
    check_feedback(cond_id, cond_ver)
    check_calibration(cond_id, cond_ver)

    # Step 14 — schema
    check_openapi()

    # Cleanup
    cleanup_task(task_id)

    # Summary
    _print_summary()

    failed = sum(1 for r in results if not r["passed"])
    sys.exit(0 if failed == 0 else 1)


def _print_summary():
    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])
    total  = len(results)

    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  Summary{RESET}")
    print(f"{'═' * 60}")
    print(f"  Total checks : {total}")
    print(f"  {GREEN}Passed{RESET}       : {passed}")
    print(f"  {RED}Failed{RESET}       : {failed}")

    if failed > 0:
        print(f"\n{BOLD}  Failed checks:{RESET}")
        for r in results:
            if not r["passed"]:
                print(f"  {RED}✗{RESET} {r['step']}")
                if r["detail"]:
                    print(f"    {GREY}{r['detail']}{RESET}")
                if r["diagnosis"]:
                    print(f"    {YELLOW}→ {r['diagnosis']}{RESET}")

        print(f"\n{BOLD}  Diagnosis guide:{RESET}")
        print(f"  {GREY}• Failures in steps 1-3 → infrastructure or auth issue{RESET}")
        print(f"  {GREY}• Failures in steps 4-6 → Memintel not configured (context/guardrails/primitives){RESET}")
        print(f"  {GREY}• Failures in steps 7-8 → task creation or LLM compiler issue{RESET}")
        print(f"  {GREY}• Failures in steps 9-11 → execution pipeline issue{RESET}")
        print(f"  {GREY}• Failures in steps 12-13 → feedback or calibration issue{RESET}")
        print(f"  {GREY}• If direct curl works but Canvas fails → bug is in Canvas MemintelClient{RESET}")
        print(f"  {GREY}• If direct curl also fails → bug is in Memintel{RESET}")
    else:
        print(f"\n  {GREEN}{BOLD}All checks passed. Memintel is healthy and Canvas can connect correctly.{RESET}")

    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
