"""
test_errors.py
------------------------------------------------------------------------------
Error handling edge case tests — HTTP against http://localhost:8000

Sections:
  1. 404 Not Found
  2. 422 Validation errors
  3. 409 Conflict
  4. 403 Auth (missing elevated key)
  5. 400 Semantic / type errors (Validator unit tests — CompileService is a stub)
"""
import json
import sys

import httpx

BASE = "http://localhost:8000"
ELEVATED_KEY = "test-elevated-key"
API_KEY = "test-key"

HEADERS = {
    "X-API-Key": API_KEY,
    "X-Elevated-Key": ELEVATED_KEY,
    "Content-Type": "application/json",
}
HEADERS_NO_ELEVATED = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
}

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

results = []


def _extract_error_type(body: dict) -> str | None:
    """Extract error_type from canonical ErrorResponse shape: body["error"]["type"]."""
    if not isinstance(body, dict):
        return None
    # Canonical shape: {"error": {"type": "...", "message": "..."}}
    error = body.get("error")
    if isinstance(error, dict):
        return error.get("type")
    # Fallback: flat shape (shouldn't happen but be defensive)
    return body.get("error_type")


def check(label: str, expected_status: int, resp: httpx.Response,
          expected_error_type: str | None = None) -> None:
    content_type = resp.headers.get("content-type", "")
    body = resp.json() if "application/json" in content_type else {}
    actual_status = resp.status_code
    actual_type = _extract_error_type(body)

    status_ok = actual_status == expected_status
    type_ok = expected_error_type is None or actual_type == expected_error_type
    passed = status_ok and type_ok

    results.append(passed)
    tag = PASS if passed else FAIL
    print(f"  {tag}  {label}")
    print(f"         expected={expected_status} got={actual_status}", end="")
    if expected_error_type:
        print(f"  error_type expected={expected_error_type!r} got={actual_type!r}", end="")
    print()
    if not passed:
        short = json.dumps(body)[:300]
        print(f"         body: {short}")


def check_unit(label: str, passed: bool, detail: str = "") -> None:
    results.append(passed)
    tag = PASS if passed else FAIL
    print(f"  {tag}  {label}")
    if not passed and detail:
        print(f"         {detail}")


# --------------------------------------------------------------------------
# SECTION 1 — 404 Not Found
# --------------------------------------------------------------------------
print("\n=== SECTION 1 — 404 Not Found ===\n")

with httpx.Client(base_url=BASE, headers=HEADERS, timeout=10) as client:

    # 1a. GET /tasks/{id} — unknown task UUID
    r = client.get("/tasks/00000000-0000-0000-0000-000000000000")
    check("GET /tasks/00000000-... (unknown task)", 404, r)

    # 1b. GET /conditions/{id}?version=... — unknown condition
    r = client.get("/conditions/no_such_condition?version=9.9")
    check("GET /conditions/no_such_condition?version=9.9", 404, r)

    # 1c. GET /registry/definitions/{id}/lineage — unknown definition
    r = client.get("/registry/definitions/does.not.exist/lineage")
    check("GET /registry/definitions/does.not.exist/lineage", 404, r)

    # 1d. GET /registry/definitions/{id}/versions — unknown definition
    #     NOTE: This endpoint returns 200 with an empty list, not 404.
    r = client.get("/registry/definitions/does.not.exist/versions")
    body = r.json()
    passed = r.status_code == 200 and body == []
    results.append(passed)
    tag = PASS if passed else FAIL
    print(f"  {tag}  GET /registry/definitions/does.not.exist/versions (returns 200 + [])")
    print(f"         got status={r.status_code} body={json.dumps(body)[:80]}")


# --------------------------------------------------------------------------
# SECTION 2 — 422 Validation errors
# --------------------------------------------------------------------------
print("\n=== SECTION 2 — 422 Validation errors ===\n")

with httpx.Client(base_url=BASE, headers=HEADERS, timeout=10) as client:

    # 2a. POST /execute/static — completely empty body
    r = client.post("/execute/static", json={})
    check("POST /execute/static (empty body)", 422, r)

    # 2b. POST /execute/static — missing required 'data' field
    r = client.post("/execute/static", json={
        "condition_id": "high_revenue",
        "condition_version": "1.0",
        "entity": "acct_1",
        # 'data' field intentionally omitted
    })
    check("POST /execute/static (missing data field)", 422, r)

    # 2c. POST /compile — missing required 'concept' field
    r = client.post("/compile", json={})
    check("POST /compile (empty body)", 422, r)

    # 2d. POST /compile — concept missing required 'output_feature' (Pydantic rejects)
    r = client.post("/compile", json={"concept": {
        "concept_id": "org.bad_concept",
        "version": "1.0",
        "namespace": "org",
        "output_type": "float",
        "primitives": {},
        "features": {},
        # 'output_feature' intentionally omitted
    }})
    check("POST /compile (concept missing output_feature)", 422, r)

    # 2e. POST /compile — output_feature not in features (Pydantic model validator)
    r = client.post("/compile", json={"concept": {
        "concept_id": "org.bad_output",
        "version": "1.0",
        "namespace": "org",
        "output_type": "float",
        "primitives": {"x": {"type": "float"}},
        "features": {
            "result": {"op": "passthrough", "inputs": {"input": "x"}, "params": {}}
        },
        "output_feature": "nonexistent_feature",  # not in features
    }})
    check("POST /compile (output_feature not in features, Pydantic rejects)", 422, r)

    # 2f. POST /registry/definitions — wrong type for body field (string not object)
    r = client.post("/registry/definitions", json={
        "definition_id": "bad_def",
        "version": "1.0",
        "definition_type": "primitive",
        "namespace": "org",
        "body": "this should be an object not a string",
    })
    check("POST /registry/definitions (body is string not object)", 422, r)


# --------------------------------------------------------------------------
# SECTION 3 — 409 Conflict
# --------------------------------------------------------------------------
print("\n=== SECTION 3 — 409 Conflict ===\n")

# Re-register existing definitions with DIFFERENT bodies → conflict.
# The registry is immutable — same (id, version) with different body → 409.

with httpx.Client(base_url=BASE, headers=HEADERS, timeout=10) as client:

    # 3a. Re-register revenue primitive v1.0 with different missing_data_policy
    r = client.post("/registry/definitions", json={
        "definition_id": "revenue",
        "version": "1.0",
        "definition_type": "primitive",
        "namespace": "org",
        "body": {
            "id": "revenue",
            "type": "float",
            "version": "1.0",
            "namespace": "org",
            "missing_data_policy": "zero",  # original is "null"
        },
    })
    check("POST /registry/definitions (revenue primitive conflict)", 409, r, "conflict")

    # 3b. Re-register high_revenue condition v1.0 with different threshold
    r = client.post("/registry/definitions", json={
        "definition_id": "high_revenue",
        "version": "1.0",
        "definition_type": "condition",
        "namespace": "org",
        "body": {
            "concept_id": "revenue_value",
            "concept_version": "1.0",
            "condition_id": "high_revenue",
            "version": "1.0",
            "namespace": "org",
            "strategy": {
                "type": "threshold",
                "threshold": 5000.0,  # original is 1000.0
                "operator": "gt",
            },
        },
    })
    check("POST /registry/definitions (high_revenue condition conflict)", 409, r, "conflict")

    # 3c. Re-register org.churn_risk_score concept v1.0 with different params
    r = client.post("/registry/definitions", json={
        "definition_id": "org.churn_risk_score",
        "version": "1.0",
        "definition_type": "concept",
        "namespace": "org",
        "body": {
            "concept_id": "org.churn_risk_score",
            "version": "1.0",
            "namespace": "org",
            "output_type": "float",
            "primitives": {
                "engagement_score": {"type": "float", "missing_data_policy": "null"}
            },
            "features": {
                "churn_score": {
                    "op": "normalize",
                    "inputs": {"input": "engagement_score"},
                    "params": {"min": 0.0, "max": 200.0},  # changed params
                }
            },
            "output_feature": "churn_score",
        },
    })
    check("POST /registry/definitions (org.churn_risk_score concept conflict)", 409, r, "conflict")


# --------------------------------------------------------------------------
# SECTION 4 — 403 Auth (missing elevated key)
# --------------------------------------------------------------------------
print("\n=== SECTION 4 — 403 Auth (missing elevated key) ===\n")

with httpx.Client(base_url=BASE, headers=HEADERS_NO_ELEVATED, timeout=10) as client:

    # 4a. POST /registry/definitions without elevated key
    r = client.post("/registry/definitions", json={
        "definition_id": "revenue",
        "version": "2.0",
        "definition_type": "primitive",
        "namespace": "org",
        "body": {"id": "revenue", "type": "float"},
    })
    check("POST /registry/definitions (no elevated key)", 403, r, "auth_error")

    # 4b. POST /compile without elevated key
    r = client.post("/compile", json={"concept": {
        "concept_id": "org.test",
        "version": "1.0",
        "namespace": "org",
        "output_type": "float",
        "primitives": {"x": {"type": "float"}},
        "features": {
            "f": {"op": "passthrough", "inputs": {"input": "x"}, "params": {}}
        },
        "output_feature": "f",
    }})
    check("POST /compile (no elevated key)", 403, r, "auth_error")

    # 4c. POST /agents/semantic-refine without elevated key
    r = client.post("/agents/semantic-refine", json={
        "definition_id": "org.churn_risk_score",
        "version": "1.0",
        "instruction": "Add recency decay",
    })
    check("POST /agents/semantic-refine (no elevated key)", 403, r, "auth_error")


# --------------------------------------------------------------------------
# SECTION 5 — 400 Semantic / type errors (Validator unit tests)
# --------------------------------------------------------------------------
# NOTE: CompileService.compile() / compile_semantic() / explain_plan() are
# unimplemented stubs (the class has only __init__). All HTTP compile paths
# return 500 AttributeError. These tests exercise the Validator directly.
print("\n=== SECTION 5 — 400 Semantic / type errors (Validator unit tests) ===\n")
print("  NOTE: CompileService methods are unimplemented (TODO stub) — testing Validator directly.\n")

from app.compiler.validator import Validator
from app.models.concept import ConceptDefinition
from app.models.errors import ErrorType

def make_concept(**overrides) -> ConceptDefinition:
    base = {
        "concept_id": "org.test",
        "version": "1.0",
        "namespace": "org",
        "output_type": "float",
        "primitives": {"x": {"type": "float", "missing_data_policy": "null"}},
        "features": {
            "result": {"op": "passthrough", "inputs": {"input": "x"}, "params": {}}
        },
        "output_feature": "result",
    }
    base.update(overrides)
    return ConceptDefinition(**base)


validator = Validator()

# 5a. Unknown operator -> reference_error
try:
    bad_op = ConceptDefinition(
        concept_id="org.bad_op",
        version="1.0",
        namespace="org",
        output_type="float",
        primitives={"x": {"type": "float", "missing_data_policy": "null"}},
        features={"result": {"op": "nonexistent_operator", "inputs": {"input": "x"}, "params": {}}},
        output_feature="result",
    )
    errors = validator.validate(bad_op)
    found = any(e.type == ErrorType.REFERENCE_ERROR for e in errors)
    check_unit("Validator: unknown operator -> reference_error",
               found,
               f"errors={[(e.type, e.message) for e in errors]}")
except Exception as exc:
    check_unit("Validator: unknown operator -> reference_error", False, str(exc))

# 5b. ConceptDefinition rejects categorical without labels at the Pydantic model level.
#     The Pydantic model validator catches this before Validator.validate() is called.
from pydantic import ValidationError as PydanticValidationError
try:
    bad_cat = ConceptDefinition(
        concept_id="org.bad_cat",
        version="1.0",
        namespace="org",
        output_type="categorical",
        labels=None,
        primitives={"x": {"type": "float", "missing_data_policy": "null"}},
        features={"result": {"op": "passthrough", "inputs": {"input": "x"}, "params": {}}},
        output_feature="result",
    )
    check_unit("ConceptDefinition: categorical without labels -> Pydantic ValidationError", False,
               "No exception raised — Pydantic should have rejected this")
except PydanticValidationError as exc:
    found = "labels" in str(exc).lower() or "categorical" in str(exc).lower()
    check_unit("ConceptDefinition: categorical without labels -> Pydantic ValidationError", found,
               str(exc)[:200])
except Exception as exc:
    check_unit("ConceptDefinition: categorical without labels -> Pydantic ValidationError", False, str(exc))

# 5c. Feature input references undeclared primitive -> syntax_error (Phase 1)
try:
    bad_prim = ConceptDefinition(
        concept_id="org.bad_prim",
        version="1.0",
        namespace="org",
        output_type="float",
        primitives={},  # empty — 'x' not declared
        features={"result": {"op": "passthrough", "inputs": {"input": "x"}, "params": {}}},
        output_feature="result",
    )
    errors = validator.validate(bad_prim)
    found = any(e.type == ErrorType.SYNTAX_ERROR for e in errors)
    check_unit("Validator: undeclared primitive input -> syntax_error",
               found,
               f"errors={[(e.type, e.message) for e in errors]}")
except Exception as exc:
    check_unit("Validator: undeclared primitive input -> syntax_error", False, str(exc))

# 5d. Circular feature dependency -> graph_error
try:
    bad_cycle = ConceptDefinition(
        concept_id="org.bad_cycle",
        version="1.0",
        namespace="org",
        output_type="float",
        primitives={"x": {"type": "float", "missing_data_policy": "null"}},
        features={
            "a": {"op": "passthrough", "inputs": {"input": "b"}, "params": {}},
            "b": {"op": "passthrough", "inputs": {"input": "a"}, "params": {}},
        },
        output_feature="a",
    )
    errors = validator.validate(bad_cycle)
    found = any(e.type == ErrorType.GRAPH_ERROR for e in errors)
    check_unit("Validator: circular dependency -> graph_error",
               found,
               f"errors={[(e.type, e.message) for e in errors]}")
except Exception as exc:
    check_unit("Validator: circular dependency -> graph_error", False, str(exc))

# 5e. CompileService stub returns 500 for unimplemented compile()
with httpx.Client(base_url=BASE, headers=HEADERS, timeout=10) as client:
    r = client.post("/compile", json={"concept": {
        "concept_id": "org.test_stub",
        "version": "1.0",
        "namespace": "org",
        "output_type": "float",
        "primitives": {"x": {"type": "float", "missing_data_policy": "null"}},
        "features": {
            "result": {"op": "passthrough", "inputs": {"input": "x"}, "params": {}}
        },
        "output_feature": "result",
    }})
    # Expect 500 because CompileService.compile() is not yet implemented.
    passed = r.status_code == 500
    results.append(passed)
    tag = PASS if passed else FAIL
    print(f"  {tag}  POST /compile (valid concept) -> 500 (CompileService is a TODO stub)")
    print(f"         got={r.status_code} (expected 500 until CompileService is implemented)")


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
total = len(results)
passed = sum(results)
failed = total - passed

print(f"\n{'=' * 60}")
print(f"Results: {passed}/{total} passed", end="")
if failed:
    print(f"  ({failed} FAILED)", end="")
print()

sys.exit(0 if failed == 0 else 1)
