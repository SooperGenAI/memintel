import httpx, json

base = "http://localhost:8000"
h = {"X-Elevated-Key": "test-elevated-key", "Content-Type": "application/json"}

results = []

def reg(label, path, body):
    r = httpx.post(f"{base}{path}", json=body, headers=h, timeout=10)
    ok = r.status_code in (200, 201)
    results.append((label, r.status_code, "PASS" if ok else "FAIL", r.text if not ok else ""))

# ── STEP 1: Primitives ──────────────────────────────────────────────────────
reg("primitive: revenue",        "/registry/definitions", {"definition_id": "revenue",        "version": "1.0", "definition_type": "primitive", "namespace": "org", "body": {"type": "float",              "missing_data_policy": "null"}})
reg("primitive: revenue_series", "/registry/definitions", {"definition_id": "revenue_series", "version": "1.0", "definition_type": "primitive", "namespace": "org", "body": {"type": "time_series<float>",   "missing_data_policy": "null"}})
reg("primitive: account_tier",   "/registry/definitions", {"definition_id": "account_tier",   "version": "1.0", "definition_type": "primitive", "namespace": "org", "body": {"type": "categorical", "labels": ["bronze", "silver", "gold"], "missing_data_policy": "null"}})
reg("primitive: status_label",   "/registry/definitions", {"definition_id": "status_label",   "version": "1.0", "definition_type": "primitive", "namespace": "org", "body": {"type": "string",             "missing_data_policy": "null"}})

# ── STEP 2: Concepts ────────────────────────────────────────────────────────
reg("concept: revenue_value",   "/registry/definitions", {
    "definition_id": "revenue_value", "version": "1.0", "definition_type": "concept", "namespace": "org",
    "body": {"concept_id": "revenue_value", "version": "1.0", "namespace": "org",
             "output_type": "float", "output_feature": "f_revenue",
             "primitives": {"revenue": {"type": "float", "missing_data_policy": "null"}},
             "features":   {"f_revenue": {"op": "passthrough", "inputs": {"input": "revenue"}}}}
})
reg("concept: account_segment", "/registry/definitions", {
    "definition_id": "account_segment", "version": "1.0", "definition_type": "concept", "namespace": "org",
    "body": {"concept_id": "account_segment", "version": "1.0", "namespace": "org",
             "output_type": "categorical", "labels": ["bronze", "silver", "gold"],
             "output_feature": "f_tier",
             "primitives": {"account_tier": {"type": "categorical", "labels": ["bronze", "silver", "gold"], "missing_data_policy": "null"}},
             "features":   {"f_tier": {"op": "passthrough", "inputs": {"input": "account_tier"}}}}
})
reg("concept: deal_status",     "/registry/definitions", {
    "definition_id": "deal_status", "version": "1.0", "definition_type": "concept", "namespace": "org",
    "body": {"concept_id": "deal_status", "version": "1.0", "namespace": "org",
             "output_type": "string", "output_feature": "f_status",
             "primitives": {"status_label": {"type": "string", "missing_data_policy": "null"}},
             "features":   {"f_status": {"op": "passthrough", "inputs": {"input": "status_label"}}}}
})

# ── STEP 3: Conditions ──────────────────────────────────────────────────────
reg("condition: high_revenue",           "/registry/definitions", {
    "definition_id": "high_revenue", "version": "1.0", "definition_type": "condition", "namespace": "org",
    "body": {"condition_id": "high_revenue", "version": "1.0", "namespace": "org",
             "concept_id": "revenue_value", "concept_version": "1.0",
             "strategy": {"type": "threshold", "params": {"direction": "above", "value": 10000}}}
})
reg("condition: top_revenue_percentile", "/registry/definitions", {
    "definition_id": "top_revenue_percentile", "version": "1.0", "definition_type": "condition", "namespace": "org",
    "body": {"condition_id": "top_revenue_percentile", "version": "1.0", "namespace": "org",
             "concept_id": "revenue_value", "concept_version": "1.0",
             "strategy": {"type": "percentile", "params": {"direction": "top", "value": 10}}}
})
reg("condition: revenue_anomaly",        "/registry/definitions", {
    "definition_id": "revenue_anomaly", "version": "1.0", "definition_type": "condition", "namespace": "org",
    "body": {"condition_id": "revenue_anomaly", "version": "1.0", "namespace": "org",
             "concept_id": "revenue_value", "concept_version": "1.0",
             "strategy": {"type": "z_score", "params": {"threshold": 2.0, "direction": "above", "window": "30d"}}}
})
reg("condition: revenue_spike",          "/registry/definitions", {
    "definition_id": "revenue_spike", "version": "1.0", "definition_type": "condition", "namespace": "org",
    "body": {"condition_id": "revenue_spike", "version": "1.0", "namespace": "org",
             "concept_id": "revenue_value", "concept_version": "1.0",
             "strategy": {"type": "change", "params": {"direction": "increase", "value": 20, "window": "1d"}}}
})
reg("condition: is_gold_account",        "/registry/definitions", {
    "definition_id": "is_gold_account", "version": "1.0", "definition_type": "condition", "namespace": "org",
    "body": {"condition_id": "is_gold_account", "version": "1.0", "namespace": "org",
             "concept_id": "account_segment", "concept_version": "1.0",
             "strategy": {"type": "equals", "params": {"value": "gold", "labels": ["bronze", "silver", "gold"]}}}
})
reg("condition: is_closed_won",          "/registry/definitions", {
    "definition_id": "is_closed_won", "version": "1.0", "definition_type": "condition", "namespace": "org",
    "body": {"condition_id": "is_closed_won", "version": "1.0", "namespace": "org",
             "concept_id": "deal_status", "concept_version": "1.0",
             "strategy": {"type": "equals", "params": {"value": "closed_won"}}}
})

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"{'Item':<38} {'HTTP':>4}  {'Result'}")
print("-" * 60)
for label, code, result, err in results:
    line = f"{label:<38} {code:>4}  {result}"
    if err:
        try:
            msg = json.loads(err).get("error", {}).get("message", "") or json.loads(err).get("detail", "")
            line += f"  -- {str(msg)[:55]}"
        except Exception:
            line += f"  -- {err[:55]}"
    print(line)

failed = sum(1 for r in results if r[2] == "FAIL")
print()
print("Overall: ALL PASS" if failed == 0 else f"Overall: {failed} FAILED")
