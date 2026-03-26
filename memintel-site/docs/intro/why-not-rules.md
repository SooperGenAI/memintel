---
id: why-not-rules
title: Why not SQL and rules?
sidebar_label: Why not SQL and rules?
---

# Why not SQL and rules?

Every technical team evaluating Memintel asks the same question early on: *"Can't we just do this with SQL and a rules engine?"*

The honest answer is: for many things, yes you can. Time-series analysis, z-score anomaly detection, multi-signal composition, threshold monitoring — all of these are technically achievable with rules. A motivated engineering team can build them.

The question is not whether rules *can* do it. The question is **how the logic gets there in the first place** — and what happens when the world changes.

---

## The fundamental difference

In a rule-based system, logic must be **specified**. Someone sits down, decides which signals matter, defines how to combine them, picks the thresholds, and writes the condition. Every decision reflects an explicit choice made by a human at authoring time. The system does exactly what it was told — no more, no less.

In Memintel, logic is **derived from intent**. A user expresses what they want to monitor in plain language. The compiler — working within the primitive vocabulary and guardrails the admin has configured — derives the signals, the combination, the strategy, and the thresholds. Nobody wrote that specific logic. It was compiled from meaning.

This is the centrepiece. Everything else — the auditability, the adaptability, the scale — is a consequence of this one architectural difference.

---

## The same problem, seen from both sides

The difference becomes concrete when you look at the same problem through both lenses across different domains.

---

### Finance — AML transaction monitoring

**Rules-based approach:**

A compliance engineer defines:

```python
if transaction.amount > customer.avg_90d * 10:
    if customer.jurisdiction_risk == "high":
        if transaction.counterparty in watchlist:
            create_alert("high_risk_transaction")
```

This logic reflects decisions made at authoring time: the 10x multiplier, the jurisdiction risk threshold, the watchlist check. Each was chosen by someone, written by someone, and must be maintained by someone.

Now the compliance team learns that structuring patterns — multiple transactions just below the reporting threshold — are more suspicious than single large transactions. Someone has to add that logic. The typology changes next month when regulators publish new guidance. Someone has to update the rules again. A new signal becomes available — the customer's Slack escalation history. Someone has to wire it in.

The system is always one step behind the current understanding of what suspicious means.

**Memintel approach:**

The compliance officer types: *"Alert me when a transaction shows unusual risk relative to this customer's established pattern and current regulatory signals."*

The compiler maps "unusual" to a statistical deviation strategy on `transaction.value_vs_baseline_ratio`, weighs it against `customer.counterparty_jurisdiction_risk` and `typology.recent_match_score`, and compiles a composite condition. The compliance officer never picked a multiplier. The system derived what "unusual" means given the registered signals and domain priors.

When new typology guidance is published, the data team updates the `typology.recent_match_score` resolver. The compiled logic automatically incorporates it on the next evaluation — because the intent was "match against current regulatory signals", not "check these three specific conditions."

---

### Healthcare — Prior authorisation monitoring

**Rules-based approach:**

A revenue cycle engineer writes:

```python
if auth.days_to_expiry < 7:
    if auth.units_remaining < 3:
        send_alert("auth_expiring_soon")
```

This fires when an auth is expiring with few units left. But it does not fire when an auth has 12 days left and 8 units remaining — even if 9 services are scheduled in the next 10 days and the auth will be exhausted on day 3. The rule captures the current state. It does not reason about the trajectory.

To catch that case, an engineer has to write another rule. To catch the case where expiry and scheduled services interact with a pending renewal that may or may not arrive in time, another rule. Each new scenario is a new specification project.

**Memintel approach:**

The care manager types: *"Alert me when an active authorisation is at risk of expiring before scheduled services are completed."*

The compiler combines `auth.days_to_expiry`, `auth.units_remaining`, `auth.units_utilization_rate`, and `auth.pending_claims_at_risk` into a composite concept that evaluates the trajectory — not just the current state. Nobody specified that `pending_claims_at_risk > units_remaining` is the relevant comparison. The compiler derived it from "at risk of expiring before scheduled services are completed."

The alert fires with enough lead time to act — not after the service has been delivered and the claim denied.

---

### DevOps / SRE — Incident early warning

**Rules-based approach:**

An SRE writes:

```yaml
alert: high_error_rate
condition: error_rate > 1.0%
severity: page
```

This fires when the error rate crosses 1%. It does not fire when the error rate is 0.7% and has been climbing 0.1% per hour for six hours. By the time the rule fires, the service may be minutes from an outage.

To catch the trajectory, the engineer has to write a second rule on the rate of change. To catch the case where a memory leak causes gradual degradation, a third rule. To catch the case where error rate is acceptable but p99 latency is deteriorating in a pattern that predicts a cascade failure, several more rules — each requiring specialised knowledge of what warning patterns look like for that specific service.

**Memintel approach:**

The SRE types: *"Alert me when this service shows early signs of degradation before it breaches SLO thresholds."*

The compiler maps "early signs of degradation" to change and z-score strategies across `service.error_rate_1h`, `service.p99_latency_trend`, and `service.memory_utilization_rate`, derives the appropriate time windows from the service's historical patterns, and compiles a composite condition. The SRE did not specify which signals matter or what trajectory threshold triggers concern. The system derived both from "early signs of degradation" given the registered primitives.

When a new signal is added — say, `service.dependent_service_latency` — tasks that include "signs of degradation" in their intent automatically become candidates for recompilation to include it.

---

### Credit risk — Portfolio early warning

**Rules-based approach:**

A risk analyst writes:

```python
if borrower.dscr < 1.25:
    flag_for_review("covenant_breach")
```

This fires at the covenant floor. But a borrower with DSCR declining from 2.41 → 2.18 → 1.87 → 1.52 over four quarters is two quarters from a breach. The rule does not fire. The relationship manager does not know. The intervention window passes.

To catch deterioration before breach, the engineer has to build time-series features, define the slope calculation, write the trend comparison. For each metric the bank monitors — DSCR, leverage, current ratio, interest coverage — a separate engineering project. And each one encodes a specific definition of "deteriorating" that may not match what an experienced credit analyst would recognise.

**Memintel approach:**

The credit analyst types: *"Alert me when a borrower's financial health is showing a significant declining trend."*

The compiler maps "financial health" to a composite of `borrower.dscr_trend_4q`, `borrower.leverage_ratio`, and `borrower.management_sentiment_score`, applies a change strategy to detect trajectory rather than current level, and derives severity from the rate of decline. The analyst did not define what "financial health" is or what "significant declining trend" means quantitatively. The system derived both.

The alert fires when the trajectory predicts a covenant breach — not when the breach has already occurred.

---

## The implications of rules-based logic specification

When logic must be specified rather than derived, a specific set of consequences follow. They are not accidental — they are structural.

### 1. Logic is always behind current understanding

Rules encode what someone understood at the time they wrote them. The world moves on. New signals become available. New patterns emerge. New regulatory guidance is published. The rules do not know any of this. They continue evaluating the same conditions they were given, against a reality that has evolved.

Keeping rules current requires continuous manual intervention — someone monitoring what has changed, deciding what it means for the logic, writing the update, testing it, deploying it. In fast-moving environments, this is not a backlog item. It is a permanent ongoing cost.

Intent-based systems face this too — but only at the primitive and guardrails layer, not at the logic layer. When a new signal becomes available, the admin registers a primitive. Tasks that were created with intent that encompasses that signal automatically benefit from it. The logic layer does not need to be rewritten.

### 2. Every edge case is an engineering project

Rule systems handle the cases their authors anticipated. Every unanticipated case — every interaction between signals that was not explicitly specified — either fires incorrectly, fails to fire, or requires a new rule.

In AML, this means the structuring pattern that was not in the original specification gets missed for months until an audit finds it. In healthcare, this means the authorisation that expires during a holiday weekend does not trigger the right escalation because nobody wrote that rule. In DevOps, this means the cascading failure that starts with a subtle latency degradation does not page anyone because the path from "latency degrading slowly" to "imminent outage" was never explicitly defined.

Intent-based systems derive logic from meaning. "At risk" means at risk — including the edge cases the user did not enumerate, because the compiler is working from the semantic intent, not from a list of specific conditions.

### 3. Complexity grows with the number of scenarios

A rules system with 10 signals and 5 severity levels requires someone to reason about every meaningful combination. Not all combinations matter — but determining which ones matter requires domain expertise, and encoding each one that does requires engineering effort. As signals are added, this grows combinatorially.

Large financial institutions routinely maintain thousands of AML rules. Healthcare payors maintain hundreds of prior authorisation rules per clinical specialty. DevOps teams maintain rule files with hundreds of alert conditions. Each was added for a reason. Each interacts with others in ways that are difficult to fully reason about. Debugging a false positive means tracing which rule fired and why — often through logic written by someone who has since left the organisation.

Intent-based systems scale differently. Adding a new signal adds one primitive. The evaluation logic for existing tasks does not need to change — the compiler incorporates the new signal where it is semantically relevant. Complexity grows linearly with new requirements, not combinatorially with new scenarios.

### 4. Adaptation requires re-specification

When a calibration is needed — thresholds are too sensitive, a new pattern needs to be detected, a clinical policy changes — rules must be rewritten. This means: identify the relevant rules, understand their current logic, decide what should change, write the change, test it, deploy it.

In practice this creates inertia. Rules that should be updated are not, because the update process is expensive. Systems drift away from current best practice because the cost of keeping them current exceeds the perceived benefit of each individual update.

Intent-based calibration works differently. Feedback on decisions — "this was a false positive", "this should have fired earlier" — adjusts parameters within the compiled condition. The intent remains the same. The system becomes more accurate without anyone rewriting logic. The adaptation is structural, not manual.

### 5. Auditability exists at the rule level, not at the meaning level

A rules engine can tell you which rule fired. It cannot tell you whether the rule still accurately reflects what the organisation means by "high risk" or "significant deterioration" or "at risk of stalling." The rule is the authoritative record, but the rule is a human artifact — it reflects an understanding that may have been correct at authoring time and may have drifted since.

Intent-based systems maintain auditability at both levels. Every decision is traceable to a specific compiled condition version — which strategy, which parameters, which primitives. And that condition version is itself traceable to the intent that produced it. When a decision is questioned, you can answer both "what fired" and "what was the system trying to detect."

---

### 6. Environment changes require logic rewrites

When the external environment changes significantly — a new regulatory framework, updated clinical guidelines, a shift in what "anomalous" means for a domain — a rules-based system requires someone to find every affected rule, understand what it was doing, decide what it should now do, and rewrite it. This is true even if the *intent* behind the rule has not changed at all. Only the parameters that give that intent its operational meaning have changed.

**In AML compliance:** Regulators tighten guidance and "unusual transaction volume" now maps to 10x the customer baseline instead of 15x. In a rules engine, someone locates every rule that encodes a transaction volume threshold, assesses whether each one reflects this concept, and rewrites the ones that do. Some rules may have been written years ago by people who have since left. Some may encode the threshold in non-obvious ways. Some may have been copied and slightly modified across business units and are now inconsistent.

**In healthcare:** CMS updates coverage criteria for a procedure. Every prior authorisation rule that encodes those criteria needs to be found and updated. If the criteria changed in a subtle way — the same procedure is now covered under condition A but not condition B — every rule that referenced the old single condition needs to be rewritten into two rules with different logic.

**In DevOps:** A service's traffic patterns shift seasonally. What was "statistically anomalous" last quarter is now normal peak behaviour. Every alert threshold that was set based on last quarter's baseline needs to be recalibrated — and in a rules engine, that means someone sitting down and manually updating numbers.

In each case: even though the *intent* has not changed — "flag unusual transactions", "enforce coverage criteria", "alert on anomalous load" — the rules must be rewritten because rules encode parameters, not intent.

**Memintel's approach:** When the environment changes, the admin updates the guardrails config — the parameter priors, bias rules, and strategy constraints that reflect current domain understanding. Then they trigger recompilation of affected tasks. The compiler re-runs against the original intent strings with the new policy constraints and derives new concept and condition versions automatically. The intent layer is untouched. The logic derives itself from current understanding.

```
Rules-based environment change:
  Find affected rules → understand current logic → decide new logic
  → rewrite rules → test → deploy → repeat for each rule

Memintel environment change:
  Update guardrails config → trigger recompilation of affected tasks
  → review delta → approve → rebind
```

The difference in scale becomes clear when an organisation has hundreds of monitoring tasks across a domain. In a rules engine, a regulatory update is a project. In Memintel, it is a config change followed by a review.

---

---

## The benefits of intent-based evaluation

The six implications above describe what goes wrong with rules. Flipped around, they describe what Memintel gets right. Here is the positive statement of each.

### 1 — Logic comes from meaning, not from engineering

Domain experts express what they want to monitor in plain language. The system derives the signals, the combination, the strategy, and the thresholds from the primitive vocabulary and guardrails the admin has configured. Nobody has to translate business intent into code. Engineers build primitives. Admins govern policy. Users express meaning. The compiler does the rest.

**In practice:** A credit analyst says "alert me when a borrower shows significant financial deterioration." A compliance officer says "flag unusual transaction patterns for this customer." An SRE says "warn me before this service breaches its SLO." None of them wrote a threshold. All of them got exactly the monitoring they intended.

### 2 — Consistency across every evaluation

This matters most for **ongoing tasks** — the primary Memintel use case. When a task evaluates thousands of entities on a schedule, every single evaluation runs against the same pinned condition version. Same threshold, same strategy, same parameters — across every entity, across every run, across months of operation. Cross-entity fairness and cross-time reproducibility are guaranteed by architecture, not by discipline.

**In practice:** When a bank's AML system evaluates 2 million customers daily, every customer is evaluated against the same compiled condition. When a health plan monitors 50,000 providers for network compliance, every provider is evaluated identically. There is no question of which version of the logic any given entity was subject to.

### 3 — Auditability at the level of meaning

Every decision is traceable to a specific compiled condition version — which strategy, which parameters, which primitives. And that condition version is itself traceable to the intent that produced it. When a decision is questioned, you can answer both "what fired" and "what was the system trying to detect."

**In practice:** A regulator asks why a specific AML alert was not raised for a specific customer on a specific date. You can replay the exact evaluation — same condition version, same primitive values, same timestamp — and show precisely what the system evaluated and why it did or did not fire. This is not post-hoc reconstruction. It is the audit trail the system produces automatically.

### 4 — Environment changes without logic rewrites

When the regulatory environment changes, clinical guidelines update, or domain understanding shifts — the admin updates the guardrails config and triggers recompilation of affected tasks. The compiler re-derives new conditions from the same original intent strings under the new policy constraints. The intent layer is untouched. Users never touch a threshold.

**In practice:** A regulatory update that would require rewriting dozens of AML rules in a rules engine becomes a guardrails config change followed by a review of the compiled delta. The compliance team approves the new thresholds. Tasks rebind. The entire process is traceable and reversible.

### 5 — Complexity grows linearly, not combinatorially

Adding a new signal adds one primitive. Existing tasks incorporate it where semantically relevant on next recompilation. Complexity grows with new requirements, not with new scenario permutations. There is no rule explosion — just a growing vocabulary of typed signals that the compiler can draw on.

**In practice:** A platform team adds a new observability signal — `service.dependent_service_latency`. Every existing SRE monitoring task that was created with "degradation" or "cascade risk" intent automatically benefits from the new signal on next recompilation. No individual task needs to be updated.

### 6 — Calibration produces clean, actionable signal

Condition versions are immutable. Feedback accumulates against a known, stable definition. When enough signal exists, the calibration recommendation is statistically meaningful — not confounded by silent changes in evaluation logic between evaluations. The system gets more accurate over time through explicit, versioned, admin-approved improvement.

**In practice:** After three months of AML monitoring, the false positive rate is 22%. The calibration engine recommends raising the threshold from 10x to 12x, estimated to reduce false positives by 30% with minimal impact on true positive detection. The admin reviews, approves, and the new version is deployed. Every step is auditable.

---

### Summary

| Benefit | Rules | Memintel |
|---|---|---|
| How logic is created | Specified by engineers | Derived from intent |
| Consistency across evaluations | Degrades as rules drift | Guaranteed by version pinning |
| Auditability | At rule level only | At meaning level |
| Response to environment change | Rewrite affected rules | Update guardrails, recompile |
| Complexity as scale grows | Combinatorial explosion | Linear growth |
| Calibration | Manual threshold adjustment | Structured feedback loop |
| Most valuable for | Bounded, static, well-defined checks | Ongoing monitoring, evolving environments |

---

## What rules do well — and where to keep them

To be precise about the boundary:

**Rules are the right tool for:**
- Bounded, well-defined threshold checks with stable definitions
- Regulatory validation rules that are explicitly specified in law or policy and do not require interpretation
- Simple event triggers with clear, unambiguous conditions
- Any decision where the logic is unlikely to evolve and does not require contextual reasoning

**Memintel is the right tool for:**
- Decisions where "what matters" requires domain interpretation, not just threshold comparison
- Monitoring that needs to evolve as signals, patterns, and regulatory expectations change
- Multi-signal evaluation where the combination reflects meaning, not just arithmetic
- Any context where the ability to express intent and have logic derived — rather than specified — reduces the engineering and governance burden to a level that makes continuous improvement practical

---

## The architectural boundary

Memintel does not replace your data pipeline. Signal extraction — turning raw transactions, clinical records, financial filings, or infrastructure metrics into clean typed values — happens upstream in your existing infrastructure. Memintel starts where typed primitives exist.

Memintel does not replace your LLM. The LLM interprets intent at task creation time, once, and produces a structured evaluation definition. It does not participate in runtime evaluation. The runtime is a pure deterministic function.

```
Data Sources  →  Signal Extraction  →  Primitives  →  Memintel  →  Actions
(your systems)   (your pipeline)       (admin cfg)    (evaluation)  (your systems)
                                                           ↑
                                                     Intent (LLM)
                                                     compiled once
                                                     at task creation
```

SQL tells you what your data is.
Rules tell you whether your data matches what someone decided mattered.
Memintel tells you whether your data means something — given current signals, current context, and current understanding of what matters.

---

## Further reading

- [Core Concepts](/docs/intro/core-concepts) — the ψ → φ → α model in detail
- [Guardrails System](/docs/intro/guardrails) — how the admin constrains intent resolution
- [Deal Intelligence Tutorial](/docs/tutorials/deal-intelligence) — intent-based monitoring applied to sales
- [Financial Risk Monitoring](/docs/tutorials/financial-risk-monitoring) — AML, credit risk, capital adequacy
- [Healthcare Payor-Provider](/docs/tutorials/healthcare-payor-provider) — claims fraud, network compliance, prior auth
