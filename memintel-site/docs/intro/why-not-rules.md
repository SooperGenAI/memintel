---
id: why-not-rules
title: Why not SQL and rules?
sidebar_label: Why not SQL and rules?
---

# Why not SQL and rules?

Every technical team evaluating Memintel asks the same question early on: *"Can't we just do this with SQL and a rules engine?"*

It is the right question. SQL and rule-based systems are well-understood, reliable, and already present in most organisations. Before adopting anything new, a technical team should be able to explain precisely why existing tools are insufficient — not in vague terms, but with specific architectural reasoning.

This page answers that question directly. It explains what SQL and rules do well, exactly where they break down, and why the problems Memintel addresses are structurally beyond what rule-based systems can solve.

---

## What SQL and rules do well

Let us be honest about this before making the case for something different.

A well-built system using SQL databases, rule engines, and validation scripts handles a significant class of decision logic correctly:

**Static threshold checks** — "Is this transaction above $10,000?" "Has this loan's LTV exceeded 80%?" "Is this server's CPU above 90%?" Rules fire correctly here. The logic is explicit, testable, and deterministic.

**Point-in-time validation** — "Is this record in a valid state?" "Does this field contain a permitted value?" "Is this configuration within spec?" SQL handles this well.

**Simple event triggers** — "When a payment fails, create a retry task." "When disk usage exceeds 95%, send an alert." Clear input, clear output, direct action.

For these problems — bounded, well-defined, static — Memintel is not needed. Use SQL. Use rules. They are faster to implement and easier to maintain.

---

## Where they fundamentally break down

The real problems are not validation problems. They are **state evolution problems** — and this is a meaningful distinction.

SQL and rules are designed to answer: **"What is true right now?"**

The problems Memintel addresses require answering: **"Has the meaning of this data changed, given everything else that has changed around it?"**

These are different questions. Here is exactly where the gap opens.

---

### Problem 1 — Time-indexed reasoning

SQL gives you the current state. But many risk and compliance decisions require reasoning about state *as of a specific historical moment* — evaluated against what the external environment looked like at that time.

**In financial compliance:** A transaction that was permissible under the sanctions regime in effect on March 1st may create liability if a counterparty was added to the OFAC list on March 3rd — but only for transactions *after* the designation, not before. The decision depends on the precise relationship between transaction timestamps and regulatory state changes.

**In healthcare:** A prior authorization approved under a clinical policy that was updated two weeks ago may no longer cover the scheduled procedure. Whether the provider is at risk depends on when the service is scheduled relative to when the policy changed.

**In DevOps:** A deployment that passed all checks at 2pm may have introduced a regression that only becomes visible at 6pm when traffic patterns shift. Root cause analysis requires reconstructing the exact system state at each point — not just the current state.

SQL can store all of this data. But evaluating decisions *as of* a historical moment — using the regulatory state, policy version, or system configuration that was current *at that time* — requires engineering that has no native SQL answer. You build it yourself, differently, for every data source.

**Memintel's approach:** Every evaluation is timestamped. Resolvers are required to return values *as of* the given timestamp. Point-in-time correctness is enforced architecturally, not by convention.

---

### Problem 2 — Cross-period dependency graphs

When something changes in a complex system, the change does not stay local. It propagates through a dependency graph — and the full extent of that propagation is rarely obvious in advance.

**In finance:** A credit rating downgrade of a single large counterparty ripples through risk-weighted assets, capital ratios, concentration limits, and reporting requirements simultaneously. Each of these is a separate calculation. Each depends on others. A rules engine fires the rule for the direct exposure. It does not automatically evaluate the second and third-order consequences across the portfolio.

**In healthcare:** A formulary change — removing a drug from covered status — affects every active prescription for that drug, every prior authorization referencing it, every care plan that includes it, and every patient whose treatment protocol depends on it. The impacted scope is not a list. It is a tree of consequences.

**In DevOps / platform engineering:** A schema migration in a shared database service affects every downstream service that reads from it. Identifying the full blast radius requires traversing a dependency graph of services, their contracts, their consumers, and their consumers' consumers.

SQL can store relationships. What it cannot do is *continuously evaluate propagation* through a dependency graph as changes occur. You end up writing cascading queries — each step a separate job, each new edge type a new engineering project. As the graph grows, the complexity grows combinatorially. It is brittle and difficult to maintain.

**Memintel's approach:** The execution graph is compiled from intent. Dependency traversal is built into the evaluation engine, not layered on top of it with hand-written queries.

---

### Problem 3 — Semantic equivalence across change

Data sources evolve. The meaning of a field, a label, or a metric changes over time — sometimes subtly, sometimes significantly. Rules that were written against the old meaning silently produce wrong answers when evaluated against the new one.

**In finance:** "Revenue" as a concept in a company's financial reporting may be split into multiple components in a new accounting standard. A rule that evaluated total revenue against a threshold now needs to evaluate the sum of multiple successor fields — but which successor fields are semantically equivalent depends on the company's specific reporting structure, not just on the standard itself.

**In healthcare:** A diagnosis code that was a single ICD-10 code in one year may be split into three more specific codes in the next revision. A rule checking for "diabetes-related complications" by code now misses two-thirds of the population unless it is explicitly updated — and that update requires clinical knowledge, not just a schema change.

**In DevOps / SRE:** A latency metric that was a single P99 value is now reported as separate P95, P99, and P99.9 values after an observability platform upgrade. Rules firing on the old metric name produce no alerts. Rules firing on the new names need to be rebuilt from scratch with correct thresholds.

In each case, the problem is not that the data changed. The problem is that the *meaning* of the data changed, and rules have no way to detect or reason about meaning — only structure and values.

**Memintel's approach:** Semantic interpretation happens at task creation time, with an LLM operating within guardrails, mapping intent to structured primitives. When a primitive's underlying source changes, the resolver is updated. The evaluation logic built on top of that primitive remains valid because it operates on meaning, not on raw field names.

---

### Problem 4 — Dual memory reconciliation

The most important class of problems Memintel addresses require simultaneously maintaining and reconciling two evolving states:

1. **Internal state** — the customer's transaction history, the borrower's financial trajectory, the patient's care history, the service's deployment history
2. **External state** — the current regulatory environment, the latest clinical guidelines, the current vendor SLA, the current threat landscape

The value does not come from either source alone. It comes from the *interaction* between them.

**In AML compliance:** A transaction is only suspicious relative to the customer's established behavior pattern. A $50,000 wire transfer is routine for a commercial real estate firm. It is anomalous for a sole trader who has never previously sent a wire above $5,000. The same value, different meaning — because the internal history is different.

**In credit risk:** A counterparty downgrade (external signal) only matters in proportion to that counterparty's share of your portfolio (internal state). A downgrade affecting a counterparty that represents 0.2% of RWA is a monitoring note. A downgrade affecting a counterparty representing 8% of RWA may require immediate capital action.

**In healthcare network compliance:** A provider's OIG exclusion (external signal) only creates liability in proportion to the claims currently in flight from that provider (internal state). The alert without the context is noise. The alert with the context is actionable.

**In DevOps / incident management:** An infrastructure event (external signal — a cloud provider's availability zone outage) only creates risk in proportion to which services have active traffic routed through that zone (internal state). The event affects everyone theoretically. The impact is specific to your topology.

SQL systems are designed around internal data. Connecting them to continuously evolving external state — and evaluating the interaction in real time — requires custom engineering for each external source, custom reconciliation logic for each interaction type, and custom alerting for each combination. This does not scale across domains.

**Memintel's approach:** Internal and external state are both expressed as typed primitives. The evaluation engine treats them identically. A new external data source is a new resolver function, not a new architecture.

---

### Problem 5 — Continuous vs batch evaluation

Traditional rule systems run checks when triggered — on a schedule or in response to a direct event. The gap between when something becomes true and when the system discovers it is structural, not accidental.

**In financial compliance:** An OIG exclusion takes effect from the moment of publication. Paying claims to an excluded provider after that moment creates federal liability — regardless of when your batch job runs. A system that checks the exclusion list nightly creates a window of exposure that is not a configuration problem; it is an architectural one.

**In healthcare:** A prior authorization approved today may exhaust its unit limit by Friday based on scheduled services. A system that checks authorization status at claim submission discovers the problem after the service has been delivered. A system that continuously evaluates trajectory discovers it with enough lead time to request an extension.

**In DevOps / SRE:** A memory leak that doubles every four hours is undetectable by a rule that checks current memory utilisation against a static threshold. By the time the threshold is crossed, the system may be minutes from failure. Detecting it requires evaluating the *rate of change* of memory utilisation over time — which is a different question from "is memory above 80%?"

**In financial markets:** A credit spread that widens by 3 basis points per day is unremarkable in isolation. Over 30 trading days, it signals a structural deterioration in counterparty credit quality that should trigger enhanced monitoring. A rule fires at a specific spread level. Memintel detects the trajectory before that level is reached.

Running batch jobs more frequently does not solve this. It multiplies compute cost while still evaluating the same static logic more often. The architecture is still reactive.

**Memintel's approach:** Tasks can be event-driven, firing immediately when a resolver returns a changed value. The evaluation is a continuous function of state, not a scheduled poll.

---

### Problem 6 — Proactive vs reactive

This is the most fundamental difference — and the hardest to solve with rules.

A rule system answers: **"Is something wrong right now?"**

Memintel is designed to answer: **"Is something becoming wrong, given the trajectory of the current state?"**

**In credit risk:** A borrower's debt service coverage ratio of 1.52 is above the covenant floor of 1.25. No rule fires. But the ratio has declined from 2.41 to 2.18 to 1.87 to 1.52 over four consecutive quarters — a -37% decline tracking toward a covenant breach in two quarters. The current value is acceptable. The trajectory is not.

**In healthcare fraud detection:** A physician billing 68% high-complexity codes this month is within tolerances. But if that proportion has risen from 31% to 45% to 57% to 68% over four months while the peer median has held steady at 31%, the trajectory indicates a systematic upcoding pattern — not a one-month anomaly that resolved itself.

**In DevOps:** A service's error rate of 0.8% is below the SLO threshold of 1.0%. No alert fires. But if the error rate has increased by 0.15% every hour for the past six hours, it will breach the SLO within the next 90 minutes. Detecting this requires evaluating the trend, not just the current value.

**In SaaS operations:** A customer's feature adoption score of 42% is above the churn risk threshold of 35%. No alert fires. But adoption has declined from 71% to 63% to 53% to 42% over four weeks. At this trajectory, they cross the threshold before their next renewal. The intervention window is now, not after they cross the line.

You can encode trend detection in rules. It is technically possible — precompute time-series features, define the slope calculation, write the comparison logic. But you have to do this individually for every metric you care about. As the number of metrics grows, the engineering cost grows linearly. As the number of trend types grows, the complexity grows combinatorially. The system becomes a collection of bespoke feature-engineering projects rather than a coherent evaluation framework.

**Memintel's approach:** Time-series primitives are a first-class type. Change and z-score strategies are built-in evaluation primitives. Trajectory detection is a configuration choice, not a custom engineering project.

---

## The honest summary

SQL and rules are not wrong. They are the right tool for bounded, well-defined, static problems. The problem they are not designed for is this:

**Continuously determining whether a system's state — evaluated in the context of its own history and an evolving external environment — has changed in a way that is meaningful enough to require action.**

This problem has six structural requirements that rule-based systems cannot satisfy at scale:

| Requirement | SQL / Rules | Memintel |
|---|---|---|
| Point-in-time historical evaluation | Manual, per-source engineering | Enforced by resolver contract |
| Cross-period dependency propagation | Combinatorial query complexity | Compiled evaluation graph |
| Semantic interpretation across change | Not possible | LLM at compile-time, deterministic at runtime |
| Dual internal/external memory reconciliation | Custom per-integration engineering | Unified primitive model |
| Continuous event-driven evaluation | Batch polling or complex streaming pipelines | Native event-driven execution |
| Trajectory and trend detection | Custom feature engineering per metric | Built-in time-series strategies |

The left column is not a criticism of SQL. SQL is excellent at what it does. The right column describes a different class of problem — one that requires a different architectural layer.

---

## What Memintel is not replacing

To be precise about the boundary:

**Memintel does not replace your data pipeline.** Signal extraction — turning raw transactions, clinical records, infrastructure metrics, or financial data into clean typed values — happens upstream in your existing infrastructure. Memintel starts where typed primitives exist.

**Memintel does not replace your LLM.** The LLM interprets intent at task creation time. It does not participate in runtime evaluation. Memintel's runtime is a pure deterministic function.

**Memintel does not replace simple rules.** For bounded, well-defined threshold checks — use SQL. The engineering is simpler and the overhead is lower. Memintel is for the problems where the answer depends on context, history, and evolving external state.

---

## The architectural position

Memintel sits between your data pipeline and your action layer:

```
Data Sources  →  Signal Extraction  →  Primitives  →  Memintel  →  Actions
(your systems)   (your pipeline)        (your config)  (evaluation)  (your systems)
                                                           ↑
                                                     Intent (LLM)
                                                     compiled once
```

Every layer to the left and right of Memintel is yours. Memintel's job is the evaluation in the middle — determining, continuously and deterministically, whether the state your data describes is meaningfully different from what it was before.

SQL tells you what your data is. Memintel tells you whether it still makes sense — given everything that has changed around it.

---

## Further reading

- [Core Concepts](/docs/intro/core-concepts) — the ψ → φ → α model in detail
- [Guardrails System](/docs/intro/guardrails) — how evaluation logic is governed
- [Deal Intelligence Tutorial](/docs/tutorials/deal-intelligence) — the architecture applied to sales pipeline monitoring
- [Financial Risk Monitoring](/docs/tutorials/financial-risk-monitoring) — AML, credit risk, and capital adequacy
- [Healthcare Payor-Provider](/docs/tutorials/healthcare-payor-provider) — claims fraud, network compliance, prior auth
