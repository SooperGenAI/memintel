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

A well-built system using SQL databases, rule engines, and validation scripts handles a significant portion of decision logic correctly:

**Static validation** — "Is this tag deprecated?" "Does this calculation add up?" "Is this value within range?" Rules are the right tool here. The logic is explicit, testable, and deterministic.

**Point-in-time threshold checks** — "Is the customer's account balance below $500?" "Has this loan's LTV exceeded 80%?" SQL handles this well. You write a query, you get an answer.

**Simple event triggers** — "When a transaction exceeds $10,000, create a compliance record." Clear input, clear output, rules fire correctly.

For these problems — static checks, simple thresholds, direct event responses — Memintel is not needed. Use SQL. Use rules. They are faster to implement and easier to maintain for bounded, well-defined problems.

---

## Where they fundamentally break down

The real problems are not validation problems. They are **state evolution problems** — and this is a meaningful distinction.

SQL and rules are designed to answer: **"What is true right now?"**

The problems Memintel addresses require answering: **"Has the meaning of this data changed, given everything else that has changed around it?"**

These are different questions. Here is exactly where the gap opens.

---

### Problem 1 — Time-indexed reasoning

SQL gives you the current state of a table. But many compliance and risk decisions require reasoning about state *as of a specific historical moment* — and that moment may need to be evaluated differently depending on what the external environment looked like at that time.

A concrete example from XBRL compliance:

A tag used correctly in a 2022 filing may be deprecated in the 2024 taxonomy update. Whether a company needs to remediate depends on:
- when exactly the tag was deprecated
- when the company's fiscal year ends
- when their filing is due
- whether the deprecation was retroactive or prospective

SQL can store all of this data. But the evaluation logic — "given this company's filing calendar and the deprecation effective date, is remediation required before the next filing?" — requires reasoning *across* these dimensions simultaneously with temporal precision. You end up writing complex procedural logic that breaks as edge cases accumulate.

More fundamentally: if you want to replay a decision from six months ago and get the exact same result — using the taxonomy that was current *at that time*, not today's taxonomy — SQL has no native answer to this. You have to engineer it yourself, every time, for every data source.

**Memintel's approach:** Every evaluation is timestamped. Resolvers are required to return values *as of* a given timestamp. The system enforces point-in-time correctness architecturally, not by convention.

---

### Problem 2 — Cross-period dependency graphs

When something changes in a complex system, that change does not stay local. It propagates.

In financial reporting: a segment restructuring in Q3 affects every prior-period filing that referenced the old segment structure. The impacted elements span multiple line items, multiple periods, multiple disclosures, and multiple calculations. The dependency graph is not a list — it is a tree of consequences.

SQL can store relationships. What it cannot do is *continuously evaluate the propagation* of a change through that dependency graph as the change occurs. You end up writing:

```sql
-- Find all filings affected by the segment change
UPDATE filings SET needs_restatement = true
WHERE period_end < '2024-09-30'
AND segment_id IN (SELECT id FROM old_segments);

-- Then find all disclosures referencing those filings
-- Then find all calculations using those disclosures
-- Then find all cross-references to those calculations
-- ...
```

Each step requires a new query. Each new edge type in the graph requires new queries. As the graph grows, the query complexity grows combinatorially. Maintaining this across rule changes, schema changes, and new use cases becomes brittle engineering work that is difficult to test and easy to break.

**Memintel's approach:** The execution graph is compiled from intent, not hand-authored. Dependency traversal is built into the evaluation engine, not layered on top of it.

---

### Problem 3 — Semantic equivalence

Taxonomy changes are not always clean mappings. An old tag does not simply map to a new tag with identical semantics. The meaning may shift subtly:

- "Revenue" in the 2021 taxonomy maps to three separate components in the 2024 taxonomy
- A dimensional structure that was a single element is now represented as a hierarchy
- A calculation relationship that was additive is now multiplicative

SQL sees columns and values. It does not understand that the *meaning* of a value has changed even when the value itself has not. Detecting this kind of semantic shift requires:
- understanding what the old definition meant
- understanding what the new definition means
- evaluating whether a company's reporting practice bridges that gap correctly

This is where LLM-assisted understanding becomes necessary — not at runtime decision evaluation, but at the stage where meaning is being interpreted and mapped to structured signals. SQL cannot perform this step. A rules engine cannot perform this step. This is precisely where the signal extraction layer sits in Memintel's architecture, upstream of any deterministic evaluation.

**Memintel's approach:** Semantic interpretation happens once, at task creation time, by the LLM operating within guardrails. The output is a structured, typed primitive that deterministic evaluation can then work with. The LLM does not participate in the runtime evaluation loop.

---

### Problem 4 — Dual memory reconciliation

The most important class of problems Memintel addresses are those that require simultaneously maintaining and reconciling two evolving states:

1. **Internal state** — the company's own filing history, the customer's transaction pattern, the borrower's financial trajectory
2. **External state** — the current taxonomy, the SEC's enforcement focus areas, the FATF grey list, the current Basel risk weights

The value does not come from either memory alone. It comes from the *interaction* between them.

- A deprecated tag is only a high-priority problem if the company has used it in 11 prior filings
- A transaction is only suspicious relative to the customer's established behavior pattern
- A counterparty downgrade is only a capital concern if that counterparty represents 8% of your RWA

SQL systems are designed around internal data. Connecting them to continuously evolving external regulatory state — and evaluating the interaction between the two in real time — requires engineering that goes well beyond queries. You end up building custom pipelines for each external source, custom reconciliation logic for each interaction type, and custom alerting logic for each combination. This scales badly. Every new external data source is a new project.

**Memintel's approach:** Internal state and external state are both expressed as typed primitives. The evaluation engine treats them identically. Adding a new external data source is a new resolver function, not a new architecture.

---

### Problem 5 — Continuous vs batch evaluation

Traditional rule systems run checks when triggered — either on a schedule or in response to an event. The SEC publishes a new comment letter. The check runs tonight in the batch job. Or next week. Or next month.

In domains where external signals evolve continuously — OIG exclusion lists updated daily, SEC comment letters published irregularly, credit ratings changed intraday — batch evaluation means there is always a gap between when something becomes true and when the system knows about it.

For an OIG exclusion, this gap is a federal liability exposure. For a credit rating downgrade affecting 8% of your RWA, this gap is a capital management decision being made without current information.

The alternative — running checks more frequently — multiplies compute cost linearly without solving the architectural problem. The system still evaluates the same static rules more often.

**Memintel's approach:** Tasks can be event-driven, firing immediately when a resolver returns a changed value. The evaluation is not a batch job polling for changes — it is a continuous function of incoming state, triggered by state transitions.

---

### Problem 6 — Proactive vs reactive

This is the most fundamental difference — and the hardest to solve with rules.

A rule system answers: **"Is something wrong right now?"**

Memintel is designed to answer: **"Is something becoming wrong, given the trajectory of the current state?"**

A borrower's DSCR of 1.52 is above the covenant floor of 1.25. No rule fires. But if the DSCR has declined from 2.41 to 2.18 to 1.87 to 1.52 over four consecutive quarters — that is a -37% decline tracking toward a breach in two quarters. The current value is fine. The trajectory is not.

Detecting this requires:
- maintaining a time series of the metric, not just its current value
- computing the rate and direction of change
- evaluating whether the trajectory, if continued, crosses a threshold within a meaningful time horizon

You can encode this in rules. It is technically possible. But:
- you need to precompute time-series features for every metric you care about
- you need to define the trajectory logic for each metric explicitly
- you need to maintain this as the system evolves

As the number of metrics grows, the engineering cost grows linearly. As the number of trajectory types grows (linear decline, accelerating decline, oscillating decline), the complexity grows combinatorially.

**Memintel's approach:** Time-series primitives are a first-class type. Change strategies — detecting meaningful movement over a time window — are built-in evaluation primitives. Trajectory detection is not a custom engineering project; it is a configuration choice.

---

## The honest summary

SQL and rules are not wrong. They are the right tool for the right problem. The problem they are not designed for is this:

**Continuously determining whether a system's state — evaluated in the context of its own history and an evolving external environment — has changed in a way that is meaningful enough to require action.**

This problem has six specific structural requirements that rule-based systems cannot satisfy at scale:

| Requirement | SQL / Rules | Memintel |
|---|---|---|
| Point-in-time historical evaluation | Manual, per-source engineering | Architectural — enforced by resolver contract |
| Cross-period dependency propagation | Combinatorial query complexity | Compiled evaluation graph |
| Semantic interpretation of changing definitions | Not possible | LLM at compile-time, deterministic at runtime |
| Dual internal/external memory reconciliation | Custom per-integration | Unified primitive model |
| Continuous event-driven evaluation | Batch polling or complex streaming pipelines | Native event-driven task execution |
| Trajectory and trend detection | Custom feature engineering per metric | Built-in time-series strategies |

The first column is not a criticism of SQL. SQL is excellent at what it does. The second column describes a different class of problem — one that requires a different architectural layer.

---

## What Memintel is not replacing

To be precise about the boundary:

**Memintel does not replace your data pipeline.** Signal extraction — turning raw emails, CRM records, XBRL filings, or transaction flows into typed, normalised values — happens upstream of Memintel in your existing infrastructure. Memintel starts where clean typed primitives exist.

**Memintel does not replace your LLM.** The LLM interprets intent at task creation time and produces a structured evaluation definition. It does not participate in runtime evaluation. Memintel's runtime is a pure deterministic function.

**Memintel does not replace simple rules.** For bounded, well-defined checks — is this value in range, does this record exist, is this flag set — use SQL. The engineering is simpler. Memintel is for the problems where the answer depends on context, history, and evolving external state.

---

## The architectural position

Memintel sits between your data pipeline and your action layer:

```
Data Sources  →  Signal Extraction  →  Primitives  →  Memintel  →  Actions
(your systems)   (your pipeline)        (your config)   (evaluation)  (your systems)
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
- [Deal Intelligence Tutorial](/docs/tutorials/deal-intelligence) — the architecture applied to a concrete use case
- [XBRL Compliance Tutorial](/docs/tutorials/xbrl-compliance) — where the taxonomy evolution and dual memory problems are most vivid
