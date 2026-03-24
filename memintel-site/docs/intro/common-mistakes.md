---
id: common-mistakes
title: Common Mistakes
sidebar_label: Common Mistakes
---

# Common Mistakes

Most issues arise when patterns from traditional LLM workflows are carried over without adapting to a deterministic decision model. This section covers the most common mistakes and how to fix them.

---

## Letting LLMs Make Decisions

**Problem:** Using LLM output directly to trigger actions.

```python
# BAD
if llm_output == "high risk":
    trigger_agent()
```

**Why this fails:** Non-deterministic, not reproducible, difficult to audit.

**Correct approach:**
```
LLM → Concept → Condition → Decision
```

LLMs can suggest meaning, but must not decide actions.

---

## Mixing Meaning and Decision

**Problem:** Embedding thresholds or evaluation logic inside concept computation.

```python
# BAD
if churn_score > 0.8:
    return "high"
```

**Why this fails:** Breaks separation of concerns, reduces reusability, hides decision logic.

**Correct approach:**
- Concept computes value
- Condition evaluates value using strategy + parameters

Meaning and decision must always be separate.

---

## Hardcoding Logic in Agent Code

**Problem:** Decision logic spread across agent workflows.

```python
# BAD
if user.login_count < 3:
    trigger_retention()
```

**Why this fails:** Logic duplication, inconsistent behavior, difficult to update.

**Correct approach:** Move all logic into `Concept + Condition`. All decisions must be centralised in Memintel.

---

## Ignoring Versioning

**Problem:** Using definitions without version control.

**Why this fails:** Results cannot be reproduced, changes break existing systems, debugging becomes difficult.

**Correct approach:**
- Always version concepts and conditions
- Treat definitions as immutable
- Evolve logic through new versions

Every execution must be tied to a specific version.

---

## Acting Directly on Raw Values

**Problem:** Using concept output directly to trigger actions.

```python
# BAD
if result.value > 0.8:
    trigger_agent()

# CORRECT
if result.decision:
    trigger_agent()
```

**Why this fails:** Bypasses the deterministic decision layer, leads to inconsistent behavior.

Agents must act only on `decision`, not `value`.

---

## Skipping Testing and Validation

**Problem:** Deploying without validating concepts and conditions.

**Why this fails:** Incorrect parameters, unstable signals, unintended actions.

**Correct approach:**
- Use `dry_run` to simulate before enabling actions
- Validate outputs across edge cases
- Test strategy parameters explicitly

Always test before enabling actions.

---

## Overcomplicating Concepts

**Problem:** Embedding too much logic into a single concept.

**Why this fails:** Difficult to understand, hard to debug, low reusability.

**Correct approach:** Break into smaller, focused concepts and compose them. Keep concepts simple and composable.

---

## Using Memintel as a Rule Engine Only

**Problem:** Treating Memintel as a basic rules system.

**Why this fails:** Ignores the meaning layer, reduces system capability.

**Correct approach:** Use the full pipeline:
```
Primitives → Concepts → Conditions → Actions
```

Memintel is a **semantic system**, not just a rules engine.

---

## Not Separating Agent and Decision Layers

**Problem:** Agents both decide and execute.

**Why this fails:** Inconsistent decisions, difficult to maintain, lack of control.

**Correct approach:**
```
Memintel → Decision
Agent    → Execution
```

Agents execute. Memintel decides.

---

## Ignoring Determinism

**Problem:** Not providing stable inputs or relying on changing data without control.

**Why this fails:** Results vary across runs, no reproducibility.

**Correct approach:**
- Use timestamps when needed
- Ensure stable inputs
- Rely on versioned definitions

Determinism is required for reliability.

---

## Skipping Calibration

**Problem:** Keeping decision logic static despite changing real-world behavior.

**Why this fails:** Increasing false positives / negatives, degraded performance over time.

**Correct approach:**
- Monitor decision outcomes regularly
- Calibrate strategy parameters when needed
- Create new condition versions via the calibration flow
- Rebind tasks explicitly

Decision logic must evolve through calibration — not ad-hoc changes.

---

## Summary

| Mistake | Fix |
|---|---|
| LLM decides actions | Route through Concept → Condition |
| Meaning mixed with decision | Separate concept (value) from condition (decision) |
| Logic in agent code | Centralise in Memintel |
| Acting on `result.value` | Always act on `result.decision` |
| No versioning | Version everything, treat definitions as immutable |
| No testing | Always `dry_run` before enabling actions |
| Static thresholds forever | Use the calibration flow |
