# Memintel

Makes agentic AI execution deterministic. Run AI agents reliably without any randomness — same inputs, same outputs, every time.

---

## What is Memintel?

Memintel is a deterministic compiler for AI-driven decisions. You describe what you want to monitor in plain language. Memintel's LLM interprets that intent within a strict guardrails policy, generates a fully specified decision definition, and compiles it into an immutable, versioned execution graph.

From that point, every evaluation is deterministic — the same data always produces the same decision, with a full computation trace. When decision quality needs to change, users give structured feedback. Calibration recommends adjusted parameters. A new version is created. Nothing changes automatically.

The result is AI that is expressive to define, deterministic to run, and safe to govern — which is precisely what production agentic systems need.

---

## Problem Memintel Solves

Current agentic frameworks struggle with a specific class of problems that Memintel addresses directly.

**Condition specification.**
Agents today typically decide when to act using the LLM's own judgment at runtime. Memintel separates that question into two parts: the LLM defines what counts as a situation worth acting on at authoring time, and a deterministic evaluator answers "is this situation one of those?" at runtime.

**Auditability and compliance.**
In regulated industries — finance, healthcare, insurance — one needs to explain every automated decision. "The model thought it was a good idea" is not an explanation. "The customer's churn score exceeded 0.85, which is the high-severity threshold for this domain, and here is the exact computation trace" is. Memintel produces the second kind of answer by default.

**Controlled adaptation.**
Memintel's calibration model is explicit: feedback is structured, adjustments are versioned, and the user approves every change before it takes effect. You get adaptation without losing control.

**Multi-agent coordination.**
If each decision boundary in a pipeline is a Memintel condition — explicit, versioned, reproducible — you can reason about the whole system.

---

## The Core Idea

Think of how traditional software works. You write source code, a compiler validates and transforms it into a deterministic executable, and the runtime runs that executable predictably every time.

Memintel applies exactly this model to AI-driven decisions.
```
Natural Language Intent  →  Compiler  →  Deterministic Execution
      (Source Code)         (Validator)      (Runtime)
```

The LLM is the author. The compiler is the authority. The runtime is the executor. Each has a strictly defined job and cannot do the other's.

---

## The Three Layers

**Layer 1 — The LLM (Interpretation)**

The LLM does one job: convert natural language intent into a structured, fully specified task definition. It selects a strategy, resolves parameters, and binds an action. Once the task is compiled, the LLM is out of the loop entirely.

**Layer 2 — The Compiler (Validation)**

The compiler enforces the type system, validates every operator and strategy against their schemas, builds a directed acyclic execution graph, and produces a deterministic intermediate representation with a cryptographic hash. If anything is invalid, the task is rejected — not approximated, not warned about. Rejected.

**Layer 3 — The Runtime (Execution)**

The runtime executes compiled graphs against live data, evaluates conditions, and triggers actions. It is a pure function — given the same concept version, condition version, entity, and timestamp, it always produces the same result. No LLM involvement. No probabilistic behaviour. Full reproducibility.

---

## Why This Matters

**The separation of concerns is clean and principled.**
The LLM cannot execute. The runtime cannot interpret. The compiler cannot be bypassed. That enforced separation is what makes the whole system provably predictable rather than just usually predictable.

**The type system does real work.**
Memintel's type system actively governs what computations are valid at compile time — preventing an entire class of subtle bugs that would otherwise only surface in production.

**The versioning model is unusually rigorous.**
Definitions are immutable once registered. All execution is version-pinned. "Latest" references are rejected at compile time. The resulting execution graph is deterministically hashed, ensuring that the graph being executed is exactly the one that was compiled.

---

## Repository Structure
```
backend/    — Python execution engine (FastAPI, Pydantic v2, asyncpg)
sdk/        — TypeScript SDK (@memintel/sdk)
```

---

## Getting Started

See [backend/memintel-backend/README.md](backend/memintel-backend/README.md) for full setup instructions.
