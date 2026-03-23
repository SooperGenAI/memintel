# memintel
What is Memintel?
Memintel is a deterministic compiler for AI-driven decisions. You describe what you want to monitor in plain language. Memintel's LLM interprets that intent within a strict guardrails policy, generates a fully specified decision definition, and compiles it into an immutable, versioned execution graph. From that point, every evaluation is deterministic — the same data always produces the same decision, with a full computation trace. When decision quality needs to change, users give structured feedback. Calibration recommends adjusted parameters. A new version is created. Nothing changes automatically. The result is AI that is expressive to define, deterministic to run, and safe to govern — which is precisely what production agentic systems need.

Problem that Memintel solves
Current agentic frameworks struggle with a specific class of problems that Memintel addresses directly:
Condition specification. 
Agents today typically decide when to act using the LLM's own judgment at runtime — "does this situation warrant sending an alert?" Memintel separates that question into two parts: the LLM defines what counts as a situation worth acting on at authoring time, and a deterministic evaluator answers is this situation one of those at runtime. That's a much stronger foundation for reliable automation.
The result - auditability and compliance. In regulated industries — finance, healthcare, insurance — one needs to be able to explain every automated decision. "The model thought it was a good idea" is not an explanation. "The customer's churn score exceeded 0.85, which is the high-severity threshold for this domain, and here is the exact computation trace" is. Memintel produces the second kind of answer by default.
Controlled adaptation. Agents that learn and adjust are appealing in theory but dangerous in practice because the adjustment process itself is typically opaque. Memintel's calibration model is explicit: feedback is structured, adjustments are versioned, and the user approves every change before it takes effect. You get adaptation without losing control.
Multi-agent coordination. As systems get more complex and agents trigger other agents, indeterminacy compounds. A non-deterministic decision in step 2 of a pipeline corrupts everything downstream in ways that are very hard to diagnose. If each decision boundary in the pipeline is a Memintel condition — explicit, versioned, reproducible — you can reason about the whole system.


The Core Idea
Think of how traditional software works. You write source code, a compiler validates and transforms it into a deterministic executable, and the runtime runs that executable predictably every time. The compiler is the guarantee of correctness.
Memintel applies exactly this model to AI-driven decisions.

Natural Language Intent  →  Compiler  →  Deterministic Execution
      (Source Code)         (Validator)      (Runtime)

The LLM is the author. The compiler is the authority. The runtime is the executor. Each has a strictly defined job and cannot do the other's.

The analogy to traditional software is apt: we don't ship source code that interprets itself at runtime — we compile it into a deterministic representation that behaves predictably. Memintel applies that same principle to AI-driven decisions. Natural language intent is the source code. The guardrails and type system are the language spec. The compiler validates and transforms. The runtime executes deterministically.
Most of the agentic frameworks being built today are essentially skipping the compiler step — they're running "interpreted" AI decisions at runtime. That's fast to prototype but fragile at scale, hard to audit, and nearly impossible to govern.
The systems that will matter in production — in enterprise, in regulated industries, in high-stakes automation — will need exactly what Memintel provides: a principled boundary between probabilistic interpretation and deterministic execution.


The Three Layers
Layer 1 — The LLM (Interpretation)
The LLM does one job: convert natural language intent into a structured, fully specified task definition. It selects a strategy, resolves parameters, and binds an action. It works within hard constraints set by the guardrails system — it cannot invent strategies, exceed parameter bounds, or produce type-invalid definitions. Once the task is compiled, the LLM is out of the loop entirely.
Layer 2 — The Compiler (Validation)
The compiler is the correctness engine. It enforces the type system, validates every operator and strategy against their schemas, checks that all references resolve, builds a directed acyclic execution graph, and produces a deterministic intermediate representation with a cryptographic hash. If anything is invalid, the task is rejected — not approximated, not warned about. Rejected. Nothing executes unless it passes the compiler.
Layer 3 — The Runtime (Execution)
The runtime executes compiled graphs against live data, evaluates conditions, and triggers actions. It is a pure function — given the same concept version, condition version, entity, and timestamp, it always produces the same result. No LLM involvement. No probabilistic behaviour. Full reproducibility.


The separation of concerns is clean and principled
The three-layer split — LLM for interpretation, Compiler for validation, Runtime for execution — isn't just a nice diagram. Each layer has a strictly defined job and cannot do the other's job. The LLM cannot execute. The runtime cannot interpret. The compiler cannot be bypassed. That kind of enforced separation is what makes the whole system provably predictable rather than just usually predictable.
Most agentic architectures blur these boundaries, especially between interpretation and execution. Memintel draws a hard line at compilation, which is architecturally unusual and genuinely important.

The type system does real work
A lot of AI system designs have a type system in the diagram but it doesn't actually enforce anything at runtime. Memintel's type system actively governs what computations are valid — you can't pass a time_series<float> where a float is expected, you can't use equals on a numeric primitive, a decision<boolean> can't flow back into a concept operator. These aren't guidelines, they're compile-time rejections. That prevents an entire class of subtle bugs that would otherwise only surface in production.
The nullable type handling — T? propagating through the graph unless explicitly resolved — is particularly mature. Most systems either ignore missing data or silently substitute zeros. Memintel forces you to declare a policy, which means the behaviour under data failure is always explicit and intentional.

The versioning model is unusually rigorous
Version pinning is easy to say and hard to actually enforce. Most systems that claim to be versioned still rely on implicit “latest” resolution or allow in-place mutation of definitions, leading to silent behavior drift.
Memintel prevents both at the compiler and API level:
· “latest” references are rejected at compile time
· definitions are immutable once registered
· all execution is version-pinned
The resulting execution graph is deterministically hashed (ir_hash), ensuring that the graph being executed is exactly the one that was compiled.
This is audit-grade, replayable versioning — not just labeling.
