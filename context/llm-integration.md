# Memintel — LLM Integration Specification
**Version:** 1.0  
**Status:** Authoritative  
**Audience:** Session 10a (TaskAuthoringService implementation); `llm/` module in the Python backend

> **Prerequisite:** Read `core-spec.md`, `py-instructions.md` section "POST /tasks",
> `memintel.guardrails.md`, and `memintel.config.md` before this file.
> This document specifies the prompt templates, output schemas, refinement loop,
> and provider abstraction for the LLM call inside `TaskAuthoringService.create_task()`.

---

## 1. Role of the LLM in Memintel

The LLM has exactly one job: **translate natural language intent into a structured,
fully-specified task definition**. It is invoked once per `POST /tasks` call. It is
never invoked at execution time.

```
LLM SCOPE:

Invoked at:   POST /tasks only (task authoring)
              POST /agents/* endpoints (agent-assisted definition)
Not invoked:  POST /evaluate/full
              POST /evaluate/condition
              POST /execute
              POST /conditions/calibrate
              POST /conditions/apply-calibration
              POST /feedback/decision
              Anything else — these are all deterministic

The LLM is an intent parser, not a decision maker.
It outputs a structured definition. The compiler validates it.
The runtime executes it deterministically. The LLM never sees execution results.
```

---

## 2. LLM Output Structure

The LLM must produce a single JSON object containing three parts: concept, condition,
and action. This is the canonical output schema — the `LLMClient` parses and validates
this structure before passing it to the compiler.

```json
{
  "concept": {
    "id": "org.churn_risk",
    "version": "1.0",
    "namespace": "org",
    "description": "Composite churn risk score based on engagement and activity trends",
    "output_type": "float",
    "features": [
      {
        "name": "user.daily_active_minutes",
        "type": "time_series<float>"
      },
      {
        "name": "user.session_count_30d",
        "type": "int"
      }
    ],
    "compute": {
      "op": "weighted_mean",
      "inputs": ["user.daily_active_minutes", "user.session_count_30d"],
      "params": { "weights": [0.6, 0.4] }
    }
  },
  "condition": {
    "id": "org.high_churn_risk",
    "version": "1.0",
    "namespace": "org",
    "concept_id": "org.churn_risk",
    "concept_version": "1.0",
    "strategy": {
      "type": "threshold",
      "params": {
        "direction": "above",
        "value": 0.80
      }
    }
  },
  "action": {
    "id": "org.notify_team",
    "version": "1.0"
  }
}
```

### Output Schema Rules

```
LLM OUTPUT SCHEMA RULES:

1. The output MUST be valid JSON. No markdown fences, no preamble,
   no explanation text. Raw JSON object only.

2. concept.id and condition.id MUST follow namespace.name format.
   The namespace prefix MUST match the namespace field.
   Example: if namespace='org', id must start with 'org.'

3. concept.output_type MUST be a valid Memintel scalar type:
   float | int | boolean | categorical
   (Concepts produce scalars — not time_series or list)

4. condition.strategy MUST be fully specified:
   - strategy.type MUST be one of: threshold | percentile | z_score |
     change | equals | composite
   - strategy.params MUST match the strategy's parameter schema exactly
   - A condition without strategy.type and strategy.params is INVALID

5. All feature names in concept.features MUST exist in the primitive registry.
   The LLM must only reference primitives provided in the context.

   CONCEPT CONSISTENCY RULE:
   concept.compute.inputs MUST be a subset of concept.features[].name.
   No input may reference a primitive not listed in concept.features.
   Violation example:
     features: [{name: "user.score"}]
     compute.inputs: ["user.score", "user.count"]   <- user.count not declared in features
   The parser validates this before passing to the compiler.

6. action.id and action.version MUST reference a registered action.
   The LLM selects an action using the action binding resolution order —
   it does not invent action IDs.

7. Version strings MUST be "1.0" for new definitions.
   The registry assigns actual versioning; "1.0" is always the starting point.
```

### Strategy Parameter Schemas

The LLM must produce exactly these parameter structures per strategy:

```json
// threshold
{ "direction": "above",    "value": 0.80 }
{ "direction": "below",    "value": 0.20 }

// percentile
{ "direction": "top",      "value": 10 }
{ "direction": "bottom",   "value": 25 }

// z_score  — note: key is "threshold", not "value"
{ "direction": "above",    "threshold": 2.5,  "window": "30d" }
{ "direction": "any",      "threshold": 2.0,  "window": "90d" }

// change
{ "direction": "decrease", "value": 0.20,     "window": "7d" }
{ "direction": "increase", "value": 0.15,     "window": "14d" }

// equals
{ "value": "high_risk" }
{ "value": "churned",  "labels": ["churned", "at_risk"] }

// composite
{ "operator": "AND",   "operands": ["org.high_churn", "org.high_value"] }
{ "operator": "OR",    "operands": ["org.payment_fail", "org.login_fail"] }
// COMPOSITE OPERAND RULE:
// operands must reference existing condition IDs in the registry
// each operand condition must produce decision<boolean>
// equals conditions produce decision<categorical> and cannot be operands
// LLM must prefer existing known conditions, not invent new operand IDs
```

---

## 3. Context Injection Order

Before calling the LLM, `build_llm_context()` assembles the system prompt context
in a strict order. **Do not reorder these sections.** Earlier sections establish
hard rules; later sections provide guidance within those rules.

```python
def build_llm_context(
    guardrails: Guardrails,
    app_context: ApplicationContext,
    primitive_registry: PrimitiveRegistry,
    type_system_summary: str,
) -> str:
    sections = []

    # [1] TYPE SYSTEM — hard rules, no exceptions
    sections.append(f"""
=== TYPE SYSTEM (HARD RULES) ===
{type_system_summary}

These rules are absolute. The compiler will reject any output that violates them.
decision<boolean> and decision<categorical> are output types of conditions only —
they cannot be used as concept compute inputs.
""")

    # [2] GUARDRAILS — strategy registry, type compatibility, bounds, priors
    sections.append(f"""
=== GUARDRAILS ===
Strategy Registry (available strategies):
{format_strategy_registry(guardrails.strategy_registry)}

Type-Strategy Compatibility (which strategies are valid for which primitive types):
{format_type_compatibility(guardrails.type_compatibility)}

Threshold Priors (default parameter values by severity):
{format_thresholds(guardrails.thresholds)}

Threshold Bounds (hard limits — parameters must stay within these):
{format_bounds(guardrails.threshold_bounds)}

Strategy Resolution Priority Order (follow this order strictly):
  1. user_explicit  — if user stated a specific strategy, use it
  2. primitive_hint — check primitive definition for strategy hint
  3. mapping_rule   — check guardrails mappings for intent keywords
  4. application_context — apply domain preferences from context below
  5. global_preferred — guardrails preferred strategy for this type
  6. global_default — fallback strategy for this type
""")

    # [3] APPLICATION CONTEXT — domain description and soft guidance
    sections.append(f"""
=== APPLICATION CONTEXT ===
Domain: {app_context.name}
Description: {app_context.description}

Instructions:
{chr(10).join(f'  - {i}' for i in app_context.instructions)}

Default entity scope: {app_context.default_entity_scope or 'not specified'}

Action preferences by severity:
{format_action_preferences(app_context.action_preferences)}

Application context is guidance, not rules. Type system and guardrails take
precedence when they conflict with application context.
""")

    # [4] PARAMETER BIAS RULES — deterministic instruction-to-severity mappings
    sections.append(f"""
=== PARAMETER BIAS RULES ===
These mappings convert natural language cues in the intent into deterministic
severity shifts. Apply them before selecting a parameter value.

{format_bias_rules(guardrails.parameter_bias_rules)}

A severity_shift of -1 means shift one tier lower (high -> medium, medium -> low).
A severity_shift of +1 means shift one tier higher (low -> medium, medium -> high).
Shifts are clamped at the boundaries (cannot go below low or above high).
""")

    # [5] PRIMITIVE REGISTRY — available data signals
    sections.append(f"""
=== AVAILABLE PRIMITIVES ===
You may ONLY reference primitives from this list. Do not invent primitive names.

{format_primitives(primitive_registry.list_all())}
""")

    return "\n".join(sections)
```

---

## 4. Context Size Management

The assembled context can be large. For models with smaller context windows or large
primitive registries, the context must be reduced without losing correctness.

```
CONTEXT SIZE MANAGEMENT:

Reduction priority order — NEVER truncate these:
  [1] Type system section    — hard rules, compiler enforces them
  [2] Strategy registry      — compiler rejects unknown strategies
  [3] Parameter bounds       — parameters must stay within bounds

May be reduced if budget is exceeded (in this order):
  [4] Primitive list         — filter to intent-relevant primitives first
  [5] Full guardrails priors — keep only the resolved severity tier
  [6] Application context instructions — keep description, truncate list
  [7] Parameter bias rules   — keep only matched rules

NEVER remove the type system section under any budget constraint.
NEVER remove strategy registry or parameter bounds.
```

```python
MAX_CONTEXT_CHARS = 80_000  # tune per model and deployment

def build_llm_context_with_budget(
    guardrails, app_context, primitive_registry,
    type_system_summary, intent, max_chars=MAX_CONTEXT_CHARS,
):
    type_sec   = format_type_system(type_system_summary)
    guard_sec  = format_guardrails(guardrails)
    app_sec    = format_app_context(app_context)
    bias_sec   = format_bias_rules(guardrails.parameter_bias_rules)
    prim_sec   = format_primitives(primitive_registry.list_all())

    full = "\n".join([type_sec, guard_sec, app_sec, bias_sec, prim_sec])
    if len(full) <= max_chars:
        return full

    # Reduction step 1: filter primitives to intent-relevant only
    relevant = filter_primitives_by_intent(primitive_registry.list_all(), intent)
    prim_sec = format_primitives(relevant)
    reduced = "\n".join([type_sec, guard_sec, app_sec, bias_sec, prim_sec])
    if len(reduced) <= max_chars:
        return reduced

    # Reduction step 2: matched bias rules only
    bias_sec = format_bias_rules(filter_bias_rules_by_intent(guardrails.parameter_bias_rules, intent))
    reduced = "\n".join([type_sec, guard_sec, app_sec, bias_sec, prim_sec])
    if len(reduced) <= max_chars:
        return reduced

    # Reduction step 3: compact app context (description only)
    app_sec = format_app_context_compact(app_context)
    reduced = "\n".join([type_sec, guard_sec, app_sec, bias_sec, prim_sec])
    if len(reduced) <= max_chars:
        return reduced

    # Last resort: type system + guardrails + minimal primitives only
    import logging
    logging.warning(f"LLM context budget exhausted ({len(reduced)} chars > {max_chars}). Using minimal context.")
    return "\n".join([type_sec, guard_sec, prim_sec])


def filter_primitives_by_intent(primitives, intent):
    words = set(intent.lower().split())
    scored = []
    for p in primitives:
        name_words = set(p.name.replace('.', ' ').replace('_', ' ').split())
        scored.append((len(words & name_words), p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:max(5, min(15, len(scored)))]]
```

---

## 5. System Prompt

This is the system prompt sent to the LLM at the start of every task creation call.
It does not change between calls — it establishes the LLM's role and output contract.

```
SYSTEM_PROMPT = """
You are the Memintel intent compiler. Your job is to translate a natural language
monitoring intent into a structured task definition consisting of three parts:
a Concept (what to compute), a Condition (when to alert), and an Action (what to do).

OUTPUT CONTRACT:
- You must output a single valid JSON object. No markdown fences, no explanation,
  no preamble. Raw JSON only.
- The JSON must conform exactly to the schema provided in the user message.
- Every field marked as required must be present.
- strategy.type and strategy.params are ALWAYS required in the condition.
  There is no default strategy. You must always specify both fields explicitly.
- You may only reference primitives listed in the AVAILABLE PRIMITIVES section.
- You may only use strategies listed in the GUARDRAILS section.
- All parameter values must be within the bounds specified in GUARDRAILS.

RESOLUTION RULES:
- Follow the strategy resolution priority order in GUARDRAILS exactly.
- Apply parameter bias rules before selecting threshold values.
- Use threshold priors as starting points; adjust using bias rules.
- If the intent contains conflicting signals, prefer the more conservative choice.

FAILURE MODES YOU MUST AVOID:
- Missing strategy.type or strategy.params in the condition
- Referencing a primitive not in the AVAILABLE PRIMITIVES list
- Using a strategy not in the GUARDRAILS strategy registry
- Setting a parameter value outside the declared bounds
- Using decision<boolean> or decision<categorical> as a concept output type
- Inventing action IDs — use only actions from the resolution order
"""
```

---

## 6. Generation Prompts

### 5.1 Concept Generation Prompt

```python
CONCEPT_GENERATION_PROMPT = """
{context}

=== TASK ===
Translate the following monitoring intent into a Concept definition.

Intent: "{intent}"
Entity scope: "{entity_scope}"

A Concept defines WHAT to compute for a given entity. It declares:
- Which primitives to use (from AVAILABLE PRIMITIVES only)
- How to combine them into a single output value
- The output type (must be a scalar: float, int, boolean, or categorical)

Output the concept as a JSON object with this exact structure:
{{
  "id": "<namespace>.<descriptive_name>",
  "version": "1.0",
  "namespace": "<namespace>",
  "description": "<one sentence describing what this measures>",
  "output_type": "<float|int|boolean|categorical>",
  "features": [
    {{"name": "<primitive_name>", "type": "<primitive_type>"}}
  ],
  "compute": {{
    "op": "<operator>",
    "inputs": ["<primitive_name>", ...],
    "params": {{...}}
  }}
}}

Use namespace "{namespace}" unless the intent specifies otherwise.
Output JSON only. No explanation.
"""
```

### 5.2 Condition Generation Prompt

```python
CONDITION_GENERATION_PROMPT = """
{context}

=== TASK ===
Given the following Concept, generate a Condition definition.

Concept:
{concept_json}

Intent: "{intent}"
Constraints: {constraints_json}

A Condition defines WHEN the concept value is significant enough to alert.
It must include a fully-specified strategy — type AND params.

Follow the strategy resolution priority order from GUARDRAILS:
1. user_explicit (did the intent specify a strategy?)
2. primitive_hint (does the concept's primary primitive have a strategy hint?)
3. mapping_rule (does the intent match any keywords in guardrails mappings?)
4. application_context (what does the domain context prefer?)
5. global_preferred (what does the guardrails specify as preferred for this type?)
6. global_default (fallback)

Apply parameter bias rules from PARAMETER BIAS RULES before setting values.
Start with the threshold prior for the resolved severity level, then adjust.

Output the condition as a JSON object with this exact structure:
{{
  "id": "<namespace>.<descriptive_name>",
  "version": "1.0",
  "namespace": "<namespace>",
  "concept_id": "{concept_id}",
  "concept_version": "{concept_version}",
  "strategy": {{
    "type": "<threshold|percentile|z_score|change|equals|composite>",
    "params": {{
      <strategy-specific parameters — see GUARDRAILS for exact schema>
    }}
  }}
}}

Output JSON only. No explanation.
"""
```

### 5.3 Action Resolution

Action resolution is **deterministic** — it does not require an LLM call. The
`resolve_action()` method follows a fixed priority order:

```python
def resolve_action(
    self,
    delivery: DeliveryConfig,
    guardrails: Guardrails,
    app_context: ApplicationContext,
    condition: dict,
) -> dict | None:
    """
    Action binding resolution order (deterministic — no LLM):
      1. user_explicit:    delivery config specifies an action type directly
      2. guardrails default: guardrails.default_action for this strategy/primitive
      3. application_context: app_context.action_preferences for resolved severity
      4. system default:   'notification' for boolean, 'notification' for categorical
      5. fail:             return None if nothing resolves (raises action_binding_failed)
    """
    # Step 1: user_explicit — delivery type maps directly to action type
    if delivery.type in ('webhook', 'workflow', 'notification', 'email'):
        action = self.action_registry.get_by_type(delivery.type)
        if action:
            return {'id': action.id, 'version': action.version}

    # Step 2: guardrails default
    strategy_type = condition['strategy']['type']
    default = guardrails.get_default_action(strategy_type)
    if default:
        return {'id': default.id, 'version': default.version}

    # Step 3: application_context preferences (subject to constraints)
    severity = self.resolve_severity(condition)
    pref = app_context.action_preferences.get(f'{severity}_severity')
    if pref:
        action = self.action_registry.get_by_type(pref)
        if action and not guardrails.is_action_disallowed(pref):
            return {'id': action.id, 'version': action.version}

    # Step 4: system default
    system_default = self.action_registry.get_system_default()
    if system_default:
        return {'id': system_default.id, 'version': system_default.version}

    # Step 5: fail
    return None


def validate_action_before_compiler(action, action_registry):
    # ACTION VALIDATION RULE:
    # Call immediately after resolve_action(), BEFORE compiler validation.
    # Fail fast with a clear typed error rather than letting a bad action
    # reach the compiler and produce a confusing reference_error.
    if action is None:
        raise HTTPException(status_code=422, detail={
            'error': {
                'type': 'action_binding_failed',
                'message': 'No action could be resolved. Check delivery config, '
                           'guardrails default_action, and action_preferences.',
                'suggestion': 'Ensure at least one action exists in the registry '
                              'and is not disallowed by guardrails constraints.',
            }
        })
    registered = action_registry.get(action['id'], action['version'])
    if registered is None:
        raise HTTPException(status_code=422, detail={
            'error': {
                'type': 'action_binding_failed',
                'message': f"Action '{action['id']}' v'{action['version']}' "
                           f"does not exist in the registry.",
                'suggestion': 'Check that the action is registered and not deprecated.',
            }
        })
```

---

## 7. Refinement Loop

When the compiler rejects an LLM-generated definition, the refinement loop sends
the validation errors back to the LLM with instructions to fix only the failing parts.

### 6.1 Loop Structure

```python
async def generate_with_refinement(
    self,
    intent: str,
    entity_scope: str,
    context: str,
    namespace: str,
    max_retries: int = 3,
) -> dict:
    """
    Generate concept + condition with compiler-guided refinement.
    Returns validated output dict or raises LLMGenerationError.
    """
    # Initial generation
    raw = await self.llm_client.complete(
        system=SYSTEM_PROMPT,
        user=build_initial_prompt(intent, entity_scope, context, namespace),
    )
    output = parse_llm_output(raw)  # raises ParseError if not valid JSON

    for attempt in range(max_retries):
        # Validate through compiler
        errors = await self.compiler.validate_llm_output(output)
        if not errors:
            return output  # success

        if attempt == max_retries - 1:
            # Final attempt failed
            raise LLMGenerationError(
                message=f"Failed to generate valid definition after {max_retries} attempts",
                last_errors=errors,
                last_output=output,
            )

        # Refine: fix only the failing parts
        raw = await self.llm_client.complete(
            system=SYSTEM_PROMPT,
            user=build_refinement_prompt(output, errors, context),
        )
        output = parse_llm_output(raw)

    raise LLMGenerationError("Refinement loop exhausted")


async def safe_llm_complete(llm_client, system: str, user: str, attempt: int) -> str:
    """
    LLM FAILURE HANDLING:
    All call failures count toward max_retries.

    Timeout       -> LLMCallError('timeout')        -> caller retries
    Empty response-> LLMCallError('empty_response') -> caller retries
    Non-JSON      -> LLMCallError('parse_error')    -> caller retries
    Provider 5xx  -> LLMCallError('provider_error') -> caller retries
    Auth 401/403  -> LLMAuthError                   -> do NOT retry, raise immediately
    """
    import asyncio
    try:
        raw = await asyncio.wait_for(
            llm_client.complete(system, user),
            timeout=getattr(llm_client, 'timeout_seconds', 30),
        )
    except asyncio.TimeoutError:
        raise LLMCallError('timeout', f"LLM timed out on attempt {attempt + 1}")
    except LLMAuthError:
        raise
    except Exception as e:
        raise LLMCallError('provider_error', str(e))

    if not raw or not raw.strip():
        raise LLMCallError('empty_response', f"LLM returned empty response on attempt {attempt + 1}")
    return raw


class LLMCallError(Exception):
    def __init__(self, failure_type: str, message: str):
        super().__init__(message)
        self.failure_type = failure_type  # timeout | empty_response | parse_error | provider_error

class LLMAuthError(Exception):
    pass
```

### 6.2 Refinement Prompt

```python
REFINEMENT_PROMPT = """
{context}

=== TASK ===
Your previous output failed compiler validation. Fix ONLY the failing parts.
Do not regenerate the entire definition from scratch.

Your previous output:
{previous_output_json}

Compiler validation errors:
{errors_formatted}

Instructions:
- Fix each error listed above
- Keep all parts of the output that passed validation unchanged
- Ensure strategy.type and strategy.params are both present in the condition
- Ensure all parameter values are within the bounds in GUARDRAILS
- Ensure all primitive names exist in AVAILABLE PRIMITIVES
- Output the complete corrected JSON object (not just the changed parts)

Output JSON only. No explanation.
"""

def format_errors_for_refinement(errors: list[ValidationError]) -> str:
    """Format compiler validation errors clearly for the LLM."""
    lines = []
    for i, err in enumerate(errors, 1):
        line = f"{i}. [{err.type}]"
        if err.location:
            line += f" at {err.location}"
        line += f": {err.message}"
        if err.suggestion:
            line += f"\n   Fix: {err.suggestion}"
        lines.append(line)
    return "\n".join(lines)
```

### 6.3 Retry Termination

```
REFINEMENT LOOP TERMINATION:

Success:   Compiler validation returns zero errors → return output
Failure:   max_retries reached without success → raise LLMGenerationError

On LLMGenerationError:
  → HTTP 422 with error.type = 'semantic_error'
  → error.message = last compiler error message
  → error.location = last compiler error location (if available)
  → error.suggestion = last compiler error suggestion (if available)
  → DO NOT persist any partial definition
  → DO NOT assign a task_id
  → System remains clean as if the request was never made

DO NOT catch LLMGenerationError silently.
DO NOT return a partially-valid output after max_retries.
DO NOT retry with the same prompt — each retry must include the errors.
```

---

## 8. Output Parsing

```python
import json
from pydantic import BaseModel, ValidationError as PydanticValidationError

class LLMConceptOutput(BaseModel):
    id: str
    version: str
    namespace: str
    description: str
    output_type: str
    features: list[dict]
    compute: dict

class LLMConditionOutput(BaseModel):
    id: str
    version: str
    namespace: str
    concept_id: str
    concept_version: str
    strategy: dict  # validated further by compiler

class LLMActionOutput(BaseModel):
    id: str
    version: str

class LLMTaskOutput(BaseModel):
    concept: LLMConceptOutput
    condition: LLMConditionOutput
    action: LLMActionOutput

def parse_llm_output(raw: str) -> dict:
    """
    Parse and structurally validate LLM output.
    Raises ParseError with a clear message if output is not valid JSON
    or does not match the expected structure.
    """
    cleaned = raw.strip()

    # Step 1: strip markdown fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        cleaned = cleaned.strip()

    # Step 2: extract JSON object — handles stray preamble or trailing explanation
    start = cleaned.find('{')
    end   = cleaned.rfind('}')
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    elif start == -1 or end == -1:
        raise ParseError(
            f"LLM output contains no JSON object. "
            f"Raw output (first 300 chars): {raw[:300]!r}"
        )

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ParseError(
            f"LLM output is not valid JSON: {e}. "
            f"Raw output (first 200 chars): {raw[:200]!r}"
        )

    try:
        validated = LLMTaskOutput.model_validate(data)
    except PydanticValidationError as e:
        raise ParseError(
            f"LLM output does not match expected structure: {e}. "
            f"Check that concept, condition, and action are all present."
        )

    # Namespace enforcement
    concept   = validated.concept
    condition = validated.condition

    if not concept.id.startswith(concept.namespace + "."):
        raise ParseError(
            f"concept.id '{concept.id}' must start with '{concept.namespace}.' "
            f"Example: '{concept.namespace}.churn_risk'"
        )
    if not condition.id.startswith(condition.namespace + "."):
        raise ParseError(
            f"condition.id '{condition.id}' must start with '{condition.namespace}.' "
            f"Example: '{condition.namespace}.high_churn'"
        )
    if condition.concept_id != concept.id:
        raise ParseError(
            f"condition.concept_id '{condition.concept_id}' must match concept.id '{concept.id}'"
        )
    if condition.concept_version != concept.version:
        raise ParseError(
            f"condition.concept_version must match concept.version '{concept.version}'"
        )

    # Concept consistency: compute.inputs must be subset of features[].name
    feature_names  = {f['name'] for f in concept.features}
    compute_inputs = set(concept.compute.get('inputs', []))
    unknown = compute_inputs - feature_names
    if unknown:
        raise ParseError(
            f"concept.compute.inputs references undeclared primitives: {sorted(unknown)}. "
            f"All compute inputs must be listed in concept.features."
        )

    return validated.model_dump()
```

---

## 9. Provider Abstraction

The LLM provider is configured in `memintel.config.md`. The `LLMClient` wraps
all provider-specific API calls behind a single interface.

```python
from abc import ABC, abstractmethod

class LLMClient(ABC):
    """Abstract interface — all providers implement this."""

    @abstractmethod
    async def complete(self, system: str, user: str) -> str:
        """Send a completion request. Returns raw text response."""
        ...

class AnthropicClient(LLMClient):
    def __init__(self, config: LLMConfig):
        import anthropic
        self.client = anthropic.AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.endpoint,
        )
        self.model = config.model
        self.timeout = config.timeout_ms / 1000

    async def complete(self, system: str, user: str) -> str:
        msg = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=0,   # always 0 — deterministic generation
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text

class OpenAIClient(LLMClient):
    def __init__(self, config: LLMConfig):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.endpoint,
        )
        self.model = config.model

    async def complete(self, system: str, user: str) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return resp.choices[0].message.content

class OllamaClient(LLMClient):
    """For local development — wraps Ollama's OpenAI-compatible API."""
    def __init__(self, config: LLMConfig):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            api_key='ollama',  # Ollama does not require a real key
            base_url=config.endpoint + '/v1',
        )
        self.model = config.model

    async def complete(self, system: str, user: str) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return resp.choices[0].message.content

def create_llm_client(config: LLMConfig) -> LLMClient:
    """Factory — returns the correct client for the configured provider."""
    clients = {
        'anthropic':    AnthropicClient,
        'openai':       OpenAIClient,
        'azure_openai': OpenAIClient,  # Azure uses OpenAI-compatible API
        'ollama':       OllamaClient,
    }
    cls = clients.get(config.provider)
    if cls is None:
        raise ConfigError(f"Unknown LLM provider: {config.provider}")
    return cls(config)
```

---

## 10. Fixture Client (Development Mode)

During development (`USE_LLM_FIXTURES=True`), the `LLMFixtureClient` returns
pre-built fixture outputs instead of calling the real LLM. This allows the full
pipeline to be built and tested without an LLM dependency.

```python
import os, json
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / 'fixtures'

class LLMFixtureClient(LLMClient):
    """
    Returns fixture outputs for development and testing.
    Routes to the appropriate fixture based on keywords in the intent.
    Active when USE_LLM_FIXTURES=True in environment.
    """

    KEYWORD_MAP = {
        'threshold_task.json': [
            'churn', 'score', 'risk', 'above', 'below', 'exceeds', 'drops below'
        ],
        'z_score_task.json': [
            'anomaly', 'spike', 'unusual', 'abnormal', 'deviation', 'outlier'
        ],
        'composite_task.json': [
            'and', 'both', 'combined', 'together', 'all of', 'multiple'
        ],
        'equals_task.json': [
            'equals', 'matches', 'is exactly', 'category', 'tier', 'plan'
        ],
    }

    async def complete(self, system: str, user: str) -> str:
        intent = self._extract_intent(user)
        fixture_file = self._route_to_fixture(intent)
        fixture_path = FIXTURES_DIR / fixture_file
        if not fixture_path.exists():
            raise FileNotFoundError(
                f"Fixture not found: {fixture_path}. "
                f"Run Session 7 to create LLM output fixtures."
            )
        return fixture_path.read_text()

    def _extract_intent(self, prompt: str) -> str:
        """Extract the intent string from the formatted prompt."""
        for line in prompt.split('\n'):
            if line.strip().startswith('Intent:'):
                return line.split(':', 1)[1].strip().strip('"')
        return prompt[:100]  # fallback: use start of prompt

    def _route_to_fixture(self, intent: str) -> str:
        """Route to fixture file based on intent keywords."""
        intent_lower = intent.lower()
        for fixture, keywords in self.KEYWORD_MAP.items():
            if any(kw in intent_lower for kw in keywords):
                return fixture
        return 'threshold_task.json'  # default fixture

def get_llm_client(config: LLMConfig) -> LLMClient:
    """
    Returns fixture client in development, real client in production.
    Check USE_LLM_FIXTURES environment variable.
    """
    if os.environ.get('USE_LLM_FIXTURES', '').lower() in ('true', '1', 'yes'):
        return LLMFixtureClient()
    return create_llm_client(config)
```

---

## 11. Fixture File Schemas

Session 7 creates these four fixture files. Each must be valid JSON that passes
full compiler validation. Use these schemas as the template.

### `fixtures/threshold_task.json`

```json
{
  "concept": {
    "id": "org.churn_risk_score",
    "version": "1.0",
    "namespace": "org",
    "description": "Composite churn risk score based on engagement and activity decline",
    "output_type": "float",
    "features": [
      { "name": "user.daily_active_minutes", "type": "time_series<float>" },
      { "name": "user.session_count_30d",    "type": "int" }
    ],
    "compute": {
      "op": "weighted_mean",
      "inputs": ["user.daily_active_minutes", "user.session_count_30d"],
      "params": { "weights": [0.6, 0.4] }
    }
  },
  "condition": {
    "id": "org.high_churn_risk",
    "version": "1.0",
    "namespace": "org",
    "concept_id": "org.churn_risk_score",
    "concept_version": "1.0",
    "strategy": {
      "type": "threshold",
      "params": { "direction": "above", "value": 0.80 }
    }
  },
  "action": {
    "id": "org.notify_team",
    "version": "1.0"
  }
}
```

### `fixtures/z_score_task.json`

```json
{
  "concept": {
    "id": "org.payment_failure_rate",
    "version": "1.0",
    "namespace": "org",
    "description": "Rolling payment failure rate over the last 30 days",
    "output_type": "float",
    "features": [
      { "name": "payment.failure_rate_30d", "type": "float" }
    ],
    "compute": {
      "op": "identity",
      "inputs": ["payment.failure_rate_30d"],
      "params": {}
    }
  },
  "condition": {
    "id": "org.payment_anomaly",
    "version": "1.0",
    "namespace": "org",
    "concept_id": "org.payment_failure_rate",
    "concept_version": "1.0",
    "strategy": {
      "type": "z_score",
      "params": { "direction": "above", "threshold": 2.5, "window": "30d" }
    }
  },
  "action": {
    "id": "org.notify_team",
    "version": "1.0"
  }
}
```

### `fixtures/composite_task.json`

```json
{
  "concept": {
    "id": "org.engagement_score",
    "version": "1.0",
    "namespace": "org",
    "description": "Overall engagement score combining activity and feature adoption",
    "output_type": "float",
    "features": [
      { "name": "user.daily_active_minutes",   "type": "time_series<float>" },
      { "name": "user.feature_adoption_score", "type": "float" }
    ],
    "compute": {
      "op": "weighted_mean",
      "inputs": ["user.daily_active_minutes", "user.feature_adoption_score"],
      "params": { "weights": [0.5, 0.5] }
    }
  },
  "condition": {
    "id": "org.critical_disengagement",
    "version": "1.0",
    "namespace": "org",
    "concept_id": "org.engagement_score",
    "concept_version": "1.0",
    "strategy": {
      "type": "composite",
      "params": {
        "operator": "AND",
        "operands": ["org.low_engagement", "org.high_churn_risk"]
      }
    }
  },
  "action": {
    "id": "org.notify_team",
    "version": "1.0"
  }
}
```

**Note:** The composite fixture references `org.low_engagement` and
`org.high_churn_risk` as operands. These conditions must exist in the registry
for the composite condition to compile successfully. Create them first using the
threshold and z_score fixtures, or mock the registry in tests.

### `fixtures/equals_task.json`

```json
{
  "concept": {
    "id": "org.account_plan_tier",
    "version": "1.0",
    "namespace": "org",
    "description": "Current subscription plan tier for the account",
    "output_type": "categorical",
    "features": [
      { "name": "account.plan_tier", "type": "categorical" }
    ],
    "compute": {
      "op": "identity",
      "inputs": ["account.plan_tier"],
      "params": {}
    }
  },
  "condition": {
    "id": "org.is_free_tier",
    "version": "1.0",
    "namespace": "org",
    "concept_id": "org.account_plan_tier",
    "concept_version": "1.0",
    "strategy": {
      "type": "equals",
      "params": { "value": "free", "labels": ["free", "trial"] }
    }
  },
  "action": {
    "id": "org.notify_team",
    "version": "1.0"
  }
}
```

---

## 12. Error Classes

```python
class LLMGenerationError(Exception):
    """
    Raised when LLM fails to produce a valid definition after max_retries.
    Maps to HTTP 422 semantic_error in the route handler.
    """
    def __init__(self, message: str, last_errors: list = None, last_output: dict = None):
        super().__init__(message)
        self.last_errors = last_errors or []
        self.last_output = last_output

    def to_error_response(self) -> dict:
        first_error = self.last_errors[0] if self.last_errors else None
        return {
            'error': {
                'type': 'semantic_error',
                'message': first_error.message if first_error else str(self),
                'location': first_error.location if first_error else None,
                'suggestion': first_error.suggestion if first_error else
                              'Try rephrasing your intent or adding more specifics',
            }
        }

class ParseError(Exception):
    """Raised when LLM output cannot be parsed as valid JSON or fails schema validation."""
    pass
```

---

## 13. Implementation Checklist for Session 10a

```
LLM INTEGRATION CHECKLIST:

/app/llm/
  client.py          → LLMClient ABC, AnthropicClient, OpenAIClient, OllamaClient,
                        create_llm_client(), get_llm_client()
  prompts.py         → SYSTEM_PROMPT, CONCEPT_GENERATION_PROMPT,
                        CONDITION_GENERATION_PROMPT, REFINEMENT_PROMPT,
                        build_llm_context(), build_initial_prompt(),
                        build_refinement_prompt(), format_errors_for_refinement()
  fixtures.py        → LLMFixtureClient, KEYWORD_MAP
  fixtures/
    threshold_task.json
    z_score_task.json
    composite_task.json
    equals_task.json

/app/services/task_authoring.py
  TaskAuthoringService.create_task():
    [1] Load guardrails + app_context
    [2] build_llm_context() in strict order
    [3] get_llm_client(config) — fixture or real
    [4] generate_with_refinement() — concept + condition
    [5] resolve_action() — deterministic, no LLM
    [6] compiler.validate() all three definitions
    [7] register all three in DefinitionStore
    [8] compiler.compile() → ExecutionGraph
    [9] task_store.create() → Task
    dry_run: return DryRunResult, skip steps 7–9

INVARIANTS:
  - LLM is NEVER called if USE_LLM_FIXTURES=True
  - LLM is NEVER called outside TaskAuthoringService
  - strategy.type and strategy.params are validated before compiler
  - On LLMGenerationError: no partial state, no task_id assigned
  - temperature=0 always — configured in LLMConfig, enforced in client
```
