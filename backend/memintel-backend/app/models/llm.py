"""
app/models/llm.py
──────────────────────────────────────────────────────────────────────────────
LLM integration models, error types, and prompt constants.

Covers:
  - Prompt constants (SYSTEM_PROMPT, CONCEPT_GENERATION_PROMPT,
    CONDITION_GENERATION_PROMPT, REFINEMENT_PROMPT) and MAX_CONTEXT_CHARS
  - LLM output parsing models (LLMFeatureRef, LLMComputeSpec,
    LLMConceptOutput, LLMConditionOutput, LLMActionOutput, LLMTaskOutput)
  - Exception hierarchy (ParseError, LLMCallError, LLMAuthError,
    LLMGenerationError)
  - Agent endpoint request/response models (AgentQueryRequest/Response,
    AgentDefineRequest/Response, SemanticRefineRequest/Response)

Design notes
────────────
LLM scope: invoked ONLY at POST /tasks (TaskAuthoringService) and
  POST /agents/* endpoints. NEVER on the execution path (POST /evaluate/*,
  POST /execute, POST /conditions/calibrate, etc.).

LLM output is parsed JSON. parse_llm_output() validates the raw string
  against LLMTaskOutput before passing it to the compiler. Two validation
  layers: structural (Pydantic) → semantic (compiler). ParseError is raised
  for invalid JSON or structure mismatches; compiler errors go through the
  refinement loop.

Refinement loop: up to max_retries attempts. Each retry includes compiler
  errors in the prompt. LLMGenerationError is raised after exhaustion →
  HTTP 422. At-most-once semantics: never persist partial definitions.

LLMCallError.failure_type is one of: 'timeout' | 'empty_response' |
  'parse_error' | 'provider_error'. LLMAuthError is non-retriable — always
  re-raised immediately without consuming a retry slot.

AgentQueryRequest / AgentDefineRequest are the wire shapes for the
  POST /agents/* endpoints. SemanticRefineRequest and SemanticRefineResponse
  use SemanticGraph from concept.py (imported here; no circular dependency
  because concept.py does not import from llm.py).
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.models.concept import SemanticGraph


# ── Prompt constants ───────────────────────────────────────────────────────────

#: Maximum character budget for the assembled LLM context string.
#: Tune per model and deployment. Reduction logic in build_llm_context_with_budget
#: trims progressively lower-priority sections when this limit is exceeded.
MAX_CONTEXT_CHARS: int = 80_000

SYSTEM_PROMPT: str = """
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

CONCEPT_GENERATION_PROMPT: str = """
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

CONDITION_GENERATION_PROMPT: str = """
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

REFINEMENT_PROMPT: str = """
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


# ── LLM output parsing models ─────────────────────────────────────────────────

class LLMFeatureRef(BaseModel):
    """
    A single feature (primitive) reference in an LLM-generated concept.

    name — the primitive name (must match a registered primitive in the registry).
    type — the primitive's declared type string (e.g. 'float', 'time_series<float>').
    """
    name: str
    type: str


class LLMComputeSpec(BaseModel):
    """
    The compute specification in an LLM-generated concept.

    op     — operator name (e.g. 'weighted_mean', 'sum', 'latest').
    inputs — list of primitive names. Must be a strict subset of the concept's
             declared features[].name. The parser enforces this before compilation.
    params — operator parameters (dict keyed by parameter name). Structure is
             operator-specific; the compiler validates the exact schema.
    """
    op: str
    inputs: list[str]
    params: dict[str, Any] = Field(default_factory=dict)


class LLMConceptOutput(BaseModel):
    """
    Structured LLM output for the concept part of a task definition.

    Produced by parse_llm_output() after validating the raw JSON string.
    Passed to the compiler after namespace and concept-consistency checks.

    output_type must be a scalar Memintel type: float | int | boolean | categorical.
    decision<boolean> and decision<categorical> are NOT valid output types here.

    Invariant enforced by parse_llm_output():
      compute.inputs ⊆ {f.name for f in features}
    """
    id: str
    version: str
    namespace: str
    description: str
    output_type: str
    features: list[LLMFeatureRef]
    compute: LLMComputeSpec


class LLMConditionOutput(BaseModel):
    """
    Structured LLM output for the condition part of a task definition.

    strategy is kept as a raw dict here; the compiler performs deep schema
    validation against the declared strategy registry. The parser only verifies
    that strategy.type and strategy.params keys are present.

    Invariants enforced by parse_llm_output():
      condition.concept_id == concept.id
      condition.concept_version == concept.version
      condition.id starts with condition.namespace + '.'
    """
    id: str
    version: str
    namespace: str
    concept_id: str
    concept_version: str
    strategy: dict[str, Any]


class LLMActionOutput(BaseModel):
    """
    Structured LLM output for the action part of a task definition.

    The LLM selects an existing registered action — it never invents action IDs.
    Action resolution is deterministic (resolve_action() in TaskAuthoringService);
    the LLM only confirms the id and version from the resolution result.
    """
    id: str
    version: str


class LLMTaskOutput(BaseModel):
    """
    The complete structured output of a single LLM generation call.

    Returned by parse_llm_output() after structural validation. Passed to
    the compiler for semantic validation (strategy params, primitive refs,
    bounds). On compiler errors, fed back into the refinement loop.
    """
    concept: LLMConceptOutput
    condition: LLMConditionOutput
    action: LLMActionOutput


# ── Exception hierarchy ────────────────────────────────────────────────────────

class ParseError(Exception):
    """
    Raised when raw LLM output cannot be parsed or structurally validated.

    Covers:
      - Output is not valid JSON (after markdown fence stripping)
      - Output does not match LLMTaskOutput structure
      - Namespace prefix mismatch (concept.id vs concept.namespace)
      - concept_id / concept_version consistency violation
      - compute.inputs references undeclared feature names

    ParseError counts as a call failure for the refinement loop. The loop
    treats it the same as LLMCallError('parse_error') for retry counting.
    """


class LLMCallError(Exception):
    """
    Raised when an individual LLM API call fails at the transport/protocol level.

    failure_type is one of:
      'timeout'        — asyncio.wait_for exceeded timeout_seconds
      'empty_response' — provider returned a blank or whitespace-only response
      'parse_error'    — response text is not valid JSON (same as ParseError path)
      'provider_error' — provider returned a non-auth exception (5xx, network error)

    All failure_type values are retriable. The refinement loop counts each
    LLMCallError against max_retries. LLMAuthError is the only non-retriable
    exception — it must be re-raised immediately without consuming a retry slot.
    """
    def __init__(self, failure_type: str, message: str = ""):
        super().__init__(message)
        self.failure_type = failure_type


class LLMAuthError(Exception):
    """
    Raised on provider authentication failure (HTTP 401 / 403).

    Non-retriable. The refinement loop re-raises this immediately without
    consuming a retry slot. Callers should surface this as a configuration
    error rather than a transient failure.
    """


class LLMGenerationError(Exception):
    """
    Raised when the refinement loop is exhausted without producing a valid output.

    Carries the final compiler errors and the last LLM output for diagnostics.
    Route handlers convert this to HTTP 422 with error.type='semantic_error'.

    The system must remain clean on LLMGenerationError — no partial definitions
    are persisted and no task_id is assigned.

    last_errors — list of compiler ValidationErrorItem dicts from the last attempt.
    last_output — the last raw dict from parse_llm_output() (may be None if
                  every attempt ended in a ParseError before reaching the compiler).
    """
    def __init__(
        self,
        message: str,
        last_errors: list[Any] | None = None,
        last_output: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.last_errors: list[Any] = last_errors or []
        self.last_output: dict[str, Any] | None = last_output


# ── Agent endpoint request / response models ───────────────────────────────────

class AgentQueryRequest(BaseModel):
    """
    Request body for POST /agents/query.

    Used for free-form natural language queries about the platform state,
    concept definitions, condition strategies, or registry contents. The
    agent answers using the current platform context without generating
    a full task definition.

    context_ids may reference concept/condition/primitive IDs to scope
    the agent's answer to a specific part of the registry.
    """
    query: str
    namespace: str | None = None
    context_ids: list[str] = Field(default_factory=list)


class AgentQueryResponse(BaseModel):
    """
    Response from POST /agents/query.

    answer is the agent's natural language response to the query.
    references lists any concept/condition/primitive IDs cited in the answer.
    """
    answer: str
    references: list[str] = Field(default_factory=list)


class AgentDefineRequest(BaseModel):
    """
    Request body for POST /agents/define and POST /agents/define-condition.

    For /agents/define: the agent generates a full task definition (concept +
    condition + action) from the natural language intent.

    For /agents/define-condition: the agent generates only a condition for a
    pre-existing concept. concept_id and concept_version must be provided.

    dry_run=True validates the definition without persisting it.
    """
    intent: str
    namespace: str
    entity_scope: str | None = None
    concept_id: str | None = None         # required for /agents/define-condition
    concept_version: str | None = None    # required for /agents/define-condition
    dry_run: bool = False


class AgentDefineResponse(BaseModel):
    """
    Response from POST /agents/define and POST /agents/define-condition.

    task_id is populated when dry_run=False and the definition was persisted.
    concept, condition, action contain the generated definition objects.
    validation carries compiler warnings (never errors — errors raise HTTP 422).
    """
    task_id: str | None = None
    concept: dict[str, Any]
    condition: dict[str, Any]
    action: dict[str, Any]
    validation: dict[str, Any] = Field(default_factory=dict)


class SemanticRefineRequest(BaseModel):
    """
    Request body for POST /agents/semantic-refine.

    Refines an existing concept definition using a natural language instruction
    applied to the semantic graph view. The agent produces a revised concept
    that preserves semantic intent while incorporating the instruction.

    concept_id + concept_version identify the source definition.
    semantic_view is the SemanticGraph produced by POST /compile/semantic for
    the source concept. Pass it to avoid a redundant compile call.
    instruction is the natural language refinement directive.
    """
    concept_id: str
    concept_version: str
    semantic_view: SemanticGraph
    instruction: str


class SemanticRefineResponse(BaseModel):
    """
    Response from POST /agents/semantic-refine.

    revised_concept is the updated concept definition (same shape as a
    ConceptDefinition body — validated against the compiler before returning).
    revised_semantic_view is the SemanticGraph for the revised concept.
    changes is a human-readable summary of what was changed and why.
    """
    revised_concept: dict[str, Any]
    revised_semantic_view: SemanticGraph
    changes: str
