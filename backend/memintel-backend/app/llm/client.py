"""
app/llm/client.py
────────────────────────────────────────────────────────────────────────────────
AnthropicClient — real LLM client for task authoring.

Reads provider configuration from memintel_config.yaml (via LLMConfig) and
the ANTHROPIC_API_KEY environment variable.

Interface mirrors LLMFixtureClient so TaskAuthoringService can swap clients
without branching (USE_LLM_FIXTURES=false activates this client).

generate_task(intent, context) -> dict
  Builds a system prompt from the context dict (type_system, guardrails,
  application context, etc.), adds the user intent, calls the Anthropic
  Messages API, and parses the JSON response.

  Raises LLMError on:
    - Anthropic API errors (auth, rate-limit, network)
    - Response text that cannot be parsed as JSON
    - Parsed JSON missing required top-level keys (concept/condition/action)

Context dict injection order (mirrors _build_context in task_authoring.py):
  context_prefix      — plain-text APPLICATION CONTEXT block (prepended first)
  type_system         — scalar/container types and strategy param specs
  guardrails          — strategy_registry, constraints, thresholds
  application_context — guardrails-level app context
  parameter_bias_rules
  primitives
  request_constraints
"""
from __future__ import annotations

import json
import os
from typing import Any

import structlog
from pydantic import ValidationError as PydanticValidationError

from app.llm.base import LLMClientBase

log = structlog.get_logger(__name__)


class LLMError(Exception):
    """
    Raised by AnthropicClient when the API call fails or the response
    cannot be parsed into the expected shape.
    """
    pass


# ── System prompt builder ──────────────────────────────────────────────────────

_TASK_INSTRUCTIONS = """\
You are a task authoring assistant for the Memintel platform.

Given an intent description and context, produce a JSON object with exactly
three top-level keys: "concept", "condition", and "action".

=== FIELD SCHEMAS — READ CAREFULLY BEFORE GENERATING ===

"concept" fields:
  - concept_id: string
  - version: string (e.g. "1.0")
  - namespace: MUST be exactly one of: "personal", "team", "org", "global".
      "personal" = per-user/individual metric.
      "org" = organisation-wide metric.
      NEVER output "default" or any other value.
  - output_type: one of: "float", "int", "boolean", "string", "categorical",
      "time_series<float>", "time_series<int>"
  - description: string
  - primitives: MUST be a JSON OBJECT (dict), NOT a list.
      Keys = primitive names. Values = objects with:
        "type": the Memintel type string (e.g. "float", "int")
        "missing_data_policy": one of "null", "zero", "forward_fill",
          "backward_fill" (optional, defaults to "null")
      CRITICAL: For FLOAT or INT primitives, ALWAYS use missing_data_policy "zero".
        Using "null" makes the type nullable (float?) which causes a type error when
        combined with operators like normalize, z_score_op, or percentile_op that
        expect non-nullable float. NEVER use "null" for float or int primitives.
      Example:
        "primitives": {
          "user.feature_adoption_score": {"type": "float", "missing_data_policy": "zero"}
        }
  - features: MUST be a JSON OBJECT (dict), NOT a list. MUST have at least one entry.
      Keys = feature node names. Values = FeatureNode objects with:
        "op": operator name — use one of the operators from the type system.
          For FLOAT primitives: ALWAYS use "normalize". NEVER use "passthrough".
          For TIME_SERIES<FLOAT> primitives: use "mean", "sum", "min", "max".
          For CATEGORICAL/STRING primitives: use "passthrough".
          passthrough is ONLY valid for categorical and string types.
        "inputs": dict mapping slot names to primitive or feature names
        "params": dict of operator params (use {} if none)
      Example for a float primitive exposed directly:
        "features": {
          "score": {
            "op": "normalize",
            "inputs": {"input": "user.feature_adoption_score"},
            "params": {}
          }
        }
  - output_feature: MUST be exactly one of the keys present in "features".
      Copy the key name EXACTLY as written — do not paraphrase or shorten it.
      In the example above, features has key "score", so output_feature MUST be "score".

"condition" fields:
  - condition_id, version, concept_id, concept_version: strings
  - namespace: MUST be exactly one of: "personal", "team", "org", "global"
  - strategy: object with "type" (from the type system) and "params" (dict)

"action" fields:
  - action_id, version: strings
  - namespace: MUST be exactly one of: "personal", "team", "org", "global"
  - config: MUST be one of these exact shapes (discriminated by "type"):
      Webhook:      {"type": "webhook", "endpoint": "https://..."}
      Notification: {"type": "notification", "channel": "default-channel"}
      Workflow:     {"type": "workflow", "workflow_id": "..."}
    Default to webhook type. "channel" is required for notification type.
  - trigger: MUST have these three fields:
      {"fire_on": "true", "condition_id": "...", "condition_version": "1.0"}
    fire_on MUST be one of: "true", "false", "any" (lowercase strings, NOT booleans).

Rules:
- Use ONLY the strategies listed in the type system.
- Respond with ONLY the raw JSON object — no markdown fences, no explanation.
"""


def _build_system_prompt(context: dict[str, Any]) -> str:
    """
    Build the full LLM system prompt from the context dict.

    Injection order matches _build_context() in task_authoring.py:
      [0] context_prefix  (APPLICATION CONTEXT plain-text block)
      [1] type_system
      [2] guardrails
      [3] application_context
      [4] parameter_bias_rules
      [5] primitives
      [+] request_constraints
    """
    parts: list[str] = []

    # [0] Application context prefix — prepended verbatim before all instructions.
    if context.get("context_prefix"):
        parts.append(context["context_prefix"])

    parts.append(_TASK_INSTRUCTIONS)

    # [1] Type system
    if context.get("type_system"):
        parts.append("=== TYPE SYSTEM ===")
        parts.append(json.dumps(context["type_system"], indent=2))
        parts.append("=== END TYPE SYSTEM ===")

    # [2] Guardrails
    if context.get("guardrails"):
        parts.append("=== GUARDRAILS ===")
        parts.append(json.dumps(context["guardrails"], indent=2))
        parts.append("=== END GUARDRAILS ===")

    # [3] Application context (guardrails-level)
    if context.get("application_context"):
        parts.append("=== APPLICATION CONTEXT (GUARDRAILS) ===")
        parts.append(json.dumps(context["application_context"], indent=2))
        parts.append("=== END APPLICATION CONTEXT (GUARDRAILS) ===")

    # [4] Parameter bias rules
    if context.get("parameter_bias_rules"):
        parts.append("=== PARAMETER BIAS RULES ===")
        parts.append(json.dumps(context["parameter_bias_rules"], indent=2))
        parts.append("=== END PARAMETER BIAS RULES ===")

    # [5] Primitives
    if context.get("primitives"):
        parts.append("=== PRIMITIVES ===")
        parts.append(json.dumps(context["primitives"], indent=2))
        parts.append("=== END PRIMITIVES ===")

    # Request constraints
    if context.get("request_constraints"):
        parts.append("=== REQUEST CONSTRAINTS ===")
        parts.append(json.dumps(context["request_constraints"], indent=2))
        parts.append("=== END REQUEST CONSTRAINTS ===")

    # Vocabulary context — MUST come last so the LLM sees it immediately before
    # generating its response. When present, concept_id and condition_id in the
    # output MUST be taken verbatim from these lists; do NOT invent new identifiers.
    if context.get("vocabulary_context"):
        vc = context["vocabulary_context"]
        parts.append("=== VOCABULARY CONTEXT ===")
        parts.append(json.dumps(vc, indent=2))
        parts.append(
            "IMPORTANT: The concept_id you output MUST be one of the identifiers "
            "listed in available_concept_ids above. Do NOT invent a new identifier. "
            "Choose the entry that best matches the intent."
        )
        parts.append("=== END VOCABULARY CONTEXT ===")

    return "\n\n".join(parts)


# ── Compile step system prompt ─────────────────────────────────────────────────

_COMPILE_STEP_INSTRUCTIONS = """\
You are a concept compilation assistant for the Memintel platform.

You are executing one step of a 4-step Chain of Reasoning (CoR) that compiles
a concept description into a formula specification.

Every response MUST be a JSON object containing at minimum:
  "summary"  — a concise plain-English sentence describing what this step concluded.
  "outcome"  — exactly "accepted" or "failed". Use "failed" only if the step
               cannot logically proceed (e.g. no valid signal found, type incompatible).
               Default to "accepted" when the step completes normally.

The remaining fields depend on which step is being executed:

=== STEP 1 — Intent Parsing ===
Parse the concept identifier and description to extract structured intent.
Return:
{
  "summary":      "One sentence describing the concept intent",
  "outcome":      "accepted",
  "intent_label": "Short human-readable label (e.g. 'Loan Repayment Ratio')",
  "metric_type":  "What kind of metric this is (e.g. 'ratio', 'count', 'score', 'flag')",
  "measurement":  "What quantity is being measured"
}

=== STEP 2 — Signal Identification ===
Analyse each signal name to understand what it represents and why it is relevant
to the concept being compiled.

Your response MUST be a JSON object containing ALL FIVE of these keys:
  step_index, label, summary, outcome, signal_rationale

signal_rationale is NOT optional. Omitting it is an error.
signal_rationale MUST be a JSON object (dict) keyed by signal name — one entry
per signal_name in the request context. Each value is one sentence.

Example response (copy this structure exactly):
{
  "step_index": 2,
  "label": "Signal Identification",
  "summary": "Selected 4 signals covering payment failures and delinquency duration.",
  "outcome": "accepted",
  "signal_rationale": {
    "emi_bounce_count": "measures frequency of EMI payment failures",
    "missed_payment_count": "tracks cumulative missed payment obligations",
    "days_past_due": "captures recency and duration of delinquency",
    "monthly_outflow": "indicates overall cash flow pressure"
  }
}

Do NOT use a list or array for signal_rationale. Do NOT omit signal_rationale.

=== STEP 3 — DAG Construction ===
Select the formula strategy and produce the complete formula specification with
explicit numerical weights for every signal.

Your response MUST be valid JSON with EXACTLY these keys at the top level:
  summary, outcome, formula_summary, output_range, signal_bindings

signal_bindings MUST be an array where every entry has EXACTLY these four keys:
  signal_name, role, weight, rationale

role MUST be one of exactly these three values:
  "numerator"            — signal adds directly to the score
  "denominator"          — signal normalises or scales the score
  "severity_multiplier"  — signal amplifies other signals

weight MUST be a number between 0.0 and 1.0.
All weights MUST sum to exactly 1.0.
Do NOT omit weight. Do NOT use strings for weight.

rationale MUST be a one-sentence plain English explanation of why this signal
has this role and weight.

Example of a correct response:
{
  "summary": "Selected weighted sum formula with four signals, weights summing to 1.0.",
  "outcome": "accepted",
  "formula_summary": "Weighted sum: emi_bounce_count (35%) + missed_payment_count (30%), amplified by days_past_due severity (25%), normalised by monthly_outflow (10%), clamped to 0-1 range",
  "output_range": "0.0 to 1.0",
  "signal_bindings": [
    {
      "signal_name": "emi_bounce_count",
      "role": "numerator",
      "weight": 0.35,
      "rationale": "Primary indicator of payment failure frequency"
    },
    {
      "signal_name": "missed_payment_count",
      "role": "numerator",
      "weight": 0.30,
      "rationale": "Cumulative measure of payment discipline breakdown"
    },
    {
      "signal_name": "days_past_due",
      "role": "severity_multiplier",
      "weight": 0.25,
      "rationale": "Amplifies score when delinquency is prolonged"
    },
    {
      "signal_name": "monthly_outflow",
      "role": "denominator",
      "weight": 0.10,
      "rationale": "Normalisation context — scales score by cash pressure"
    }
  ]
}

formula_summary MUST explicitly state the percentage weight of each signal inline.

If the concept description or feedback contains explicit percentage weights for signals
(e.g. "signal_x: 40%"), you MUST use those exact weights. Do not approximate or
redistribute. The weights you assign MUST match any explicit instructions in the
description exactly.

Do NOT produce any other JSON structure. Do NOT omit any of the four keys from
any signal_bindings entry.

=== STEP 4 — Type Validation ===
Confirm that the declared output_type is compatible with the formula produced in Step 3.
Return:
{
  "summary":    "One sentence confirming or flagging the type compatibility",
  "outcome":    "accepted" or "failed",
  "compatible": true or false,
  "reason":     "Concise explanation of why the output_type is or is not compatible"
}

Rules:
- Respond with ONLY the raw JSON object — no markdown fences, no explanation text.
- Include only the fields listed for the current step plus "summary" and "outcome".
- All fields shown in the example responses above are required — do not omit any.
"""


# ── AnthropicClient ────────────────────────────────────────────────────────────

class AnthropicClient(LLMClientBase):
    """
    Real Anthropic LLM client for task authoring.

    Parameters
    ----------
    config:
        LLMConfig from memintel_config.yaml. When provided, model,
        api_key, and timeout_seconds are sourced from the config.
        Falls back to environment variables / defaults when None.
    model:
        Anthropic model ID. Ignored when config is provided.
        Defaults to ANTHROPIC_MODEL env var or "claude-sonnet-4-20250514".
    api_key:
        Anthropic API key. Ignored when config is provided.
        Defaults to ANTHROPIC_API_KEY env var.
    temperature:
        Sampling temperature. Defaults to 0 for deterministic output.
    max_tokens:
        Maximum tokens in the completion. Defaults to 4096.
    timeout:
        Request timeout in seconds. Used when config is None. Defaults to 60.
    """

    def __init__(
        self,
        config: Any = None,      # LLMConfig | None — typed as Any to avoid circular import
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0,
        max_tokens: int = 4096,
        timeout: float = 60.0,
    ) -> None:
        try:
            import anthropic as _anthropic_module
            self._anthropic = _anthropic_module
        except ImportError as exc:
            raise LLMError(
                "anthropic package is not installed. "
                "Run: pip install anthropic"
            ) from exc

        if config is not None:
            resolved_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
            self._model = config.model or os.environ.get("ANTHROPIC_MODEL") or "claude-sonnet-4-20250514"
            self._timeout = float(config.timeout_seconds)
        else:
            resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            self._model = model or os.environ.get("ANTHROPIC_MODEL") or "claude-sonnet-4-20250514"
            self._timeout = timeout

        if not resolved_key:
            raise LLMError(
                "ANTHROPIC_API_KEY environment variable is not set."
            )

        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = self._anthropic.Anthropic(api_key=resolved_key)

    # ── Public interface (mirrors LLMFixtureClient) ────────────────────────────

    def generate_task(self, intent: str, context: dict) -> dict:
        """
        Generate a task definition (concept + condition + action) from intent.

        Builds a system prompt from ``context``, sends the user intent to the
        Anthropic Messages API, and parses the JSON response.

        Parameters
        ----------
        intent:
            Natural language description of the task to create.
        context:
            LLM prompt context built by TaskAuthoringService._build_context().
            Keys: context_prefix, type_system, guardrails, application_context,
            parameter_bias_rules, primitives, request_constraints.

        Returns
        -------
        dict
            Parsed response with keys ``concept``, ``condition``, ``action``.

        Raises
        ------
        LLMError
            On Anthropic API failure, invalid JSON response, or missing keys.
        """
        system_prompt = _build_system_prompt(context)
        user_message = f"Create a task for the following intent:\n\n{intent}"

        log.info("anthropic_request", model=self._model, intent_len=len(intent))

        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
        except self._anthropic.APIError as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"Unexpected error calling Anthropic API: {exc}") from exc

        # Extract text content from the response.
        raw_text = ""
        for block in message.content:
            if hasattr(block, "text"):
                raw_text += block.text

        raw_text = raw_text.strip()

        # Strip markdown code fences if the model wrapped the JSON.
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            # Drop opening fence (```json or ```) and closing fence (```)
            inner = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                if line.startswith("```") and in_block:
                    break
                if in_block:
                    inner.append(line)
            raw_text = "\n".join(inner).strip()

        try:
            result = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"Anthropic response is not valid JSON: {exc}. "
                f"Raw response (first 500 chars): {raw_text[:500]!r}"
            ) from exc

        if not isinstance(result, dict):
            raise LLMError(
                f"Anthropic response must be a JSON object; "
                f"got {type(result).__name__}"
            )

        # Validate against LLMTaskOutput — checks concept/condition/action keys exist
        # and are dicts. Raises LLMError on structural mismatch.
        from app.models.llm import LLMTaskOutput
        try:
            validated = LLMTaskOutput.model_validate(result)
        except (PydanticValidationError, ValueError) as exc:
            raise LLMError(
                f"Anthropic response does not match expected structure: {exc}. "
                f"Response must contain 'concept', 'condition', 'action' as JSON objects."
            ) from exc

        log.info("anthropic_response_ok", model=self._model)
        return validated.model_dump()

    def generate_compile_step(self, prompt: str, context: dict) -> dict:
        """
        Generate a single CoR compile step output.

        Uses _COMPILE_STEP_INSTRUCTIONS as the system prompt. Returns a dict
        containing at minimum 'summary' and 'outcome'. Step-specific fields
        (signal_rationale, formula_summary, signal_bindings, etc.) are also
        present when the LLM produces them.

        Parameters
        ----------
        prompt:
            Natural language prompt describing what this step should do.
        context:
            Step context dict. Must include 'step' (int 1–4).

        Raises
        ------
        LLMError
            On Anthropic API failure, invalid JSON, or missing 'summary' key.
        """
        step = context.get("step", 0)
        context_json = json.dumps(context, indent=2)
        user_message = (
            f"Execute Step {step} of the compilation pipeline.\n\n"
            f"Step context:\n{context_json}\n\n"
            f"Prompt: {prompt}"
        )

        log.info("anthropic_compile_step_request", model=self._model, step=step)

        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                temperature=0,
                system=_COMPILE_STEP_INSTRUCTIONS,
                messages=[{"role": "user", "content": user_message}],
            )
        except self._anthropic.APIError as exc:
            raise LLMError(f"Anthropic API error at compile step {step}: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"Unexpected error at compile step {step}: {exc}") from exc

        raw_text = ""
        for block in message.content:
            if hasattr(block, "text"):
                raw_text += block.text

        raw_text = raw_text.strip()

        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            inner = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                if line.startswith("```") and in_block:
                    break
                if in_block:
                    inner.append(line)
            raw_text = "\n".join(inner).strip()

        try:
            result = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"Compile step {step}: response is not valid JSON: {exc}. "
                f"Raw (first 500 chars): {raw_text[:500]!r}"
            ) from exc

        if not isinstance(result, dict):
            raise LLMError(
                f"Compile step {step}: response must be a JSON object; "
                f"got {type(result).__name__}"
            )

        # Diagnostic: log raw LLM output immediately after parsing, before any
        # field extraction — shows exactly what the LLM returned for each step.
        log.info(
            "raw_step_output",
            step=step,
            keys=list(result.keys()),
            full=result,
        )

        if "summary" not in result:
            raise LLMError(
                f"Compile step {step}: response missing required 'summary' key."
            )

        log.info("anthropic_compile_step_ok", model=self._model, step=step)
        return result
