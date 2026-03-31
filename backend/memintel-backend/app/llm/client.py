"""
app/llm/client.py
────────────────────────────────────────────────────────────────────────────────
AnthropicClient — real LLM client for task authoring.

Reads provider configuration from memintel_config.yaml (via LLMConfig) and
the ANTHROPIC_API_KEY environment variable.

Interface mirrors LLMFixtureClient so TaskAuthoringService can swap clients
without branching (USE_LLM_FIXTURES=false activates this client).

generate_task(intent, context) → dict
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

Rules:
- "concept" must be a valid ConceptDefinition (concept_id, version, namespace,
  output_type, description, primitives, features, output_feature).
- "condition" must be a valid ConditionDefinition (condition_id, version,
  concept_id, concept_version, namespace, strategy).
  strategy MUST include "type" and "params".
- "action" must be a valid ActionDefinition (action_id, version, namespace,
  config, trigger).
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

    return "\n\n".join(parts)


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
