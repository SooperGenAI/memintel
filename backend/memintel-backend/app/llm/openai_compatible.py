"""
app/llm/openai_compatible.py
──────────────────────────────────────────────────────────────────────────────
OpenAICompatibleClient — LLM client for any server that exposes the OpenAI
chat completions API.

Supported deployments:
  - Ollama (on-premise, e.g. http://localhost:11434)
  - vLLM   (GPU inference server, e.g. http://gpu-server:8000)
  - LM Studio (developer desktop, e.g. http://localhost:1234)
  - Azure OpenAI (cloud, e.g. https://<resource>.openai.azure.com)
  - Any OpenAI-compatible endpoint

This is the primary on-premise deployment path — banks and enterprises running
Llama, Mistral, or any open model on their own GPU servers use this client.

API contract:
  POST {base_url}/chat/completions
  Body: {"model": ..., "messages": [...], "temperature": 0}
  Response: {"choices": [{"message": {"content": "<json>"}}]}

Authentication:
  If api_key is set and non-empty: Authorization: Bearer {api_key}
  Otherwise: no Authorization header (internal deployments need no auth)

Configuration fields used from LLMConfig:
  base_url        — required; target server base URL
  api_key         — optional; Bearer token if needed
  model           — model name to send in the request
  ssl_verify      — set False for self-signed certificates
  timeout_seconds — HTTP timeout; increase for slower on-premise GPUs

temperature=0 is intentional — deterministic compilation output only.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
import structlog
from pydantic import ValidationError as PydanticValidationError

from app.llm.base import LLMClientBase
from app.llm.client import LLMError, _build_system_prompt, _COMPILE_STEP_INSTRUCTIONS

log = structlog.get_logger(__name__)

# Strip markdown code fences — same logic as AnthropicClient.
def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if present."""
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    inner: list[str] = []
    in_block = False
    for line in lines:
        if line.startswith("```") and not in_block:
            in_block = True
            continue
        if line.startswith("```") and in_block:
            break
        if in_block:
            inner.append(line)
    return "\n".join(inner).strip()


class OpenAICompatibleClient(LLMClientBase):
    """
    LLM client for OpenAI-compatible chat completions endpoints.

    Use this for any on-premise or cloud server that speaks the OpenAI
    chat completions protocol: Ollama, vLLM, LM Studio, Azure OpenAI, etc.

    Parameters
    ----------
    config:
        LLMConfig from memintel_config.yaml. Uses base_url, api_key,
        model, ssl_verify, and timeout_seconds.
    """

    def __init__(self, config: Any) -> None:  # config: LLMConfig
        if not config.base_url:
            raise LLMError(
                "openai_compatible provider requires base_url to be set in llm config. "
                "Example: base_url: http://localhost:11434"
            )
        self._base_url = config.base_url.rstrip("/")
        self._api_key = config.api_key or ""
        self._model = config.model
        self._ssl_verify = config.ssl_verify
        self._timeout = float(config.timeout_seconds)

    # ── Public interface ────────────────────────────────────────────────────────

    def generate_task(self, intent: str, context: dict) -> dict:
        """
        Generate a task definition via the OpenAI chat completions endpoint.

        Builds a system prompt from ``context``, sends the intent as the
        user message, and parses the JSON response.

        Raises LLMError on connection error, timeout, HTTP 4xx/5xx,
        invalid JSON, or missing required keys (concept/condition/action).
        """
        system_prompt = _build_system_prompt(context)
        user_message = f"Create a task for the following intent:\n\n{intent}"

        log.info(
            "openai_compatible_request",
            model=self._model,
            base_url=self._base_url,
            intent_len=len(intent),
        )

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0,
        }

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = f"{self._base_url}/chat/completions"

        try:
            with httpx.Client(
                timeout=self._timeout,
                verify=self._ssl_verify,
            ) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"OpenAI-compatible endpoint timed out after {self._timeout}s: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"OpenAI-compatible endpoint returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Cannot connect to OpenAI-compatible endpoint {url}: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMError(
                f"Request error calling OpenAI-compatible endpoint {url}: {exc}"
            ) from exc

        try:
            resp_body = response.json()
        except Exception as exc:
            raise LLMError(
                f"OpenAI-compatible endpoint returned non-JSON response: {exc}"
            ) from exc

        try:
            raw_text = resp_body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"Unexpected response shape from OpenAI-compatible endpoint "
                f"(missing choices[0].message.content): {exc}"
            ) from exc

        raw_text = _strip_code_fences(raw_text.strip())

        try:
            result = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"OpenAI-compatible response is not valid JSON: {exc}. "
                f"Raw response (first 500 chars): {raw_text[:500]!r}"
            ) from exc

        if not isinstance(result, dict):
            raise LLMError(
                f"OpenAI-compatible response must be a JSON object; "
                f"got {type(result).__name__}"
            )

        # Validate against LLMTaskOutput — checks concept/condition/action keys.
        from app.models.llm import LLMTaskOutput
        try:
            validated = LLMTaskOutput.model_validate(result)
        except (PydanticValidationError, ValueError) as exc:
            raise LLMError(
                f"OpenAI-compatible response does not match expected structure: {exc}. "
                f"Response must contain 'concept', 'condition', 'action' as JSON objects."
            ) from exc

        log.info("openai_compatible_response_ok", model=self._model)
        return validated.model_dump()

    def generate_compile_step(self, prompt: str, context: dict) -> dict:
        """
        Generate a single CoR compile step via the OpenAI-compatible endpoint.

        Uses _COMPILE_STEP_INSTRUCTIONS as the system prompt.

        Raises LLMError on connection error, timeout, HTTP error, invalid JSON,
        or missing 'summary' key in the response.
        """
        step = context.get("step", 0)
        context_json = json.dumps(context, indent=2)
        user_message = (
            f"Execute Step {step} of the compilation pipeline.\n\n"
            f"Step context:\n{context_json}\n\n"
            f"Prompt: {prompt}"
        )

        log.info(
            "openai_compatible_compile_step_request",
            model=self._model,
            base_url=self._base_url,
            step=step,
        )

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _COMPILE_STEP_INSTRUCTIONS},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0,
        }

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = f"{self._base_url}/chat/completions"

        try:
            with httpx.Client(
                timeout=self._timeout,
                verify=self._ssl_verify,
            ) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"OpenAI-compatible endpoint timed out at compile step {step}: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"OpenAI-compatible endpoint returned HTTP {exc.response.status_code} "
                f"at compile step {step}: {exc.response.text[:200]}"
            ) from exc
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Cannot connect to OpenAI-compatible endpoint {url}: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMError(
                f"Request error at compile step {step} calling {url}: {exc}"
            ) from exc

        try:
            resp_body = response.json()
        except Exception as exc:
            raise LLMError(
                f"OpenAI-compatible endpoint returned non-JSON at compile step {step}: {exc}"
            ) from exc

        try:
            raw_text = resp_body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"Unexpected response shape at compile step {step} "
                f"(missing choices[0].message.content): {exc}"
            ) from exc

        raw_text = _strip_code_fences(raw_text.strip())

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

        if "summary" not in result:
            raise LLMError(
                f"Compile step {step}: response missing required 'summary' key."
            )

        log.info("openai_compatible_compile_step_ok", model=self._model, step=step)
        return result
