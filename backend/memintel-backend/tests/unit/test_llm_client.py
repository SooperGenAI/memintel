"""
tests/unit/test_llm_client.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for AnthropicClient.

Coverage:
  1. Context dict is serialised into the system prompt sent to the API.
  2. LLMError is raised when the Anthropic API returns an error.
  3. LLMError is raised when the API returns non-JSON text.

Test isolation: all tests use a MockAnthropic that replaces the real
``anthropic.Anthropic`` client; no network calls are made.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.llm.client import AnthropicClient, LLMError, _build_system_prompt


# ── Minimal valid LLM response ─────────────────────────────────────────────────

_VALID_RESPONSE_BODY = {
    "concept": {
        "concept_id": "org.test_concept",
        "version": "1.0",
        "namespace": "org",
        "output_type": "float",
        "description": "Test concept",
        "primitives": {},
        "features": {},
        "output_feature": "test_feature",
    },
    "condition": {
        "condition_id": "org.test_condition",
        "version": "1.0",
        "concept_id": "org.test_concept",
        "concept_version": "1.0",
        "namespace": "org",
        "strategy": {"type": "threshold", "params": {"direction": "above", "value": 0.5}},
    },
    "action": {
        "action_id": "org.test_action",
        "version": "1.0",
        "namespace": "org",
        "config": {"type": "notification", "channel": "test"},
        "trigger": {
            "fire_on": "true",
            "condition_id": "org.test_condition",
            "condition_version": "1.0",
        },
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────────

import json


def _make_mock_anthropic(response_text: str) -> MagicMock:
    """
    Build a mock anthropic module that returns ``response_text`` as the
    content of the Messages API response.
    """
    mock_block = MagicMock()
    mock_block.text = response_text

    mock_message = MagicMock()
    mock_message.content = [mock_block]

    mock_messages = MagicMock()
    mock_messages.create.return_value = mock_message

    mock_sdk_client = MagicMock()
    mock_sdk_client.messages = mock_messages

    mock_anthropic_module = MagicMock()
    mock_anthropic_module.Anthropic.return_value = mock_sdk_client
    # Make APIError available on the module so the except clause can catch it.
    mock_anthropic_module.APIError = Exception

    return mock_anthropic_module


def _make_client_with_mock(
    mock_anthropic: MagicMock,
    model: str = "claude-test",
) -> AnthropicClient:
    """Construct an AnthropicClient whose _anthropic attribute is the mock module."""
    client = AnthropicClient.__new__(AnthropicClient)
    client._anthropic = mock_anthropic
    client._client = mock_anthropic.Anthropic(api_key="test-key")
    client._model = model
    client._temperature = 0
    client._max_tokens = 4096
    client._timeout = 60.0
    return client


# ── Test 1: context dict is passed through to the API call ────────────────────


def test_generate_task_passes_context_to_api():
    """
    The context dict must be serialised into the system prompt that is sent
    to the Anthropic Messages API.

    Specifically:
    - context["context_prefix"] is included verbatim at the start of the prompt.
    - context["type_system"] is JSON-serialised into the prompt.
    - context["guardrails"] is JSON-serialised into the prompt.
    """
    context = {
        "context_prefix": "=== APPLICATION CONTEXT ===\nDomain: e-commerce\n=== END APPLICATION CONTEXT ===\n",
        "type_system": {"scalar_types": ["float", "int"], "strategies": ["threshold"]},
        "guardrails": {"strategy_registry": {"threshold": {"description": "threshold strategy"}}},
    }

    mock_anthropic = _make_mock_anthropic(json.dumps(_VALID_RESPONSE_BODY))
    client = _make_client_with_mock(mock_anthropic)

    result = client.generate_task("flag high churn risk users", context)

    # Verify the Anthropic SDK was called once.
    create_call = mock_anthropic.Anthropic.return_value.messages.create
    assert create_call.call_count == 1

    call_kwargs = create_call.call_args.kwargs
    system_prompt: str = call_kwargs["system"]

    # context_prefix must appear at the start of the system prompt.
    assert "=== APPLICATION CONTEXT ===" in system_prompt
    assert "Domain: e-commerce" in system_prompt

    # type_system must be serialised into the prompt.
    assert "scalar_types" in system_prompt
    assert "threshold" in system_prompt

    # guardrails must be serialised into the prompt.
    assert "strategy_registry" in system_prompt

    # The user message must contain the intent.
    user_messages = [m for m in call_kwargs["messages"] if m["role"] == "user"]
    assert len(user_messages) == 1
    assert "high churn risk" in user_messages[0]["content"]

    # The returned dict must have the required keys.
    assert set(result.keys()) >= {"concept", "condition", "action"}


# ── Test 2: LLMError raised on API failure ────────────────────────────────────


def test_generate_task_raises_llm_error_on_api_failure():
    """
    When the Anthropic API raises an APIError (auth failure, rate limit, etc.)
    AnthropicClient must re-raise it as LLMError.
    """
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value.messages.create.side_effect = Exception(
        "API error: 401 Unauthorized"
    )
    # Make APIError the same as Exception for the isinstance check in client.
    mock_anthropic.APIError = Exception

    client = _make_client_with_mock(mock_anthropic)

    with pytest.raises(LLMError, match="401 Unauthorized"):
        client.generate_task("flag churn", {})


# ── Test 3: LLMError raised on invalid JSON response ─────────────────────────


def test_generate_task_raises_llm_error_on_invalid_json():
    """
    When the Anthropic API returns a response that cannot be parsed as JSON,
    AnthropicClient must raise LLMError with a descriptive message.
    """
    mock_anthropic = _make_mock_anthropic("Sorry, I cannot help with that.")
    client = _make_client_with_mock(mock_anthropic)

    with pytest.raises(LLMError, match="not valid JSON"):
        client.generate_task("flag churn", {})
