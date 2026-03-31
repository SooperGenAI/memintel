"""
tests/unit/test_llm_client_factory.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for the multi-provider LLM client architecture.

Coverage:
  1.  create_llm_client returns LLMFixtureClient when use_fixtures=True
      regardless of provider
  2.  create_llm_client returns AnthropicClient for provider=anthropic
  3.  create_llm_client returns OpenAICompatibleClient for
      provider=openai_compatible
  4.  create_llm_client raises LLMError for unknown provider
  5.  OpenAICompatibleClient: successful response parsed and validated
      against LLMTaskOutput correctly
  6.  OpenAICompatibleClient: omits Authorization header when api_key is
      absent or empty string
  7.  OpenAICompatibleClient: connection error raises LLMError
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.llm.base import LLMClientBase
from app.llm.client import LLMError
from app.llm.client_factory import create_llm_client
from app.llm.fixtures import LLMFixtureClient
from app.llm.openai_compatible import OpenAICompatibleClient
from app.models.config import LLMConfig


# ── Helpers ────────────────────────────────────────────────────────────────────

def _config(
    provider: str = "anthropic",
    model: str = "test-model",
    api_key: str | None = None,
    base_url: str | None = None,
    ssl_verify: bool = True,
    timeout_seconds: int = 30,
) -> LLMConfig:
    """Build an LLMConfig from resolved values (bypasses ${ENV_VAR} check)."""
    return LLMConfig.model_validate(
        {
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "ssl_verify": ssl_verify,
            "timeout_seconds": timeout_seconds,
        },
        context={"resolved": True},
    )


_VALID_LLM_RESPONSE = {
    "concept": {"concept_id": "org.test", "version": "1.0"},
    "condition": {"condition_id": "org.cond", "version": "1.0"},
    "action": {"action_id": "org.act", "version": "1.0"},
}


def _make_openai_http_mock(response_body: dict, status_code: int = 200) -> MagicMock:
    """Return a mock httpx.Client context manager that yields a fake response."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = response_body
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post = MagicMock(return_value=mock_response)

    return mock_client


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_factory_returns_fixture_client_when_use_fixtures_true():
    """use_fixtures=True always returns LLMFixtureClient, regardless of provider."""
    for provider in ("anthropic", "openai_compatible"):
        cfg = _config(provider=provider, base_url="http://localhost:11434")
        client = create_llm_client(cfg, use_fixtures=True)
        assert isinstance(client, LLMFixtureClient)
        assert isinstance(client, LLMClientBase)


def test_factory_returns_anthropic_client_for_anthropic_provider(monkeypatch):
    """provider=anthropic returns AnthropicClient when use_fixtures=False."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

    from app.llm.client import AnthropicClient

    cfg = _config(provider="anthropic", api_key="sk-test-key")

    with patch("app.llm.client.AnthropicClient.__init__", return_value=None):
        client = create_llm_client(cfg, use_fixtures=False)

    assert isinstance(client, AnthropicClient)
    assert isinstance(client, LLMClientBase)


def test_factory_returns_openai_compatible_client(monkeypatch):
    """provider=openai_compatible returns OpenAICompatibleClient."""
    cfg = _config(
        provider="openai_compatible",
        base_url="http://localhost:11434",
        model="llama3.2",
    )
    client = create_llm_client(cfg, use_fixtures=False)

    assert isinstance(client, OpenAICompatibleClient)
    assert isinstance(client, LLMClientBase)


def test_factory_raises_llm_error_for_unknown_provider():
    """An unknown provider string raises LLMError with descriptive message."""
    cfg = LLMConfig.model_validate(
        {"provider": "ollama", "model": "llama3"},
        context={"resolved": True},
    )

    with pytest.raises(LLMError, match="Unknown LLM provider"):
        create_llm_client(cfg, use_fixtures=False)


def test_openai_compatible_parses_and_validates_response():
    """Successful response is parsed from choices[0].message.content and validated."""
    cfg = _config(
        provider="openai_compatible",
        base_url="http://gpu-server:8000",
        model="mistral-7b",
    )

    openai_response = {
        "choices": [
            {"message": {"content": json.dumps(_VALID_LLM_RESPONSE)}}
        ]
    }
    mock_client = _make_openai_http_mock(openai_response)

    with patch("app.llm.openai_compatible.httpx.Client", return_value=mock_client):
        client = OpenAICompatibleClient(cfg)
        result = client.generate_task("flag high churn risk", {})

    assert set(result.keys()) >= {"concept", "condition", "action"}
    assert isinstance(result["concept"], dict)
    assert isinstance(result["condition"], dict)
    assert isinstance(result["action"], dict)


def test_openai_compatible_omits_auth_header_when_no_api_key():
    """When api_key is absent or empty, no Authorization header is sent."""
    for api_key in (None, ""):
        cfg = _config(
            provider="openai_compatible",
            base_url="http://internal-llm:8080",
            model="llama3",
            api_key=api_key,
        )

        openai_response = {
            "choices": [
                {"message": {"content": json.dumps(_VALID_LLM_RESPONSE)}}
            ]
        }
        mock_client = _make_openai_http_mock(openai_response)

        with patch("app.llm.openai_compatible.httpx.Client", return_value=mock_client):
            client = OpenAICompatibleClient(cfg)
            client.generate_task("test intent", {})

        call_kwargs = mock_client.post.call_args.kwargs
        headers_sent = call_kwargs.get("headers", {})
        assert "Authorization" not in headers_sent, (
            f"Authorization header must be omitted when api_key={api_key!r}"
        )


def test_openai_compatible_connection_error_raises_llm_error():
    """A connection failure from httpx raises LLMError (non-fatal contract)."""
    import httpx

    cfg = _config(
        provider="openai_compatible",
        base_url="http://unreachable-server:8000",
        model="llama3",
    )

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.side_effect = httpx.ConnectError("Connection refused")

    with patch("app.llm.openai_compatible.httpx.Client", return_value=mock_client):
        client = OpenAICompatibleClient(cfg)
        with pytest.raises(LLMError, match="Cannot connect"):
            client.generate_task("test intent", {})
