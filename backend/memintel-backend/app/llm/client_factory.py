"""
app/llm/client_factory.py
──────────────────────────────────────────────────────────────────────────────
create_llm_client() — factory that selects and constructs the correct LLM
client for the active configuration.

Usage:
    from app.llm.client_factory import create_llm_client
    client = create_llm_client(config, use_fixtures)

The factory is the single authoritative point where provider → client class
resolution happens. Services call this function; they never instantiate clients
directly (except in tests, where a specific client can be injected).
"""
from __future__ import annotations

from app.llm.base import LLMClientBase
from app.llm.client import LLMError
from app.models.config import LLMConfig


def create_llm_client(config: LLMConfig, use_fixtures: bool) -> LLMClientBase:
    """
    Construct and return the appropriate LLM client.

    Parameters
    ----------
    config:
        LLMConfig from the active memintel_config.yaml (or built from env vars).
    use_fixtures:
        When True, return LLMFixtureClient regardless of provider.
        This is the default for development and testing (USE_LLM_FIXTURES=true).

    Returns
    -------
    LLMClientBase
        A concrete client implementing generate_task().

    Raises
    ------
    LLMError
        When provider is unknown or unsupported.
    NotImplementedError
        For reserved providers (bedrock, vertex) not yet implemented.
    """
    if use_fixtures:
        from app.llm.fixtures import LLMFixtureClient
        return LLMFixtureClient()

    provider = config.provider

    if provider == "anthropic":
        from app.llm.client import AnthropicClient
        return AnthropicClient(config=config)

    if provider == "openai_compatible":
        from app.llm.openai_compatible import OpenAICompatibleClient
        return OpenAICompatibleClient(config)

    if provider == "bedrock":
        raise NotImplementedError(
            "Bedrock provider not yet implemented. "
            "Use openai_compatible with an OpenAI-compatible Bedrock endpoint "
            "if available."
        )

    if provider == "vertex":
        raise NotImplementedError(
            "Vertex AI provider not yet implemented. "
            "Use openai_compatible with a Vertex AI OpenAI-compatible endpoint "
            "if available."
        )

    raise LLMError(
        f"Unknown LLM provider: {config.provider!r}. "
        f"Supported: anthropic, openai_compatible"
    )
