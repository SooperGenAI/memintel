"""
app/llm/base.py
──────────────────────────────────────────────────────────────────────────────
LLMClientBase — abstract base class for all LLM client implementations.

All LLM clients (AnthropicClient, LLMFixtureClient, OpenAICompatibleClient)
inherit from this class and implement generate_task().

The base class enforces a single contract:
  generate_task(intent, context) -> dict

Both the task authoring service and the agent service depend on this interface,
not on any concrete client class. Swap the implementation via create_llm_client()
in app/llm/client_factory.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClientBase(ABC):
    """
    Abstract base class for all LLM clients.

    All concrete clients must implement generate_task().  Additional methods
    (generate_query, generate_define, refine_task, …) may be added by subclasses
    but are not required by the base contract.
    """

    @abstractmethod
    def generate_task(self, intent: str, context: dict) -> dict:
        """
        Generate a task definition (concept + condition + action) from a
        natural language intent and a prompt context dict.

        Parameters
        ----------
        intent:
            Natural language description of the task to create.
        context:
            Prompt context dict built by TaskAuthoringService._build_context().
            Keys: context_prefix, type_system, guardrails, application_context,
            parameter_bias_rules, primitives, request_constraints.

        Returns
        -------
        dict
            Parsed response with top-level keys: concept, condition, action.
            Each value is a dict representing the respective definition.

        Raises
        ------
        LLMError (from app.llm.client)
            On delivery failure, invalid JSON response, or missing required keys.
        """
        ...
