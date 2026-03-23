"""
app/services/agents.py
──────────────────────────────────────────────────────────────────────────────
AgentService — LLM-backed definition authoring and query endpoints.

Provides LLM-driven endpoints for:
  - Querying the definition registry using natural language
  - Defining concepts from a description
  - Defining conditions with strategy selection
  - Semantic refinement of existing definitions
  - Workflow compilation

These endpoints invoke the LLM for intent resolution. They are the only
agent endpoints that use the LLM — unlike TaskAuthoringService, which drives
the full task creation pipeline, AgentService handles individual definition
authoring operations.

TODO: full implementation in a future session.
"""
from __future__ import annotations

import asyncpg


class AgentService:
    """
    LLM-backed definition authoring and semantic search.

    query()            — natural language query over registry; returns AgentQueryResponse.
    define()           — LLM-assisted concept definition; returns AgentDefineResponse.
    define_condition() — LLM-assisted condition definition; returns AgentDefineResponse.
    semantic_refine()  — refines an existing definition; returns SemanticRefineResponse.
    compile_workflow() — compiles a multi-step workflow; returns ExecutionPlan.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
