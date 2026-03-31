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

LLM client selection follows the same pattern as TaskAuthoringService:
  USE_LLM_FIXTURES=true (default) → LLMFixtureClient  (development / test)
  USE_LLM_FIXTURES=false          → LLMFixtureClient  (real client not yet
                                    implemented; falls back to fixtures)

DB access
─────────
query()          — ILIKE search on definition_id and body->>'description'.
                   Falls back to LLM fixture when DB returns no matches.
define()         — no DB read required; LLM generates draft from description.
define_condition() — fetches the linked concept body to pass as LLM context.
                   Raises NotFoundError when concept not found.
semantic_refine() — fetches the existing definition body as the refinement
                   base. Raises NotFoundError when definition not found.
compile_workflow() — no DB read required; LLM compiles from description.
"""
from __future__ import annotations

import json
import os
from typing import Any

import asyncpg
import structlog

from app.llm.client_factory import create_llm_client
from app.models.concept import ExecutionPlan
from app.models.errors import NotFoundError

log = structlog.get_logger(__name__)


def _extract_description(body: Any) -> str:
    """Pull 'description' from a JSONB body (str or dict). Returns '' if absent."""
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except Exception:
            return ""
    if isinstance(body, dict):
        return body.get("description") or ""
    return ""


def _parse_body(raw: Any) -> dict:
    """Deserialise a JSONB value that may arrive as str or dict."""
    if isinstance(raw, str):
        return json.loads(raw)
    return raw or {}


class AgentService:
    """
    LLM-backed definition authoring and semantic search.

    query()            — natural language query over registry; returns AgentQueryResponse.
    define()           — LLM-assisted concept definition; returns AgentDefineResponse.
    define_condition() — LLM-assisted condition definition; returns AgentDefineResponse.
    semantic_refine()  — refines an existing definition; returns SemanticRefineResponse.
    compile_workflow() — compiles a multi-step workflow; returns ExecutionPlan.

    Parameters
    ──────────
    pool       — asyncpg connection pool for DB reads.
    llm_client — LLM client with generate_* methods. When None, auto-selected
                 from USE_LLM_FIXTURES env var (defaults to LLMFixtureClient).
    """

    def __init__(self, pool: asyncpg.Pool, llm_client: Any = None) -> None:
        self._pool = pool
        self._llm = llm_client if llm_client is not None else self._select_llm_client()

    @staticmethod
    def _select_llm_client() -> Any:
        """Select LLM client from USE_LLM_FIXTURES and LLM_PROVIDER env vars."""
        use_fixtures = os.environ.get("USE_LLM_FIXTURES", "true").lower() != "false"
        from app.models.config import LLMConfig
        config = LLMConfig.model_validate({
            "provider": os.environ.get("LLM_PROVIDER", "anthropic"),
            "model": os.environ.get("ANTHROPIC_MODEL") or "claude-sonnet-4-20250514",
            "api_key": os.environ.get("ANTHROPIC_API_KEY"),
            "base_url": os.environ.get("LLM_BASE_URL"),
            "ssl_verify": os.environ.get("LLM_SSL_VERIFY", "true").lower() == "true",
            "timeout_seconds": int(os.environ.get("LLM_TIMEOUT_SECONDS", "30")),
        }, context={"resolved": True})
        return create_llm_client(config, use_fixtures)

    # ── query ─────────────────────────────────────────────────────────────────

    async def query(self, req: Any) -> Any:
        """
        Natural language query over the definition registry.

        First searches the DB with an ILIKE match on definition_id and
        body->>'description'. Falls back to the LLM fixture when the DB
        returns no matches (e.g. empty registry in development).

        Returns AgentQueryResponse(results, query, total_count).
        """
        from app.api.routes.agents import AgentQueryResponse

        rows = await self._pool.fetch(
            """
            SELECT definition_id, version, definition_type, body
            FROM definitions
            WHERE ($1::text IS NULL OR definition_type = $1)
              AND (
                    definition_id ILIKE $2
                 OR (body->>'description') ILIKE $2
              )
            ORDER BY created_at DESC
            LIMIT $3
            """,
            req.definition_type,
            f"%{req.query}%",
            req.limit,
        )

        if rows:
            results = [
                {
                    "definition_id": r["definition_id"],
                    "version": r["version"],
                    "score": 1.0,
                    "summary": _extract_description(r["body"]),
                }
                for r in rows
            ]
            log.info("agent_query_db_hit", query=req.query, count=len(results))
            return AgentQueryResponse(
                results=results,
                query=req.query,
                total_count=len(results),
            )

        # DB returned nothing — use LLM fixture for a representative response.
        log.info("agent_query_fixture_fallback", query=req.query)
        output = self._llm.generate_query(req.query, {})
        return AgentQueryResponse(
            results=output.get("results", []),
            query=req.query,
            total_count=output.get("total_count", 0),
        )

    # ── define ────────────────────────────────────────────────────────────────

    async def define(self, req: Any) -> Any:
        """
        LLM-assisted concept definition.

        Calls the LLM with the natural language description and the target
        namespace. Returns a concept draft that the caller should validate via
        POST /compile/semantic before registering.

        Returns AgentDefineResponse(draft, validation_notes, requires_review).
        """
        from app.api.routes.agents import AgentDefineResponse

        context = {"namespace": req.namespace}
        output = self._llm.generate_define(req.description, context)

        log.info("agent_define", namespace=req.namespace)
        return AgentDefineResponse(
            draft=output["draft"],
            validation_notes=output.get("validation_notes", []),
            requires_review=output.get("requires_review", True),
        )

    # ── define_condition ──────────────────────────────────────────────────────

    async def define_condition(self, req: Any) -> Any:
        """
        LLM-assisted condition definition with strategy selection.

        Fetches the linked concept body from the DB and passes it to the LLM
        as context for strategy selection. Patches concept_id and
        concept_version from the request into the draft (the fixture values are
        overridden so the draft always references the caller's concept).

        Raises NotFoundError when (concept_id, concept_version) is not found.

        Returns AgentDefineResponse(draft, validation_notes, requires_review).
        """
        from app.api.routes.agents import AgentDefineResponse

        concept_row = await self._pool.fetchrow(
            """
            SELECT body FROM definitions
            WHERE definition_id = $1 AND version = $2
              AND definition_type = 'concept'
            """,
            req.concept_id,
            req.concept_version,
        )
        if concept_row is None:
            raise NotFoundError(
                f"Concept '{req.concept_id}' version '{req.concept_version}' not found.",
                location="concept_id",
                suggestion="Register the concept first via POST /registry/definitions.",
            )

        concept_body = _parse_body(concept_row["body"])
        context = {"namespace": req.namespace, "concept": concept_body}
        output = self._llm.generate_define_condition(req.description, concept_body, context)

        # Always bind the draft to the caller's concept, overriding the fixture.
        draft = {**output["draft"], "concept_id": req.concept_id, "concept_version": req.concept_version}

        log.info(
            "agent_define_condition",
            concept_id=req.concept_id,
            concept_version=req.concept_version,
        )
        return AgentDefineResponse(
            draft=draft,
            validation_notes=output.get("validation_notes", []),
            requires_review=output.get("requires_review", False),
        )

    # ── semantic_refine ───────────────────────────────────────────────────────

    async def semantic_refine(self, req: Any) -> Any:
        """
        Refine an existing definition using a natural language instruction.

        Fetches the current definition body from the DB as the refinement base.
        Raises NotFoundError when (definition_id, version) is not found.

        Returns SemanticRefineResponse(original, proposed, changes, breaking).
        breaking=True signals that the change affects the semantic_hash and a
        new version should be created rather than updating in place.
        """
        from app.api.routes.agents import SemanticRefineResponse

        row = await self._pool.fetchrow(
            """
            SELECT body, definition_type FROM definitions
            WHERE definition_id = $1 AND version = $2
            """,
            req.definition_id,
            req.version,
        )
        if row is None:
            raise NotFoundError(
                f"Definition '{req.definition_id}' version '{req.version}' not found.",
                location="definition_id",
                suggestion="Check available versions via GET /registry/definitions/{id}/versions.",
            )

        original = _parse_body(row["body"])
        output = self._llm.generate_semantic_refine(original, req.instruction, {})

        log.info(
            "agent_semantic_refine",
            definition_id=req.definition_id,
            version=req.version,
            breaking=output.get("breaking", False),
        )
        return SemanticRefineResponse(
            original=original,
            proposed=output["proposed"],
            changes=output.get("changes", []),
            breaking=output.get("breaking", False),
        )

    # ── compile_workflow ──────────────────────────────────────────────────────

    async def compile_workflow(self, req: Any) -> ExecutionPlan:
        """
        Compile a multi-step workflow description into an ExecutionPlan.

        Calls the LLM with the natural language workflow description and the
        target namespace. Returns an ExecutionPlan with execution_order and
        parallelizable_groups ready for the runtime.

        Returns ExecutionPlan.
        """
        context = {"namespace": req.namespace}
        output = self._llm.generate_workflow(req.description, context)

        log.info("agent_compile_workflow", namespace=req.namespace)
        return ExecutionPlan(**output)
