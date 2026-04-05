"""
tests/integration/test_v7_cross_module.py
──────────────────────────────────────────────────────────────────────────────
V7 cross-layer integration tests.

These tests verify that V7 features (two-phase compile+register, vocabulary
context, reasoning trace, SSE streaming) work correctly across module
boundaries.  Tests use the service layer directly (no HTTP stack) against a
real PostgreSQL database except where noted.

Test organisation
─────────────────
Section 1 — Model→Store     (pure Python, no DB)
Section 2 — Service→Store   (real DB via db_pool + run fixtures)
Section 3 — Service→Route   (service layer, real DB for stores)
Section 4 — Route→Route     (two-service flows, real DB)

Fixtures (from conftest.py + conftest_v7.py)
────────────────────────────────────────────
db_pool              — per-test asyncpg pool (tables truncated before each test)
run                  — runs a coroutine in the test's event loop
llm_mock             — LLMMockClient (call_count=0)
loan_compile_request — CompileConceptRequest for loan.repayment_ratio
loan_task_request    — CreateTaskRequest for overdue loan alert
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.models.concept import MAX_VOCABULARY_IDS, VocabularyContext
from app.models.concept_compile import (
    CompileConceptRequest,
    CompileToken,
    RegisterConceptRequest,
)
from app.models.errors import (
    ErrorType,
    IdentifierMismatchError,
    VocabularyMismatchError,
    http_status_for,
)
from app.models.task import CreateTaskRequest, DeliveryConfig, DeliveryType
from app.services.concept_compiler import ConceptCompilerService
from app.services.concept_registration import ConceptRegistrationService
from app.stores.compile_token import CompileTokenStore


# ── Helper ────────────────────────────────────────────────────────────────────

async def _collect_events(gen) -> list[dict]:
    """Drain an async generator, collecting all dicts."""
    events: list[dict] = []
    async for item in gen:
        events.append(item)
    return events


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Model → Store  (pure Python, no DB required)
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelToStore:
    """V7 model invariants that can be verified without a database."""

    def test_compile_token_model_roundtrip(self):
        """CompileToken fields survive model_dump → model_validate roundtrip."""
        now = datetime.now(tz=timezone.utc)
        original = CompileToken(
            token_id=str(uuid.uuid4()),
            token_string="tok_abc123xyz",
            identifier="loan.repayment_ratio",
            ir_hash="a" * 64,
            output_type="float",
            expires_at=now + timedelta(minutes=30),
            used=False,
            created_at=now,
        )
        data = original.model_dump()
        restored = CompileToken.model_validate(data)

        assert restored.token_id == original.token_id
        assert restored.token_string == original.token_string
        assert restored.identifier == original.identifier
        assert restored.ir_hash == original.ir_hash
        assert restored.output_type == original.output_type
        assert restored.used is False

    def test_v7_error_http_status_mapping(self):
        """Locked cross-team contract: EXPIRED→400, NOT_FOUND→404, CONSUMED→409."""
        assert http_status_for(ErrorType.COMPILE_TOKEN_EXPIRED)   == 400
        assert http_status_for(ErrorType.COMPILE_TOKEN_NOT_FOUND) == 404
        assert http_status_for(ErrorType.COMPILE_TOKEN_CONSUMED)  == 409

    def test_vocabulary_context_validated_at_model_layer(self):
        """VocabularyContext rejects lists exceeding MAX_VOCABULARY_IDS (500 per list)."""
        # Boundary: exactly 500 IDs is valid
        vc = VocabularyContext(
            available_concept_ids=["c"] * MAX_VOCABULARY_IDS,
            available_condition_ids=[],
        )
        assert len(vc.available_concept_ids) == MAX_VOCABULARY_IDS

        # 501 concept IDs → rejected
        with pytest.raises(PydanticValidationError):
            VocabularyContext(
                available_concept_ids=["c"] * (MAX_VOCABULARY_IDS + 1),
                available_condition_ids=[],
            )

        # 501 condition IDs → rejected
        with pytest.raises(PydanticValidationError):
            VocabularyContext(
                available_concept_ids=[],
                available_condition_ids=["d"] * (MAX_VOCABULARY_IDS + 1),
            )

        # 499 + 499 = 998 total is valid (cap is per-list, not combined)
        vc2 = VocabularyContext(
            available_concept_ids=["c"] * 499,
            available_condition_ids=["d"] * 499,
        )
        assert len(vc2.available_concept_ids)   == 499
        assert len(vc2.available_condition_ids) == 499


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Service → Store  (real DB)
# ═══════════════════════════════════════════════════════════════════════════════

class TestServiceToStore:
    """Verify that V7 service calls persist correctly to the DB stores."""

    def test_concept_compiler_persists_token_to_store(
        self, db_pool, run, llm_mock, loan_compile_request
    ):
        """compile() creates a compile_token row that is readable via the store."""
        token_store = CompileTokenStore(db_pool)
        svc = ConceptCompilerService(llm_client=llm_mock, token_store=token_store)

        resp = run(svc.compile(loan_compile_request, db_pool))

        token = run(token_store.get(resp.compile_token))
        assert token is not None
        assert token.identifier == "loan.repayment_ratio"
        assert token.used is False
        assert token.token_string == resp.compile_token

    def test_ir_hash_is_deterministic(self, db_pool, run, llm_mock):
        """
        Compiling the same (identifier, description, signal_names, output_type)
        twice produces the same ir_hash — the compiler is deterministic.

        Note: ir_hash includes the identifier in formula_data, so both calls
        must use the same identifier for the hashes to be equal.
        """
        token_store = CompileTokenStore(db_pool)

        def _req() -> CompileConceptRequest:
            return CompileConceptRequest(
                identifier="loan.determinism_test",   # same identifier both times
                description="Determinism test — same formula",
                output_type="float",
                signal_names=["payments_on_time", "payments_due"],
            )

        svc = ConceptCompilerService(llm_client=llm_mock, token_store=token_store)
        resp_a = run(svc.compile(_req(), db_pool))
        resp_b = run(svc.compile(_req(), db_pool))

        token_a = run(token_store.get(resp_a.compile_token))
        token_b = run(token_store.get(resp_b.compile_token))
        assert token_a.ir_hash == token_b.ir_hash

    def test_concept_registration_consumes_token_atomically(
        self, db_pool, run, llm_mock, loan_compile_request
    ):
        """After register(), the compile_token.used flag is True."""
        token_store = CompileTokenStore(db_pool)
        compiler = ConceptCompilerService(llm_client=llm_mock, token_store=token_store)
        resp = run(compiler.compile(loan_compile_request, db_pool))

        reg_svc = ConceptRegistrationService()
        run(reg_svc.register(
            RegisterConceptRequest(
                compile_token=resp.compile_token,
                identifier="loan.repayment_ratio",
            ),
            db_pool,
        ))

        token_after = run(token_store.get(resp.compile_token))
        assert token_after is not None
        assert token_after.used is True

    def test_ir_hash_idempotency_across_registrations(self, db_pool, run, llm_mock):
        """
        Same (identifier, ir_hash) registered from two separate compile flows
        returns HTTP 201 both times with the same concept_id.
        """
        token_store = CompileTokenStore(db_pool)
        svc = ConceptCompilerService(llm_client=llm_mock, token_store=token_store)

        def _req() -> CompileConceptRequest:
            return CompileConceptRequest(
                identifier="loan.idempotent_concept",
                description="Idempotency test — same formula both times",
                output_type="float",
                signal_names=["payments_on_time", "payments_due"],
            )

        resp1 = run(svc.compile(_req(), db_pool))
        resp2 = run(svc.compile(_req(), db_pool))

        reg_svc = ConceptRegistrationService()
        result1 = run(reg_svc.register(
            RegisterConceptRequest(
                compile_token=resp1.compile_token,
                identifier="loan.idempotent_concept",
            ),
            db_pool,
        ))
        result2 = run(reg_svc.register(
            RegisterConceptRequest(
                compile_token=resp2.compile_token,
                identifier="loan.idempotent_concept",
            ),
            db_pool,
        ))

        assert result1.concept_id == result2.concept_id


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Service → Route  (service layer, real DB for stores)
# ═══════════════════════════════════════════════════════════════════════════════

class TestServiceToRoute:
    """
    Verify that V7 features work correctly at the service boundary.

    All tests use the ConceptCompilerService directly (not the HTTP route)
    to keep the test scope focused on the feature behaviour, not HTTP plumbing.
    """

    def test_reasoning_trace_absent_when_not_requested(
        self, db_pool, run, llm_mock
    ):
        """return_reasoning=False → reasoning_trace is None in the response."""
        token_store = CompileTokenStore(db_pool)
        svc = ConceptCompilerService(llm_client=llm_mock, token_store=token_store)
        req = CompileConceptRequest(
            identifier="loan.trace_absent_test",
            description="Trace absence test",
            output_type="float",
            signal_names=["payments_on_time", "payments_due"],
            return_reasoning=False,
        )
        resp = run(svc.compile(req, db_pool))
        assert resp.reasoning_trace is None

    def test_reasoning_trace_present_when_requested(self, db_pool, run, llm_mock):
        """return_reasoning=True → reasoning_trace has 4 steps in order 1-4."""
        token_store = CompileTokenStore(db_pool)
        svc = ConceptCompilerService(llm_client=llm_mock, token_store=token_store)
        req = CompileConceptRequest(
            identifier="loan.trace_present_test",
            description="Trace presence test",
            output_type="float",
            signal_names=["payments_on_time", "payments_due"],
            return_reasoning=True,
        )
        resp = run(svc.compile(req, db_pool))
        assert resp.reasoning_trace is not None
        assert len(resp.reasoning_trace.steps) == 4
        assert [s.step_index for s in resp.reasoning_trace.steps] == [1, 2, 3, 4]
        assert all(s.outcome == "accepted" for s in resp.reasoning_trace.steps)

    def test_vocabulary_context_rejected_before_llm(
        self, db_pool, run, llm_mock
    ):
        """
        vocabulary_context with both lists empty → VocabularyMismatchError raised
        before any LLM call.
        """
        from app.registry.definitions import DefinitionRegistry
        from app.services.task_authoring import TaskAuthoringService
        from app.stores.definition import DefinitionStore
        from app.stores.task import TaskStore

        svc = TaskAuthoringService(
            task_store=TaskStore(db_pool),
            definition_registry=DefinitionRegistry(store=DefinitionStore(db_pool)),
            llm_client=llm_mock,
        )
        req = CreateTaskRequest(
            intent="alert when loan repayment drops below threshold",
            entity_scope="loan",
            delivery=DeliveryConfig(
                type=DeliveryType.WEBHOOK,
                endpoint="https://loan.example.com/hook",
            ),
            vocabulary_context=VocabularyContext(
                available_concept_ids=[],
                available_condition_ids=[],
            ),
        )

        calls_before = llm_mock.call_count
        with pytest.raises(VocabularyMismatchError):
            run(svc.create_task(req))

        # LLM must NOT have been called — error is raised before Step 2.
        assert llm_mock.call_count == calls_before, (
            "LLM was called despite empty vocabulary_context — pre-LLM guard failed"
        )

    def test_concept_id_skips_steps_1_and_2(self, db_pool, run, llm_mock):
        """
        When concept_id is provided, CoR steps 1 and 2 have outcome='skipped'.

        The test registers a concept first so that the definition_registry lookup
        succeeds, then uses create_task_stream() and inspects the step events.
        Steps 1 and 2 are emitted before any LLM call and before step 3 logic,
        so we can assert 'skipped' even if later steps fail or raise.
        """
        from app.registry.definitions import DefinitionRegistry
        from app.services.task_authoring import TaskAuthoringService
        from app.stores.definition import DefinitionStore
        from app.stores.task import TaskStore

        # Phase 1: compile + register a concept so the concept_id is known.
        token_store = CompileTokenStore(db_pool)
        compiler = ConceptCompilerService(llm_client=llm_mock, token_store=token_store)
        compile_resp = run(compiler.compile(
            CompileConceptRequest(
                identifier="loan.skip_steps_concept",
                description="Skip steps 1 and 2 test concept",
                output_type="float",
                signal_names=["payments_on_time", "payments_due"],
            ),
            db_pool,
        ))
        run(ConceptRegistrationService().register(
            RegisterConceptRequest(
                compile_token=compile_resp.compile_token,
                identifier="loan.skip_steps_concept",
            ),
            db_pool,
        ))

        # Phase 2: create task with the pre-compiled concept_id.
        svc = TaskAuthoringService(
            task_store=TaskStore(db_pool),
            definition_registry=DefinitionRegistry(store=DefinitionStore(db_pool)),
            llm_client=llm_mock,
        )
        req = CreateTaskRequest(
            intent="alert when loan repayment drops below threshold",
            entity_scope="loan",
            delivery=DeliveryConfig(
                type=DeliveryType.WEBHOOK,
                endpoint="https://loan.example.com/hook",
            ),
            concept_id="loan.skip_steps_concept",
        )

        events = run(_collect_events(svc.create_task_stream(req)))

        # Find step events for indices 1 and 2.
        step_events = {
            e["data"]["step_index"]: e["data"]
            for e in events
            if e.get("event_type") == "cor_step"
        }

        assert 1 in step_events, "step_index=1 event not found"
        assert 2 in step_events, "step_index=2 event not found"
        assert step_events[1]["outcome"] == "skipped", (
            f"Step 1 expected 'skipped', got {step_events[1]['outcome']!r}"
        )
        assert step_events[2]["outcome"] == "skipped", (
            f"Step 2 expected 'skipped', got {step_events[2]['outcome']!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — Route → Route  (two-service end-to-end flows, real DB)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRouteToRoute:
    """
    Two-service flows that span multiple V7 endpoints.

    All tests exercise the service layer directly (no HTTP transport) with a
    real PostgreSQL database — the closest equivalent to Route→Route without
    requiring the FastAPI lifespan (which needs a config file at startup).
    """

    def test_compile_token_single_use_at_http_boundary(
        self, db_pool, run, llm_mock, loan_compile_request
    ):
        """
        POST /concepts/compile → POST /concepts/register (OK) →
        POST /concepts/register again with same token → CompileTokenConsumedError.
        """
        from app.models.errors import CompileTokenConsumedError

        token_store = CompileTokenStore(db_pool)
        compiler = ConceptCompilerService(llm_client=llm_mock, token_store=token_store)
        resp = run(compiler.compile(loan_compile_request, db_pool))
        token_string = resp.compile_token

        reg_svc = ConceptRegistrationService()
        run(reg_svc.register(
            RegisterConceptRequest(
                compile_token=token_string,
                identifier="loan.repayment_ratio",
            ),
            db_pool,
        ))

        # Second register with the same (consumed) token → 409
        with pytest.raises(CompileTokenConsumedError):
            run(reg_svc.register(
                RegisterConceptRequest(
                    compile_token=token_string,
                    identifier="loan.repayment_ratio",
                ),
                db_pool,
            ))

    def test_identifier_not_locked_at_compile_time(self, db_pool, run, llm_mock):
        """
        The identifier is freely chosen at compile time (no pre-registration
        required).  At register time, mismatching the identifier → HTTP 422.

        Verifies two things:
          1. Any identifier compiles without error (no lock/pre-validation at POST /concepts/compile).
          2. Using a different identifier at POST /concepts/register → IdentifierMismatchError.
        """
        token_store = CompileTokenStore(db_pool)
        compiler = ConceptCompilerService(llm_client=llm_mock, token_store=token_store)
        resp = run(compiler.compile(
            CompileConceptRequest(
                identifier="loan.freely_chosen_identifier",
                description="Identifier mismatch test concept",
                output_type="float",
                signal_names=["payments_on_time", "payments_due"],
            ),
            db_pool,
        ))
        # Compile succeeds — identifier is not pre-validated against any registry.
        assert resp.compile_token, "Compile should have succeeded"

        # Register with a DIFFERENT identifier → IdentifierMismatchError (HTTP 422).
        reg_svc = ConceptRegistrationService()
        with pytest.raises(IdentifierMismatchError):
            run(reg_svc.register(
                RegisterConceptRequest(
                    compile_token=resp.compile_token,
                    identifier="loan.different_identifier",  # mismatch!
                ),
                db_pool,
            ))

    def test_sse_wire_format_compliance(
        self, db_pool, run, llm_mock, loan_compile_request
    ):
        """
        SSE events from compile_stream() pass through sse_event() and produce
        wire-format strings conforming to RFC 8895:
          event: <type>\n
          data: <json>\n
          \n
        """
        import json

        from app.api.routes.utils import sse_event

        token_store = CompileTokenStore(db_pool)
        svc = ConceptCompilerService(llm_client=llm_mock, token_store=token_store)
        events = run(_collect_events(svc.compile_stream(loan_compile_request, db_pool)))

        assert len(events) > 0, "compile_stream yielded no events"

        for ev in events:
            event_type = ev["event_type"]
            data       = ev["data"]
            wire       = sse_event(event_type, data)

            # Must start with "event: <type>\n"
            assert wire.startswith(f"event: {event_type}\n"), (
                f"SSE event line malformed: {wire!r}"
            )
            # Must have "data: <json>\n"
            lines = wire.split("\n")
            data_line = next((l for l in lines if l.startswith("data: ")), None)
            assert data_line is not None, f"No 'data:' line in SSE event: {wire!r}"
            json_payload = data_line[len("data: "):]
            parsed = json.loads(json_payload)   # must be valid JSON
            assert isinstance(parsed, dict)

            # Must end with "\n\n" (blank line separator)
            assert wire.endswith("\n\n"), (
                f"SSE event must end with blank line, got: {wire!r}"
            )
