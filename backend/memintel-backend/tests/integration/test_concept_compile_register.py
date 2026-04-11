"""
tests/integration/test_concept_compile_register.py
──────────────────────────────────────────────────────────────────────────────
Integration tests for the full V7 two-phase concept compile + register flow.

These tests exercise the complete pipeline against a real PostgreSQL database
(memintel_test) — no mocks for the token store or definition store.

The ConceptCompilerService uses MockIntegrationLLM (injected) to avoid
any LLM API dependency during integration tests.

Coverage
────────
  1. Full two-phase flow: compile → register → concept_id returned (HTTP 201)
  2. Idempotent registration: same (identifier, ir_hash) via two separate
     compile→register flows both return HTTP 201 with the same concept_id
  3. Identifier conflict: same identifier + different ir_hash → HTTP 409
  4. Token consumed on first register; second register raises HTTP 409
  5. Identifier mismatch: register with wrong identifier → HTTP 422
"""
from __future__ import annotations

import pytest

from app.models.concept_compile import CompileConceptRequest, RegisterConceptRequest
from app.models.errors import (
    CompileTokenConsumedError,
    IdentifierConflictError,
    IdentifierMismatchError,
)
from app.services.concept_compiler import ConceptCompilerService
from app.services.concept_registration import ConceptRegistrationService
from app.stores.compile_token import CompileTokenStore
from app.stores.definition import DefinitionStore


# ── Mock LLM for integration tests ────────────────────────────────────────────

class MockIntegrationLLM:
    """
    Minimal LLM mock for integration tests.

    Routes responses by context["step"]. Accepts a `formula_variant`
    parameter so that tests can produce two different ir_hashes from the
    same identifier by using different formulas.
    """

    def __init__(self, formula_variant: str = "ratio") -> None:
        self._formula_variant = formula_variant

    def generate_compile_step(self, prompt: str, context: dict) -> dict:
        step = context.get("step", 0)
        identifier = context.get("identifier", "test.concept")
        signal_names: list[str] = context.get("signal_names") or ["sig_a", "sig_b"]
        output_type: str = context.get("output_type", "float")

        if step == 1:
            return {
                "summary": f"Parsed intent for {identifier}",
                "outcome": "accepted",
                "entity_type": "entity",
                "desired_signal": identifier.split(".")[-1],
                "direction": "above",
                "urgency": "normal",
            }
        elif step == 2:
            return {
                "summary": "Identified signals",
                "outcome": "accepted",
                "domain_context": "test domain",
            }
        elif step == 3:
            formula = (
                f"{signal_names[0]} / {signal_names[1]}"
                if len(signal_names) >= 2
                else identifier
            )
            bindings = (
                [
                    {"signal_name": signal_names[0], "role": "numerator"},
                    {"signal_name": signal_names[1], "role": "denominator"},
                ]
                if len(signal_names) >= 2
                else [{"signal_name": s, "role": "input"} for s in signal_names]
            )
            return {
                "summary": f"DAG strategy: {self._formula_variant}",
                "outcome": "accepted",
                "formula_summary": f"{formula} [{self._formula_variant}]",
                "signal_bindings": bindings,
            }
        else:  # step 4
            return {
                "summary": f"Validated output_type '{output_type}'",
                "outcome": "accepted",
            }


# ── Request builders ───────────────────────────────────────────────────────────

def _compile_request(
    identifier: str = "loan.repayment_ratio",
    signal_names: list[str] | None = None,
) -> CompileConceptRequest:
    return CompileConceptRequest(
        identifier=identifier,
        description="Ratio of on-time payments to total payments due",
        output_type="float",
        signal_names=signal_names or ["payments_on_time", "payments_due"],
    )


def _register_request(
    compile_token: str,
    identifier: str = "loan.repayment_ratio",
) -> RegisterConceptRequest:
    return RegisterConceptRequest(
        compile_token=compile_token,
        identifier=identifier,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCompileRegisterIntegration:

    def test_full_two_phase_flow_returns_concept_id(self, db_pool, run):
        """
        Phase 1 (compile) → Phase 2 (register): happy path.

        Verifies that a compile→register round trip returns a stable concept_id
        equal to the identifier, version '1.0.0', and the correct output_type.
        """
        token_store = CompileTokenStore(db_pool)
        def_store = DefinitionStore(db_pool)

        compiler = ConceptCompilerService(
            llm_client=MockIntegrationLLM(),
            token_store=token_store,
        )
        compile_resp = run(compiler.compile(_compile_request(), pool=db_pool))

        assert compile_resp.compile_token
        assert compile_resp.compiled_concept.identifier == "loan.repayment_ratio"

        registrar = ConceptRegistrationService(
            token_store=token_store,
            definition_store=def_store,
        )
        reg_resp = run(
            registrar.register(
                _register_request(compile_resp.compile_token),
                pool=db_pool,
            )
        )

        assert reg_resp.concept_id == "loan.repayment_ratio"
        assert reg_resp.identifier == "loan.repayment_ratio"
        assert reg_resp.version == "1.0.0"
        assert reg_resp.output_type == "float"
        assert reg_resp.registered_at is not None

    def test_idempotent_registration_returns_same_concept_id(self, db_pool, run):
        """
        Two separate compile→register flows for the same (identifier, ir_hash)
        both succeed with HTTP 201 and return the SAME concept_id.
        No duplicate row is created.
        """
        token_store = CompileTokenStore(db_pool)
        def_store = DefinitionStore(db_pool)

        # Compile twice with the same LLM (same formula → same ir_hash)
        llm = MockIntegrationLLM(formula_variant="ratio")
        compiler1 = ConceptCompilerService(llm_client=llm, token_store=token_store)
        compiler2 = ConceptCompilerService(llm_client=llm, token_store=token_store)

        resp1 = run(compiler1.compile(_compile_request(), pool=db_pool))
        resp2 = run(compiler2.compile(_compile_request(), pool=db_pool))

        # Both tokens refer to the same identifier
        assert resp1.compile_token != resp2.compile_token

        registrar = ConceptRegistrationService(
            token_store=token_store,
            definition_store=def_store,
        )

        reg1 = run(registrar.register(_register_request(resp1.compile_token), pool=db_pool))
        reg2 = run(registrar.register(_register_request(resp2.compile_token), pool=db_pool))

        # Idempotent: same concept_id returned
        assert reg1.concept_id == reg2.concept_id == "loan.repayment_ratio"
        assert reg1.version == reg2.version == "1.0.0"

    def test_identifier_conflict_raises_identifier_conflict_error(self, db_pool, run):
        """
        Same identifier + different ir_hash (different formula) → IdentifierConflictError.

        Simulated by using two LLM mocks with different formula_variants, which
        produces different formula_summary values → different ir_hashes.
        """
        token_store = CompileTokenStore(db_pool)
        def_store = DefinitionStore(db_pool)

        # Two compiles with different formulas (different ir_hash)
        compiler_a = ConceptCompilerService(
            llm_client=MockIntegrationLLM(formula_variant="ratio"),
            token_store=token_store,
        )
        compiler_b = ConceptCompilerService(
            llm_client=MockIntegrationLLM(formula_variant="average"),
            token_store=token_store,
        )

        resp_a = run(compiler_a.compile(_compile_request(), pool=db_pool))
        resp_b = run(compiler_b.compile(_compile_request(), pool=db_pool))

        # Verify different ir_hashes
        tk_a = run(token_store.get(resp_a.compile_token))
        tk_b = run(token_store.get(resp_b.compile_token))
        assert tk_a.ir_hash != tk_b.ir_hash, (
            "Integration test requires different ir_hashes — "
            "check that MockIntegrationLLM formula_variant produces different formula_summary"
        )

        registrar = ConceptRegistrationService(
            token_store=token_store,
            definition_store=def_store,
        )

        # First registration succeeds
        run(registrar.register(_register_request(resp_a.compile_token), pool=db_pool))

        # Second registration for the same identifier with a different formula → conflict
        with pytest.raises(IdentifierConflictError) as exc_info:
            run(registrar.register(_register_request(resp_b.compile_token), pool=db_pool))

        assert exc_info.value.http_status == 409

    def test_token_consumed_on_first_register_second_raises(self, db_pool, run):
        """
        A compile_token is single-use. Presenting it a second time raises
        CompileTokenConsumedError (HTTP 409).
        """
        token_store = CompileTokenStore(db_pool)
        def_store = DefinitionStore(db_pool)

        compiler = ConceptCompilerService(
            llm_client=MockIntegrationLLM(),
            token_store=token_store,
        )
        compile_resp = run(compiler.compile(_compile_request(), pool=db_pool))
        token_string = compile_resp.compile_token

        registrar = ConceptRegistrationService(
            token_store=token_store,
            definition_store=def_store,
        )

        # First use: succeeds
        run(registrar.register(_register_request(token_string), pool=db_pool))

        # Second use of the same token: consumed
        with pytest.raises(CompileTokenConsumedError) as exc_info:
            run(registrar.register(_register_request(token_string), pool=db_pool))

        assert exc_info.value.http_status == 409

    def test_identifier_mismatch_raises_identifier_mismatch_error(self, db_pool, run):
        """
        Register with an identifier that differs from the one locked at compile
        time → IdentifierMismatchError (HTTP 422). No definition is written.
        """
        token_store = CompileTokenStore(db_pool)
        def_store = DefinitionStore(db_pool)

        compiler = ConceptCompilerService(
            llm_client=MockIntegrationLLM(),
            token_store=token_store,
        )
        compile_resp = run(
            compiler.compile(_compile_request(identifier="loan.repayment_ratio"), pool=db_pool)
        )

        registrar = ConceptRegistrationService(
            token_store=token_store,
            definition_store=def_store,
        )

        with pytest.raises(IdentifierMismatchError) as exc_info:
            run(
                registrar.register(
                    _register_request(
                        compile_resp.compile_token,
                        identifier="loan.something_else",  # wrong identifier
                    ),
                    pool=db_pool,
                )
            )

        assert exc_info.value.http_status == 422
