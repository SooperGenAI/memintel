"""
tests/integration/test_stores_integration.py
──────────────────────────────────────────────────────────────────────────────
Integration tests for all six store classes + migration integrity.

These tests run against a real PostgreSQL database (memintel_test) using a
real asyncpg connection pool.  No mocks.  Every test is independent — the
clean_tables fixture truncates all tables before each test runs.

Run integration tests only::

    pytest tests/integration/test_stores_integration.py -m integration

Run with custom DB URL::

    TEST_DATABASE_URL=postgresql://... pytest tests/integration/ -m integration

Bugs found / confirmed
──────────────────────
BUG-6-1   decisions.concept_value is DOUBLE PRECISION — Python bool True is
          stored as 1.0 and retrieved as 1.0 (not True). Union ordering in
          DecisionRecord cannot recover the original type.

BUG-6-2   decisions.concept_value is DOUBLE PRECISION — Python str values
          (e.g. concept_value="high_risk") cannot be encoded by asyncpg for a
          DOUBLE PRECISION column. record() raises an asyncpg encoding error
          at INSERT time.

Note: BUG-6-1 and BUG-6-2 share the same root cause: the column type must be
changed (e.g. to TEXT or a separate concept_value_text column) to support the
full bool | float | int | str | None union declared in DecisionRecord.

BUG-7-1   definitions.uq_definition_version is UNIQUE (definition_id, version)
          without namespace. promote() tries to INSERT a second row with the
          same (definition_id, version) but a different namespace, which always
          raises UniqueViolationError. Fix: change constraint to
          UNIQUE (definition_id, version, namespace).
"""
from __future__ import annotations

import asyncpg
import pytest

from app.models.calibration import FeedbackRecord, FeedbackValue
from app.models.decision import DecisionRecord
from app.models.task import DeliveryConfig, DeliveryType, Task, TaskStatus
from app.stores.concept_result import ConceptResultStore
from app.stores.decision import DecisionStore
from app.stores.definition import DefinitionStore
from app.stores.feedback import FeedbackStore
from app.stores.job import JobStore
from app.stores.task import TaskStore

pytestmark = pytest.mark.integration


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _task(suffix: str = "a", *, status: TaskStatus = TaskStatus.ACTIVE) -> Task:
    """Minimal valid Task for testing."""
    return Task(
        intent=f"Alert when churn risk is high ({suffix})",
        concept_id=f"churn_risk_{suffix}",
        concept_version="1.0",
        condition_id=f"churn_cond_{suffix}",
        condition_version="1.0",
        action_id=f"alert_action_{suffix}",
        action_version="1.0",
        entity_scope="user",
        delivery=DeliveryConfig(
            type=DeliveryType.WEBHOOK,
            endpoint=f"https://hooks.example.com/{suffix}",
        ),
        status=status,
        context_version=None,
        guardrails_version="1.0",
    )


def _definition_body(label: str = "test") -> dict:
    """Minimal JSONB body for a definition row."""
    return {"label": label, "version": "1.0"}


def _decision(
    entity: str = "ent_001",
    concept_value: float = 0.75,
    condition_version: str = "1.0",
) -> DecisionRecord:
    """Minimal valid DecisionRecord for testing."""
    return DecisionRecord(
        concept_id="churn_risk",
        concept_version="1.0",
        condition_id="churn_cond",
        condition_version=condition_version,
        entity_id=entity,
        fired=True,
        concept_value=concept_value,
        threshold_applied={"direction": "above", "value": 0.7},
        ir_hash="abc123",
        input_primitives={"days_since_login": 30},
        signal_errors=None,
        reason=None,
        action_ids_fired=["alert_action"],
        dry_run=False,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STORE 1 — DefinitionStore
# ═══════════════════════════════════════════════════════════════════════════════

class TestDefinitionStore:

    def test_register_and_retrieve_primitive(self, db_pool, clean_tables, run):
        """Register a primitive and retrieve it — assert all metadata fields match."""
        store = DefinitionStore(db_pool)
        body = {"type": "float", "description": "Days since last login"}

        resp = run(store.register(
            definition_id="days_since_login",
            version="1.0",
            definition_type="primitive",
            namespace="org",
            body=body,
            meaning_hash="hash_001",
            ir_hash=None,
        ))

        assert resp.definition_id == "days_since_login"
        assert resp.version == "1.0"
        assert resp.definition_type == "primitive"
        assert resp.namespace.value == "org"
        assert resp.meaning_hash == "hash_001"
        assert resp.ir_hash is None
        assert resp.deprecated is False
        assert resp.deprecated_at is None
        assert resp.replacement_version is None
        assert resp.created_at is not None

        # Verify body round-trip via get()
        retrieved_body = run(store.get("days_since_login", "1.0"))
        assert retrieved_body == body

    def test_register_feature_type_accepted(self, db_pool, clean_tables, run):
        """
        definition_type='feature' must be accepted by the DB after migration 0005.

        This was bug 3-B: the CHECK constraint only allowed concept/condition/action/
        primitive. Migration 0005 added 'feature'.
        """
        store = DefinitionStore(db_pool)
        resp = run(store.register(
            definition_id="revenue_feature",
            version="1.0",
            definition_type="feature",
            namespace="org",
            body={"feature": "revenue_30d"},
        ))
        assert resp.definition_type == "feature"

        # Also verify it can be retrieved
        retrieved = run(store.get("revenue_feature", "1.0"))
        assert retrieved is not None
        assert retrieved["feature"] == "revenue_30d"

    def test_register_invalid_type_rejected(self, db_pool, clean_tables, run):
        """
        definition_type='invalid_type' must raise an error.

        The DB CHECK constraint enforces valid types — asyncpg raises
        CheckViolationError when the constraint is violated.
        """
        store = DefinitionStore(db_pool)
        with pytest.raises(Exception):
            run(store.register(
                definition_id="bad_def",
                version="1.0",
                definition_type="invalid_type",
                namespace="org",
                body={"x": 1},
            ))

    def test_list_definitions_returns_correct_page(self, db_pool, clean_tables, run):
        """
        Register 5 definitions.  First page (limit=3) returns 3 with has_more=True.
        Second page (using next_cursor) returns remaining 2 with has_more=False.

        DefinitionStore uses cursor-based pagination, not offset-based.
        """
        store = DefinitionStore(db_pool)
        for i in range(5):
            run(store.register(
                definition_id=f"prim_{i:02d}",
                version="1.0",
                definition_type="primitive",
                namespace="org",
                body={"index": i},
            ))

        page1 = run(store.list(definition_type="primitive", namespace="org", limit=3))
        assert len(page1.items) == 3
        assert page1.has_more is True
        assert page1.next_cursor is not None
        assert page1.total_count == 5

        page2 = run(store.list(
            definition_type="primitive",
            namespace="org",
            limit=3,
            cursor=page1.next_cursor,
        ))
        assert len(page2.items) == 2
        assert page2.has_more is False
        assert page2.total_count == 5

    def test_count_actions_returns_db_total_not_page_count(
        self, db_pool, clean_tables, run
    ):
        """
        count_actions() must return the DB total regardless of the page size
        used by list_actions().

        This was bug 4-B: a prior implementation returned len(page) instead of
        the real count when pagination limited the result set.
        """
        store = DefinitionStore(db_pool)
        # Register a valid ActionDefinition body that model_validate accepts.
        # ActionDefinition requires: action_id, version, config (discriminated
        # union — needs 'type' field), trigger (fire_on + condition_id +
        # condition_version), and namespace.
        def _action_body(i: int) -> dict:
            return {
                "action_id": f"action_{i}",
                "version": "1.0",
                "config": {
                    "type": "webhook",
                    "endpoint": f"https://example.com/hook/{i}",
                    "method": "POST",
                },
                "trigger": {
                    "fire_on": "true",
                    "condition_id": f"cond_{i}",
                    "condition_version": "1.0",
                },
                "namespace": "org",
            }
        for i in range(5):
            run(store.register(
                definition_id=f"action_{i}",
                version="1.0",
                definition_type="action",
                namespace="org",
                body=_action_body(i),
            ))

        # count_actions sees all 5 regardless of limit
        total = run(store.count_actions(namespace="org"))
        assert total == 5

        # list_actions with limit=2 still returns only 2 rows
        page = run(store.list_actions(namespace="org", limit=2, offset=0))
        assert len(page) == 2
        # but count_actions is not affected by limit — still 5
        assert total == 5

    def test_deprecate_definition(self, db_pool, clean_tables, run):
        """Register → deprecate → retrieve: deprecated flag must be True."""
        store = DefinitionStore(db_pool)
        run(store.register(
            definition_id="old_prim",
            version="1.0",
            definition_type="primitive",
            namespace="org",
            body={"x": 1},
        ))

        deprecated = run(store.deprecate(
            definition_id="old_prim",
            version="1.0",
            replacement_version="2.0",
            reason="Superseded by new_prim v2.0",
        ))
        assert deprecated.deprecated is True
        assert deprecated.deprecated_at is not None
        assert deprecated.replacement_version == "2.0"

        # get_metadata also reflects the deprecated state
        meta = run(store.get_metadata("old_prim", "1.0"))
        assert meta is not None
        assert meta.deprecated is True

    def test_promote_definition(self, db_pool, clean_tables, run):
        """
        BUG-7-1: promote() cannot copy a definition to another namespace.

        promote() tries to INSERT a new row with the target namespace, but the
        DB constraint uq_definition_version is on (definition_id, version)
        WITHOUT namespace.  Inserting a second row with the same definition_id
        and version always raises UniqueViolationError regardless of namespace.

        Root cause: the unique constraint should be
            UNIQUE (definition_id, version, namespace)
        to allow the same logical definition to exist in multiple namespaces.

        Until the schema is corrected, promote() raises ConflictError for any
        definition that already exists in any namespace.
        """
        from app.models.errors import ConflictError

        store = DefinitionStore(db_pool)
        run(store.register(
            definition_id="shared_concept",
            version="1.0",
            definition_type="concept",
            namespace="org",
            body={"concept": "shared"},
        ))

        with pytest.raises(ConflictError):
            run(store.promote(
                definition_id="shared_concept",
                version="1.0",
                from_namespace="org",
                to_namespace="team",
                elevated_key=False,
            ))


# ═══════════════════════════════════════════════════════════════════════════════
# STORE 2 — TaskStore
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskStore:

    def test_create_and_retrieve_task(self, db_pool, clean_tables, run):
        """
        Create a task with all fields including guardrails_version.
        Retrieve by task_id and assert guardrails_version is persisted.

        This was bug 3-A: guardrails_version was never written to the DB INSERT
        statement, so it was always NULL on retrieval even when set on creation.
        """
        store = TaskStore(db_pool)
        task = _task("gv")
        task.guardrails_version = "v2"

        created = run(store.create(task))
        assert created.task_id is not None

        fetched = run(store.get(created.task_id))
        assert fetched is not None
        assert fetched.task_id == created.task_id
        assert fetched.intent == task.intent
        assert fetched.concept_id == task.concept_id
        assert fetched.concept_version == task.concept_version
        assert fetched.condition_id == task.condition_id
        assert fetched.condition_version == task.condition_version
        assert fetched.action_id == task.action_id
        assert fetched.action_version == task.action_version
        assert fetched.entity_scope == task.entity_scope
        assert fetched.status == TaskStatus.ACTIVE
        # Bug 3-A: guardrails_version must survive the round-trip
        assert fetched.guardrails_version == "v2"

    def test_list_tasks_returns_all_fields(self, db_pool, clean_tables, run):
        """
        Create 3 tasks.  List all tasks.  Every returned task must have
        guardrails_version populated.
        """
        store = TaskStore(db_pool)
        for i in range(3):
            t = _task(str(i))
            t.guardrails_version = f"gv_{i}"
            run(store.create(t))

        task_list = run(store.list())
        assert task_list.total_count == 3
        assert len(task_list.items) == 3

        for task in task_list.items:
            # guardrails_version must be present on every task (bug 3-A check)
            assert task.guardrails_version is not None
            assert task.guardrails_version.startswith("gv_")

    def test_task_status_filter(self, db_pool, clean_tables, run):
        """
        Create active and paused tasks.  List with status filter returns
        only the tasks matching the requested status.
        """
        store = TaskStore(db_pool)

        active_task = run(store.create(_task("act", status=TaskStatus.ACTIVE)))
        paused_task = run(store.create(_task("pau", status=TaskStatus.PAUSED)))

        active_list = run(store.list(status="active"))
        paused_list = run(store.list(status="paused"))

        active_ids = {t.task_id for t in active_list.items}
        paused_ids = {t.task_id for t in paused_list.items}

        assert active_task.task_id in active_ids
        assert paused_task.task_id not in active_ids
        assert paused_task.task_id in paused_ids
        assert active_task.task_id not in paused_ids

    def test_update_task_condition_version(self, db_pool, clean_tables, run):
        """
        Create a task bound to condition v1.  Update condition_version to v2.
        Retrieve and assert condition_version == 'v2'.
        """
        store = TaskStore(db_pool)
        task = _task("upd")
        created = run(store.create(task))
        assert created.condition_version == "1.0"

        updated = run(store.update(created.task_id, {"condition_version": "2.0"}))
        assert updated.condition_version == "2.0"

        fetched = run(store.get(created.task_id))
        assert fetched.condition_version == "2.0"


# ═══════════════════════════════════════════════════════════════════════════════
# STORE 3 — DecisionStore
# ═══════════════════════════════════════════════════════════════════════════════

class TestDecisionStore:

    def test_record_and_retrieve_decision(self, db_pool, clean_tables, run):
        """
        Record a decision with all fields populated (concept_value as float).
        Retrieve by decision_id and assert every field matches.

        Uses concept_value=0.75 (float) to avoid the DOUBLE PRECISION column
        type-mismatch bugs (see BUG-6-1, BUG-6-2).
        """
        store = DecisionStore(db_pool)
        d = DecisionRecord(
            concept_id="churn_risk",
            concept_version="2.0",
            condition_id="churn_cond",
            condition_version="2.0",
            entity_id="ent_full",
            fired=True,
            concept_value=0.87,
            threshold_applied={"direction": "above", "value": 0.80},
            ir_hash="ir_abc",
            input_primitives={"days_since_login": 45, "avg_spend": 120.5},
            signal_errors={"weather_signal": "timeout"},
            reason=None,
            action_ids_fired=["webhook_alert", "email_alert"],
            dry_run=False,
        )

        decision_id = run(store.record(d))
        assert decision_id is not None
        assert len(decision_id) > 0

        fetched = run(store.get(decision_id))
        assert fetched is not None
        assert fetched.decision_id == decision_id
        assert fetched.concept_id == "churn_risk"
        assert fetched.concept_version == "2.0"
        assert fetched.condition_id == "churn_cond"
        assert fetched.condition_version == "2.0"
        assert fetched.entity_id == "ent_full"
        assert fetched.fired is True
        assert fetched.concept_value == pytest.approx(0.87)
        assert fetched.threshold_applied == {"direction": "above", "value": 0.80}
        assert fetched.ir_hash == "ir_abc"
        assert fetched.input_primitives == {"days_since_login": 45, "avg_spend": 120.5}
        assert fetched.signal_errors == {"weather_signal": "timeout"}
        assert fetched.reason is None
        assert set(fetched.action_ids_fired) == {"webhook_alert", "email_alert"}
        assert fetched.dry_run is False
        assert fetched.evaluated_at is not None

    def test_concept_value_bool_persisted_correctly(
        self, db_pool, clean_tables, run
    ):
        """
        BUG-6-1 — concept_value=True should round-trip as True (bool), not 1.0 (float).

        The decisions.concept_value column is DOUBLE PRECISION. PostgreSQL stores
        Python bool True as 1.0 (float). On retrieval, asyncpg returns 1.0, and
        Pydantic's DecisionRecord.concept_value: bool | float | int | str | None
        resolves 1.0 as float — not as bool — because 1.0 is not an exact bool.

        EXPECTED (correct): fetched.concept_value is True
        ACTUAL (buggy):     fetched.concept_value == 1.0

        Fix: change the column type from DOUBLE PRECISION to TEXT or add a
        separate concept_value_bool / concept_value_text column.
        """
        store = DecisionStore(db_pool)
        d = DecisionRecord(
            concept_id="bool_concept",
            concept_version="1.0",
            condition_id="bool_cond",
            condition_version="1.0",
            entity_id="ent_bool",
            fired=True,
            concept_value=True,   # bool — stored in DOUBLE PRECISION column
            dry_run=False,
        )
        decision_id = run(store.record(d))
        fetched = run(store.get(decision_id))
        assert fetched is not None
        # BUG-6-1: this assertion fails — actual value is 1.0 (float), not True (bool)
        assert fetched.concept_value is True, (
            f"BUG-6-1: concept_value=True was stored as {fetched.concept_value!r} "
            f"({type(fetched.concept_value).__name__}). "
            "The decisions.concept_value column type (DOUBLE PRECISION) "
            "cannot preserve Python bool identity."
        )

    def test_concept_value_str_persisted_correctly(
        self, db_pool, clean_tables, run
    ):
        """
        BUG-6-2 — concept_value='high_risk' should round-trip as 'high_risk' (str).

        The decisions.concept_value column is DOUBLE PRECISION. asyncpg cannot
        encode a Python str as float8 — it raises a DataError at INSERT time.
        This means string concept values (e.g. from equals strategy on categorical
        concepts) cannot be recorded in the decisions table at all.

        EXPECTED (correct): fetched.concept_value == "high_risk"
        ACTUAL (buggy):     asyncpg.DataError raised at store.record() time.

        Fix: same as BUG-6-1 — change the column type or add a parallel text column.
        """
        store = DecisionStore(db_pool)
        d = DecisionRecord(
            concept_id="segment_concept",
            concept_version="1.0",
            condition_id="segment_cond",
            condition_version="1.0",
            entity_id="ent_str",
            fired=True,
            concept_value="high_risk",   # str — cannot encode for DOUBLE PRECISION
            dry_run=False,
        )
        decision_id = run(store.record(d))
        fetched = run(store.get(decision_id))
        assert fetched is not None
        # BUG-6-2: this assertion is never reached — record() raises an encoding error
        assert fetched.concept_value == "high_risk", (
            f"BUG-6-2: concept_value='high_risk' was stored as {fetched.concept_value!r}. "
            "The decisions.concept_value DOUBLE PRECISION column cannot store strings."
        )

    def test_list_decisions_by_entity(self, db_pool, clean_tables, run):
        """
        Record 5 decisions for entity ent_001 and 3 for ent_002.
        list_for_entity('ent_001', concept_id) returns exactly 5.
        """
        store = DecisionStore(db_pool)

        for i in range(5):
            run(store.record(_decision(entity="ent_001")))
        for i in range(3):
            run(store.record(_decision(entity="ent_002")))

        results = run(store.list_for_entity("ent_001", "churn_risk", limit=20))
        assert len(results) == 5
        assert all(r.entity_id == "ent_001" for r in results)

        results_002 = run(store.list_for_entity("ent_002", "churn_risk", limit=20))
        assert len(results_002) == 3

    def test_list_decisions_by_condition_version(
        self, db_pool, clean_tables, run
    ):
        """
        Record decisions for condition v1 and v2.
        Query by condition_version='v1' returns only v1 decisions.

        DecisionStore has no list_by_condition_version() method — this test
        uses a raw SQL query to verify the data is persisted correctly and
        queryable by condition_version.
        """
        store = DecisionStore(db_pool)

        for _ in range(3):
            run(store.record(_decision(entity="ent_cv", condition_version="v1")))
        for _ in range(2):
            run(store.record(_decision(entity="ent_cv", condition_version="v2")))

        # Use raw SQL since DecisionStore has no list_by_condition_version
        async def _fetch_by_version(version: str) -> list:
            return await db_pool.fetch(
                "SELECT * FROM decisions WHERE condition_version = $1",
                version,
            )

        v1_rows = run(_fetch_by_version("v1"))
        v2_rows = run(_fetch_by_version("v2"))

        assert len(v1_rows) == 3
        assert len(v2_rows) == 2
        assert all(r["condition_version"] == "v1" for r in v1_rows)
        assert all(r["condition_version"] == "v2" for r in v2_rows)


# ═══════════════════════════════════════════════════════════════════════════════
# STORE 4 — FeedbackStore
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeedbackStore:

    def _feedback_record(
        self,
        entity: str = "user:1",
        ts: str = "2026-01-15T10:00:00",
        feedback: FeedbackValue = FeedbackValue.FALSE_POSITIVE,
    ) -> FeedbackRecord:
        return FeedbackRecord(
            feedback_id="placeholder",   # overwritten by DB
            condition_id="churn_cond",
            condition_version="1.0",
            entity=entity,
            timestamp=ts,
            feedback=feedback,
            note=None,
            recorded_at="placeholder",   # overwritten by DB
        )

    def test_record_and_retrieve_feedback(self, db_pool, clean_tables, run):
        """
        Record one feedback record.  Retrieve it via get_by_condition().
        Assert condition_id, condition_version, entity, and feedback type match.
        """
        store = FeedbackStore(db_pool)
        record = self._feedback_record()
        created = run(store.create(record))

        assert created.feedback_id != "placeholder"
        assert created.condition_id == "churn_cond"
        assert created.condition_version == "1.0"
        assert created.entity == "user:1"
        assert created.feedback == FeedbackValue.FALSE_POSITIVE
        assert created.recorded_at != "placeholder"

        # Retrieve via get_by_condition
        records = run(store.get_by_condition("churn_cond", "1.0"))
        assert len(records) == 1
        r = records[0]
        assert r.condition_id == "churn_cond"
        assert r.condition_version == "1.0"
        assert r.entity == "user:1"
        assert r.feedback == FeedbackValue.FALSE_POSITIVE

    def test_feedback_accumulates_for_calibration(
        self, db_pool, clean_tables, run
    ):
        """
        Record 5 feedback records for the same condition.
        get_by_condition() returns all 5 — this is what CalibrationService
        reads to derive a calibration direction.

        Verifies the feedback → calibration chain (bug 5): if get_by_condition()
        returned fewer than all records, CalibrationService would receive
        insufficient data and return no_recommendation even when enough feedback
        exists.
        """
        store = FeedbackStore(db_pool)

        timestamps = [
            f"2026-01-{15 + i:02d}T10:00:00"
            for i in range(5)
        ]
        entities = [f"user:{i}" for i in range(5)]

        for ts, entity in zip(timestamps, entities):
            rec = self._feedback_record(entity=entity, ts=ts)
            run(store.create(rec))

        records = run(store.get_by_condition("churn_cond", "1.0"))
        assert len(records) == 5, (
            f"Expected 5 feedback records for calibration but got {len(records)}. "
            "Bug 5: CalibrationService would have seen insufficient data."
        )

        # Records are returned oldest-first (recorded_at ASC) as required by
        # CalibrationService.derive_direction() for majority-vote computation.
        timestamps_returned = [r.timestamp for r in records]
        assert timestamps_returned == sorted(timestamps_returned)

    def test_duplicate_feedback_raises_conflict(
        self, db_pool, clean_tables, run
    ):
        """
        Submitting feedback twice for the same (condition, entity, timestamp)
        raises ConflictError — the uniqueness constraint is enforced.
        """
        from app.models.errors import ConflictError
        store = FeedbackStore(db_pool)

        rec = self._feedback_record()
        run(store.create(rec))

        with pytest.raises(ConflictError):
            run(store.create(rec))


# ═══════════════════════════════════════════════════════════════════════════════
# STORE 5 — ConceptResultStore
# ═══════════════════════════════════════════════════════════════════════════════

class TestConceptResultStore:

    def test_store_and_fetch_history(self, db_pool, clean_tables, run):
        """
        Store 5 concept results for entity 'ent_001'.
        fetch_history() returns all 5 in oldest-first order.

        Oldest-first ordering is required by ChangeStrategy (history[-1] is the
        previous value) and consistent with ZScoreStrategy/PercentileStrategy.
        """
        store = ConceptResultStore(db_pool)

        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        for v in values:
            run(store.store(
                concept_id="churn_risk",
                version="1.0",
                entity="ent_001",
                value=v,
                output_type="float",
            ))

        history = run(store.fetch_history("churn_risk", "ent_001", limit=30))

        assert len(history) == 5
        # Verify oldest-first: values should be in insertion order
        retrieved_values = [row["value"] for row in history]
        assert retrieved_values == pytest.approx(values)

    def test_history_excludes_null_values(self, db_pool, clean_tables, run):
        """
        Store 3 results with real values and 2 with value=None.
        fetch_history() returns only the 3 non-null rows.

        This was bug 7-1: fetch_history() should filter WHERE value IS NOT NULL
        so strategies never receive null entries in their reference frame.

        The SQL in ConceptResultStore.fetch_history() explicitly has:
          WHERE concept_id = $1 AND entity = $2 AND value IS NOT NULL
        This test verifies the filter works end-to-end.
        """
        store = ConceptResultStore(db_pool)

        # 3 real values
        for v in [0.7, 0.8, 0.9]:
            run(store.store(
                concept_id="churn_risk",
                version="1.0",
                entity="ent_null",
                value=v,
                output_type="float",
            ))

        # 2 null values (categorical output)
        for _ in range(2):
            run(store.store(
                concept_id="churn_risk",
                version="1.0",
                entity="ent_null",
                value=None,
                output_type="categorical",
                output_text="high",
            ))

        history = run(store.fetch_history("churn_risk", "ent_null", limit=30))

        assert len(history) == 3, (
            f"Bug 7-1: expected 3 non-null rows but got {len(history)}. "
            "fetch_history() must filter WHERE value IS NOT NULL."
        )
        assert all(row["value"] is not None for row in history)

    def test_history_minimum_not_met(self, db_pool, clean_tables, run):
        """
        Store 2 results.  fetch_history() with limit=30 returns those 2 rows.

        The store itself does NOT enforce a minimum history count — that is the
        responsibility of the strategy layer (PercentileStrategy enforces
        _HISTORY_MIN_RESULTS=3; ZScoreStrategy and ChangeStrategy rely on the
        service layer).  This test verifies the store returns exactly what exists.

        When the service layer calls fetch_history() and gets fewer than the
        minimum required results, it sets reason='insufficient_history' on the
        DecisionValue — not the store's concern.
        """
        store = ConceptResultStore(db_pool)

        run(store.store("churn_risk", "1.0", "ent_few", 0.5, "float"))
        run(store.store("churn_risk", "1.0", "ent_few", 0.6, "float"))

        history = run(store.fetch_history("churn_risk", "ent_few", limit=30))

        # Store returns exactly 2 — the strategy layer enforces minimums
        assert len(history) == 2, (
            f"Expected 2 rows from store (strategy enforces min, not store); "
            f"got {len(history)}."
        )

    def test_history_is_entity_scoped(self, db_pool, clean_tables, run):
        """
        Results for entity A must not appear in history for entity B.
        """
        store = ConceptResultStore(db_pool)

        run(store.store("churn_risk", "1.0", "ent_A", 0.9, "float"))
        run(store.store("churn_risk", "1.0", "ent_B", 0.1, "float"))

        hist_A = run(store.fetch_history("churn_risk", "ent_A", limit=30))
        hist_B = run(store.fetch_history("churn_risk", "ent_B", limit=30))

        assert len(hist_A) == 1
        assert len(hist_B) == 1
        assert hist_A[0]["value"] == pytest.approx(0.9)
        assert hist_B[0]["value"] == pytest.approx(0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# STORE 6 — JobStore
# ═══════════════════════════════════════════════════════════════════════════════

class TestJobStore:

    def test_create_and_retrieve_job(self, db_pool, clean_tables, run):
        """
        Enqueue a job.  Retrieve it by job_id.
        Assert all fields — especially poll_interval_seconds which maps from
        the DB column poll_interval_s.

        This was bug 3-1: poll_interval_s (DB column) was not mapped to
        poll_interval_seconds (Python field) in _row_to_job(), so jobs always
        returned the default value (2) even if the column had a different value.
        """
        store = JobStore(db_pool)
        request = {"concept_id": "churn", "entity": "user:1"}

        job = run(store.enqueue(request))

        assert job.job_id is not None
        assert job.job_type == "execute"
        assert job.status.value == "queued"
        assert job.poll_interval_seconds == 2   # DB default poll_interval_s = 2

        # Retrieve and verify the same mapping
        fetched = run(store.get(job.job_id))
        assert fetched is not None
        assert fetched.job_id == job.job_id
        assert fetched.status.value == "queued"
        assert fetched.poll_interval_seconds == 2
        assert fetched.request_body == request
        assert fetched.enqueued_at is not None

    def test_job_result_body_persisted(self, db_pool, clean_tables, run):
        """
        Enqueue → transition to running → transition to completed with result_body.
        Retrieve — assert result_body is populated on the returned Job.

        This was bug 4-A: result_body was never populated in the Job returned
        by update_status() or get() even when the DB had the value.  The field
        is excluded from JSON serialisation (Field(exclude=True)) but must be
        accessible as a Python attribute.
        """
        store = JobStore(db_pool)

        job = run(store.enqueue({"concept_id": "revenue", "entity": "user:99"}))
        running = run(store.update_status(job.job_id, "running"))
        assert running.status.value == "running"
        assert running.started_at is not None

        result_payload = {"value": 0.95, "type": "float", "entity": "user:99"}
        completed = run(store.update_status(
            job.job_id, "completed", result=result_payload
        ))
        assert completed.status.value == "completed"
        assert completed.completed_at is not None

        # Bug 4-A: result_body must be accessible on the returned Job
        assert completed.result_body == result_payload, (
            f"Bug 4-A: result_body was {completed.result_body!r} on the returned Job "
            "after update_status('completed', result=...). "
            "Expected the result payload to be persisted and returned."
        )

        # Verify get() also returns the result_body
        retrieved = run(store.get(completed.job_id))
        assert retrieved.result_body == result_payload, (
            "Bug 4-A: result_body was not returned by get() after update_status."
        )

    def test_job_state_machine_invalid_transition(
        self, db_pool, clean_tables, run
    ):
        """
        Attempt to transition from 'completed' (terminal) to any other state.
        ConflictError must be raised — the state machine is enforced by JobStore.
        """
        from app.models.errors import ConflictError
        store = JobStore(db_pool)

        job = run(store.enqueue({"x": 1}))
        run(store.update_status(job.job_id, "running"))
        run(store.update_status(job.job_id, "completed"))

        with pytest.raises(ConflictError):
            run(store.update_status(job.job_id, "running"))


# ═══════════════════════════════════════════════════════════════════════════════
# MIGRATION INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestMigrationIntegrity:

    def test_all_migrations_run_cleanly(self, db_pool, clean_tables, run):
        """
        Verify that all expected tables exist after alembic upgrade head.

        This implicitly tests that conftest.py successfully ran all 5 migrations
        (0001–0005) without errors.  The fixture would have skipped all tests if
        any migration failed.
        """
        async def _table_names() -> set[str]:
            rows = await db_pool.fetch(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
                """
            )
            return {r["table_name"] for r in rows}

        tables = run(_table_names())

        expected = {
            "definitions",
            "tasks",
            "decisions",
            "feedback_records",
            "concept_results",
            "guardrails_versions",
            "application_context",   # singular — migration 0002
            "jobs",
            "calibration_tokens",
            "execution_graphs",
        }
        missing = expected - tables
        assert not missing, (
            f"Missing tables after all migrations: {sorted(missing)}. "
            "One or more migrations may not have run."
        )

    def test_migration_0005_feature_constraint(self, db_pool, clean_tables, run):
        """
        After migration 0005, the definitions CHECK constraint must accept
        'feature' but reject any other unrecognised type.

        This was bug 3-B: the CHECK constraint in migration 0001 only allowed
        concept/condition/action/primitive.  RegistryService.register_feature()
        inserted definition_type='feature', which triggered a CheckViolationError
        at runtime.  Migration 0005 fixed the constraint.
        """
        # 'feature' must be accepted (no exception)
        async def _insert(dtype: str) -> None:
            await db_pool.execute(
                """
                INSERT INTO definitions
                    (definition_id, version, definition_type, namespace, body)
                VALUES
                    ($1, '1.0', $2, 'org', '{}')
                """,
                f"test_{dtype}",
                dtype,
            )

        # Should succeed
        run(_insert("feature"))

        # 'invalid_type' must be rejected by the CHECK constraint
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            run(_insert("invalid_type"))

    def test_guardrails_version_column_exists_on_tasks(
        self, db_pool, clean_tables, run
    ):
        """
        The guardrails_version column must exist on the tasks table (migration 0003).

        This is the column that bug 3-A was about: the column was added by
        migration 0003 but the INSERT in TaskStore.create() initially didn't
        include it in the VALUES list, so it was always NULL.
        """
        async def _columns() -> set[str]:
            rows = await db_pool.fetch(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'tasks'
                  AND table_schema = 'public'
                """
            )
            return {r["column_name"] for r in rows}

        cols = run(_columns())
        assert "guardrails_version" in cols, (
            "guardrails_version column missing from tasks table — "
            "migration 0003 may not have run."
        )
        assert "context_version" in cols, (
            "context_version column missing from tasks table — "
            "migration 0002 may not have run."
        )

    def test_concept_results_output_text_column_exists(
        self, db_pool, clean_tables, run
    ):
        """
        The output_text column must exist on concept_results (migration 0004).

        ConceptResultStore.store() uses output_text to store categorical string
        values separately from the numeric DOUBLE PRECISION value column.
        """
        async def _columns() -> set[str]:
            rows = await db_pool.fetch(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'concept_results'
                  AND table_schema = 'public'
                """
            )
            return {r["column_name"] for r in rows}

        cols = run(_columns())
        assert "output_text" in cols, (
            "output_text column missing from concept_results table — "
            "migration 0004 may not have run correctly."
        )
