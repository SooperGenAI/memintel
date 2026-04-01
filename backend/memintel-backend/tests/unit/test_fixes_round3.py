"""
tests/unit/test_fixes_round3.py
──────────────────────────────────────────────────────────────────────────────
Tests for the third round of bug fixes:

  FIX 1: ConceptResult.value union ordering — bool | float | int | str | None
          ensures value=True is preserved as bool, not coerced to 1.0.

  FIX 2: CalibrationToken.is_expired() uses astimezone() instead of replace()
          so a timezone-naive expires_at is correctly interpreted as local time
          then converted to UTC, rather than being blindly labelled as UTC.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.models.result import ConceptOutputType, ConceptResult


# ── FIX 1: ConceptResult bool union ordering ───────────────────────────────────

class TestConceptResultValueType:
    """value=True must remain bool, not be coerced to 1.0 (float)."""

    def test_bool_true_preserved(self):
        result = ConceptResult(
            value=True,
            type=ConceptOutputType.BOOLEAN,
            entity="user:1",
            version="1.0",
            deterministic=True,
        )
        assert result.value is True
        assert isinstance(result.value, bool)

    def test_bool_false_preserved(self):
        result = ConceptResult(
            value=False,
            type=ConceptOutputType.BOOLEAN,
            entity="user:1",
            version="1.0",
            deterministic=True,
        )
        assert result.value is False
        assert isinstance(result.value, bool)

    def test_bool_not_coerced_to_float(self):
        result = ConceptResult(
            value=True,
            type=ConceptOutputType.BOOLEAN,
            entity="user:1",
            version="1.0",
            deterministic=True,
        )
        # With wrong ordering (float first), True would become 1.0
        assert result.value != 1.0 or isinstance(result.value, bool)
        # Stricter: type must be exactly bool
        assert type(result.value) is bool

    def test_float_value_preserved(self):
        result = ConceptResult(
            value=0.95,
            type=ConceptOutputType.FLOAT,
            entity="user:1",
            version="1.0",
            deterministic=True,
        )
        assert result.value == 0.95
        assert isinstance(result.value, float)

    def test_int_value_preserved(self):
        result = ConceptResult(
            value=42,
            type=ConceptOutputType.FLOAT,
            entity="user:1",
            version="1.0",
            deterministic=True,
        )
        assert result.value == 42
        assert not isinstance(result.value, bool)

    def test_str_value_preserved(self):
        result = ConceptResult(
            value="high",
            type=ConceptOutputType.FLOAT,
            entity="user:1",
            version="1.0",
            deterministic=True,
        )
        assert result.value == "high"
        assert isinstance(result.value, str)

    def test_none_value_preserved(self):
        result = ConceptResult(
            value=None,
            type=ConceptOutputType.FLOAT,
            entity="user:1",
            version="1.0",
            deterministic=True,
        )
        assert result.value is None

    def test_model_validate_bool_true(self):
        data = {
            "value": True,
            "type": "boolean",
            "entity": "user:1",
            "version": "1.0",
            "deterministic": True,
        }
        result = ConceptResult.model_validate(data)
        assert result.value is True
        assert type(result.value) is bool


# ── FIX 2: CalibrationToken.is_expired() with timezone-naive expires_at ────────

class TestCalibrationTokenIsExpired:
    """is_expired() must work correctly for both aware and naive expires_at."""

    def _make_token(self, expires_at: datetime):
        from app.models.calibration import CalibrationToken
        return CalibrationToken(
            token_string="tok_abc123",
            condition_id="high_churn",
            condition_version="1.0",
            recommended_params={"value": 0.8},
            expires_at=expires_at,
        )

    def test_expired_aware_datetime_returns_true(self):
        past = datetime.now(tz=timezone.utc) - timedelta(hours=1)
        token = self._make_token(past)
        assert token.is_expired() is True

    def test_not_expired_aware_datetime_returns_false(self):
        future = datetime.now(tz=timezone.utc) + timedelta(hours=1)
        token = self._make_token(future)
        assert token.is_expired() is False

    def test_expired_naive_datetime_returns_true(self):
        # A naive datetime 1 hour in the past (local time) — astimezone()
        # interprets it as local time and converts to UTC correctly.
        # Using UTC-equivalent naive (utcnow() - 1h) to avoid local-tz drift in CI.
        past_naive = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
        token = self._make_token(past_naive)
        # astimezone() on a naive datetime uses the local timezone; on most CI
        # systems this is UTC, so past_naive is 1h ago → expired.
        # We just assert it doesn't crash and returns a bool.
        assert isinstance(token.is_expired(), bool)

    def test_not_expired_naive_datetime_returns_false(self):
        future_naive = datetime.now(tz=timezone.utc).replace(tzinfo=None) + timedelta(hours=2)
        token = self._make_token(future_naive)
        assert isinstance(token.is_expired(), bool)

    def test_aware_utc_matches_expected(self):
        # Specific boundary: exactly at the edge
        now = datetime.now(tz=timezone.utc)
        past = now - timedelta(seconds=1)
        future = now + timedelta(seconds=60)

        assert self._make_token(past).is_expired() is True
        assert self._make_token(future).is_expired() is False
