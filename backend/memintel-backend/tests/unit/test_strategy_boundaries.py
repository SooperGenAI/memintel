"""
tests/unit/test_strategy_boundaries.py
──────────────────────────────────────────────────────────────────────────────
Comprehensive boundary and edge-case tests for all six strategy
implementations.  Every test exercises the strategy class directly — no HTTP
stack, no database, no service layer.

Ambiguous behaviours discovered while reading implementations
─────────────────────────────────────────────────────────────
1. All numeric strategies (threshold/percentile/z_score/change) use strict
   inequalities (> / <).  When the current value sits EXACTLY at the computed
   threshold, the condition does NOT fire.

2. EqualsStrategy null_input returns decision<categorical> with value=None —
   consistent with all non-null equals results (which are also CATEGORICAL).
   value=None signals that input was absent; reason='null_input' is also set.

3. EqualsStrategy non-match returns value="" (empty string) with
   decision_type=CATEGORICAL, NOT False with decision_type=BOOLEAN.

4. PercentileStrategy cutoff=0, direction='top': the 100th-percentile of the
   history equals the maximum value.  The condition fires only when
   current > max(history), not "never fires."

5. PercentileStrategy cutoff=100, direction='top': the 0th-percentile equals
   the minimum value.  The condition fires when current > min(history),
   meaning it fires for most values — not "always fires" (it still requires
   strictly above the minimum).

6. ZScoreStrategy with all-identical history produces std=0.  The strategy
   returns False without crashing.

7. ChangeStrategy with previous=0 returns False without crashing (division
   guard).

8. The service layer in execute.py guards _HISTORY_MIN_RESULTS=3 before
   calling strategy classes for z_score/percentile/change.  The strategy
   classes themselves do NOT enforce this floor — they evaluate with whatever
   history is provided (or return False for empty history).
"""
from __future__ import annotations

import pytest

from app.models.condition import DecisionType, DecisionValue
from app.models.errors import ErrorType, MemintelError
from app.models.result import ConceptOutputType, ConceptResult
from app.strategies.change import ChangeStrategy
from app.strategies.composite import CompositeStrategy
from app.strategies.equals import EqualsStrategy
from app.strategies.percentile import PercentileStrategy
from app.strategies.threshold import ThresholdStrategy
from app.strategies.z_score import ZScoreStrategy


# ── Shared helpers ────────────────────────────────────────────────────────────

def _r(value, rtype: str = "float") -> ConceptResult:
    """Build a ConceptResult with the given value and type."""
    return ConceptResult(
        value=value,
        type=ConceptOutputType(rtype),
        entity="user:1",
        version="1.0",
        deterministic=True,
    )


def _cat(value: str) -> ConceptResult:
    """Build a categorical ConceptResult."""
    return _r(value, rtype="categorical")


def _null() -> ConceptResult:
    """Build a ConceptResult with value=None."""
    return ConceptResult(
        value=None,
        type=ConceptOutputType("float"),
        entity="user:1",
        version="1.0",
        deterministic=True,
    )


def _hist(*values: float) -> list[ConceptResult]:
    """Build a history list of float ConceptResults."""
    return [_r(v) for v in values]


def _bool_decision(fired: bool) -> DecisionValue:
    """Build a boolean DecisionValue (for composite operands)."""
    return DecisionValue(
        value=fired,
        decision_type=DecisionType.BOOLEAN,
        condition_id="sub",
        condition_version="1.0",
        entity="user:1",
    )


# ── ThresholdStrategy boundaries ─────────────────────────────────────────────

class TestThresholdBoundaries:
    """Strict > / < inequalities and edge cases for ThresholdStrategy."""

    @pytest.fixture(autouse=True)
    def _strategy(self):
        self.s = ThresholdStrategy()

    def test_exactly_at_threshold_below_direction_does_not_fire(self):
        """
        direction='below', value=5.0, current=5.0 → NOT fired.
        Strict < means the boundary value itself does not cross the threshold.
        """
        r = self.s.evaluate(
            _r(5.0), [],
            {"direction": "below", "value": 5.0},
            condition_id="c", condition_version="1.0",
        )
        assert r.value is False

    def test_just_below_threshold_below_direction_fires(self):
        """
        direction='below', value=5.0, current=4.999 → fires.
        One epsilon below the threshold should fire.
        """
        r = self.s.evaluate(
            _r(4.999), [],
            {"direction": "below", "value": 5.0},
        )
        assert r.value is True

    def test_exactly_at_threshold_above_direction_does_not_fire(self):
        """
        direction='above', value=10.0, current=10.0 → NOT fired.
        Strict > means the boundary value does not fire.
        """
        r = self.s.evaluate(
            _r(10.0), [],
            {"direction": "above", "value": 10.0},
        )
        assert r.value is False

    def test_zero_threshold_zero_current_does_not_fire(self):
        """
        direction='above', value=0.0, current=0.0 → NOT fired (0 > 0 is False).
        """
        r = self.s.evaluate(
            _r(0.0), [],
            {"direction": "above", "value": 0.0},
        )
        assert r.value is False

    def test_zero_threshold_positive_current_fires(self):
        """
        direction='above', value=0.0, current=0.001 → fires (0.001 > 0).
        """
        r = self.s.evaluate(
            _r(0.001), [],
            {"direction": "above", "value": 0.0},
        )
        assert r.value is True

    def test_null_input_returns_false_with_reason(self):
        """
        result.value=None → does not fire; reason='null_input'.
        The strategy short-circuits before any threshold comparison.
        """
        r = self.s.evaluate(
            _null(), [],
            {"direction": "above", "value": 5.0},
            condition_id="c", condition_version="1.0",
        )
        assert r.value is False
        assert r.reason == "null_input"

    def test_null_input_decision_type_is_boolean(self):
        """
        null_input result still produces decision<boolean> (consistent with
        all other strategies).
        """
        r = self.s.evaluate(
            _null(), [],
            {"direction": "above", "value": 5.0},
        )
        assert r.decision_type == DecisionType.BOOLEAN

    def test_reason_is_none_on_normal_fire(self):
        """
        When the condition fires normally, reason must be None (not set).
        """
        r = self.s.evaluate(
            _r(20.0), [],
            {"direction": "above", "value": 10.0},
        )
        assert r.value is True
        assert r.reason is None

    def test_reason_is_none_on_normal_no_fire(self):
        """
        When the condition does not fire normally, reason must be None.
        """
        r = self.s.evaluate(
            _r(5.0), [],
            {"direction": "above", "value": 10.0},
        )
        assert r.value is False
        assert r.reason is None

    def test_history_is_ignored(self):
        """
        ThresholdStrategy does not use history — providing values has no effect.
        Result depends only on current value and threshold.
        """
        r_with = self.s.evaluate(
            _r(15.0), _hist(100.0, 200.0),
            {"direction": "above", "value": 10.0},
        )
        r_without = self.s.evaluate(
            _r(15.0), [],
            {"direction": "above", "value": 10.0},
        )
        assert r_with.value == r_without.value is True

    def test_very_large_positive_threshold(self):
        """Current below sys.maxsize threshold — does not fire."""
        r = self.s.evaluate(
            _r(1e300), [],
            {"direction": "above", "value": 1e308},
        )
        assert r.value is False

    def test_negative_current_below_negative_threshold(self):
        """
        direction='above', value=-10.0, current=-5.0 → fires (-5 > -10).
        """
        r = self.s.evaluate(
            _r(-5.0), [],
            {"direction": "above", "value": -10.0},
        )
        assert r.value is True

    def test_negative_current_at_negative_threshold_does_not_fire(self):
        """
        direction='above', value=-10.0, current=-10.0 → NOT fired (strict >).
        """
        r = self.s.evaluate(
            _r(-10.0), [],
            {"direction": "above", "value": -10.0},
        )
        assert r.value is False


# ── PercentileStrategy boundaries ─────────────────────────────────────────────

class TestPercentileBoundaries:
    """Rank-based percentile edge cases."""

    @pytest.fixture(autouse=True)
    def _strategy(self):
        self.s = PercentileStrategy()

    def test_current_exactly_at_computed_percentile_does_not_fire_top(self):
        """
        direction='top', value=50 (top 50%) with history=[1,2,3,4,5].
        50th-percentile of [1,2,3,4,5] = 3.0.
        current=3.0 → 3.0 > 3.0 is False → does NOT fire.
        """
        r = self.s.evaluate(
            _r(3.0), _hist(1.0, 2.0, 3.0, 4.0, 5.0),
            {"direction": "top", "value": 50},
        )
        assert r.value is False

    def test_current_exactly_at_computed_percentile_does_not_fire_bottom(self):
        """
        direction='bottom', value=50 with history=[1,2,3,4,5].
        50th-percentile = 3.0. current=3.0 → 3.0 < 3.0 is False → does NOT fire.
        """
        r = self.s.evaluate(
            _r(3.0), _hist(1.0, 2.0, 3.0, 4.0, 5.0),
            {"direction": "bottom", "value": 50},
        )
        assert r.value is False

    def test_cutoff_zero_top_fires_only_above_max(self):
        """
        direction='top', value=0 → threshold = percentile(100) = max(history).
        current must be STRICTLY greater than max to fire.
        current=100.0, max=10.0 → fires.
        """
        r = self.s.evaluate(
            _r(100.0), _hist(1.0, 5.0, 10.0),
            {"direction": "top", "value": 0},
        )
        assert r.value is True

    def test_cutoff_zero_top_at_max_does_not_fire(self):
        """
        direction='top', value=0, current=max(history)=10.0 → does NOT fire.
        current > percentile(100) means current > max, which is False at max.
        """
        r = self.s.evaluate(
            _r(10.0), _hist(1.0, 5.0, 10.0),
            {"direction": "top", "value": 0},
        )
        assert r.value is False

    def test_cutoff_zero_bottom_fires_only_below_min(self):
        """
        direction='bottom', value=0 → threshold = percentile(0) = min(history).
        current must be STRICTLY less than min to fire.
        current=0.0, min=1.0 → fires.
        """
        r = self.s.evaluate(
            _r(0.0), _hist(1.0, 5.0, 10.0),
            {"direction": "bottom", "value": 0},
        )
        assert r.value is True

    def test_cutoff_zero_bottom_at_min_does_not_fire(self):
        """
        direction='bottom', value=0, current=min(history)=1.0 → does NOT fire.
        """
        r = self.s.evaluate(
            _r(1.0), _hist(1.0, 5.0, 10.0),
            {"direction": "bottom", "value": 0},
        )
        assert r.value is False

    def test_cutoff_100_top_fires_above_min(self):
        """
        direction='top', value=100 → threshold = percentile(0) = min(history) = 1.0.
        Any current > 1.0 fires.  current=2.0 → fires.
        (cutoff=100 does NOT mean 'always fires' — it still requires > min.)
        """
        r = self.s.evaluate(
            _r(2.0), _hist(1.0, 5.0, 10.0),
            {"direction": "top", "value": 100},
        )
        assert r.value is True

    def test_cutoff_100_top_at_min_does_not_fire(self):
        """
        direction='top', value=100, current=min(history)=1.0 → does NOT fire.
        Strict > means exactly at min does not fire.
        """
        r = self.s.evaluate(
            _r(1.0), _hist(1.0, 5.0, 10.0),
            {"direction": "top", "value": 100},
        )
        assert r.value is False

    def test_all_identical_history_top_does_not_fire_at_history_value(self):
        """
        All history values identical → percentile(any p) = that value.
        current = history value → strict > fails → does NOT fire.
        """
        r = self.s.evaluate(
            _r(5.0), _hist(5.0, 5.0, 5.0, 5.0),
            {"direction": "top", "value": 25},
        )
        assert r.value is False

    def test_all_identical_history_top_fires_above_history_value(self):
        """
        All history values identical = 5.0. current=5.001 → fires (strict >).
        """
        r = self.s.evaluate(
            _r(5.001), _hist(5.0, 5.0, 5.0, 5.0),
            {"direction": "top", "value": 25},
        )
        assert r.value is True

    def test_empty_history_returns_false(self):
        """
        Empty history → cannot compute percentile → does NOT fire.
        No error is raised.
        """
        r = self.s.evaluate(
            _r(100.0), [],
            {"direction": "top", "value": 10},
        )
        assert r.value is False

    def test_single_history_item_top(self):
        """
        Single-item history: percentile(any p) = that single value.
        direction='top', value=10, history=[5.0], current=6.0 → fires (6 > 5).
        Strategy class handles single-item history without error.
        """
        r = self.s.evaluate(
            _r(6.0), _hist(5.0),
            {"direction": "top", "value": 10},
        )
        assert r.value is True

    def test_null_input_returns_false_with_reason(self):
        """result.value=None → does not fire; reason='null_input'."""
        r = self.s.evaluate(
            _null(), _hist(1.0, 2.0, 3.0),
            {"direction": "top", "value": 25},
        )
        assert r.value is False
        assert r.reason == "null_input"

    def test_two_history_items_linear_interpolation(self):
        """
        history=[2.0, 8.0], value=50 (top 50%).
        percentile(50) of [2.0, 8.0] = 2.0*0.5 + 8.0*0.5 = 5.0.
        current=5.0 → 5.0 > 5.0 → False.
        current=5.001 → True.
        """
        s = self.s
        params = {"direction": "top", "value": 50}
        hist = _hist(2.0, 8.0)

        assert s.evaluate(_r(5.0), hist, params).value is False
        assert s.evaluate(_r(5.001), hist, params).value is True


# ── ZScoreStrategy boundaries ─────────────────────────────────────────────────

class TestZScoreBoundaries:
    """Standard-deviation and z-score edge cases."""

    @pytest.fixture(autouse=True)
    def _strategy(self):
        self.s = ZScoreStrategy()

    def test_std_zero_all_identical_returns_false_no_crash(self):
        """
        All history values identical → std=0 → z is undefined.
        Strategy returns False without raising an exception.
        """
        r = self.s.evaluate(
            _r(5.0), _hist(3.0, 3.0, 3.0, 3.0, 3.0),
            {"direction": "above", "threshold": 1.5, "window": "30d"},
        )
        assert r.value is False

    def test_z_exactly_at_threshold_does_not_fire_above(self):
        """
        direction='above': fires when z > threshold (strict >).
        With history=[0.0]*n and current=mean+threshold*std, z equals threshold
        exactly, so it does NOT fire.
        history=[0,2,4], mean=2.0, std=sqrt(8/3)≈1.633.
        threshold=1.0. current = mean + 1.0*std = 2.0 + 1.633 = 3.633.
        z = (3.633 - 2.0) / 1.633 = 1.0 → 1.0 > 1.0 is False.
        """
        import math
        values = [0.0, 2.0, 4.0]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance)
        current_at_z1 = mean + 1.0 * std  # z exactly 1.0

        r = self.s.evaluate(
            _r(current_at_z1), _hist(*values),
            {"direction": "above", "threshold": 1.0, "window": "30d"},
        )
        assert r.value is False

    def test_just_above_threshold_fires_above(self):
        """
        Same setup as above but current is epsilon above the z=threshold crossing.
        """
        import math
        values = [0.0, 2.0, 4.0]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance)
        current_above_z1 = mean + 1.0 * std + 0.001

        r = self.s.evaluate(
            _r(current_above_z1), _hist(*values),
            {"direction": "above", "threshold": 1.0, "window": "30d"},
        )
        assert r.value is True

    def test_direction_below_fires_when_z_negative_enough(self):
        """
        direction='below': fires when z < -threshold (not when z < threshold).
        history=[2,2,2,2,10], current=-100 → large negative z → fires.
        """
        r = self.s.evaluate(
            _r(-100.0), _hist(2.0, 2.0, 2.0, 2.0, 10.0),
            {"direction": "below", "threshold": 1.5, "window": "30d"},
        )
        assert r.value is True

    def test_direction_below_does_not_fire_on_positive_z(self):
        """
        direction='below': a large POSITIVE z does NOT fire (only negative z fires).
        current = very large positive value → z >> 0 → does NOT fire for 'below'.
        """
        r = self.s.evaluate(
            _r(1000.0), _hist(2.0, 2.0, 2.0, 2.0, 10.0),
            {"direction": "below", "threshold": 1.5, "window": "30d"},
        )
        assert r.value is False

    def test_direction_any_fires_on_large_negative_z(self):
        """
        direction='any': fires when abs(z) > threshold.
        Very negative current → abs(z) large → fires.
        """
        r = self.s.evaluate(
            _r(-1000.0), _hist(0.0, 1.0, 2.0, 3.0, 4.0),
            {"direction": "any", "threshold": 2.0, "window": "30d"},
        )
        assert r.value is True

    def test_direction_any_does_not_fire_at_zero_z(self):
        """
        direction='any': current equals the history mean → z=0.
        0 > threshold (>0) is False → does NOT fire.
        """
        values = [2.0, 4.0, 6.0]
        mean = sum(values) / len(values)  # 4.0
        r = self.s.evaluate(
            _r(mean), _hist(*values),
            {"direction": "any", "threshold": 0.5, "window": "30d"},
        )
        assert r.value is False

    def test_empty_history_returns_false(self):
        """Empty history → cannot compute z → does NOT fire."""
        r = self.s.evaluate(
            _r(100.0), [],
            {"direction": "above", "threshold": 1.5, "window": "30d"},
        )
        assert r.value is False

    def test_null_input_returns_false_with_reason(self):
        """result.value=None → does not fire; reason='null_input'."""
        r = self.s.evaluate(
            _null(), _hist(1.0, 2.0, 3.0, 4.0, 5.0),
            {"direction": "above", "threshold": 1.5, "window": "30d"},
        )
        assert r.value is False
        assert r.reason == "null_input"

    def test_single_history_item_returns_false(self):
        """
        Single-item history: std=0 (only one data point, zero spread).
        Strategy returns False without error.
        """
        r = self.s.evaluate(
            _r(100.0), _hist(5.0),
            {"direction": "above", "threshold": 1.5, "window": "30d"},
        )
        assert r.value is False

    def test_reason_none_on_normal_fire(self):
        """When the condition fires, reason must be None."""
        r = self.s.evaluate(
            _r(100.0), _hist(0.0, 1.0, 2.0, 1.0, 0.0),
            {"direction": "above", "threshold": 1.0, "window": "30d"},
        )
        assert r.value is True
        assert r.reason is None


# ── ChangeStrategy boundaries ─────────────────────────────────────────────────

class TestChangeBoundaries:
    """Percentage-change edge cases."""

    @pytest.fixture(autouse=True)
    def _strategy(self):
        self.s = ChangeStrategy()

    def test_previous_zero_returns_false_no_crash(self):
        """
        history[-1].value == 0 → division by zero guard → does NOT fire.
        No ZeroDivisionError raised.
        """
        r = self.s.evaluate(
            _r(5.0), _hist(1.0, 2.0, 0.0),   # last item = 0
            {"direction": "increase", "value": 0.1, "window": "7d"},
        )
        assert r.value is False

    def test_exactly_at_change_threshold_does_not_fire_increase(self):
        """
        direction='increase', value=0.5 (50%).
        previous=10.0, current=15.0 → pct_change = 0.5 exactly.
        0.5 > 0.5 is False → does NOT fire.
        """
        r = self.s.evaluate(
            _r(15.0), _hist(5.0, 10.0),      # previous = history[-1] = 10.0
            {"direction": "increase", "value": 0.5, "window": "7d"},
        )
        assert r.value is False

    def test_just_above_change_threshold_fires(self):
        """
        direction='increase', value=0.5. previous=10.0, current=15.001.
        pct_change = 0.5001 > 0.5 → fires.
        """
        r = self.s.evaluate(
            _r(15.001), _hist(5.0, 10.0),
            {"direction": "increase", "value": 0.5, "window": "7d"},
        )
        assert r.value is True

    def test_decrease_fires_on_large_drop(self):
        """
        direction='decrease', value=0.3 (30%).
        previous=10.0, current=6.0 → pct_change = (6-10)/10 = -0.4.
        -0.4 < -0.3 → fires.
        """
        r = self.s.evaluate(
            _r(6.0), _hist(5.0, 10.0),
            {"direction": "decrease", "value": 0.3, "window": "7d"},
        )
        assert r.value is True

    def test_decrease_does_not_fire_on_increase(self):
        """
        direction='decrease', current > previous → positive pct_change → does NOT fire.
        """
        r = self.s.evaluate(
            _r(15.0), _hist(5.0, 10.0),
            {"direction": "decrease", "value": 0.1, "window": "7d"},
        )
        assert r.value is False

    def test_any_fires_on_large_increase(self):
        """
        direction='any': abs(pct_change) > threshold.
        previous=10.0, current=20.0 → pct_change=1.0 → abs=1.0 > 0.5 → fires.
        """
        r = self.s.evaluate(
            _r(20.0), _hist(10.0),
            {"direction": "any", "value": 0.5, "window": "7d"},
        )
        assert r.value is True

    def test_any_fires_on_large_decrease(self):
        """
        direction='any': large negative change also fires.
        previous=10.0, current=3.0 → pct_change=-0.7 → abs=0.7 > 0.5 → fires.
        """
        r = self.s.evaluate(
            _r(3.0), _hist(10.0),
            {"direction": "any", "value": 0.5, "window": "7d"},
        )
        assert r.value is True

    def test_uses_most_recent_history_item_as_previous(self):
        """
        history[-1] is the most recent value used as 'previous'.
        history=[2.0, 100.0], previous=100.0.
        current=101.0 → pct_change=0.01 → < threshold 0.5 → does NOT fire.
        Would fire if previous=2.0: (101-2)/2=49.5 > 0.5.
        """
        r = self.s.evaluate(
            _r(101.0), _hist(2.0, 100.0),    # previous = 100.0 (last)
            {"direction": "increase", "value": 0.5, "window": "7d"},
        )
        assert r.value is False   # confirms history[-1] is used

    def test_negative_previous_abs_in_denominator(self):
        """
        previous is negative: pct_change uses abs(previous) in denominator.
        previous=-10.0, current=-5.0 → pct_change = (-5 - (-10)) / 10 = 0.5.
        0.5 > 0.3 → fires (direction='increase').
        """
        r = self.s.evaluate(
            _r(-5.0), _hist(-10.0),
            {"direction": "increase", "value": 0.3, "window": "7d"},
        )
        assert r.value is True

    def test_empty_history_returns_false(self):
        """Empty history → no previous → does NOT fire."""
        r = self.s.evaluate(
            _r(10.0), [],
            {"direction": "increase", "value": 0.1, "window": "7d"},
        )
        assert r.value is False

    def test_null_input_returns_false_with_reason(self):
        """result.value=None → does not fire; reason='null_input'."""
        r = self.s.evaluate(
            _null(), _hist(1.0, 2.0, 3.0),
            {"direction": "increase", "value": 0.1, "window": "7d"},
        )
        assert r.value is False
        assert r.reason == "null_input"

    def test_zero_threshold_fires_on_any_nonzero_change(self):
        """
        value=0.0 → threshold=0.0. pct_change > 0.0 fires on any increase.
        previous=10.0, current=10.001 → pct_change=0.0001 > 0.0 → fires.
        """
        r = self.s.evaluate(
            _r(10.001), _hist(10.0),
            {"direction": "increase", "value": 0.0, "window": "7d"},
        )
        assert r.value is True

    def test_zero_threshold_does_not_fire_when_unchanged(self):
        """
        value=0.0, previous=10.0, current=10.0 → pct_change=0.0 → 0.0 > 0.0 False.
        """
        r = self.s.evaluate(
            _r(10.0), _hist(10.0),
            {"direction": "increase", "value": 0.0, "window": "7d"},
        )
        assert r.value is False


# ── EqualsStrategy boundaries ─────────────────────────────────────────────────

class TestEqualsBoundaries:
    """Categorical match edge cases."""

    @pytest.fixture(autouse=True)
    def _strategy(self):
        self.s = EqualsStrategy()

    def test_null_input_returns_categorical_with_none_value(self):
        """
        BOUNDARY: EqualsStrategy null_input returns decision<categorical> with
        value=None — consistent with all non-null equals results (which are also
        CATEGORICAL).  This differs from numeric strategies, where null_input
        returns decision<boolean> with value=False.

        value=None distinguishes null-input from a non-match (value="") so
        callers can tell WHY no label was produced.
        """
        r = self.s.evaluate(
            _null(), [],
            {"value": "high"},
        )
        assert r.value is None
        assert r.decision_type == DecisionType.CATEGORICAL

    def test_null_input_has_null_input_reason(self):
        """null_input path sets reason='null_input'."""
        r = self.s.evaluate(_null(), [], {"value": "high"})
        assert r.reason == "null_input"

    def test_non_match_returns_empty_string_not_false(self):
        """
        BOUNDARY: When the label does not match, EqualsStrategy returns
        value="" (empty string) with decision_type=CATEGORICAL — NOT False
        with decision_type=BOOLEAN.
        """
        r = self.s.evaluate(
            _cat("low"), [],
            {"value": "high"},
        )
        assert r.value == ""
        assert r.decision_type == DecisionType.CATEGORICAL

    def test_match_returns_the_label_string(self):
        """
        On a match, value is the matched label string, not True.
        decision_type remains CATEGORICAL.
        """
        r = self.s.evaluate(
            _cat("high"), [],
            {"value": "high"},
        )
        assert r.value == "high"
        assert isinstance(r.value, str)
        assert r.decision_type == DecisionType.CATEGORICAL

    def test_case_sensitive_no_match(self):
        """
        "High" and "high" are different strings — case-sensitive comparison.
        result.value="High", target="high" → does NOT match → returns "".
        """
        r = self.s.evaluate(
            _cat("High"), [],
            {"value": "high"},
        )
        assert r.value == ""

    def test_case_sensitive_match(self):
        """
        Exact case match → fires.  "HIGH"=="HIGH" → returns "HIGH".
        """
        r = self.s.evaluate(
            _cat("HIGH"), [],
            {"value": "HIGH"},
        )
        assert r.value == "HIGH"

    def test_whitespace_is_significant(self):
        """
        " high" (leading space) != "high" → does NOT match.
        """
        r = self.s.evaluate(
            _cat(" high"), [],
            {"value": "high"},
        )
        assert r.value == ""

    def test_empty_string_target_matches_empty_result(self):
        """
        Target value="" and result.value="" → both empty strings → matches.
        Returns "" which is the matched label (and also the non-match sentinel),
        but the path is through _categorical_decision with the matched label.
        """
        r = self.s.evaluate(
            _cat(""), [],
            {"value": ""},
        )
        # "" matches "" — fired_label == target == ""
        assert r.value == ""
        assert r.decision_type == DecisionType.CATEGORICAL

    def test_reason_is_none_on_match(self):
        """reason must be None when the condition fires normally."""
        r = self.s.evaluate(
            _cat("active"), [],
            {"value": "active"},
        )
        assert r.reason is None

    def test_reason_is_none_on_non_match(self):
        """reason must be None on a non-match (empty string result)."""
        r = self.s.evaluate(
            _cat("inactive"), [],
            {"value": "active"},
        )
        assert r.reason is None

    def test_labels_set_value_not_in_labels_raises_type_error(self):
        """
        result.value is not in the declared labels set → type_error.
        This is a label closure violation — the concept produced a label that
        was not in the defined label space.
        """
        with pytest.raises(MemintelError) as exc:
            self.s.evaluate(
                _cat("unknown"), [],
                {"value": "high", "labels": ["low", "medium", "high"]},
            )
        assert exc.value.error_type == ErrorType.TYPE_ERROR

    def test_labels_set_match_and_in_labels_fires(self):
        """result.value in labels AND equals target → fires."""
        r = self.s.evaluate(
            _cat("high"), [],
            {"value": "high", "labels": ["low", "medium", "high"]},
        )
        assert r.value == "high"


# ── CompositeStrategy boundaries ─────────────────────────────────────────────

class TestCompositeBoundaries:
    """Logical operator edge cases."""

    @pytest.fixture(autouse=True)
    def _strategy(self):
        self.s = CompositeStrategy()

    def _params(self, operator: str, *fired_flags: bool) -> dict:
        """Build params dict with operand_results as pre-evaluated DecisionValues."""
        operands = [_bool_decision(f) for f in fired_flags]
        return {"operator": operator, "operand_results": operands}

    def test_and_all_true_fires(self):
        """AND: all operands True → fires."""
        params = self._params("AND", True, True, True)
        r = self.s.evaluate(_null(), [], params)
        assert r.value is True

    def test_and_one_false_does_not_fire(self):
        """AND: any False operand → does NOT fire."""
        params = self._params("AND", True, False, True)
        r = self.s.evaluate(_null(), [], params)
        assert r.value is False

    def test_and_all_false_does_not_fire(self):
        """AND: all False → does NOT fire."""
        params = self._params("AND", False, False)
        r = self.s.evaluate(_null(), [], params)
        assert r.value is False

    def test_or_all_false_does_not_fire(self):
        """OR: all False → does NOT fire."""
        params = self._params("OR", False, False, False)
        r = self.s.evaluate(_null(), [], params)
        assert r.value is False

    def test_or_one_true_fires(self):
        """OR: any True operand → fires."""
        params = self._params("OR", False, True, False)
        r = self.s.evaluate(_null(), [], params)
        assert r.value is True

    def test_or_all_true_fires(self):
        """OR: all True → fires."""
        params = self._params("OR", True, True)
        r = self.s.evaluate(_null(), [], params)
        assert r.value is True

    def test_not_operator_raises_semantic_error(self):
        """
        BOUNDARY: CompositeOperator only has AND and OR — there is no NOT.
        Passing operator='NOT' must raise MemintelError(SEMANTIC_ERROR).
        operand_results must still be present so the operator check is reached.
        """
        params = self._params("AND", True, False)
        params["operator"] = "NOT"
        with pytest.raises(MemintelError) as exc:
            self.s.evaluate(_null(), [], params)
        assert exc.value.error_type == ErrorType.SEMANTIC_ERROR

    def test_two_operands_minimum_is_accepted(self):
        """
        Two operands is the minimum valid case — confirms no error is raised.
        """
        params = self._params("AND", True, True)
        r = self.s.evaluate(_null(), [], params)
        assert r.value is True

    def test_result_decision_type_is_boolean(self):
        """
        Composite always returns decision<boolean> regardless of operand values.
        """
        params = self._params("OR", True, False)
        r = self.s.evaluate(_null(), [], params)
        assert r.decision_type == DecisionType.BOOLEAN

    def test_categorical_operand_raises_type_error(self):
        """
        EqualsStrategy produces decision<categorical>. Using it as a composite
        operand must raise MemintelError(TYPE_ERROR).
        """
        categorical_operand = DecisionValue(
            value="high",
            decision_type=DecisionType.CATEGORICAL,
            condition_id="seg",
            condition_version="1.0",
            entity="user:1",
        )
        params = {
            "operator": "AND",
            "operand_results": [_bool_decision(True), categorical_operand],
        }
        with pytest.raises(MemintelError) as exc:
            self.s.evaluate(_null(), [], params)
        assert exc.value.error_type == ErrorType.TYPE_ERROR

    def test_reason_none_on_fire(self):
        """reason must be None when composite condition fires."""
        params = self._params("AND", True, True)
        r = self.s.evaluate(_null(), [], params)
        assert r.reason is None

    def test_reason_none_on_no_fire(self):
        """reason must be None when composite condition does not fire."""
        params = self._params("AND", True, False)
        r = self.s.evaluate(_null(), [], params)
        assert r.reason is None


# ── Cross-strategy invariants ─────────────────────────────────────────────────

class TestCrossStrategyInvariants:
    """Invariants that must hold across all strategy implementations."""

    @pytest.mark.parametrize("strategy_cls,result,history,params", [
        (
            ThresholdStrategy,
            _r(20.0),
            [],
            {"direction": "above", "value": 10.0},
        ),
        (
            PercentileStrategy,
            _r(10.0),
            _hist(1.0, 2.0, 3.0, 4.0, 5.0),
            {"direction": "top", "value": 25},
        ),
        (
            ZScoreStrategy,
            _r(50.0),
            _hist(0.0, 1.0, 2.0, 1.0, 0.0),
            {"direction": "above", "threshold": 1.0, "window": "30d"},
        ),
        (
            ChangeStrategy,
            _r(20.0),
            _hist(10.0),
            {"direction": "increase", "value": 0.5, "window": "7d"},
        ),
    ])
    def test_reason_is_none_on_normal_fire(self, strategy_cls, result, history, params):
        """When a numeric strategy fires normally, reason must be None."""
        s = strategy_cls()
        r = s.evaluate(result, history, params)
        assert r.value is True
        assert r.reason is None, (
            f"{strategy_cls.__name__}: expected reason=None on fire, got {r.reason!r}"
        )

    @pytest.mark.parametrize("strategy_cls,null_params", [
        (ThresholdStrategy, {"direction": "above", "value": 5.0}),
        (PercentileStrategy, {"direction": "top", "value": 25}),
        (ZScoreStrategy, {"direction": "above", "threshold": 1.5, "window": "30d"}),
        (ChangeStrategy, {"direction": "increase", "value": 0.1, "window": "7d"}),
    ])
    def test_null_input_always_returns_false_with_null_input_reason(
        self, strategy_cls, null_params
    ):
        """
        All numeric strategies: result.value=None → value=False, reason='null_input'.
        This is the null_input short-circuit before any type check.
        """
        s = strategy_cls()
        r = s.evaluate(_null(), [], null_params)
        assert r.value is False
        assert r.reason == "null_input", (
            f"{strategy_cls.__name__}: expected reason='null_input', got {r.reason!r}"
        )

    @pytest.mark.parametrize("strategy_cls,null_params", [
        (ThresholdStrategy, {"direction": "above", "value": 5.0}),
        (PercentileStrategy, {"direction": "top", "value": 25}),
        (ZScoreStrategy, {"direction": "above", "threshold": 1.5, "window": "30d"}),
        (ChangeStrategy, {"direction": "increase", "value": 0.1, "window": "7d"}),
    ])
    def test_numeric_null_input_produces_boolean_decision(self, strategy_cls, null_params):
        """
        Numeric strategies (threshold/percentile/z_score/change): null_input path
        produces decision<boolean> with value=False.
        """
        s = strategy_cls()
        r = s.evaluate(_null(), [], null_params)
        assert r.decision_type == DecisionType.BOOLEAN, (
            f"{strategy_cls.__name__}: expected BOOLEAN on null_input, "
            f"got {r.decision_type!r}"
        )

    def test_equals_null_input_produces_categorical_decision(self):
        """
        EqualsStrategy: null_input produces decision<categorical> with value=None.
        This is different from numeric strategies (which return BOOLEAN/False) and
        reflects that equals always returns CATEGORICAL regardless of input state.
        """
        s = EqualsStrategy()
        r = s.evaluate(_null(), [], {"value": "high"})
        assert r.decision_type == DecisionType.CATEGORICAL, (
            f"EqualsStrategy: expected CATEGORICAL on null_input, "
            f"got {r.decision_type!r}"
        )
        assert r.value is None

    @pytest.mark.parametrize("strategy_cls,fired_result,history,params", [
        (
            ThresholdStrategy,
            _r(20.0),
            [],
            {"direction": "above", "value": 10.0},
        ),
        (
            PercentileStrategy,
            _r(10.0),
            _hist(1.0, 2.0, 3.0, 4.0, 5.0),
            {"direction": "top", "value": 25},
        ),
    ])
    def test_determinism_same_inputs_same_output(
        self, strategy_cls, fired_result, history, params
    ):
        """
        Strategy evaluation is deterministic: identical inputs always produce
        the same output.  Run the same evaluation 3 times and confirm identity.
        """
        s = strategy_cls()
        results = [s.evaluate(fired_result, history, params) for _ in range(3)]
        values = [r.value for r in results]
        assert len(set(values)) == 1, (
            f"{strategy_cls.__name__}: non-deterministic output across calls: {values}"
        )
