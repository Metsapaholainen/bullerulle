"""The core look-ahead-bias fix, directly tested: fundamentals_as_of(quarters,
D) must never see a quarter reported after D. This is the fix for the
parent project's own documented look-ahead-bias risk in
data/fundamentals_cache.py (which stores only "most recently known" growth,
not what was actually knowable ON THE SIGNAL DATE).
"""
from __future__ import annotations

from bullarullaep.point_in_time_fundamentals import fundamentals_as_of

# Five quarters, most-recent-first, each with a clean 2x eps/revenue growth
# step-up vs. the quarter 4 back so growth% differs cleanly pre/post the
# newest report -- report dates spaced ~3 months apart, plausible EPS-beat
# figures on each.
QUARTERS = [
    {"report_date": "2026-05-20", "eps_actual": 2.00, "eps_estimated": 1.80, "revenue_actual": 200.0, "revenue_estimated": 190.0},
    {"report_date": "2026-02-20", "eps_actual": 1.50, "eps_estimated": 1.40, "revenue_actual": 150.0, "revenue_estimated": 145.0},
    {"report_date": "2025-11-20", "eps_actual": 1.30, "eps_estimated": 1.25, "revenue_actual": 130.0, "revenue_estimated": 128.0},
    {"report_date": "2025-08-20", "eps_actual": 1.20, "eps_estimated": 1.15, "revenue_actual": 120.0, "revenue_estimated": 118.0},
    {"report_date": "2025-05-20", "eps_actual": 1.00, "eps_estimated": 0.95, "revenue_actual": 100.0, "revenue_estimated": 98.0},
]


def test_day_before_latest_report_does_not_see_it():
    before = fundamentals_as_of(QUARTERS, "2026-05-19")
    assert before["quarters_known_as_of"] == 4
    # Latest visible beat should be the 2026-02-20 quarter, not 2026-05-20's.
    assert before["eps_beat_date"] == "2026-02-20"
    assert before["eps_beat_pct"] is not None


def test_report_date_itself_does_see_it():
    on_day = fundamentals_as_of(QUARTERS, "2026-05-20")
    assert on_day["quarters_known_as_of"] == 5
    assert on_day["eps_beat_date"] == "2026-05-20"


def test_growth_pct_changes_across_the_look_ahead_boundary():
    """Not just "different quarter counted" -- the actual growth % computed
    (quarter vs. same quarter 4 back) must differ before vs. on the report
    date, since a different "current" quarter is used in each case."""
    before = fundamentals_as_of(QUARTERS, "2026-05-19")
    on_day = fundamentals_as_of(QUARTERS, "2026-05-20")
    assert before["eps_growth_pct"] != on_day["eps_growth_pct"]
    assert before["revenue_growth_pct"] != on_day["revenue_growth_pct"]


def test_no_quarters_known_returns_all_none():
    far_past = fundamentals_as_of(QUARTERS, "2020-01-01")
    assert far_past["quarters_known_as_of"] == 0
    assert far_past["revenue_growth_pct"] is None
    assert far_past["eps_growth_pct"] is None
    assert far_past["eps_beat_pct"] is None
    assert far_past["eps_beat_date"] is None


def test_fewer_than_five_known_quarters_skips_growth_but_keeps_beat():
    """With only 2 quarters known, YoY growth (needs 5) can't be computed,
    but the most recent quarter's EPS-beat still can be."""
    result = fundamentals_as_of(QUARTERS, "2025-11-21")
    assert result["quarters_known_as_of"] == 3
    assert result["revenue_growth_pct"] is None
    assert result["eps_growth_pct"] is None
    assert result["eps_beat_pct"] is not None
    assert result["eps_beat_date"] == "2025-11-20"


def test_quarters_missing_report_date_are_ignored():
    quarters_with_gap = QUARTERS + [
        {"report_date": None, "eps_actual": 99.0, "eps_estimated": 1.0, "revenue_actual": 999.0, "revenue_estimated": 1.0}
    ]
    result = fundamentals_as_of(quarters_with_gap, "2026-05-20")
    # The report_date=None row must never be counted or leak into "latest".
    assert result["quarters_known_as_of"] == 5
    assert result["eps_beat_date"] == "2026-05-20"
