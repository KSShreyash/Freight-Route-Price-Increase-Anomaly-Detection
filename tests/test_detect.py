import datetime as dt

import pytest

from ftanomaly import config
from ftanomaly.detect import detect, own_history_bar
from ftanomaly.loading import load_route_weeks
from ftanomaly.metrics import build_metrics


@pytest.fixture(scope="module")
def candidates():
    rws = load_route_weeks(config.PATHS.shipments)
    return detect(build_metrics(rws))


def test_flags_exactly_the_expected_count(candidates):
    assert len(candidates) == 21


def test_includes_every_sample_output_row(candidates):
    keys = {(c.metrics.route, c.metrics.week_of) for c in candidates}
    for route, week in [
        ("Delhi-Jaipur", "2024-11-11"),
        ("Ahmedabad-Mumbai", "2025-01-20"),
        ("Mumbai-Pune", "2025-09-15"),
    ]:
        assert (route, dt.date.fromisoformat(week)) in keys, (route, week)


def test_slow_drift_is_caught_by_peers_not_by_own_history(candidates):
    """Mumbai-Pune creeps up slowly, so its rolling baseline follows it and
    own-history stays quiet. Only the peer comparison sees the drift."""
    c = next(
        c for c in candidates
        if c.metrics.route == "Mumbai-Pune"
        and c.metrics.week_of == dt.date(2025, 9, 15)
    )
    assert c.peer_triggered is True
    assert c.own_history_triggered is False


def test_cost_drops_are_never_flagged(candidates):
    for c in candidates:
        rose = (c.metrics.vs_own_history or 0) > 0 or (c.metrics.vs_similar_routes or 0) > 0
        assert rose, (c.metrics.route, c.metrics.week_of)


def test_adaptive_bar_rises_for_a_volatile_route():
    """A route whose recent weeks are turbulent needs a bigger move to count."""
    from ftanomaly.metrics import WeekMetrics

    quiet = WeekMetrics("R", "Short", dt.date(2025, 1, 6), 3.0, 12.0, 0.0, 1.9, 8)
    noisy = WeekMetrics("R", "Short", dt.date(2025, 1, 6), 3.0, 12.0, 0.0, 6.3, 8)
    assert own_history_bar(quiet) == 10.0           # floor binds
    assert own_history_bar(noisy) == pytest.approx(18.9)  # 3 x MAD binds


def test_rows_without_enough_history_are_not_history_flagged():
    from ftanomaly.metrics import WeekMetrics

    warmup = WeekMetrics("R", "Short", dt.date(2024, 1, 1), 3.0, None, 0.0, None, 0)
    assert own_history_bar(warmup) is None


def test_output_is_sorted_deterministically(candidates):
    keys = [(c.metrics.week_of, c.metrics.route) for c in candidates]
    assert keys == sorted(keys)
