import datetime as dt

from ftanomaly.detect import Candidate
from ftanomaly.metrics import WeekMetrics
from ftanomaly.notes import EnrichedNote, Note
from ftanomaly.verdict import note_applies


def _candidate(route, week, own=True, peer=False):
    m = WeekMetrics(route, "Medium", week, 3.0, 25.0, 5.0, 2.5, 8)
    return Candidate(metrics=m, own_history_triggered=own, peer_triggered=peer)


def _note(nid, applies_to, direction, scope, start, end, in_dataset=True):
    n = Note(nid, dt.date.fromisoformat(start), applies_to, "text")
    return EnrichedNote(
        note=n, direction=direction, scope=scope,
        window_start=dt.date.fromisoformat(start),
        window_end=dt.date.fromisoformat(end) if end else None,
        in_dataset=in_dataset, magnitude_pct=None,
    )


def test_matching_route_inside_window_applies():
    e = _note("N001", "Chennai-Bangalore", "raises", "route", "2025-02-24", "2025-03-08")
    c = _candidate("Chennai-Bangalore", dt.date(2025, 3, 3))
    assert note_applies(e, c) is True


def test_week_starting_after_the_window_closes_does_not_apply():
    """N001's disruption ended 2025-03-08. The week of 2025-03-10 is still
    elevated but has no cover - it must come out unexplained."""
    e = _note("N001", "Chennai-Bangalore", "raises", "route", "2025-02-24", "2025-03-08")
    c = _candidate("Chennai-Bangalore", dt.date(2025, 3, 10))
    assert note_applies(e, c) is False


def test_week_before_the_window_opens_does_not_apply():
    e = _note("N003", "All Routes", "raises", "all_routes", "2025-05-05", None)
    c = _candidate("Mumbai-Delhi", dt.date(2025, 4, 21))
    assert note_applies(e, c) is False


def test_wrong_route_does_not_apply():
    e = _note("N002", "Ahmedabad-Mumbai", "raises", "route", "2025-01-20", "2025-01-26")
    c = _candidate("Mumbai-Pune", dt.date(2025, 1, 20))
    assert note_applies(e, c) is False


def test_a_note_that_does_not_raise_costs_never_justifies():
    for direction in ("none", "lowers"):
        e = _note("N006", "All Routes", direction, "all_routes", "2025-09-22", None)
        c = _candidate("Mumbai-Pune", dt.date(2025, 9, 22))
        assert note_applies(e, c) is False, direction


def test_a_note_about_routes_outside_the_dataset_never_justifies():
    e = _note("N004", "All Routes", "raises", "all_routes", "2024-03-11", None,
              in_dataset=False)
    c = _candidate("Mumbai-Delhi", dt.date(2024, 3, 11))
    assert note_applies(e, c) is False


def test_an_all_routes_note_cannot_justify_a_peer_based_flag():
    """A nationwide fuel rise lifts this route's peers too, so it cannot
    explain this route rising RELATIVE to them."""
    e = _note("N003", "All Routes", "raises", "all_routes", "2025-05-05", None)
    peer_only = _candidate("Mumbai-Pune", dt.date(2025, 9, 15), own=False, peer=True)
    assert note_applies(e, peer_only) is False

    own_based = _candidate("Mumbai-Delhi", dt.date(2025, 5, 12), own=True, peer=False)
    assert note_applies(e, own_based) is True


def test_a_route_specific_note_can_justify_either_kind_of_flag():
    e = _note("N001", "Chennai-Bangalore", "raises", "route", "2025-02-24", "2025-03-08")
    peer_only = _candidate("Chennai-Bangalore", dt.date(2025, 3, 3), own=False, peer=True)
    assert note_applies(e, peer_only) is True


def test_open_ended_window_applies_to_any_later_week():
    e = _note("N003", "All Routes", "raises", "all_routes", "2025-05-05", None)
    c = _candidate("Mumbai-Delhi", dt.date(2025, 5, 12))
    assert note_applies(e, c) is True


def test_a_week_overlapping_only_the_window_start_applies():
    """The window opens mid-week: Mon 2025-02-24 .. Sun 2025-03-02 overlaps a
    window that opens on the Friday."""
    e = _note("N001", "Chennai-Bangalore", "raises", "route", "2025-02-28", "2025-03-08")
    c = _candidate("Chennai-Bangalore", dt.date(2025, 2, 24))
    assert note_applies(e, c) is True
