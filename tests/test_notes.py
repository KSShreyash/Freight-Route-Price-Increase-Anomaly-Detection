import datetime as dt

import pytest

from ftanomaly import config
from ftanomaly.notes import load_notes, parse_enrichment


@pytest.fixture(scope="module")
def notes():
    return load_notes(config.PATHS.notes)


def test_loads_all_ten_notes(notes):
    assert len(notes) == 10
    assert notes[0].note_id == "N001"
    assert notes[0].date == dt.date(2025, 2, 24)
    assert notes[0].applies_to == "Chennai-Bangalore"


def test_parse_enrichment_reads_a_well_formed_payload(notes):
    payload = (
        '{"direction": "raises", "scope": "route", '
        '"window_start": "2025-02-24", "window_end": "2025-03-08", '
        '"in_dataset": true, "magnitude_pct": null}'
    )
    e = parse_enrichment(notes[0], payload)
    assert e.direction == "raises"
    assert e.window_end == dt.date(2025, 3, 8)
    assert e.magnitude_pct is None


def test_parse_enrichment_accepts_a_fenced_payload(notes):
    """Models often wrap JSON in a markdown fence - strip it, do not crash."""
    payload = (
        '```json\n{"direction": "none", "scope": "route", '
        '"window_start": "2024-07-29", "window_end": "2024-08-04", '
        '"in_dataset": true, "magnitude_pct": null}\n```'
    )
    e = parse_enrichment(notes[4], payload)
    assert e.direction == "none"


def test_parse_enrichment_extracts_json_from_surrounding_prose(notes):
    """gpt-oss-120b is a reasoning model and may narrate before answering."""
    payload = (
        "Let me think about this. The note says costs were unaffected.\n"
        '{"direction": "none", "scope": "route", '
        '"window_start": "2024-07-29", "window_end": "2024-08-04", '
        '"in_dataset": true, "magnitude_pct": null}\n'
        "That is my answer."
    )
    e = parse_enrichment(notes[4], payload)
    assert e.direction == "none"
    assert e.window_start == dt.date(2024, 7, 29)


def test_parse_enrichment_rejects_an_unknown_direction(notes):
    bad = (
        '{"direction": "maybe", "scope": "route", '
        '"window_start": "2025-02-24", "window_end": null, '
        '"in_dataset": true, "magnitude_pct": null}'
    )
    with pytest.raises(ValueError, match="direction"):
        parse_enrichment(notes[0], bad)


def test_parse_enrichment_rejects_an_unknown_scope(notes):
    bad = (
        '{"direction": "raises", "scope": "galaxy", '
        '"window_start": "2025-02-24", "window_end": null, '
        '"in_dataset": true, "magnitude_pct": null}'
    )
    with pytest.raises(ValueError, match="scope"):
        parse_enrichment(notes[0], bad)


def test_open_ended_window_is_none_not_a_sentinel_date(notes):
    payload = (
        '{"direction": "raises", "scope": "all_routes", '
        '"window_start": "2025-05-05", "window_end": null, '
        '"in_dataset": true, "magnitude_pct": 6.0}'
    )
    e = parse_enrichment(notes[2], payload)
    assert e.window_end is None
    assert e.scope == "all_routes"
    assert e.magnitude_pct == 6.0
