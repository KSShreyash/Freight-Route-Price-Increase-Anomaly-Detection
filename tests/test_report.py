import csv
import datetime as dt

from ftanomaly import config
from ftanomaly.detect import Candidate
from ftanomaly.metrics import WeekMetrics
from ftanomaly.notes import EnrichedNote, Note
from ftanomaly.report import (
    COLUMNS,
    format_own_history,
    format_similar,
    to_row,
    write_report,
)
from ftanomaly.verdict import Verdict


def test_columns_match_the_sample_header_exactly():
    with config.PATHS.sample_output.open(newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert list(COLUMNS) == header


def test_percentage_formatting_matches_the_sample_strings():
    assert format_own_history(35.52) == "+35.5% vs this route's past average"
    assert format_similar(21.04) == "+21.0% vs similar-length routes this week"
    assert format_own_history(-4.0) == "-4.0% vs this route's past average"


def test_missing_history_formats_without_crashing():
    assert "n/a" in format_own_history(None)
    assert "n/a" in format_similar(None)


def _verdict(matched):
    m = WeekMetrics("Delhi-Jaipur", "Short", dt.date(2024, 11, 11), 4.1712,
                    35.52, 21.04, 2.79, 8)
    c = Candidate(metrics=m, own_history_triggered=True, peer_triggered=True)
    return Verdict(candidate=c, matched=matched, retrieved=[])


def test_unexplained_row_has_yes_and_an_empty_note_id():
    row = to_row(_verdict(None), "No matching note found.")
    assert row["flagged"] == "Yes"
    assert row["matched_note_id"] == ""
    assert row["cost_per_tonne_km"] == "4.17"
    assert row["week_of"] == "2024-11-11"


def test_justified_row_has_the_other_vocabulary_value():
    note = Note("N002", dt.date(2025, 1, 20), "Ahmedabad-Mumbai", "text")
    e = EnrichedNote(note, "raises", "route", dt.date(2025, 1, 20),
                     dt.date(2025, 1, 26), True, None)
    row = to_row(_verdict(e), "Matches note N002.")
    assert row["flagged"] == "No (justified)"
    assert row["matched_note_id"] == "N002"


def test_written_file_is_byte_identical_on_a_second_write(tmp_path):
    rows = [to_row(_verdict(None), "No matching note found.")]
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write_report(rows, a)
    write_report(rows, b)
    assert a.read_bytes() == b.read_bytes()


def test_rows_are_sorted_by_week_then_route(tmp_path):
    m1 = WeekMetrics("Zeta", "Short", dt.date(2025, 2, 3), 1.0, 30.0, 30.0, 2.0, 8)
    m2 = WeekMetrics("Alpha", "Short", dt.date(2025, 1, 6), 1.0, 30.0, 30.0, 2.0, 8)
    m3 = WeekMetrics("Alpha", "Short", dt.date(2025, 2, 3), 1.0, 30.0, 30.0, 2.0, 8)
    rows = [
        to_row(Verdict(Candidate(m, True, True), None, []), "x")
        for m in (m1, m2, m3)
    ]
    out = tmp_path / "o.csv"
    write_report(rows, out)
    with out.open(newline="", encoding="utf-8") as fh:
        got = [(r["week_of"], r["route"]) for r in csv.DictReader(fh)]
    assert got == [
        ("2025-01-06", "Alpha"),
        ("2025-02-03", "Alpha"),
        ("2025-02-03", "Zeta"),
    ]


def test_written_rows_round_trip_through_the_csv_reader(tmp_path):
    """Reasons contain commas and apostrophes; quoting must survive a read."""
    reason = 'The closest note (N006, 2025-09-22) mentions "stable demand".'
    rows = [to_row(_verdict(None), reason)]
    out = tmp_path / "o.csv"
    write_report(rows, out)
    with out.open(newline="", encoding="utf-8") as fh:
        back = list(csv.DictReader(fh))
    assert back[0]["reason"] == reason
    assert len(back) == 1
