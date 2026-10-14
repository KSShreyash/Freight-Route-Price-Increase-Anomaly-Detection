import csv
import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from run_eval import score  # noqa: E402

from ftanomaly import config  # noqa: E402
from ftanomaly.detect import detect  # noqa: E402
from ftanomaly.loading import load_route_weeks  # noqa: E402
from ftanomaly.metrics import build_metrics  # noqa: E402

load_dotenv()

LABELS = ROOT / "eval" / "labeled_set.csv"

needs_credentials = pytest.mark.skipif(
    not (os.environ.get("GROQ_API_KEY") and os.environ.get("GEMINI_API_KEY")),
    reason="needs GROQ_API_KEY and GEMINI_API_KEY to generate a report to score",
)


# --------------------------------------------------------------- offline
def test_labeled_set_covers_exactly_the_detected_candidates():
    """The labels must track detection. If the rule changes and nobody
    updates the labels, the eval silently stops measuring the real output."""
    candidates = detect(build_metrics(load_route_weeks(config.PATHS.shipments)))
    detected = {(c.metrics.route, c.metrics.week_of.isoformat()) for c in candidates}

    with LABELS.open(newline="", encoding="utf-8") as fh:
        labeled = {(r["route"], r["week_of"]) for r in csv.DictReader(fh)}

    assert labeled == detected, {
        "labeled but not detected": sorted(labeled - detected),
        "detected but not labeled": sorted(detected - labeled),
    }


def test_labels_only_use_the_two_allowed_verdict_values():
    with LABELS.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["expected_flagged"] for r in rows} <= {"Yes", "No (justified)"}
    for r in rows:
        if r["expected_flagged"] == "Yes":
            assert r["expected_note_id"] == "", r
        else:
            assert r["expected_note_id"].startswith("N"), r


def _write(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def test_score_reports_a_clean_sheet_when_everything_matches(tmp_path):
    report = tmp_path / "r.csv"
    labels = tmp_path / "l.csv"
    _write(report, [{"route": "A", "week_of": "2025-01-06",
                     "flagged": "No (justified)", "matched_note_id": "N002"}],
           ["route", "week_of", "flagged", "matched_note_id"])
    _write(labels, [{"route": "A", "week_of": "2025-01-06",
                     "expected_flagged": "No (justified)", "expected_note_id": "N002"}],
           ["route", "week_of", "expected_flagged", "expected_note_id"])

    r = score(report, labels)
    assert r["mismatches"] == []
    assert r["verdict_correct"] == 1
    assert r["precision_justified"] == 1.0
    assert r["recall_justified"] == 1.0


def test_score_catches_a_wrong_note_id(tmp_path):
    report = tmp_path / "r.csv"
    labels = tmp_path / "l.csv"
    _write(report, [{"route": "A", "week_of": "2025-01-06",
                     "flagged": "No (justified)", "matched_note_id": "N006"}],
           ["route", "week_of", "flagged", "matched_note_id"])
    _write(labels, [{"route": "A", "week_of": "2025-01-06",
                     "expected_flagged": "No (justified)", "expected_note_id": "N002"}],
           ["route", "week_of", "expected_flagged", "expected_note_id"])

    r = score(report, labels)
    assert any("note" in m for m in r["mismatches"])


def test_score_catches_a_fabricated_justification(tmp_path):
    """Report says justified, truth says unexplained - precision must drop."""
    report = tmp_path / "r.csv"
    labels = tmp_path / "l.csv"
    _write(report, [{"route": "A", "week_of": "2025-01-06",
                     "flagged": "No (justified)", "matched_note_id": "N006"}],
           ["route", "week_of", "flagged", "matched_note_id"])
    _write(labels, [{"route": "A", "week_of": "2025-01-06",
                     "expected_flagged": "Yes", "expected_note_id": ""}],
           ["route", "week_of", "expected_flagged", "expected_note_id"])

    r = score(report, labels)
    assert r["precision_justified"] == 0.0
    assert r["mismatches"]


def test_score_flags_a_report_row_that_is_not_labeled(tmp_path):
    report = tmp_path / "r.csv"
    labels = tmp_path / "l.csv"
    _write(report, [{"route": "A", "week_of": "2025-01-06",
                     "flagged": "Yes", "matched_note_id": ""},
                    {"route": "B", "week_of": "2025-01-06",
                     "flagged": "Yes", "matched_note_id": ""}],
           ["route", "week_of", "flagged", "matched_note_id"])
    _write(labels, [{"route": "A", "week_of": "2025-01-06",
                     "expected_flagged": "Yes", "expected_note_id": ""}],
           ["route", "week_of", "expected_flagged", "expected_note_id"])

    r = score(report, labels)
    assert any("not labeled" in m for m in r["mismatches"])


# ------------------------------------------------------------------ live
@pytest.mark.live
@needs_credentials
def test_generated_report_matches_every_hand_label():
    result = score(config.PATHS.output_csv, LABELS)
    assert result["mismatches"] == []
    assert result["verdict_correct"] == result["total"]
    assert result["note_correct"] == result["total"]
    assert result["precision_justified"] == 1.0
    assert result["recall_justified"] == 1.0
