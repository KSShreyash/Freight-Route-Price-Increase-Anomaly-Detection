"""Score the generated report against a hand-labeled set.

Run after the pipeline:  python eval/run_eval.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftanomaly import config  # noqa: E402


def _read(path: Path) -> dict[tuple[str, str], dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return {(r["route"], r["week_of"]): r for r in csv.DictReader(fh)}


def score(report_path: Path, labels_path: Path) -> dict:
    report = _read(report_path)
    labels = _read(labels_path)

    verdict_correct = note_correct = 0
    tp = fp = fn = 0
    mismatches: list[str] = []

    for key, want in labels.items():
        got = report.get(key)
        if got is None:
            mismatches.append(f"{key}: expected in report, missing")
            fn += int(want["expected_flagged"] == "No (justified)")
            continue

        if got["flagged"] == want["expected_flagged"]:
            verdict_correct += 1
        else:
            mismatches.append(
                f"{key}: verdict {got['flagged']!r} != {want['expected_flagged']!r}"
            )
        if got["matched_note_id"] == want["expected_note_id"]:
            note_correct += 1
        else:
            mismatches.append(
                f"{key}: note {got['matched_note_id']!r} != {want['expected_note_id']!r}"
            )

        want_j = want["expected_flagged"] == "No (justified)"
        got_j = got["flagged"] == "No (justified)"
        tp += int(want_j and got_j)
        fp += int(got_j and not want_j)
        fn += int(want_j and not got_j)

    for key in report.keys() - labels.keys():
        mismatches.append(f"{key}: present in report but not labeled")

    return {
        "total": len(labels),
        "verdict_correct": verdict_correct,
        "note_correct": note_correct,
        "precision_justified": tp / (tp + fp) if (tp + fp) else 1.0,
        "recall_justified": tp / (tp + fn) if (tp + fn) else 1.0,
        "mismatches": mismatches,
    }


if __name__ == "__main__":
    if not config.PATHS.output_csv.exists():
        print("no report found - run `python -m ftanomaly.cli` first", file=sys.stderr)
        raise SystemExit(1)

    result = score(config.PATHS.output_csv, ROOT / "eval" / "labeled_set.csv")
    print(f"rows labeled           : {result['total']}")
    print(f"verdicts correct       : {result['verdict_correct']}/{result['total']}")
    print(f"note ids correct       : {result['note_correct']}/{result['total']}")
    print(f"justified precision    : {result['precision_justified']:.3f}")
    print(f"justified recall       : {result['recall_justified']:.3f}")
    for m in result["mismatches"]:
        print("  MISMATCH", m)
    raise SystemExit(1 if result["mismatches"] else 0)
