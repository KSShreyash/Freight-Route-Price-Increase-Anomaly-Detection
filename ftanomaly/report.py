"""Emit the output CSV. The column contract is not negotiable."""
from __future__ import annotations

import csv
from pathlib import Path

from .verdict import Verdict

COLUMNS = (
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
)


def format_own_history(pct: float | None) -> str:
    if pct is None:
        return "n/a (insufficient prior weeks)"
    return f"{pct:+.1f}% vs this route's past average"


def format_similar(pct: float | None) -> str:
    if pct is None:
        return "n/a (no similar routes)"
    return f"{pct:+.1f}% vs similar-length routes this week"


def to_row(v: Verdict, reason: str) -> dict[str, str]:
    m = v.candidate.metrics
    return {
        "route": m.route,
        "week_of": m.week_of.isoformat(),
        "cost_per_tonne_km": f"{m.cost_per_tonne_km:.2f}",
        "vs_own_history": format_own_history(m.vs_own_history),
        "vs_similar_routes": format_similar(m.vs_similar_routes),
        "flagged": v.flagged_value,
        "matched_note_id": v.matched.note.note_id if v.matched else "",
        "reason": reason,
    }


def write_report(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: (r["week_of"], r["route"]))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)
