"""Decide whether a retrieved note genuinely justifies a cost rise.

Every rule here is deterministic. Retrieval proposes; these rules dispose.
A note justifies a candidate only when all of the following hold:

  1. it describes costs going UP                (direction == "raises")
  2. its routes are part of this dataset        (in_dataset)
  3. it covers this route                       (scope / applies_to)
  4. its window overlaps this shipping week     (dates)
  5. it can explain THIS KIND of rise           (see the scope rule below)

Rule 5 is the subtle one. A nationwide note lifts every route together, so
it cannot explain a route rising relative to its peers - only a rise
against the route's own history. A route-specific note can explain either.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .detect import Candidate
from .llm import LLMClient
from .notes import EnrichedNote
from .retrieval import NoteIndex, query_text_for


@dataclass(frozen=True)
class Verdict:
    candidate: Candidate
    matched: EnrichedNote | None
    retrieved: list[tuple[EnrichedNote, float]]

    @property
    def is_justified(self) -> bool:
        return self.matched is not None

    @property
    def flagged_value(self) -> str:
        return "No (justified)" if self.is_justified else "Yes"


def note_applies(e: EnrichedNote, candidate: Candidate) -> bool:
    if e.direction != "raises":
        return False
    if not e.in_dataset:
        return False

    route = candidate.metrics.route
    if e.scope == "route" and e.note.applies_to != route:
        return False

    # A nationwide cause moves peers equally, so it cannot explain a
    # peer-relative rise. Only an own-history rise can be covered by it.
    if e.scope == "all_routes" and not candidate.own_history_triggered:
        return False

    week_start = candidate.metrics.week_of
    week_end = week_start + dt.timedelta(days=6)
    if week_end < e.window_start:
        return False
    if e.window_end is not None and week_start > e.window_end:
        return False
    return True


def decide(
    candidates: list[Candidate], index: NoteIndex, client: LLMClient
) -> list[Verdict]:
    verdicts: list[Verdict] = []
    for c in candidates:
        retrieved = index.query(
            query_text_for(c.metrics.route, c.metrics.week_of), client
        )
        applicable = [e for e, _ in retrieved if note_applies(e, c)]
        # deterministic pick: best-ranked applicable note, ties on note_id
        matched = applicable[0] if applicable else None
        verdicts.append(Verdict(candidate=c, matched=matched, retrieved=retrieved))
    return verdicts
