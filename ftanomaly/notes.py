"""Load context notes and enrich each one into a structured record.

The LLM is asked only to read language: which direction does this note
push costs, over what window, for which routes. All date comparison and
applicability logic lives in verdict.py as plain Python.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .llm import LLMClient

_DIRECTIONS = {"raises", "lowers", "none"}
_SCOPES = {"route", "all_routes"}

ENRICH_PROMPT = """You are classifying a logistics context note.

Note id: {note_id}
Note date: {date}
Applies to: {applies_to}
Text: {text}

Answer with JSON only, no prose, using exactly these keys:

  direction     "raises" if the note describes something that pushed freight
                costs UP; "lowers" if it describes something that pushed them
                DOWN; "none" if it explicitly says costs were unaffected, or
                describes normal/stable conditions, or a return to normal.
  scope         "all_routes" if it applies to every route, else "route".
  window_start  ISO date the effect starts. Default to the note date.
  window_end    ISO date the effect ends.
                - If the text names an end date, or says when things returned
                  to normal, use that date.
                - If it describes a bounded event (a festival week, a
                  temporary surcharge, maintenance lasting about a week),
                  infer the end of that period from the note date.
                - Use null ONLY for a genuinely open-ended change with no
                  stated end, such as a nationwide price level shift.
  in_dataset    false if the note says the affected routes are NOT part of
                this dataset, otherwise true.
  magnitude_pct the percentage rise the note states, as a number, or null.
                If a range is given, use its midpoint.

JSON:"""


@dataclass(frozen=True)
class Note:
    note_id: str
    date: dt.date
    applies_to: str
    text: str


@dataclass(frozen=True)
class EnrichedNote:
    note: Note
    direction: str
    scope: str
    window_start: dt.date
    window_end: dt.date | None
    in_dataset: bool
    magnitude_pct: float | None


def load_notes(path: Path) -> list[Note]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [
            Note(
                note_id=row["note_id"],
                date=dt.date.fromisoformat(row["date"]),
                applies_to=row["applies_to"],
                text=row["note"],
            )
            for row in csv.DictReader(fh)
        ]


def _extract_json(raw: str) -> str:
    """Pull the JSON object out of a model reply.

    Handles a bare object, a ```json fence, and a reasoning model that
    narrates around its answer.
    """
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    braced = re.search(r"\{.*\}", raw, re.DOTALL)
    if braced:
        return braced.group(0).strip()
    return raw.strip()


def parse_enrichment(note: Note, raw: str) -> EnrichedNote:
    data = json.loads(_extract_json(raw))

    direction = str(data["direction"]).lower()
    if direction not in _DIRECTIONS:
        raise ValueError(f"unknown direction {direction!r} for {note.note_id}")

    scope = str(data["scope"]).lower()
    if scope not in _SCOPES:
        raise ValueError(f"unknown scope {scope!r} for {note.note_id}")

    end = data.get("window_end")
    mag = data.get("magnitude_pct")
    return EnrichedNote(
        note=note,
        direction=direction,
        scope=scope,
        window_start=dt.date.fromisoformat(data["window_start"]),
        window_end=dt.date.fromisoformat(end) if end else None,
        in_dataset=bool(data.get("in_dataset", True)),
        magnitude_pct=float(mag) if mag is not None else None,
    )


def enrich_notes(notes: list[Note], client: LLMClient) -> list[EnrichedNote]:
    out = []
    for note in notes:
        prompt = ENRICH_PROMPT.format(
            note_id=note.note_id,
            date=note.date.isoformat(),
            applies_to=note.applies_to,
            text=note.text,
        )
        raw = client.chat(
            prompt, cache_key=client.key_for("enrich", note.note_id, note.text)
        )
        out.append(parse_enrichment(note, raw))
    return out
