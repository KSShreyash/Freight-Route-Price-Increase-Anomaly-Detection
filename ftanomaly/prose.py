"""Plain-English reasons, with a guard against ungrounded claims.

The model writes the sentence; the validator checks it did not invent a
justification. A violation aborts the run rather than shipping a confident
wrong reason, which the brief scores worse than admitting none was found.
"""
from __future__ import annotations

import re

from .llm import LLMClient
from .verdict import Verdict

_JUSTIFY_WORDS = re.compile(
    r"\b(matches|justified by|explained by|supported by|confirms)\b", re.IGNORECASE
)


class GroundingError(ValueError):
    """The generated reason makes a claim the evidence does not support."""


# Models reach for typographic punctuation. Left in the CSV it renders as
# mojibake when the file is opened under cp1252, so the output is kept ASCII.
_ASCII_MAP = {
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "‘": "'", "’": "'", "‚": "'",
    "“": '"', "”": '"', "„": '"',
    "…": "...", " ": " ", "­": "",
}


def to_ascii(text: str) -> str:
    """Fold a model reply down to plain ASCII.

    A BOM would be the other way to keep Excel happy, but it would also
    prepend a stray character to the first column name and break a strict
    header comparison against the sample output.
    """
    for bad, good in _ASCII_MAP.items():
        text = text.replace(bad, good)
    return text.encode("ascii", "replace").decode("ascii")


JUSTIFIED_PROMPT = """Write one or two plain sentences explaining why this
freight cost rise is justified. State the verdict, name the note by id, and
reflect what the note actually says including its dates.

Route: {route}
Week beginning: {week}
Cost rise vs this route's own 8-week average: {own}
Cost rise vs similar-length routes this week: {peer}

The note that justifies it:
  id: {note_id}
  date: {note_date}
  applies to: {applies_to}
  text: {note_text}

Do not mention any other note id. Do not invent facts that are not in the
note text. Reply with the explanation only, no preamble."""

UNEXPLAINED_PROMPT = """Write one or two plain sentences explaining that this
freight cost rise has no supporting note and needs a human review.

Route: {route}
Week beginning: {week}
Cost rise vs this route's own 8-week average: {own}
Cost rise vs similar-length routes this week: {peer}

The closest notes that were retrieved and REJECTED:
{rejected}

You may say briefly why the closest note does not apply (wrong route, wrong
dates, or it describes no cost rise). Never claim the rise is justified or
explained, and never use the words "matches", "justified by", "explained by",
"supported by" or "confirms". Reply with the explanation only, no preamble."""


def validate_reason(text: str, matched_id: str | None, all_note_ids: set[str]) -> None:
    # Search for the ids this corpus actually uses rather than assuming they
    # look like N001 - a different notes file may id its rows any way it likes.
    # Longest first so "N12" is not credited when only "N1" was cited.
    mentioned = {
        nid
        for nid in sorted(all_note_ids, key=len, reverse=True)
        if re.search(rf"(?<!\w){re.escape(nid)}(?!\w)", text)
    }

    if matched_id is None:
        if _JUSTIFY_WORDS.search(text):
            raise GroundingError(f"unexplained row claims justification: {text!r}")
        return

    if matched_id not in mentioned:
        raise GroundingError(
            f"justified row must cite {matched_id} in the reason text: {text!r}"
        )
    extra = mentioned - {matched_id}
    if extra:
        raise GroundingError(f"reason cites unrelated note(s) {sorted(extra)}: {text!r}")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1f}%"


def write_reason(v: Verdict, client: LLMClient, all_note_ids: set[str]) -> str:
    m = v.candidate.metrics
    common = {
        "route": m.route,
        "week": m.week_of.isoformat(),
        "own": _fmt(m.vs_own_history),
        "peer": _fmt(m.vs_similar_routes),
    }

    if v.matched is not None:
        e = v.matched
        prompt = JUSTIFIED_PROMPT.format(
            note_id=e.note.note_id,
            note_date=e.note.date.isoformat(),
            applies_to=e.note.applies_to,
            note_text=e.note.text,
            **common,
        )
        key = client.key_for("reason", m.route, m.week_of.isoformat(), e.note.note_id)
    else:
        rejected = "\n".join(
            f"  - {e.note.note_id} ({e.note.date.isoformat()}, "
            f"{e.note.applies_to}): {e.note.text}"
            for e, _ in v.retrieved
        ) or "  - none"
        prompt = UNEXPLAINED_PROMPT.format(rejected=rejected, **common)
        key = client.key_for("reason", m.route, m.week_of.isoformat(), "unexplained")

    text = to_ascii(" ".join(client.chat(prompt, cache_key=key).split()))
    validate_reason(text, v.matched.note.note_id if v.matched else None, all_note_ids)
    return text
