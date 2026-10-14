"""Embedding-backed retrieval over the context notes.

A persisted embedding matrix plus cosine similarity is the whole vector
store. With ten notes a hosted vector database would add operational
weight without changing a single result; the NoteIndex interface is kept
narrow so Chroma or FAISS could be dropped in unchanged if the corpus grew.

Retrieval only proposes candidates. Whether a note actually justifies a
cost rise is decided in verdict.py by explicit rules, never by similarity.
"""
from __future__ import annotations

import datetime as dt
import math

from . import config
from .llm import LLMClient
from .notes import EnrichedNote


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return num / (na * nb)


def query_text_for(route: str, week_of: dt.date) -> str:
    return (
        f"Freight cost on the {route} corridor rose during the week "
        f"beginning {week_of.isoformat()}. What happened?"
    )


class NoteIndex:
    """Dense index over enriched notes."""

    def __init__(self) -> None:
        self._notes: list[EnrichedNote] = []
        self._vectors: list[list[float]] = []

    @staticmethod
    def _document(e: EnrichedNote) -> str:
        return f"{e.note.applies_to} ({e.note.date.isoformat()}): {e.note.text}"

    def build(self, enriched: list[EnrichedNote], client: LLMClient) -> None:
        self._notes = list(enriched)
        self._vectors = client.embed([self._document(e) for e in self._notes])

    def query(
        self, text: str, client: LLMClient, k: int = config.RETRIEVAL_TOP_K
    ) -> list[tuple[EnrichedNote, float]]:
        if not self._notes:
            raise RuntimeError("NoteIndex.query called before build")
        qv = client.embed([text])[0]
        scored = [(e, cosine(qv, v)) for e, v in zip(self._notes, self._vectors)]
        # descending score, ties broken on note_id so the order is total
        scored.sort(key=lambda pair: (-pair[1], pair[0].note.note_id))
        return scored[:k]
