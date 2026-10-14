import datetime as dt

import pytest

from ftanomaly.notes import EnrichedNote, Note
from ftanomaly.retrieval import NoteIndex, cosine, query_text_for


def test_cosine_basics():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert round(cosine([1.0, 1.0], [1.0, 0.0]), 4) == 0.7071


def test_cosine_handles_a_zero_vector_without_dividing_by_zero():
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_query_text_mentions_the_route_and_the_week():
    q = query_text_for("Chennai-Bangalore", dt.date(2025, 2, 24))
    assert "Chennai-Bangalore" in q
    assert "2025-02-24" in q


# --------------------------------------------------------------------------
# NoteIndex.query ordering, exercised with a stub client so no network is
# involved and the vectors are chosen to force an exact-tie.
# --------------------------------------------------------------------------

def _note(nid: str) -> EnrichedNote:
    n = Note(nid, dt.date(2025, 1, 1), "All Routes", f"text for {nid}")
    return EnrichedNote(n, "none", "all_routes", dt.date(2025, 1, 1), None, True, None)


class StubClient:
    """Returns a canned vector per document, keyed by call order."""

    def __init__(self, doc_vectors, query_vector):
        self._doc_vectors = doc_vectors
        self._query_vector = query_vector
        self.embed_calls = 0

    def embed(self, texts):
        self.embed_calls += 1
        if len(texts) == 1 and self.embed_calls > 1:
            return [self._query_vector]
        return list(self._doc_vectors)


def test_query_ranks_by_descending_similarity():
    index = NoteIndex()
    client = StubClient(
        doc_vectors=[[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]],
        query_vector=[1.0, 0.0],
    )
    index.build([_note("N001"), _note("N002"), _note("N003")], client)
    ranked = index.query("q", client, k=3)
    assert [e.note.note_id for e, _ in ranked] == ["N001", "N002", "N003"]


def test_equal_scores_break_ties_on_note_id_so_order_is_total():
    """Three runs must produce the same CSV; an unstable sort here would
    let a tie flip the chosen note between runs."""
    index = NoteIndex()
    client = StubClient(
        doc_vectors=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        query_vector=[1.0, 0.0],
    )
    index.build([_note("N003"), _note("N001"), _note("N002")], client)
    ranked = index.query("q", client, k=3)
    assert [e.note.note_id for e, _ in ranked] == ["N001", "N002", "N003"]


def test_query_before_build_is_an_error_not_an_empty_result():
    with pytest.raises(RuntimeError, match="build"):
        NoteIndex().query("q", StubClient([], []))
