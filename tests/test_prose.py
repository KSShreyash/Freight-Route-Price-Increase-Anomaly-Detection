import pytest

from ftanomaly.prose import GroundingError, validate_reason

ALL = {"N001", "N002", "N003", "N004", "N005", "N006"}


def test_accepts_a_justified_reason_citing_its_own_note():
    validate_reason("Matches note N002 dated 2025-01-20: a festival surcharge.",
                    "N002", ALL)


def test_rejects_a_justified_reason_that_omits_its_note_id():
    with pytest.raises(GroundingError, match="must cite"):
        validate_reason("A festival surcharge explains the rise.", "N002", ALL)


def test_rejects_a_justified_reason_citing_a_different_note():
    with pytest.raises(GroundingError, match="unrelated note"):
        validate_reason("Matches note N002 and also N004.", "N002", ALL)


def test_accepts_an_unexplained_reason_that_names_no_note():
    validate_reason("No matching note found for this route or date range.",
                    None, ALL)


def test_rejects_an_unexplained_reason_that_claims_a_note_justifies_it():
    """The nearest note MAY be discussed, but never as a justification."""
    with pytest.raises(GroundingError, match="unexplained"):
        validate_reason("Matches note N003: diesel prices rose.", None, ALL)


def test_allows_an_unexplained_reason_to_mention_why_a_note_was_rejected():
    validate_reason(
        "The closest note (N006, 2025-09-22) mentions stable demand and does "
        "not describe a cost rise on this route. No genuine justification found.",
        None, ALL,
    )


def test_ignores_a_note_id_that_does_not_exist_in_the_corpus():
    """A hallucinated id like N999 is not in ALL, so it must not be mistaken
    for a real citation that satisfies the requirement."""
    with pytest.raises(GroundingError, match="must cite"):
        validate_reason("Matches note N999, which explains everything.",
                        "N002", ALL)
