import pytest

from ftanomaly.prose import GroundingError, to_ascii, validate_reason


def test_to_ascii_replaces_typographic_punctuation():
    """The model emits non-breaking hyphens and em dashes. Left in, they
    render as mojibake when the CSV is opened in Excel under cp1252."""
    assert to_ascii("Ahmedabad‑Mumbai") == "Ahmedabad-Mumbai"
    assert to_ascii("justified – note N003") == "justified - note N003"
    assert to_ascii("rise — see note") == "rise - see note"
    assert to_ascii("the week") == "the week"
    assert to_ascii("“stable” demand") == '"stable" demand'
    assert to_ascii("it’s fine") == "it's fine"


def test_to_ascii_output_is_always_encodable_as_ascii():
    """Any character the map misses must still not reach the CSV."""
    weird = "cost rose ≥ 5% ± 2 中文"
    assert to_ascii(weird).encode("ascii")


def test_to_ascii_leaves_plain_text_untouched():
    plain = "Matches note N002 dated 2025-01-20: a festival surcharge."
    assert to_ascii(plain) == plain

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


def test_works_with_note_ids_that_are_not_the_nNNN_format():
    """A different notes file may id its rows differently. The validator must
    read the ids it was given, not assume the shape of this corpus."""
    ids = {"NOTE-1", "NOTE-2", "fuel-2025-05"}
    validate_reason("Matches note NOTE-2 dated 2025-01-20.", "NOTE-2", ids)
    validate_reason("Matches note fuel-2025-05: diesel rose.", "fuel-2025-05", ids)

    with pytest.raises(GroundingError, match="must cite"):
        validate_reason("A surcharge explains it.", "NOTE-2", ids)
    with pytest.raises(GroundingError, match="unrelated note"):
        validate_reason("Matches NOTE-2 and NOTE-1.", "NOTE-2", ids)


def test_a_note_id_that_is_a_substring_of_another_is_not_confused():
    ids = {"N1", "N12"}
    with pytest.raises(GroundingError, match="must cite"):
        validate_reason("Matches note N12.", "N1", ids)


def test_ignores_a_note_id_that_does_not_exist_in_the_corpus():
    """A hallucinated id like N999 is not in ALL, so it must not be mistaken
    for a real citation that satisfies the requirement."""
    with pytest.raises(GroundingError, match="must cite"):
        validate_reason("Matches note N999, which explains everything.",
                        "N002", ALL)
