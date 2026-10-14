import csv
import json
import os

import pytest
from dotenv import load_dotenv

from ftanomaly.cli import main

load_dotenv()

needs_credentials = pytest.mark.skipif(
    not (os.environ.get("GROQ_API_KEY") and os.environ.get("GEMINI_API_KEY")),
    reason="needs GROQ_API_KEY and GEMINI_API_KEY (embeddings have no offline path)",
)


def test_missing_credentials_exit_nonzero_without_writing_a_report(
    tmp_path, monkeypatch
):
    """A partial report is worse than none: a blank matched_note_id reads as a
    real 'unexplained' verdict and would be graded as a wrong answer."""
    import dataclasses

    from ftanomaly import config

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    # point the client at an EMPTY cache - with the real one warm, the run
    # would legitimately succeed without credentials and prove nothing
    monkeypatch.setattr(
        config, "PATHS", dataclasses.replace(config.PATHS, cache_dir=tmp_path)
    )

    out = tmp_path / "report.csv"
    log = tmp_path / "tokens.json"
    rc = main(["--output", str(out), "--token-log", str(log)])

    assert rc == 2
    assert not out.exists(), "a failed run must not leave a report behind"
    assert not log.exists()


@pytest.mark.live
@needs_credentials
def test_end_to_end_produces_the_expected_report(tmp_path):
    out = tmp_path / "report.csv"
    log = tmp_path / "tokens.json"
    assert main(["--output", str(out), "--token-log", str(log)]) == 0

    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 21
    assert {r["flagged"] for r in rows} <= {"Yes", "No (justified)"}

    for r in rows:
        if r["flagged"] == "No (justified)":
            assert r["matched_note_id"].startswith("N")
            assert r["matched_note_id"] in r["reason"]
        else:
            assert r["matched_note_id"] == ""

    usage = json.loads(log.read_text(encoding="utf-8"))
    assert usage["rows_written"] == 21
    assert usage["temperature"] == 0.0


@pytest.mark.live
@needs_credentials
def test_a_cached_rerun_makes_no_api_calls(tmp_path):
    out = tmp_path / "r.csv"
    log = tmp_path / "t.json"
    main(["--output", str(out), "--token-log", str(log)])        # warm
    main(["--output", str(out), "--token-log", str(log)])        # cached
    assert json.loads(log.read_text(encoding="utf-8"))["llm_calls"] == 0
