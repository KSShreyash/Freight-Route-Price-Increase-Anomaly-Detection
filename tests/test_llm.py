"""Tests for the cache and failure behaviour of the model client.

None of these touch the network: the cache path is exercised with a
pre-seeded cache file, and the failure paths are exercised by removing
the keys.
"""
import json

import pytest

from ftanomaly.llm import LLMClient, LLMUnavailable


def test_key_for_is_stable_and_order_sensitive():
    assert LLMClient.key_for("a", "b") == LLMClient.key_for("a", "b")
    assert LLMClient.key_for("a", "b") != LLMClient.key_for("b", "a")
    # the separator must prevent "ab" + "c" colliding with "a" + "bc"
    assert LLMClient.key_for("ab", "c") != LLMClient.key_for("a", "bc")


def test_a_cached_prompt_is_returned_without_calling_out(tmp_path, monkeypatch):
    """A warm cache must not need credentials at all - this is what makes
    repeat runs free and byte-identical."""
    client = LLMClient(cache_dir=tmp_path)
    (tmp_path / "chat.json").write_text(
        json.dumps(
            {client.effective_key("k1", "p"): {"prompt": "p", "response": "cached answer"}}
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)

    assert client.chat("p", cache_key="k1") == "cached answer"
    assert client.usage.cached_hits == 1
    assert client.usage.chat_calls == 0


def test_changing_the_prompt_invalidates_the_cached_answer(tmp_path, monkeypatch):
    """A cache keyed only on the caller's label would serve a stale answer
    after a prompt fix, silently discarding the fix."""
    client = LLMClient(cache_dir=tmp_path)
    stale_key = client.effective_key("k1", "ORIGINAL PROMPT")
    (tmp_path / "chat.json").write_text(
        json.dumps({stale_key: {"prompt": "ORIGINAL PROMPT", "response": "old"}}),
        encoding="utf-8",
    )
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)

    # same cache_key, same prompt -> cached
    assert client.chat("ORIGINAL PROMPT", cache_key="k1") == "old"

    # same cache_key, EDITED prompt -> must not reuse it
    with pytest.raises(LLMUnavailable):
        client.chat("EDITED PROMPT", cache_key="k1")


def test_missing_groq_key_aborts_rather_than_returning_empty(tmp_path, monkeypatch):
    """A silent failure here would surface as a blank matched_note_id, which
    reads as a real 'unexplained' verdict and would be graded as wrong."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)

    client = LLMClient(cache_dir=tmp_path)
    with pytest.raises(LLMUnavailable, match="GROQ_API_KEY"):
        client.chat("uncached prompt", cache_key="miss")


def test_missing_gemini_key_aborts_on_embed(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)

    client = LLMClient(cache_dir=tmp_path)
    with pytest.raises(LLMUnavailable, match="GEMINI_API_KEY"):
        client.embed(["uncached text"])


def test_usage_reports_a_combined_call_count(tmp_path):
    client = LLMClient(cache_dir=tmp_path)
    client.usage.chat_calls = 3
    client.usage.embed_calls = 4
    assert client.usage.as_dict()["llm_calls"] == 7
