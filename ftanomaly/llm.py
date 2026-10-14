"""Two-provider model client with a content-hash disk cache.

Chat runs on Groq (open-weight gpt-oss-120b, free tier) through the
OpenAI-compatible SDK. Embeddings run on Gemini, because Groq serves no
embedding models.

The cache is what makes repeat runs byte-identical and free: a warm run
makes zero API calls. Cache files are committed; keys never are.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import config


class LLMUnavailable(RuntimeError):
    """Raised when a model call cannot be completed.

    Never swallowed: a missing verdict must abort the run rather than be
    written out as an empty matched_note_id, which would read as a real
    'unexplained' answer.
    """


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    chat_calls: int = 0
    embed_calls: int = 0
    cached_hits: int = 0

    def as_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "chat_calls": self.chat_calls,
            "embed_calls": self.embed_calls,
            "llm_calls": self.chat_calls + self.embed_calls,
            "cache_hits": self.cached_hits,
        }


@dataclass
class LLMClient:
    cache_dir: Path = field(default_factory=lambda: config.PATHS.cache_dir)
    usage: Usage = field(default_factory=Usage)
    _chat_client: object | None = field(default=None, repr=False)
    _embed_ready: bool = field(default=False, repr=False)

    # ---------------------------------------------------------------- cache
    def _cache_path(self, kind: str) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{kind}.json"

    def _load_cache(self, kind: str) -> dict:
        p = self._cache_path(kind)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def _save_cache(self, kind: str, cache: dict) -> None:
        self._cache_path(kind).write_text(
            json.dumps(cache, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def key_for(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _require_key(name: str) -> str:
        from dotenv import load_dotenv

        load_dotenv()
        value = os.environ.get(name)
        if not value:
            raise LLMUnavailable(
                f"{name} is not set and the answer is not cached. "
                "Copy .env.example to .env and add your key."
            )
        return value

    # ----------------------------------------------------------------- chat
    def _ensure_chat(self):
        if self._chat_client is None:
            from openai import OpenAI

            # NOTE: never use urllib here - Groq's Cloudflare rejects it
            # with 403 error code 1010. The OpenAI SDK is fine.
            self._chat_client = OpenAI(
                api_key=self._require_key("GROQ_API_KEY"),
                base_url=config.CHAT_BASE_URL,
            )
        return self._chat_client

    @classmethod
    def effective_key(cls, cache_key: str, prompt: str) -> str:
        """Fold the prompt into the cache key.

        Keying on the caller's label alone would serve a stale answer after
        a prompt is edited, silently discarding the edit. Including the
        prompt means a reworded prompt re-asks the model exactly once.
        """
        return cls.key_for(cache_key, prompt)

    def chat(self, prompt: str, *, cache_key: str) -> str:
        key = self.effective_key(cache_key, prompt)
        cache = self._load_cache("chat")
        if key in cache:
            self.usage.cached_hits += 1
            return cache[key]["response"]

        client = self._ensure_chat()
        try:
            resp = client.chat.completions.create(
                model=config.CHAT_MODEL,
                temperature=config.TEMPERATURE,
                seed=config.SEED,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content
        except LLMUnavailable:
            raise
        except Exception as exc:
            raise LLMUnavailable(f"chat call failed: {exc}") from exc
        if not text:
            raise LLMUnavailable("chat returned an empty response")

        self.usage.input_tokens += resp.usage.prompt_tokens
        self.usage.output_tokens += resp.usage.completion_tokens
        self.usage.chat_calls += 1

        cache[key] = {"prompt": prompt, "response": text}
        self._save_cache("chat", cache)
        return text

    # ----------------------------------------------------------- embeddings
    def _ensure_embed(self) -> None:
        if not self._embed_ready:
            import google.generativeai as genai

            genai.configure(api_key=self._require_key("GEMINI_API_KEY"))
            self._embed_ready = True

    def embed(self, texts: list[str]) -> list[list[float]]:
        cache = self._load_cache("embeddings")
        out: list[list[float]] = []
        dirty = False
        for text in texts:
            k = self.key_for("embed", config.EMBED_MODEL, text)
            if k in cache:
                self.usage.cached_hits += 1
                out.append(cache[k])
                continue
            try:
                import google.generativeai as genai

                self._ensure_embed()
                resp = genai.embed_content(model=config.EMBED_MODEL, content=text)
                vec = [float(x) for x in resp["embedding"]]
            except LLMUnavailable:
                raise
            except Exception as exc:
                raise LLMUnavailable(f"embedding call failed: {exc}") from exc
            self.usage.embed_calls += 1
            cache[k] = vec
            out.append(vec)
            dirty = True
        if dirty:
            self._save_cache("embeddings", cache)
        return out
