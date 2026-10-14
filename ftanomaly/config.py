"""Single source of truth for paths and tuned constants.

No other module in src/ may contain a threshold literal.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- weekly statistics -------------------------------------------------
TRAILING_WEEKS = 8           # brief-mandated rolling window
MIN_PRIOR_WEEKS_FOR_MAD = 4  # below this, MAD is too noisy to trust

# --- flagging rule (derivation in docs/THRESHOLD.md) -------------------
OWN_HISTORY_FLOOR_PCT = 10.0
MAD_MULTIPLIER = 3.0
PEER_THRESHOLD_PCT = 20.0

# --- LLM ---------------------------------------------------------------
# Chat runs on Groq (open-weight model, free tier). Groq serves no
# embedding models, so retrieval borrows Gemini's free embedding endpoint.
CHAT_BASE_URL = "https://api.groq.com/openai/v1"
CHAT_MODEL = "openai/gpt-oss-120b"
EMBED_MODEL = "models/text-embedding-004"
TEMPERATURE = 0.0
SEED = 42
RETRIEVAL_TOP_K = 3


@dataclass(frozen=True)
class Paths:
    shipments: Path = ROOT / "data" / "shipment_records.csv"
    notes: Path = ROOT / "data" / "context_notes.csv"
    sample_output: Path = ROOT / "data" / "sample_output_format_v2.csv"
    cache_dir: Path = ROOT / "cache"
    output_csv: Path = ROOT / "output" / "anomaly_report.csv"
    token_log: Path = ROOT / "reports" / "token_log.json"


PATHS = Paths()
