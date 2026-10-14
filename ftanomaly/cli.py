"""Run the whole pipeline and record what it cost."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config
from .detect import detect
from .llm import LLMClient, LLMUnavailable
from .loading import load_route_weeks
from .metrics import build_metrics
from .notes import enrich_notes, load_notes
from .prose import write_reason
from .report import to_row, write_report
from .retrieval import NoteIndex
from .verdict import decide

# Both providers are used on their free tiers, which bill zero. Set these
# to the published paid rates if that ever changes, so the reported cost
# stays measured rather than assumed. Token counts are always real.
INPUT_RATE_PER_1M = 0.0
OUTPUT_RATE_PER_1M = 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FreightTiger cost anomaly assistant")
    parser.add_argument("--output", type=Path, default=config.PATHS.output_csv)
    parser.add_argument("--token-log", type=Path, default=config.PATHS.token_log)
    args = parser.parse_args(argv)

    client = LLMClient()

    try:
        route_weeks = load_route_weeks(config.PATHS.shipments)
        metrics = build_metrics(route_weeks)
        candidates = detect(metrics)

        raw_notes = load_notes(config.PATHS.notes)
        enriched = enrich_notes(raw_notes, client)
        all_ids = {n.note_id for n in raw_notes}

        index = NoteIndex()
        index.build(enriched, client)

        verdicts = decide(candidates, index, client)
        rows = [to_row(v, write_reason(v, client, all_ids)) for v in verdicts]
    except LLMUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Refusing to emit a partial report.", file=sys.stderr)
        return 2

    write_report(rows, args.output)

    usage = client.usage.as_dict()
    usage["cost_usd"] = round(
        usage["input_tokens"] / 1e6 * INPUT_RATE_PER_1M
        + usage["output_tokens"] / 1e6 * OUTPUT_RATE_PER_1M,
        6,
    )
    usage["chat_model"] = config.CHAT_MODEL
    usage["chat_provider"] = config.CHAT_BASE_URL
    usage["embed_model"] = config.EMBED_MODEL
    usage["temperature"] = config.TEMPERATURE
    usage["seed"] = config.SEED
    usage["rows_written"] = len(rows)
    args.token_log.parent.mkdir(parents=True, exist_ok=True)
    args.token_log.write_text(
        json.dumps(usage, indent=2, sort_keys=True), encoding="utf-8"
    )

    justified = sum(1 for r in rows if r["flagged"] == "No (justified)")
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"  flagged (unexplained): {len(rows) - justified}")
    print(f"  justified by a note:   {justified}")
    print(f"  llm calls: {usage['llm_calls']}  cache hits: {usage['cache_hits']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
