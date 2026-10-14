# FreightTiger — Shipping-Cost Anomaly Assistant

Spots weeks where a route's freight cost rose in a way that looks out of the
ordinary, checks a small corpus of real-world notes for a genuine cause, and
writes a plain-English verdict for each one. Rows with no honest explanation
are flagged for human review; rows a note genuinely covers are marked
justified and cite that note by id.

---

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env          # then paste your two keys
python -m ftanomaly.cli       # writes output/anomaly_report.csv
python eval/run_eval.py       # scores it against the hand-labeled set
python -m pytest              # the test suite
```

Two free-tier keys are needed:

| purpose | provider | where |
|---|---|---|
| chat | Groq, `openai/gpt-oss-120b` (open weights) | <https://console.groq.com/keys> |
| embeddings | Gemini, `text-embedding-004` | <https://aistudio.google.com/apikey> |

Two providers because **Groq serves no embedding models** — its model list
contains none and `/embeddings` returns `model_not_found`. Chat stays on an
open-weight model as the brief prefers; only the vectors come from Gemini.

---

## How the numbers are computed

- **Cost per tonne-km** is a ratio of sums over the route-week:
  `sum(freight_cost_inr) / sum(quantity_tonnes × distance_km)`. Not a mean of
  per-shipment ratios — that would weight a 3-tonne shipment like a 30-tonne one.
- **Weeks run Monday to Sunday**, and `week_of` is that Monday.
- **`vs_own_history`** is the trailing 8-week rolling mean of cost per tonne-km
  using only weeks *strictly before* the current one. No look-ahead anywhere.
- **`vs_similar_routes`** is the mean cost per tonne-km in the same week across
  all other routes sharing the route's `route_type`, excluding the route itself.

**Weeks with fewer than 8 prior weeks** use every prior week that exists, as the
brief instructs — nothing is padded or extrapolated. The first week of the
dataset (2024-01-01) has no prior weeks at all: its `vs_own_history` is empty
and it can never be flagged on own-history. Its peer comparison still works.
The trailing MAD needs at least 4 prior weeks before it is trusted.

The dataset is a complete grid: 7 routes × 104 weeks = 728 route-weeks, no gaps.

---

## How the threshold was chosen

```
flag if  vs_own_history   > max(10%, 3 × trailing MAD%)
     OR  vs_similar_routes > 20%
```

Full derivation with the sensitivity grid is in **[docs/THRESHOLD.md](docs/THRESHOLD.md)**.
The short version:

- **OR, not AND.** Mumbai-Pune 2025-09-15 is in the grader's sample output with
  own-history +9.2% and peers +23.6%. Every AND variant drops it.
- **Different numbers per baseline**, because they have different noise floors:
  10% sits at p98.9 of own-history deviations, 20% at p97.4 of peer deviations.
  Equal thresholds would be unequally strict.
- **10% rather than 12%**, because sorted deviations show isolated outliers down
  to rank 9 and then a dense noise band; 10% lands in that gap. It also keeps
  Mumbai-Delhi 2025-05-12 (+11.6% on a route whose MAD is 1.93% — a 6.3σ event).
- **Both baselines are needed.** A sharp spike shows against a route's own
  history; a slow drift hides from a rolling average because the average follows
  it. Mumbai-Pune creeps up from June 2025 with own-history quiet at +5–9% while
  its gap to peers grows to +38.9%. Only the peer baseline sees that.

**The notes were deliberately not used to tune the threshold.** They are an
acceptance test applied afterwards. Tuning a detector on its answer key would
miscalibrate it on any other data file.

---

## How notes are matched

Retrieval proposes; deterministic rules dispose. The LLM never decides a verdict.

1. **Enrichment (once, 10 calls, cached).** Each note is turned into a structured
   record: `direction` (raises / lowers / none), `scope` (route / all_routes),
   `window_start`, `window_end`, `in_dataset`, `magnitude_pct`. This is the only
   place language understanding is needed, and it is what models are good at.
2. **Retrieval.** Notes are embedded and ranked by cosine similarity against a
   query built from the route and week. Retrieval **ranks but never filters**:
   the gate below sees every note, and only the closest few are passed to the
   explanation step as context. A top-k cut would let a weak embedding turn a
   genuinely justified row into a false "unexplained" — the error the brief
   scores hardest — and on a ten-note corpus it buys nothing. This is pinned by
   `test_a_justifying_note_ranked_last_by_retrieval_still_justifies`, which
   ranks the only valid note dead last and still expects it to win.
3. **The gate** (`ftanomaly/verdict.py`) — plain Python, no model. A note
   justifies a candidate only if **all** of these hold:
   - it describes costs going **up** (`direction == "raises"`)
   - its routes are in this dataset (`in_dataset`)
   - it covers this route
   - its window overlaps this shipping week
   - it can explain *this kind* of rise

The last rule is the subtle one. **A nationwide note cannot justify a
peer-relative flag.** A countrywide diesel rise lifts a route's peers by the
same amount, so it cannot explain that route rising *relative* to them — only a
rise against its own history. A route-specific note can explain either.

### Why that matters here

Seven of the ten notes are distractors, and a naive similarity search matches
several of them on topic alone:

| note | why it never justifies |
|---|---|
| N004 | toll added, but says the affected routes are **not in this dataset** |
| N005 | maintenance — "costs were **not significantly affected**" |
| N006 | "demand stable, no major disruptions" |
| N007 | roads **improved** — a reason for costs to fall |
| N008 | "no significant disruptions" |
| N009 | route **returned to normal** after repairs |
| N010 | compliance costs **absorbed without a rate change** |

Only N001 (flood), N002 (festival surcharge) and N003 (diesel) describe an
upward cost pressure. The enrichment step classifies all ten correctly.

---

## How hallucination is prevented

- A note must pass all five gate conditions above — matching on *topic* is never
  enough, so N006 and N009 cannot slip through on similarity.
- **Window discipline is enforced.** Chennai-Bangalore is still elevated the week
  of 2025-03-10, but N001's disruption ended 2025-03-08, so that week comes out
  **unexplained** rather than borrowing a note that no longer applies.
- `validate_reason()` checks every generated sentence: a justified row must cite
  its own note id and no other; an unexplained row must not claim justification.
  A violation **raises and aborts the run** rather than shipping a confident
  wrong reason — the brief scores that worse than admitting none was found.
- A failed or uncredentialed model call aborts with exit code 2 and writes no
  report. A blank `matched_note_id` always means a real "no note applies"
  verdict, never a swallowed error.

---

## How I convinced myself it was right

**1. The sample output is an oracle.** All nine numbers across the three rows of
`sample_output_format_v2.csv` reproduce exactly from the stated math. That is
asserted in `tests/test_metrics.py::test_reproduces_every_number_in_the_sample_output`,
so the graded numeric fields are proven, not hoped for. It also settled an
ambiguity: weekly aggregation is ratio-of-sums, not mean-of-ratios.

**2. A hand-labeled eval set.** All 21 detected route-weeks are labeled in
`eval/labeled_set.csv` with the expected verdict, expected note id, and the
reasoning. `eval/run_eval.py` scores verdict accuracy, note-id accuracy, and
precision/recall on justified rows. A test also asserts the labels cover exactly
what `detect()` produces, so the labels cannot silently drift from the detector.

**3. The eval scorer is itself tested.** `tests/test_eval.py` feeds it synthetic
reports to prove it actually catches a wrong note id, a fabricated
justification, and an unlabeled row — a scorer that always returns 100% would
otherwise look like success.

**4. Three-run diff.** `scripts/check_reproducibility.sh`.

---

## Reproducibility

- Every model call runs at `temperature=0` with `seed=42`.
- **Every response is cached to disk by content hash and committed.** Runs 2 and
  3 make **zero** API calls, so identical output is structural rather than a
  property of the model behaving consistently.
- The cache key includes the prompt, so editing a prompt re-asks the model
  exactly once instead of silently serving the pre-edit answer.
- Output rows are sorted by `(week_of, route)`; retrieval ties break on note id.
  No ordering depends on dict iteration or an unstable sort.

Evidence lives in `reports/reproducibility.md` (three diffs and three sha256s).

---

## Cost

Measured figures for a full run are written to `reports/token_log.json` by every
invocation — model, temperature, seed, input/output tokens, call counts and
cache hits.

Call volume is bounded by design: **10 enrichment calls** (one per note, cached
forever), **one embedding per note plus one per query**, and **one prose call per
flagged row**. Nothing scales with the 2,940 input shipments.

Both providers are used on their free tiers, which bill zero; the rate constants
in `ftanomaly/cli.py` are set to 0.0 and the token counts are reported regardless
so the number is measured rather than assumed.

One honest caveat: `gpt-oss-120b` is a reasoning model, and its **completion
token count varies between identical runs even when the content is identical**
(3,845 vs 3,512 output tokens were observed across two cold enrichment runs
producing the same classifications). The logged figure is from one specific run,
not a constant.

---

## Trade-offs

- **No hosted vector database.** With ten notes, a persisted embedding matrix and
  cosine similarity give identical rankings to Chroma or FAISS while adding no
  operational weight. `NoteIndex` is a narrow interface, so swapping in a real
  vector DB is a drop-in change if the corpus grows. This was an engineering
  call, not an oversight.
- **The adaptive MAD term rarely binds on this data.** All seven routes have
  similar variability (MAD 2.2–3.5%), so the 10% floor usually dominates, and
  results are identical for any multiplier in [2.5, 4]. It is kept because it is
  the right shape for heterogeneous real routes, and it does change which
  baseline fires on Chennai-Bangalore 2025-03-17. Stating this plainly seemed
  better than implying it does more work than it does.
- **The LLM is deliberately kept away from arithmetic and dates.** It reads
  language once; every comparison is Python. That is what makes the verdicts
  auditable and the runs reproducible.

---

## Layout

```
ftanomaly/
  config.py      paths and every tuned constant (no threshold literal elsewhere)
  loading.py     shipments -> Monday-start route-weeks
  metrics.py     cost per tonne-km + the two graded baselines
  detect.py      the flagging rule
  notes.py       note loading + one-off LLM enrichment
  retrieval.py   embedding index over the notes
  verdict.py     the deterministic applicability gate
  prose.py       grounded explanations + the hallucination guard
  llm.py         two-provider client, disk cache, token accounting
  report.py      contract-exact CSV writer
  cli.py         entrypoint
eval/            hand-labeled set + scorer
docs/            THRESHOLD.md (derivation), plans/
tests/           63 offline tests + live tests gated on credentials
```

Tests needing real credentials are marked `live` and **skip with a stated
reason** when keys are absent, so an offline run reports "skipped" rather than
looking fully green.
