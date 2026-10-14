# FreightTiger Shipping-Cost Anomaly Assistant — Implementation Plan


**Goal:** Detect week-over-week freight cost anomalies per route, decide which are justified by a retrieved context note, and emit a CSV matching the grader's contract exactly.

**Architecture:** A deterministic statistics core computes weekly cost-per-tonne-km and two baselines, then flags anomalies with a fixed rule. An LLM is used in exactly two narrow places — enriching the 10 context notes into structured records once, and writing prose for each flagged row — both disk-cached and keyed by content hash so repeat runs are byte-identical and make zero API calls. All verdict logic (does this note apply to this route, in this week, in this direction) is plain Python, never the LLM.

**Tech Stack:** Python 3.11, stdlib `csv`/`datetime`, pytest. Two free-tier providers, both verified working against this machine:
- **Chat — Groq**, model `openai/gpt-oss-120b` (open-weight), reached through the installed `openai` SDK pointed at `https://api.groq.com/openai/v1`.
- **Embeddings — Gemini**, model `models/text-embedding-004`, via `google-generativeai`. Groq serves **no** embedding models, so retrieval needs this second provider.

**Verified environment facts (do not re-derive):**
- `urllib` is blocked by Groq's Cloudflare with `403 error code: 1010`. Use the `openai` SDK or `requests`; never `urllib`.
- `openai/gpt-oss-120b` is a *reasoning* model: identical prompts give identical **content** but different **completion_token** counts (observed 93 vs 122 for one prompt). The cost log must report measured tokens and say this plainly.
- Temperature 0 with `seed=42` returned identical content across repeated calls in testing. The disk cache is what *guarantees* reproducibility; the seed only reduces drift on a cold cache.

**Spec:** `../AI Intern Case Study/AI Intern Case Study/FreightTiger_Intern_CaseStudy.pdf` (the 24-hour challenge brief). Numeric decisions derived from it are pinned in Global Constraints below.

## Global Constraints

- **Cost per tonne-km is ratio-of-sums**, never mean-of-ratios: `sum(freight_cost_inr) / sum(quantity_tonnes * distance_km)` over the route-week. Verified against the sample output.
- **Weeks are Monday-start.** `week_of` is that Monday's date, ISO format `YYYY-MM-DD`.
- **`vs_own_history`** = trailing **8-week** rolling mean of cost-per-tonne-km using only weeks **strictly before** the current week. No look-ahead, ever. Fewer than 8 prior weeks → use all available and state it in the README.
- **`vs_similar_routes`** = mean cost-per-tonne-km in the **same week** across all other routes of the same `route_type`, **excluding the route itself**.
- **Flagging rule (locked):** `vs_own_history > max(10.0%, 3.0 * trailing_MAD_pct)` **OR** `vs_similar_routes > 20.0%`. OR, never AND.
- **Thresholds live in `config.py` only.** No numeric threshold literal anywhere else in `src/`.
- **Output columns, exact order:** `route,week_of,cost_per_tonne_km,vs_own_history,vs_similar_routes,flagged,matched_note_id,reason`
- **`flagged` vocabulary is exactly two values:** `Yes` and `No (justified)`. Never invent a third.
- **`cost_per_tonne_km`** formats to 2 decimals. Comparison columns format as `+X.Y% vs this route's past average` and `+X.Y% vs similar-length routes this week` — signed, one decimal.
- **`matched_note_id`** is the note id (e.g. `N002`) or empty string. Never a placeholder like `NONE`.
- **LLM temperature is 0 and `seed=42`** on every call. Any deviation must be recorded in the README with evidence it does not move a verdict.
- **Every LLM response is cached to disk and committed.** Runs 2 and 3 must make zero API calls.
- **Output rows are sorted by `(week_of, route)`** before writing. No reliance on dict insertion order.
- **No secret ever reaches the repository.** This is submitted to GitHub, whose secret scanner will flag a committed Groq or Gemini key within minutes of a push. `.env` is gitignored; only `.env.example` with placeholder values is committed. Task 12 Step 5 verifies this before the first push.

## Review Focus

1. **Warm-up weeks (first 8 Mondays of 2024) have fewer than 8 prior weeks.** `vs_own_history` must compute from what exists, and the very first week (zero priors) must not divide by zero — it emits `None` and can never be flagged on own-history. Test owned by Task 3.
2. **An "All Routes" note cannot justify a peer-based flag.** A nationwide fuel rise lifts a route's peers equally, so it cannot explain that route rising *relative* to them. Test owned by Task 7.
3. **Note windows must exclude weeks starting after the window closes.** Chennai-Bangalore is still elevated the week of 2025-03-10, but N001's disruption ended 2025-03-08 — that week must come out unexplained. Test owned by Task 7.
4. **LLM failure must be loud.** A timeout, rate-limit, or missing API key must abort with a non-zero exit, never silently leave `matched_note_id` blank — a blank reads as a real "unexplained" verdict and would be graded as a wrong answer. Test owned by Task 8.
5. **Ordering must be total and deterministic.** Equal cosine scores in retrieval and equal sort keys in the report must break ties on a stable field (note id, then route), so three runs produce byte-identical CSVs. Test owned by Task 9.

---

### Task 1: Project scaffold, config, and input data

**Files:**
- Create: `.gitignore`, `requirements.txt`, `.env.example`, `src/ftanomaly/__init__.py`, `src/ftanomaly/config.py`, `pytest.ini`
- Create: `data/shipment_records.csv`, `data/context_notes.csv`, `data/sample_output_format_v2.csv` (copied from the case study folder)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `config.PATHS` (frozen dataclass with `.shipments`, `.notes`, `.sample_output`, `.cache_dir`, `.output_csv`, `.token_log`) and constants `OWN_HISTORY_FLOOR_PCT = 10.0`, `MAD_MULTIPLIER = 3.0`, `PEER_THRESHOLD_PCT = 20.0`, `TRAILING_WEEKS = 8`, `MIN_PRIOR_WEEKS_FOR_MAD = 4`, `CHAT_MODEL`, `EMBED_MODEL`, `TEMPERATURE = 0.0`, `RETRIEVAL_TOP_K = 3`.

- [ ] **Step 1: Initialise the repository and copy inputs**

```bash
cd "C:/Drive D/College/Applied Companies Project/Freight Tiger/freight-tiger-anomaly"
git init
mkdir -p src/ftanomaly tests eval cache output reports data
CS="../AI Intern Case Study/AI Intern Case Study"
cp "$CS/shipment_records.csv" "$CS/context_notes.csv" "$CS/sample_output_format_v2.csv" data/
touch src/ftanomaly/__init__.py output/.gitkeep
ls -la data/
```

- [ ] **Step 2: Write `pytest.ini` so `src/` is importable**

```ini
[pytest]
pythonpath = src
testpaths = tests
```

- [ ] **Step 3: Write `.gitignore`, `requirements.txt`, `.env.example`**

`.gitignore`:
```
__pycache__/
*.pyc
.env
.pytest_cache/
output/*.csv
!output/.gitkeep
```

`requirements.txt`:
```
openai>=2.0
google-generativeai>=0.8.3
python-dotenv>=1.0
pytest>=8.0
```

`.env.example` (**placeholders only — never commit real values**):
```
# Chat. Free key from https://console.groq.com/keys
GROQ_API_KEY=your-groq-key-here

# Embeddings only. Free key from https://aistudio.google.com/apikey
GEMINI_API_KEY=your-gemini-key-here
```

- [ ] **Step 4: Write `src/ftanomaly/config.py`**

```python
"""Single source of truth for paths and tuned constants.

No other module in src/ may contain a threshold literal.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

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
```

- [ ] **Step 5: Write the test `tests/test_config.py`**

```python
from ftanomaly import config


def test_input_files_are_present():
    assert config.PATHS.shipments.exists()
    assert config.PATHS.notes.exists()
    assert config.PATHS.sample_output.exists()


def test_thresholds_are_the_locked_values():
    assert config.OWN_HISTORY_FLOOR_PCT == 10.0
    assert config.PEER_THRESHOLD_PCT == 20.0
    assert config.MAD_MULTIPLIER == 3.0
    assert config.TRAILING_WEEKS == 8
```

- [ ] **Step 6: Run it**

Run: `python -m pytest tests/test_config.py -v`
Expected: 2 passed. A failure means the `cp` in Step 1 used wrong paths — fix before continuing.

- [ ] **Step 7: Put both keys in `.env` and probe both providers**

`.env` already exists with the Groq key. Add a free Gemini key from <https://aistudio.google.com/apikey> — it is needed only for embeddings.

```bash
cat .env   # must contain GROQ_API_KEY and GEMINI_API_KEY
python -c "
import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
load_dotenv()

c = OpenAI(api_key=os.environ['GROQ_API_KEY'], base_url='https://api.groq.com/openai/v1')
r = c.chat.completions.create(model='openai/gpt-oss-120b', temperature=0, seed=42,
                              messages=[{'role':'user','content':'Reply with one word: ok'}])
print('groq chat  :', r.choices[0].message.content.strip())

genai.configure(api_key=os.environ['GEMINI_API_KEY'])
v = genai.embed_content(model='models/text-embedding-004', content='test')['embedding']
print('gemini embed: dim', len(v))
"
```

Expected: `groq chat : ok` and an embedding dimension (768). If the Groq model id is rejected, pick another chat model from `c.models.list()` — `qwen/qwen3.8-27b` is the fallback.

**Do not use `urllib` anywhere in this project.** Groq's Cloudflare returns `403 error code: 1010` to it. The `openai` SDK and `requests` both work.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: scaffold project, pin thresholds, vendor case-study inputs"
```

---

### Task 2: Load shipments into weekly route series

**Files:**
- Create: `src/ftanomaly/loading.py`
- Test: `tests/test_loading.py`

**Interfaces:**
- Consumes: `config.PATHS.shipments`.
- Produces:
  - `RouteWeek` frozen dataclass: `route: str`, `route_type: str`, `week_of: datetime.date`, `total_cost: float`, `total_tonne_km: float`, `shipment_count: int`
  - `monday_of(d: datetime.date) -> datetime.date`
  - `load_route_weeks(path: Path) -> list[RouteWeek]` sorted by `(route, week_of)`

- [ ] **Step 1: Write the failing test `tests/test_loading.py`**

```python
import datetime as dt

from ftanomaly import config
from ftanomaly.loading import load_route_weeks, monday_of


def test_monday_of_snaps_to_week_start():
    assert monday_of(dt.date(2024, 1, 3)) == dt.date(2024, 1, 1)   # Wednesday
    assert monday_of(dt.date(2024, 1, 1)) == dt.date(2024, 1, 1)   # Monday
    assert monday_of(dt.date(2024, 1, 7)) == dt.date(2024, 1, 1)   # Sunday


def test_dataset_shape_is_a_complete_grid():
    rws = load_route_weeks(config.PATHS.shipments)
    assert len(rws) == 728                      # 7 routes x 104 weeks
    assert len({r.route for r in rws}) == 7
    assert len({r.week_of for r in rws}) == 104
    for route in {r.route for r in rws}:
        assert sum(1 for r in rws if r.route == route) == 104


def test_route_types_come_from_the_file_not_derived():
    types = {r.route: r.route_type for r in load_route_weeks(config.PATHS.shipments)}
    assert types["Mumbai-Pune"] == "Short"
    assert types["Ahmedabad-Mumbai"] == "Medium"
    assert types["Mumbai-Delhi"] == "Long"


def test_aggregates_are_sums():
    rws = {(r.route, r.week_of): r for r in load_route_weeks(config.PATHS.shipments)}
    wk = rws[("Mumbai-Pune", dt.date(2024, 1, 1))]
    assert wk.shipment_count >= 3
    assert wk.total_cost > 0
    assert wk.total_tonne_km > 0
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/test_loading.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ftanomaly.loading'`

- [ ] **Step 3: Write `src/ftanomaly/loading.py`**

```python
"""Read raw shipment rows and fold them into Monday-start route-weeks."""
from __future__ import annotations

import csv
import datetime as dt
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RouteWeek:
    route: str
    route_type: str
    week_of: dt.date
    total_cost: float
    total_tonne_km: float
    shipment_count: int


def monday_of(d: dt.date) -> dt.date:
    """Monday of the week containing `d`."""
    return d - dt.timedelta(days=d.weekday())


def load_route_weeks(path: Path) -> list[RouteWeek]:
    cost: dict[tuple[str, dt.date], float] = defaultdict(float)
    tonne_km: dict[tuple[str, dt.date], float] = defaultdict(float)
    count: dict[tuple[str, dt.date], int] = defaultdict(int)
    route_type: dict[str, str] = {}

    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            route = f"{row['origin']}-{row['destination']}"
            route_type[route] = row["route_type"]
            week = monday_of(dt.date.fromisoformat(row["shipment_date"]))
            key = (route, week)
            cost[key] += float(row["freight_cost_inr"])
            tonne_km[key] += float(row["quantity_tonnes"]) * float(row["distance_km"])
            count[key] += 1

    return sorted(
        (
            RouteWeek(
                route=route,
                route_type=route_type[route],
                week_of=week,
                total_cost=cost[(route, week)],
                total_tonne_km=tonne_km[(route, week)],
                shipment_count=count[(route, week)],
            )
            for (route, week) in cost
        ),
        key=lambda r: (r.route, r.week_of),
    )
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_loading.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/ftanomaly/loading.py tests/test_loading.py pytest.ini
git commit -m "feat: aggregate shipments into Monday-start route-weeks"
```

---

### Task 3: Weekly metrics and the two graded baselines

This task carries the **oracle test**. All nine numbers across the three rows of `sample_output_format_v2.csv` reproduce exactly from the brief's math, so they are a free correctness check on the whole statistics core. If this task passes, the graded numeric fields are right.

**Files:**
- Create: `src/ftanomaly/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `loading.RouteWeek`, `config.TRAILING_WEEKS`, `config.MIN_PRIOR_WEEKS_FOR_MAD`.
- Produces:
  - `WeekMetrics` frozen dataclass: `route`, `route_type`, `week_of`, `cost_per_tonne_km: float`, `vs_own_history: float | None`, `vs_similar_routes: float | None`, `trailing_mad_pct: float | None`, `n_prior_weeks: int`. Percentages are signed numbers like `35.5`, not strings.
  - `median(xs: list[float]) -> float`
  - `mad_pct(xs: list[float]) -> float`
  - `build_metrics(route_weeks: list[RouteWeek]) -> list[WeekMetrics]`

- [ ] **Step 1: Write the failing test `tests/test_metrics.py`**

```python
import csv
import datetime as dt

import pytest

from ftanomaly import config
from ftanomaly.loading import load_route_weeks
from ftanomaly.metrics import build_metrics, mad_pct, median


@pytest.fixture(scope="module")
def by_key():
    rws = load_route_weeks(config.PATHS.shipments)
    return {(m.route, m.week_of): m for m in build_metrics(rws)}


def test_median_and_mad_helpers():
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([4.0, 1.0, 2.0, 3.0]) == 2.5
    assert mad_pct([5.0, 5.0, 5.0]) == 0.0


def test_reproduces_every_number_in_the_sample_output(by_key):
    """The grader's sample rows are real reference values - treat as an oracle."""
    with config.PATHS.sample_output.open(newline="", encoding="utf-8") as fh:
        expected = list(csv.DictReader(fh))
    assert len(expected) == 3

    for row in expected:
        key = (row["route"], dt.date.fromisoformat(row["week_of"]))
        m = by_key[key]
        assert round(m.cost_per_tonne_km, 2) == float(row["cost_per_tonne_km"]), key
        own = float(row["vs_own_history"].split("%")[0])     # "+35.5% vs ..." -> 35.5
        peer = float(row["vs_similar_routes"].split("%")[0])
        assert round(m.vs_own_history, 1) == own, key
        assert round(m.vs_similar_routes, 1) == peer, key


def test_first_week_has_no_history_and_never_divides_by_zero(by_key):
    for (route, week), m in by_key.items():
        if week == dt.date(2024, 1, 1):
            assert m.vs_own_history is None
            assert m.n_prior_weeks == 0
            assert m.vs_similar_routes is not None   # peers exist in week 1


def test_warmup_weeks_use_all_available_prior_weeks(by_key):
    m = by_key[("Mumbai-Pune", dt.date(2024, 1, 29))]
    assert m.n_prior_weeks == 4          # Jan 1, 8, 15, 22
    assert m.vs_own_history is not None


def test_trailing_window_never_exceeds_eight(by_key):
    assert by_key[("Mumbai-Pune", dt.date(2025, 6, 2))].n_prior_weeks == 8


def test_peer_average_excludes_the_route_itself(by_key):
    """Mumbai-Pune and Delhi-Jaipur are the only two Short routes, so each
    one's peer average is exactly the other's cost."""
    wk = dt.date(2025, 9, 15)
    mp = by_key[("Mumbai-Pune", wk)]
    dj = by_key[("Delhi-Jaipur", wk)]
    implied_peer = mp.cost_per_tonne_km / (1 + mp.vs_similar_routes / 100)
    assert round(implied_peer, 6) == round(dj.cost_per_tonne_km, 6)
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ftanomaly.metrics'`

- [ ] **Step 3: Write `src/ftanomaly/metrics.py`**

```python
"""Weekly cost-per-tonne-km and the two baselines the brief grades.

No look-ahead anywhere: a week's own-history baseline is built strictly
from weeks before it.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass

from . import config
from .loading import RouteWeek

_MAD_SCALE = 1.4826  # makes MAD comparable to a standard deviation


@dataclass(frozen=True)
class WeekMetrics:
    route: str
    route_type: str
    week_of: dt.date
    cost_per_tonne_km: float
    vs_own_history: float | None
    vs_similar_routes: float | None
    trailing_mad_pct: float | None
    n_prior_weeks: int


def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        raise ValueError("median of empty sequence")
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def mad_pct(xs: list[float]) -> float:
    """Scaled median absolute deviation as a percentage of the median.

    Relative, so it compares against a percentage threshold regardless of
    a route's absolute price level.
    """
    med = median(xs)
    if med == 0:
        return 0.0
    spread = median([abs(x - med) for x in xs]) * _MAD_SCALE
    return spread / med * 100.0


def build_metrics(route_weeks: list[RouteWeek]) -> list[WeekMetrics]:
    cpk = {
        (rw.route, rw.week_of): rw.total_cost / rw.total_tonne_km
        for rw in route_weeks
    }
    route_type = {rw.route: rw.route_type for rw in route_weeks}
    weeks = sorted({rw.week_of for rw in route_weeks})
    routes = sorted({rw.route for rw in route_weeks})

    by_type: dict[str, list[str]] = defaultdict(list)
    for r in routes:
        by_type[route_type[r]].append(r)

    out: list[WeekMetrics] = []
    for route in routes:
        for i, week in enumerate(weeks):
            current = cpk[(route, week)]
            prior = [
                cpk[(route, weeks[j])]
                for j in range(max(0, i - config.TRAILING_WEEKS), i)
            ]

            own = (current / (sum(prior) / len(prior)) - 1) * 100 if prior else None
            mad = (
                mad_pct(prior)
                if len(prior) >= config.MIN_PRIOR_WEEKS_FOR_MAD
                else None
            )

            peers = [
                cpk[(other, week)]
                for other in by_type[route_type[route]]
                if other != route
            ]
            peer = (current / (sum(peers) / len(peers)) - 1) * 100 if peers else None

            out.append(
                WeekMetrics(
                    route=route,
                    route_type=route_type[route],
                    week_of=week,
                    cost_per_tonne_km=current,
                    vs_own_history=own,
                    vs_similar_routes=peer,
                    trailing_mad_pct=mad,
                    n_prior_weeks=len(prior),
                )
            )
    return out
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: 6 passed. **If the oracle test fails, stop and fix the math — every downstream task depends on it.**

- [ ] **Step 5: Commit**

```bash
git add src/ftanomaly/metrics.py tests/test_metrics.py
git commit -m "feat: weekly cost-per-tonne-km with 8-week and peer baselines

Validated against all nine reference numbers in the provided sample output."
```

---

### Task 4: Anomaly detection rule

**Files:**
- Create: `src/ftanomaly/detect.py`
- Create: `docs/THRESHOLD.md` (the derivation, for the README and walkthrough)
- Test: `tests/test_detect.py`

**Interfaces:**
- Consumes: `metrics.WeekMetrics`, the four threshold constants from `config`.
- Produces:
  - `Candidate` frozen dataclass: `metrics: WeekMetrics`, `own_history_triggered: bool`, `peer_triggered: bool`
  - `own_history_bar(m: WeekMetrics) -> float | None` — the adaptive bar for that route-week
  - `detect(all_metrics: list[WeekMetrics]) -> list[Candidate]` sorted by `(week_of, route)`

- [ ] **Step 1: Write the failing test `tests/test_detect.py`**

```python
import datetime as dt

import pytest

from ftanomaly import config
from ftanomaly.detect import detect, own_history_bar
from ftanomaly.loading import load_route_weeks
from ftanomaly.metrics import build_metrics


@pytest.fixture(scope="module")
def candidates():
    rws = load_route_weeks(config.PATHS.shipments)
    return detect(build_metrics(rws))


def test_flags_exactly_the_expected_count(candidates):
    assert len(candidates) == 21


def test_includes_every_sample_output_row(candidates):
    keys = {(c.metrics.route, c.metrics.week_of) for c in candidates}
    for route, week in [
        ("Delhi-Jaipur", "2024-11-11"),
        ("Ahmedabad-Mumbai", "2025-01-20"),
        ("Mumbai-Pune", "2025-09-15"),
    ]:
        assert (route, dt.date.fromisoformat(week)) in keys, (route, week)


def test_slow_drift_is_caught_by_peers_not_by_own_history(candidates):
    """Mumbai-Pune creeps up slowly, so its rolling baseline follows it and
    own-history stays quiet. Only the peer comparison sees the drift."""
    c = next(
        c for c in candidates
        if c.metrics.route == "Mumbai-Pune"
        and c.metrics.week_of == dt.date(2025, 9, 15)
    )
    assert c.peer_triggered is True
    assert c.own_history_triggered is False


def test_cost_drops_are_never_flagged(candidates):
    for c in candidates:
        rose = (c.metrics.vs_own_history or 0) > 0 or (c.metrics.vs_similar_routes or 0) > 0
        assert rose, (c.metrics.route, c.metrics.week_of)


def test_adaptive_bar_rises_for_a_volatile_route():
    """A route whose recent weeks are turbulent needs a bigger move to count."""
    from ftanomaly.metrics import WeekMetrics

    quiet = WeekMetrics("R", "Short", dt.date(2025, 1, 6), 3.0, 12.0, 0.0, 1.9, 8)
    noisy = WeekMetrics("R", "Short", dt.date(2025, 1, 6), 3.0, 12.0, 0.0, 6.3, 8)
    assert own_history_bar(quiet) == 10.0           # floor binds
    assert own_history_bar(noisy) == pytest.approx(18.9)  # 3 x MAD binds


def test_rows_without_enough_history_are_not_history_flagged():
    from ftanomaly.metrics import WeekMetrics

    warmup = WeekMetrics("R", "Short", dt.date(2024, 1, 1), 3.0, None, 0.0, None, 0)
    assert own_history_bar(warmup) is None


def test_output_is_sorted_deterministically(candidates):
    keys = [(c.metrics.week_of, c.metrics.route) for c in candidates]
    assert keys == sorted(keys)
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/test_detect.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ftanomaly.detect'`

- [ ] **Step 3: Write `src/ftanomaly/detect.py`**

```python
"""Decide which route-weeks look out of the ordinary.

Rule (locked, see docs/THRESHOLD.md):
    vs_own_history   > max(OWN_HISTORY_FLOOR_PCT, MAD_MULTIPLIER * trailing_MAD)
 OR vs_similar_routes > PEER_THRESHOLD_PCT

The two baselines are combined with OR because they catch different
failures: a sharp spike shows up against the route's own history, while a
slow drift hides from a rolling average and only shows against peers.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config
from .metrics import WeekMetrics


@dataclass(frozen=True)
class Candidate:
    metrics: WeekMetrics
    own_history_triggered: bool
    peer_triggered: bool


def own_history_bar(m: WeekMetrics) -> float | None:
    """How big a rise must be, on this route, this week, to count.

    Returns None when there is not enough history to judge.
    """
    if m.vs_own_history is None or m.trailing_mad_pct is None:
        return None
    return max(
        config.OWN_HISTORY_FLOOR_PCT,
        config.MAD_MULTIPLIER * m.trailing_mad_pct,
    )


def detect(all_metrics: list[WeekMetrics]) -> list[Candidate]:
    out: list[Candidate] = []
    for m in all_metrics:
        bar = own_history_bar(m)
        own = bar is not None and m.vs_own_history > bar
        peer = (
            m.vs_similar_routes is not None
            and m.vs_similar_routes > config.PEER_THRESHOLD_PCT
        )
        if own or peer:
            out.append(Candidate(metrics=m, own_history_triggered=own,
                                 peer_triggered=peer))
    return sorted(out, key=lambda c: (c.metrics.week_of, c.metrics.route))
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_detect.py -v`
Expected: 7 passed

- [ ] **Step 5: Write `docs/THRESHOLD.md`**

Record the derivation so the README and the walkthrough can cite it. Content to write:

```markdown
# How the flagging threshold was chosen

Rule: `vs_own_history > max(10%, 3 x trailing MAD%)` OR `vs_similar_routes > 20%`.

## Why OR, not AND
Mumbai-Pune 2025-09-15 appears in the grader's sample output with
own-history +9.2% and peers +23.6%. Every AND variant drops it. The brief
also says "its own history **or** similar routes".

## Why the two baselines need different numbers
They have different noise floors, so equal thresholds are not equally strict:

| threshold | percentile of own-history noise | percentile of peer noise |
|---|---|---|
| 10% | 98.9% | 87.8% |
| 20% | 99.2% | 97.4% |

10% on own-history and 20% on peers both sit near p98.

## Why 10% and not 12% or 15%
Sorted own-history deviations show isolated points down to rank 9, then a
dense noise band from rank 10 (gaps collapse from ~2pp to <0.4pp). A 10%
cut lands inside that gap. It also preserves Mumbai-Delhi 2025-05-12,
which is +11.6% on a route whose MAD is only 1.93% — a 6.3-sigma event.
A 12% floor would discard a 6-sigma outlier for no statistical reason.

Under this rule every own-history flag clears 6 sigma for its own route.

## What the adaptive term does
On this dataset all seven routes have similar variability (MAD 2.2-3.5%),
so the 10% floor usually binds. The MAD term still matters in two places:
it keeps the quiet-route event above (Mumbai-Delhi) and it suppresses
Chennai-Bangalore 2025-03-17, where recent flood weeks lifted the route's
own volatility so +13.7% is no longer remarkable for it.

## Note on methodology
The threshold was derived from the distribution alone. Context notes were
deliberately **not** used to tune it - they are an acceptance test applied
after the fact, never an input to detection.
```

- [ ] **Step 6: Commit**

```bash
git add src/ftanomaly/detect.py tests/test_detect.py docs/THRESHOLD.md
git commit -m "feat: adaptive anomaly detection rule with derivation doc"
```

---

### Task 5: Context notes and one-off LLM enrichment

The LLM reads each note once and turns it into a structured record. Date arithmetic and applicability are **not** delegated to it — it only does language understanding, which is what it is good at. Ten calls total, cached to disk and committed.

**Files:**
- Create: `src/ftanomaly/llm.py` (client, token accounting, content-hash cache)
- Create: `src/ftanomaly/notes.py` (load + enrich)
- Test: `tests/test_notes.py`

**Interfaces:**
- Consumes: `config.PATHS.notes`, `config.CHAT_MODEL`, `config.TEMPERATURE`.
- Produces:
  - `llm.LLMClient` with `.chat(prompt: str, *, cache_key: str) -> str`, `.embed(texts: list[str]) -> list[list[float]]`, `.usage -> Usage` (fields `input_tokens`, `output_tokens`, `calls`, `cached_hits`)
  - `llm.LLMUnavailable(RuntimeError)`
  - `notes.Note` frozen dataclass: `note_id`, `date: dt.date`, `applies_to: str`, `text: str`
  - `notes.EnrichedNote` frozen dataclass: `note: Note`, `direction: str` (`raises` | `lowers` | `none`), `scope: str` (`route` | `all_routes`), `window_start: dt.date`, `window_end: dt.date | None` (None = open-ended), `in_dataset: bool`, `magnitude_pct: float | None`
  - `notes.load_notes(path) -> list[Note]`
  - `notes.enrich_notes(notes: list[Note], client: LLMClient) -> list[EnrichedNote]`

- [ ] **Step 1: Write `src/ftanomaly/llm.py`**

```python
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

    def chat(self, prompt: str, *, cache_key: str) -> str:
        cache = self._load_cache("chat")
        if cache_key in cache:
            self.usage.cached_hits += 1
            return cache[cache_key]["response"]

        client = self._ensure_chat()
        try:
            resp = client.chat.completions.create(
                model=config.CHAT_MODEL,
                temperature=config.TEMPERATURE,
                seed=config.SEED,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content
        except Exception as exc:
            raise LLMUnavailable(f"chat call failed: {exc}") from exc
        if not text:
            raise LLMUnavailable("chat returned an empty response")

        self.usage.input_tokens += resp.usage.prompt_tokens
        self.usage.output_tokens += resp.usage.completion_tokens
        self.usage.chat_calls += 1

        cache[cache_key] = {"prompt": prompt, "response": text}
        self._save_cache("chat", cache)
        return text

    # ----------------------------------------------------------- embeddings
    def _ensure_embed(self):
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
```

- [ ] **Step 2: Write the failing test `tests/test_notes.py`**

```python
import datetime as dt

import pytest

from ftanomaly import config
from ftanomaly.notes import EnrichedNote, load_notes, parse_enrichment


def test_loads_all_ten_notes():
    ns = load_notes(config.PATHS.notes)
    assert len(ns) == 10
    assert ns[0].note_id == "N001"
    assert ns[0].date == dt.date(2025, 2, 24)
    assert ns[0].applies_to == "Chennai-Bangalore"


def test_parse_enrichment_reads_a_well_formed_payload():
    payload = (
        '{"direction": "raises", "scope": "route", '
        '"window_start": "2025-02-24", "window_end": "2025-03-08", '
        '"in_dataset": true, "magnitude_pct": null}'
    )
    e = parse_enrichment(load_notes(config.PATHS.notes)[0], payload)
    assert e.direction == "raises"
    assert e.window_end == dt.date(2025, 3, 8)
    assert e.magnitude_pct is None


def test_parse_enrichment_accepts_a_fenced_payload():
    """Models often wrap JSON in a markdown fence - strip it, do not crash."""
    payload = (
        '```json\n{"direction": "none", "scope": "route", '
        '"window_start": "2024-07-29", "window_end": "2024-08-04", '
        '"in_dataset": true, "magnitude_pct": null}\n```'
    )
    e = parse_enrichment(load_notes(config.PATHS.notes)[4], payload)
    assert e.direction == "none"


def test_parse_enrichment_rejects_an_unknown_direction():
    bad = (
        '{"direction": "maybe", "scope": "route", '
        '"window_start": "2025-02-24", "window_end": null, '
        '"in_dataset": true, "magnitude_pct": null}'
    )
    with pytest.raises(ValueError, match="direction"):
        parse_enrichment(load_notes(config.PATHS.notes)[0], bad)


def test_open_ended_window_is_none_not_a_sentinel_date():
    payload = (
        '{"direction": "raises", "scope": "all_routes", '
        '"window_start": "2025-05-05", "window_end": null, '
        '"in_dataset": true, "magnitude_pct": 6.0}'
    )
    e = parse_enrichment(load_notes(config.PATHS.notes)[2], payload)
    assert e.window_end is None
    assert e.scope == "all_routes"
    assert e.magnitude_pct == 6.0
```

- [ ] **Step 3: Run it to confirm it fails**

Run: `python -m pytest tests/test_notes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ftanomaly.notes'`

- [ ] **Step 4: Write `src/ftanomaly/notes.py`**

```python
"""Load context notes and enrich each one into a structured record.

The LLM is asked only to read language: which direction does this note
push costs, over what window, for which routes. All date comparison and
applicability logic lives in verdict.py as plain Python.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .llm import LLMClient

_DIRECTIONS = {"raises", "lowers", "none"}
_SCOPES = {"route", "all_routes"}

ENRICH_PROMPT = """You are classifying a logistics context note.

Note id: {note_id}
Note date: {date}
Applies to: {applies_to}
Text: {text}

Answer with JSON only, no prose, using exactly these keys:

  direction     "raises" if the note describes something that pushed freight
                costs UP; "lowers" if it describes something that pushed them
                DOWN; "none" if it explicitly says costs were unaffected, or
                describes normal/stable conditions, or a return to normal.
  scope         "all_routes" if it applies to every route, else "route".
  window_start  ISO date the effect starts. Default to the note date.
  window_end    ISO date the effect ends, or null if open-ended/ongoing.
                If the text names an end date or says when things returned
                to normal, use that date.
  in_dataset    false if the note says the affected routes are NOT part of
                this dataset, otherwise true.
  magnitude_pct the percentage rise the note states, as a number, or null.
                If a range is given, use its midpoint.

JSON:"""


@dataclass(frozen=True)
class Note:
    note_id: str
    date: dt.date
    applies_to: str
    text: str


@dataclass(frozen=True)
class EnrichedNote:
    note: Note
    direction: str
    scope: str
    window_start: dt.date
    window_end: dt.date | None
    in_dataset: bool
    magnitude_pct: float | None


def load_notes(path: Path) -> list[Note]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [
            Note(
                note_id=row["note_id"],
                date=dt.date.fromisoformat(row["date"]),
                applies_to=row["applies_to"],
                text=row["note"],
            )
            for row in csv.DictReader(fh)
        ]


def _strip_fence(raw: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    return (m.group(1) if m else raw).strip()


def parse_enrichment(note: Note, raw: str) -> EnrichedNote:
    data = json.loads(_strip_fence(raw))

    direction = str(data["direction"]).lower()
    if direction not in _DIRECTIONS:
        raise ValueError(f"unknown direction {direction!r} for {note.note_id}")

    scope = str(data["scope"]).lower()
    if scope not in _SCOPES:
        raise ValueError(f"unknown scope {scope!r} for {note.note_id}")

    end = data.get("window_end")
    mag = data.get("magnitude_pct")
    return EnrichedNote(
        note=note,
        direction=direction,
        scope=scope,
        window_start=dt.date.fromisoformat(data["window_start"]),
        window_end=dt.date.fromisoformat(end) if end else None,
        in_dataset=bool(data.get("in_dataset", True)),
        magnitude_pct=float(mag) if mag is not None else None,
    )


def enrich_notes(notes: list[Note], client: LLMClient) -> list[EnrichedNote]:
    out = []
    for note in notes:
        prompt = ENRICH_PROMPT.format(
            note_id=note.note_id,
            date=note.date.isoformat(),
            applies_to=note.applies_to,
            text=note.text,
        )
        raw = client.chat(prompt, cache_key=client.key_for("enrich", note.note_id, note.text))
        out.append(parse_enrichment(note, raw))
    return out
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_notes.py -v`
Expected: 5 passed

- [ ] **Step 6: Run the enrichment once for real and inspect it**

```bash
python -c "
from ftanomaly import config
from ftanomaly.llm import LLMClient
from ftanomaly.notes import load_notes, enrich_notes
c = LLMClient()
for e in enrich_notes(load_notes(config.PATHS.notes), c):
    print(f'{e.note.note_id} {e.direction:7s} {e.scope:11s} {e.window_start} -> {e.window_end} in_dataset={e.in_dataset} mag={e.magnitude_pct}')
print(c.usage.as_dict())
"
```

Expected: exactly **N001, N002, N003 classified `raises`**; N004 `in_dataset=false`; N005, N006, N008, N009, N010 `none`; N007 `lowers`. N001 must end on or near 2025-03-08.

**If any of the seven distractor notes comes back `raises`, fix the prompt before continuing** — that is the single highest-risk failure in the whole system.

- [ ] **Step 7: Commit the code and the cache**

```bash
git add src/ftanomaly/llm.py src/ftanomaly/notes.py tests/test_notes.py cache/
git commit -m "feat: structured note enrichment via one cached LLM pass"
```

---

### Task 6: Vector retrieval over the notes

**Files:**
- Create: `src/ftanomaly/retrieval.py`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `notes.EnrichedNote`, `llm.LLMClient.embed`, `config.RETRIEVAL_TOP_K`.
- Produces:
  - `NoteIndex` class with `.build(enriched: list[EnrichedNote], client) -> None` and `.query(text: str, client, k: int) -> list[tuple[EnrichedNote, float]]` sorted by score descending, ties broken by `note_id` ascending
  - `cosine(a: list[float], b: list[float]) -> float`
  - `query_text_for(route: str, week_of: dt.date) -> str`

- [ ] **Step 1: Write the failing test `tests/test_retrieval.py`**

```python
import datetime as dt

from ftanomaly.retrieval import cosine, query_text_for


def test_cosine_basics():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert round(cosine([1.0, 1.0], [1.0, 0.0]), 4) == 0.7071


def test_cosine_handles_a_zero_vector_without_dividing_by_zero():
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_query_text_mentions_the_route_and_the_month():
    q = query_text_for("Chennai-Bangalore", dt.date(2025, 2, 24))
    assert "Chennai-Bangalore" in q
    assert "2025-02-24" in q
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/test_retrieval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ftanomaly.retrieval'`

- [ ] **Step 3: Write `src/ftanomaly/retrieval.py`**

```python
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
        scored = [
            (e, cosine(qv, v)) for e, v in zip(self._notes, self._vectors)
        ]
        # descending score, ties broken on note_id so the order is total
        scored.sort(key=lambda pair: (-pair[1], pair[0].note.note_id))
        return scored[:k]
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_retrieval.py -v`
Expected: 3 passed

- [ ] **Step 5: Sanity-check retrieval against a known event**

```bash
python -c "
import datetime as dt
from ftanomaly import config
from ftanomaly.llm import LLMClient
from ftanomaly.notes import load_notes, enrich_notes
from ftanomaly.retrieval import NoteIndex, query_text_for
c = LLMClient(); idx = NoteIndex()
idx.build(enrich_notes(load_notes(config.PATHS.notes), c), c)
for e, s in idx.query(query_text_for('Chennai-Bangalore', dt.date(2025,2,24)), c):
    print(f'{e.note.note_id} {s:.4f} {e.note.applies_to}')
"
```

Expected: N001 or N009 ranked top (both mention the Chennai-Bangalore flood). **Both appearing is correct and expected** — retrieval is topical, and it is the Task 7 gate that rejects N009 because its direction is `none`.

- [ ] **Step 6: Commit**

```bash
git add src/ftanomaly/retrieval.py tests/test_retrieval.py cache/
git commit -m "feat: embedding retrieval over context notes"
```

---

### Task 7: The applicability gate and verdicts

This is the task the hallucination grading actually targets. Every rule here is deterministic and inspectable.

**Files:**
- Create: `src/ftanomaly/verdict.py`
- Test: `tests/test_verdict.py`

**Interfaces:**
- Consumes: `detect.Candidate`, `notes.EnrichedNote`, `retrieval.NoteIndex`.
- Produces:
  - `Verdict` frozen dataclass: `candidate: Candidate`, `matched: EnrichedNote | None`, `retrieved: list[tuple[EnrichedNote, float]]`
  - `note_applies(e: EnrichedNote, candidate: Candidate) -> bool`
  - `decide(candidates, index, client) -> list[Verdict]`
  - `Verdict.is_justified -> bool` property, `Verdict.flagged_value -> str` returning `"Yes"` or `"No (justified)"`

- [ ] **Step 1: Write the failing test `tests/test_verdict.py`**

```python
import datetime as dt

import pytest

from ftanomaly.detect import Candidate
from ftanomaly.metrics import WeekMetrics
from ftanomaly.notes import EnrichedNote, Note
from ftanomaly.verdict import note_applies


def _candidate(route, week, own=True, peer=False):
    m = WeekMetrics(route, "Medium", week, 3.0, 25.0, 5.0, 2.5, 8)
    return Candidate(metrics=m, own_history_triggered=own, peer_triggered=peer)


def _note(nid, applies_to, direction, scope, start, end, in_dataset=True):
    n = Note(nid, dt.date.fromisoformat(start), applies_to, "text")
    return EnrichedNote(
        note=n, direction=direction, scope=scope,
        window_start=dt.date.fromisoformat(start),
        window_end=dt.date.fromisoformat(end) if end else None,
        in_dataset=in_dataset, magnitude_pct=None,
    )


def test_matching_route_inside_window_applies():
    e = _note("N001", "Chennai-Bangalore", "raises", "route", "2025-02-24", "2025-03-08")
    c = _candidate("Chennai-Bangalore", dt.date(2025, 3, 3))
    assert note_applies(e, c) is True


def test_week_starting_after_the_window_closes_does_not_apply():
    """N001's disruption ended 2025-03-08. The week of 2025-03-10 is still
    elevated but has no cover - it must come out unexplained."""
    e = _note("N001", "Chennai-Bangalore", "raises", "route", "2025-02-24", "2025-03-08")
    c = _candidate("Chennai-Bangalore", dt.date(2025, 3, 10))
    assert note_applies(e, c) is False


def test_week_before_the_window_opens_does_not_apply():
    e = _note("N003", "All Routes", "raises", "all_routes", "2025-05-05", None)
    c = _candidate("Mumbai-Delhi", dt.date(2025, 4, 21))
    assert note_applies(e, c) is False


def test_wrong_route_does_not_apply():
    e = _note("N002", "Ahmedabad-Mumbai", "raises", "route", "2025-01-20", "2025-01-26")
    c = _candidate("Mumbai-Pune", dt.date(2025, 1, 20))
    assert note_applies(e, c) is False


def test_a_note_that_does_not_raise_costs_never_justifies():
    for direction in ("none", "lowers"):
        e = _note("N006", "All Routes", direction, "all_routes", "2025-09-22", None)
        c = _candidate("Mumbai-Pune", dt.date(2025, 9, 22))
        assert note_applies(e, c) is False, direction


def test_a_note_about_routes_outside_the_dataset_never_justifies():
    e = _note("N004", "All Routes", "raises", "all_routes", "2024-03-11", None,
              in_dataset=False)
    c = _candidate("Mumbai-Delhi", dt.date(2024, 3, 11))
    assert note_applies(e, c) is False


def test_an_all_routes_note_cannot_justify_a_peer_based_flag():
    """A nationwide fuel rise lifts this route's peers too, so it cannot
    explain this route rising RELATIVE to them."""
    e = _note("N003", "All Routes", "raises", "all_routes", "2025-05-05", None)
    peer_only = _candidate("Mumbai-Pune", dt.date(2025, 9, 15), own=False, peer=True)
    assert note_applies(e, peer_only) is False

    own_based = _candidate("Mumbai-Delhi", dt.date(2025, 5, 12), own=True, peer=False)
    assert note_applies(e, own_based) is True


def test_a_route_specific_note_can_justify_either_kind_of_flag():
    e = _note("N001", "Chennai-Bangalore", "raises", "route", "2025-02-24", "2025-03-08")
    peer_only = _candidate("Chennai-Bangalore", dt.date(2025, 3, 3), own=False, peer=True)
    assert note_applies(e, peer_only) is True


def test_open_ended_window_applies_to_any_later_week():
    e = _note("N003", "All Routes", "raises", "all_routes", "2025-05-05", None)
    c = _candidate("Mumbai-Delhi", dt.date(2025, 5, 12))
    assert note_applies(e, c) is True
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/test_verdict.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ftanomaly.verdict'`

- [ ] **Step 3: Write `src/ftanomaly/verdict.py`**

```python
"""Decide whether a retrieved note genuinely justifies a cost rise.

Every rule here is deterministic. Retrieval proposes; these rules dispose.
A note justifies a candidate only when all of the following hold:

  1. it describes costs going UP                (direction == "raises")
  2. its routes are part of this dataset        (in_dataset)
  3. it covers this route                       (scope / applies_to)
  4. its window overlaps this shipping week     (dates)
  5. it can explain THIS KIND of rise           (see the scope rule below)

Rule 5 is the subtle one. A nationwide note lifts every route together, so
it cannot explain a route rising relative to its peers - only a rise
against the route's own history. A route-specific note can explain either.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .detect import Candidate
from .llm import LLMClient
from .notes import EnrichedNote
from .retrieval import NoteIndex, query_text_for


@dataclass(frozen=True)
class Verdict:
    candidate: Candidate
    matched: EnrichedNote | None
    retrieved: list[tuple[EnrichedNote, float]]

    @property
    def is_justified(self) -> bool:
        return self.matched is not None

    @property
    def flagged_value(self) -> str:
        return "No (justified)" if self.is_justified else "Yes"


def note_applies(e: EnrichedNote, candidate: Candidate) -> bool:
    if e.direction != "raises":
        return False
    if not e.in_dataset:
        return False

    route = candidate.metrics.route
    if e.scope == "route" and e.note.applies_to != route:
        return False

    # A nationwide cause moves peers equally, so it cannot explain a
    # peer-relative rise. Only an own-history rise can be covered by it.
    if e.scope == "all_routes" and not candidate.own_history_triggered:
        return False

    week_start = candidate.metrics.week_of
    week_end = week_start + dt.timedelta(days=6)
    if week_end < e.window_start:
        return False
    if e.window_end is not None and week_start > e.window_end:
        return False
    return True


def decide(
    candidates: list[Candidate], index: NoteIndex, client: LLMClient
) -> list[Verdict]:
    verdicts: list[Verdict] = []
    for c in candidates:
        retrieved = index.query(
            query_text_for(c.metrics.route, c.metrics.week_of), client
        )
        applicable = [e for e, _ in retrieved if note_applies(e, c)]
        # deterministic pick: best-ranked applicable note, ties on note_id
        matched = applicable[0] if applicable else None
        verdicts.append(Verdict(candidate=c, matched=matched, retrieved=retrieved))
    return verdicts
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_verdict.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/ftanomaly/verdict.py tests/test_verdict.py
git commit -m "feat: deterministic note applicability gate

Retrieval proposes candidates; explicit rules decide. Nationwide notes
cannot justify peer-relative rises."
```

---

### Task 8: Grounded explanations and the hallucination guard

**Files:**
- Create: `src/ftanomaly/prose.py`
- Test: `tests/test_prose.py`

**Interfaces:**
- Consumes: `verdict.Verdict`, `llm.LLMClient`, `llm.LLMUnavailable`.
- Produces:
  - `validate_reason(text: str, matched_id: str | None, all_note_ids: set[str]) -> None` — raises `GroundingError` on a violation
  - `GroundingError(ValueError)`
  - `write_reason(v: Verdict, client: LLMClient, all_note_ids: set[str]) -> str`

- [ ] **Step 1: Write the failing test `tests/test_prose.py`**

```python
import pytest

from ftanomaly.prose import GroundingError, validate_reason

ALL = {"N001", "N002", "N003", "N004", "N005"}


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
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/test_prose.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ftanomaly.prose'`

- [ ] **Step 3: Write `src/ftanomaly/prose.py`**

```python
"""Plain-English reasons, with a guard against ungrounded claims.

The model writes the sentence; the validator checks it did not invent a
justification. A violation aborts the run rather than shipping a confident
wrong reason, which the brief scores worse than admitting none was found.
"""
from __future__ import annotations

import re

from .llm import LLMClient
from .verdict import Verdict

_JUSTIFY_WORDS = re.compile(
    r"\b(matches|justified by|explained by|supported by|confirms)\b", re.IGNORECASE
)


class GroundingError(ValueError):
    """The generated reason makes a claim the evidence does not support."""


JUSTIFIED_PROMPT = """Write one or two plain sentences explaining why this
freight cost rise is justified. State the verdict, name the note by id, and
reflect what the note actually says including its dates.

Route: {route}
Week beginning: {week}
Cost rise vs this route's own 8-week average: {own}
Cost rise vs similar-length routes this week: {peer}

The note that justifies it:
  id: {note_id}
  date: {note_date}
  applies to: {applies_to}
  text: {note_text}

Do not mention any other note id. Do not invent facts that are not in the
note text. Reply with the explanation only."""

UNEXPLAINED_PROMPT = """Write one or two plain sentences explaining that this
freight cost rise has no supporting note and needs a human review.

Route: {route}
Week beginning: {week}
Cost rise vs this route's own 8-week average: {own}
Cost rise vs similar-length routes this week: {peer}

The closest notes that were retrieved and REJECTED:
{rejected}

You may say briefly why the closest note does not apply (wrong route, wrong
dates, or it describes no cost rise). Never claim the rise is justified or
explained. Reply with the explanation only."""


def validate_reason(text: str, matched_id: str | None, all_note_ids: set[str]) -> None:
    mentioned = {m for m in re.findall(r"\bN\d{3}\b", text) if m in all_note_ids}

    if matched_id is None:
        if _JUSTIFY_WORDS.search(text):
            raise GroundingError(
                f"unexplained row claims justification: {text!r}"
            )
        return

    if matched_id not in mentioned:
        raise GroundingError(
            f"justified row must cite {matched_id} in the reason text: {text!r}"
        )
    extra = mentioned - {matched_id}
    if extra:
        raise GroundingError(
            f"reason cites unrelated note(s) {sorted(extra)}: {text!r}"
        )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1f}%"


def write_reason(v: Verdict, client: LLMClient, all_note_ids: set[str]) -> str:
    m = v.candidate.metrics
    common = {
        "route": m.route,
        "week": m.week_of.isoformat(),
        "own": _fmt(m.vs_own_history),
        "peer": _fmt(m.vs_similar_routes),
    }

    if v.matched is not None:
        e = v.matched
        prompt = JUSTIFIED_PROMPT.format(
            note_id=e.note.note_id,
            note_date=e.note.date.isoformat(),
            applies_to=e.note.applies_to,
            note_text=e.note.text,
            **common,
        )
        key = client.key_for("reason", m.route, m.week_of.isoformat(), e.note.note_id)
    else:
        rejected = "\n".join(
            f"  - {e.note.note_id} ({e.note.date.isoformat()}, {e.note.applies_to}): {e.note.text}"
            for e, _ in v.retrieved
        ) or "  - none"
        prompt = UNEXPLAINED_PROMPT.format(rejected=rejected, **common)
        key = client.key_for("reason", m.route, m.week_of.isoformat(), "unexplained")

    text = " ".join(client.chat(prompt, cache_key=key).split())
    validate_reason(text, v.matched.note.note_id if v.matched else None, all_note_ids)
    return text
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_prose.py -v`
Expected: 6 passed

- [ ] **Step 5: Add the loud-failure test to `tests/test_prose.py`**

```python
def test_missing_api_key_aborts_instead_of_returning_a_blank_reason(tmp_path, monkeypatch):
    """A blank matched_note_id reads as a real 'unexplained' verdict, so an
    API failure must never be swallowed into one."""
    from ftanomaly.llm import LLMClient, LLMUnavailable

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    client = LLMClient(cache_dir=tmp_path)   # empty cache, so it must call out
    with pytest.raises(LLMUnavailable, match="GROQ_API_KEY"):
        client.chat("anything", cache_key="miss")


def test_missing_embedding_key_also_aborts(tmp_path, monkeypatch):
    from ftanomaly.llm import LLMClient, LLMUnavailable

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    client = LLMClient(cache_dir=tmp_path)
    with pytest.raises(LLMUnavailable, match="GEMINI_API_KEY"):
        client.embed(["anything"])
```

- [ ] **Step 6: Run the tests again**

Run: `python -m pytest tests/test_prose.py -v`
Expected: 7 passed

- [ ] **Step 7: Commit**

```bash
git add src/ftanomaly/prose.py tests/test_prose.py
git commit -m "feat: grounded explanations with a hallucination guard"
```

---

### Task 9: Report writer matching the grader's contract

**Files:**
- Create: `src/ftanomaly/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `verdict.Verdict`.
- Produces:
  - `COLUMNS: tuple[str, ...]`
  - `format_own_history(pct: float | None) -> str`
  - `format_similar(pct: float | None) -> str`
  - `to_row(v: Verdict, reason: str) -> dict[str, str]`
  - `write_report(rows: list[dict[str, str]], path: Path) -> None`

- [ ] **Step 1: Write the failing test `tests/test_report.py`**

```python
import csv
import datetime as dt

from ftanomaly import config
from ftanomaly.detect import Candidate
from ftanomaly.metrics import WeekMetrics
from ftanomaly.notes import EnrichedNote, Note
from ftanomaly.report import COLUMNS, format_own_history, format_similar, to_row, write_report
from ftanomaly.verdict import Verdict


def test_columns_match_the_sample_header_exactly():
    with config.PATHS.sample_output.open(newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert list(COLUMNS) == header


def test_percentage_formatting_matches_the_sample_strings():
    assert format_own_history(35.52) == "+35.5% vs this route's past average"
    assert format_similar(21.04) == "+21.0% vs similar-length routes this week"
    assert format_own_history(-4.0) == "-4.0% vs this route's past average"


def test_missing_history_formats_without_crashing():
    assert "n/a" in format_own_history(None)


def _verdict(matched):
    m = WeekMetrics("Delhi-Jaipur", "Short", dt.date(2024, 11, 11), 4.1712,
                    35.52, 21.04, 2.79, 8)
    c = Candidate(metrics=m, own_history_triggered=True, peer_triggered=True)
    return Verdict(candidate=c, matched=matched, retrieved=[])


def test_unexplained_row_has_yes_and_an_empty_note_id():
    row = to_row(_verdict(None), "No matching note found.")
    assert row["flagged"] == "Yes"
    assert row["matched_note_id"] == ""
    assert row["cost_per_tonne_km"] == "4.17"
    assert row["week_of"] == "2024-11-11"


def test_justified_row_has_the_other_vocabulary_value():
    note = Note("N002", dt.date(2025, 1, 20), "Ahmedabad-Mumbai", "text")
    e = EnrichedNote(note, "raises", "route", dt.date(2025, 1, 20),
                     dt.date(2025, 1, 26), True, None)
    row = to_row(_verdict(e), "Matches note N002.")
    assert row["flagged"] == "No (justified)"
    assert row["matched_note_id"] == "N002"


def test_written_file_is_byte_identical_on_a_second_write(tmp_path):
    rows = [to_row(_verdict(None), "No matching note found.")]
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write_report(rows, a)
    write_report(rows, b)
    assert a.read_bytes() == b.read_bytes()


def test_rows_are_sorted_by_week_then_route(tmp_path):
    m1 = WeekMetrics("Zeta", "Short", dt.date(2025, 2, 3), 1.0, 30.0, 30.0, 2.0, 8)
    m2 = WeekMetrics("Alpha", "Short", dt.date(2025, 1, 6), 1.0, 30.0, 30.0, 2.0, 8)
    m3 = WeekMetrics("Alpha", "Short", dt.date(2025, 2, 3), 1.0, 30.0, 30.0, 2.0, 8)
    rows = [
        to_row(Verdict(Candidate(m, True, True), None, []), "x")
        for m in (m1, m2, m3)
    ]
    out = tmp_path / "o.csv"
    write_report(rows, out)
    with out.open(newline="", encoding="utf-8") as fh:
        got = [(r["week_of"], r["route"]) for r in csv.DictReader(fh)]
    assert got == [("2025-01-06", "Alpha"), ("2025-02-03", "Alpha"), ("2025-02-03", "Zeta")]
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ftanomaly.report'`

- [ ] **Step 3: Write `src/ftanomaly/report.py`**

```python
"""Emit the output CSV. The column contract is not negotiable."""
from __future__ import annotations

import csv
from pathlib import Path

from .verdict import Verdict

COLUMNS = (
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
)


def format_own_history(pct: float | None) -> str:
    if pct is None:
        return "n/a (insufficient prior weeks)"
    return f"{pct:+.1f}% vs this route's past average"


def format_similar(pct: float | None) -> str:
    if pct is None:
        return "n/a (no similar routes)"
    return f"{pct:+.1f}% vs similar-length routes this week"


def to_row(v: Verdict, reason: str) -> dict[str, str]:
    m = v.candidate.metrics
    return {
        "route": m.route,
        "week_of": m.week_of.isoformat(),
        "cost_per_tonne_km": f"{m.cost_per_tonne_km:.2f}",
        "vs_own_history": format_own_history(m.vs_own_history),
        "vs_similar_routes": format_similar(m.vs_similar_routes),
        "flagged": v.flagged_value,
        "matched_note_id": v.matched.note.note_id if v.matched else "",
        "reason": reason,
    }


def write_report(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: (r["week_of"], r["route"]))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_report.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/ftanomaly/report.py tests/test_report.py
git commit -m "feat: contract-exact CSV report writer"
```

---

### Task 10: CLI orchestration and the token/cost log

**Files:**
- Create: `src/ftanomaly/cli.py`
- Create: `reports/.gitkeep`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: every module above.
- Produces: `main(argv: list[str] | None = None) -> int`, runnable as `python -m ftanomaly.cli`. Writes `output/anomaly_report.csv` and `reports/token_log.json`.

- [ ] **Step 1: Write `src/ftanomaly/cli.py`**

```python
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
    args.token_log.write_text(json.dumps(usage, indent=2, sort_keys=True), encoding="utf-8")

    justified = sum(1 for r in rows if r["flagged"] == "No (justified)")
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"  flagged (unexplained): {len(rows) - justified}")
    print(f"  justified by a note:   {justified}")
    print(f"  llm calls: {usage['llm_calls']}  cache hits: {usage['cache_hits']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Write `tests/test_cli.py`**

```python
import csv
import json

from ftanomaly import config
from ftanomaly.cli import main


def test_end_to_end_produces_the_expected_report(tmp_path):
    out = tmp_path / "report.csv"
    log = tmp_path / "tokens.json"
    assert main(["--output", str(out), "--token-log", str(log)]) == 0

    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 21
    assert {r["flagged"] for r in rows} <= {"Yes", "No (justified)"}

    # every justified row cites a note; every flagged row cites none
    for r in rows:
        if r["flagged"] == "No (justified)":
            assert r["matched_note_id"].startswith("N")
            assert r["matched_note_id"] in r["reason"]
        else:
            assert r["matched_note_id"] == ""

    usage = json.loads(log.read_text(encoding="utf-8"))
    assert usage["rows_written"] == 21
    assert usage["temperature"] == 0.0


def test_a_cached_rerun_makes_no_api_calls(tmp_path):
    out = tmp_path / "r.csv"
    log = tmp_path / "t.json"
    main(["--output", str(out), "--token-log", str(log)])        # warm
    main(["--output", str(out), "--token-log", str(log)])        # cached
    assert json.loads(log.read_text(encoding="utf-8"))["llm_calls"] == 0
```

- [ ] **Step 3: Run the pipeline for real**

Run: `python -m ftanomaly.cli`
Expected: `wrote 21 rows`, with **4 justified** (N002 once, N001 twice, N003 once) and 17 unexplained.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_cli.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit the code, the cache, and the token log**

```bash
git add src/ftanomaly/cli.py tests/test_cli.py cache/ reports/
git commit -m "feat: pipeline entrypoint with token and cost accounting"
```

---

### Task 11: Eval harness and labeled set

**Files:**
- Create: `eval/labeled_set.csv`
- Create: `eval/run_eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: the generated `output/anomaly_report.csv`.
- Produces: `eval.run_eval.score(report_path, labels_path) -> dict` with keys `total`, `verdict_correct`, `note_correct`, `precision_justified`, `recall_justified`, `mismatches`.

- [ ] **Step 1: Write `eval/labeled_set.csv`**

Hand-labeled expectations for all 21 detected rows, derived by reading the ten notes directly.

```csv
route,week_of,expected_flagged,expected_note_id,why
Delhi-Jaipur,2024-11-11,Yes,,N007 is the only Delhi-Jaipur note: wrong date and it describes improved roads
Delhi-Jaipur,2024-11-18,Yes,,same as the prior week; no note covers Nov 2024
Ahmedabad-Mumbai,2025-01-20,No (justified),N002,festival surcharge on this exact corridor that week
Chennai-Bangalore,2025-02-24,No (justified),N001,flooding Feb 24 to Mar 8 on this corridor
Chennai-Bangalore,2025-03-03,No (justified),N001,week of Mar 3 to Mar 9 overlaps the disruption window
Chennai-Bangalore,2025-03-10,Yes,,still elevated but N001 ended Mar 8; N009 describes a return to normal
Chennai-Bangalore,2025-03-17,Yes,,outside N001; N009 is a recovery note not a cost-rise cause
Mumbai-Delhi,2025-05-12,No (justified),N003,nationwide diesel rise from May 5; this is an own-history flag
Mumbai-Pune,2025-06-23,Yes,,peer-relative drift; N003 lifts peers equally so it cannot explain it
Mumbai-Pune,2025-09-15,Yes,,peer-relative drift with no covering note
Mumbai-Pune,2025-10-06,Yes,,peer-relative drift with no covering note
Mumbai-Pune,2025-10-20,Yes,,peer-relative drift with no covering note
Mumbai-Pune,2025-10-27,Yes,,N010 says compliance costs were absorbed with no rate change
Mumbai-Pune,2025-11-03,Yes,,peer-relative drift with no covering note
Mumbai-Pune,2025-11-10,Yes,,peer-relative drift with no covering note
Mumbai-Pune,2025-11-17,Yes,,peer-relative drift with no covering note
Mumbai-Pune,2025-11-24,Yes,,peer-relative drift with no covering note
Mumbai-Pune,2025-12-01,Yes,,peer-relative drift with no covering note
Mumbai-Pune,2025-12-08,Yes,,peer-relative drift with no covering note
Mumbai-Pune,2025-12-15,Yes,,peer-relative drift with no covering note
Mumbai-Pune,2025-12-22,Yes,,peer-relative drift with no covering note
```

- [ ] **Step 2: Write `eval/run_eval.py`**

```python
"""Score the generated report against a hand-labeled set.

Run after the pipeline:  python eval/run_eval.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ftanomaly import config  # noqa: E402


def _read(path: Path) -> dict[tuple[str, str], dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return {(r["route"], r["week_of"]): r for r in csv.DictReader(fh)}


def score(report_path: Path, labels_path: Path) -> dict:
    report = _read(report_path)
    labels = _read(labels_path)

    verdict_correct = note_correct = 0
    tp = fp = fn = 0
    mismatches: list[str] = []

    for key, want in labels.items():
        got = report.get(key)
        if got is None:
            mismatches.append(f"{key}: expected in report, missing")
            fn += int(want["expected_flagged"] == "No (justified)")
            continue

        if got["flagged"] == want["expected_flagged"]:
            verdict_correct += 1
        else:
            mismatches.append(
                f"{key}: verdict {got['flagged']!r} != {want['expected_flagged']!r}"
            )
        if got["matched_note_id"] == want["expected_note_id"]:
            note_correct += 1
        else:
            mismatches.append(
                f"{key}: note {got['matched_note_id']!r} != {want['expected_note_id']!r}"
            )

        want_j = want["expected_flagged"] == "No (justified)"
        got_j = got["flagged"] == "No (justified)"
        tp += int(want_j and got_j)
        fp += int(got_j and not want_j)
        fn += int(want_j and not got_j)

    for key in report.keys() - labels.keys():
        mismatches.append(f"{key}: present in report but not labeled")

    return {
        "total": len(labels),
        "verdict_correct": verdict_correct,
        "note_correct": note_correct,
        "precision_justified": tp / (tp + fp) if (tp + fp) else 1.0,
        "recall_justified": tp / (tp + fn) if (tp + fn) else 1.0,
        "mismatches": mismatches,
    }


if __name__ == "__main__":
    result = score(config.PATHS.output_csv, ROOT / "eval" / "labeled_set.csv")
    print(f"rows labeled           : {result['total']}")
    print(f"verdicts correct       : {result['verdict_correct']}/{result['total']}")
    print(f"note ids correct       : {result['note_correct']}/{result['total']}")
    print(f"justified precision    : {result['precision_justified']:.3f}")
    print(f"justified recall       : {result['recall_justified']:.3f}")
    for m in result["mismatches"]:
        print("  MISMATCH", m)
    raise SystemExit(1 if result["mismatches"] else 0)
```

- [ ] **Step 3: Write `tests/test_eval.py`**

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from run_eval import score  # noqa: E402

from ftanomaly import config  # noqa: E402


def test_generated_report_matches_every_hand_label():
    result = score(config.PATHS.output_csv, ROOT / "eval" / "labeled_set.csv")
    assert result["mismatches"] == []
    assert result["verdict_correct"] == result["total"]
    assert result["note_correct"] == result["total"]
    assert result["precision_justified"] == 1.0
    assert result["recall_justified"] == 1.0
```

- [ ] **Step 4: Run the eval**

Run: `python -m ftanomaly.cli && python eval/run_eval.py`
Expected: 21/21 verdicts, 21/21 note ids, precision and recall 1.000, no mismatches.

If a mismatch appears, the fix belongs in whichever stage is wrong — do not edit the labels to match the output.

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add eval/ tests/test_eval.py
git commit -m "test: hand-labeled eval set scoring every detected row"
```

---

### Task 12: Reproducibility proof, README, and submission bundle

**Files:**
- Create: `scripts/check_reproducibility.sh`
- Create: `reports/reproducibility.md`
- Create: `README.md`

**Interfaces:**
- Consumes: the finished pipeline.
- Produces: the three hand-in artefacts beyond the code — README, reproducibility evidence, token/cost log.

- [ ] **Step 1: Write `scripts/check_reproducibility.sh`**

```bash
#!/usr/bin/env bash
# Three untouched runs must produce identical output.
set -euo pipefail
cd "$(dirname "$0")/.."

for i in 1 2 3; do
  python -m ftanomaly.cli --output "output/run_${i}.csv" --token-log "reports/tokens_run_${i}.json"
done

echo
echo "--- diff run 1 vs run 2 ---"
diff output/run_1.csv output/run_2.csv && echo "identical"
echo "--- diff run 1 vs run 3 ---"
diff output/run_1.csv output/run_3.csv && echo "identical"

echo
echo "sha256:"
sha256sum output/run_1.csv output/run_2.csv output/run_3.csv
```

- [ ] **Step 2: Run it**

```bash
bash scripts/check_reproducibility.sh | tee reports/reproducibility.md
```

Expected: both diffs empty, three identical sha256 hashes. Runs 2 and 3 should report `llm_calls: 0`.

- [ ] **Step 3: Write `README.md`**

Must cover, in this order:

1. **What it does** — one paragraph.
2. **How to run it** — `pip install -r requirements.txt`, copy `.env.example` to `.env`, add a free Gemini key, `python -m ftanomaly.cli`.
3. **How the numbers are computed** — cost per tonne-km as ratio-of-sums; Monday weeks; the 8-week trailing baseline with no look-ahead; the peer baseline excluding self. **State explicitly that weeks with fewer than 8 prior weeks use all available prior weeks**, as the brief requires, and that the first week of the dataset has no own-history value at all.
4. **How the threshold was chosen** — summarise `docs/THRESHOLD.md`, including that notes were deliberately not used to tune detection.
5. **How notes are matched** — retrieval proposes, deterministic rules dispose; list the five gate conditions; call out the nationwide-note rule and why a nationwide cause cannot explain a peer-relative rise.
6. **How hallucination is prevented** — structured enrichment, the gate, and `validate_reason` aborting the run on an ungrounded claim.
7. **How I convinced myself it was right** — the sample-output oracle (all nine numbers), the 21-row hand-labeled eval set, and the three-run diff.
8. **Reproducibility** — temperature 0, disk cache, byte-identical runs, hashes from `reports/reproducibility.md`.
9. **Models used** — chat on Groq's `openai/gpt-oss-120b` (open-weight, free tier), embeddings on Gemini `text-embedding-004` because Groq serves no embedding endpoint. Say why two providers rather than pretending it was a preference.
10. **Cost** — numbers straight from `reports/token_log.json`: chat calls (~10 enrichment + ~21 prose), embedding calls, input/output tokens, and the rate used. State plainly that both free tiers bill zero, and give the measured token counts anyway. **Note honestly that `gpt-oss-120b` is a reasoning model whose completion-token count varies between identical runs even though the content does not** — so the token figure is from one specific run, not a constant.
11. **Trade-offs and what I would do next** — cosine over a 10-note embedding matrix instead of a hosted vector DB and why; the adaptive MAD term rarely binding on this dataset; the stretch Q&A not built (or built, if Task 13 was done).

- [ ] **Step 4: Verify the output file one last time against the contract**

```bash
head -1 output/anomaly_report.csv
head -1 data/sample_output_format_v2.csv
python -c "
import csv
a=open('output/anomaly_report.csv',newline='',encoding='utf-8')
b=open('data/sample_output_format_v2.csv',newline='',encoding='utf-8')
assert next(csv.reader(a)) == next(csv.reader(b)), 'HEADER MISMATCH'
print('header matches the contract')
"
```

Expected: identical header lines, then `header matches the contract`.

- [ ] **Step 5: Prove no secret is about to be committed**

This repo is going to GitHub. A committed Groq or Gemini key is caught by GitHub secret scanning and, more importantly, is live until revoked.

```bash
# .env must be ignored, and must NOT appear in the index
git check-ignore -v .env || echo "DANGER: .env is not ignored"
git ls-files | grep -E '(^|/)\.env$' && echo "DANGER: .env is tracked" || echo "ok: .env not tracked"

# no key-shaped string anywhere in tracked content
git add -A
git diff --cached -U0 | grep -nE 'gsk_[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_-]{20,}' \
  && echo "DANGER: key material in staged diff" || echo "ok: no key material staged"
```

Expected: `.env` ignored and untracked, and no key material staged. **If anything prints DANGER, stop and fix it before committing.** The cache files are safe to commit — they hold prompts and responses, never credentials.

- [ ] **Step 6: Commit**

```bash
git add README.md scripts/ reports/
git commit -m "docs: README, reproducibility proof, and cost log"
```

- [ ] **Step 7: Final full verification before submitting**

```bash
python -m pytest -v
python -m ftanomaly.cli
python eval/run_eval.py
bash scripts/check_reproducibility.sh
```

Expected: all tests pass, 21 rows written, eval clean, three identical hashes. Only claim the submission is ready after seeing all four succeed.

- [ ] **Step 8: Push to GitHub**

```bash
git log --oneline          # sanity-check the history reads well
gh repo create freight-tiger-anomaly --private --source=. --push
```

Use `--private` unless the brief asked for a public repo; share the link with HR. **After submitting, rotate the Groq key at <https://console.groq.com/keys>** — it was pasted into a chat transcript during development.

---

## Optional Task 13: Plain-English Q&A over the findings (stretch goal)

Build only if time remains after Task 12 is green. The brief lists this as a stretch goal, and a broken stretch feature costs more than a missing one.

**Files:**
- Create: `src/ftanomaly/ask.py`
- Test: `tests/test_ask.py`

**Interfaces:**
- Consumes: `output/anomaly_report.csv`.
- Produces: `answer(question: str, rows: list[dict], client: LLMClient) -> str`, CLI `python -m ftanomaly.ask "why did Mumbai-Pune get pricier in October?"`

- [ ] **Step 1: Write the failing test `tests/test_ask.py`**

```python
import datetime as dt

from ftanomaly.ask import select_rows

ROWS = [
    {"route": "Mumbai-Pune", "week_of": "2025-10-06", "reason": "drift"},
    {"route": "Mumbai-Pune", "week_of": "2025-03-03", "reason": "other"},
    {"route": "Delhi-Jaipur", "week_of": "2025-10-06", "reason": "x"},
]


def test_filters_by_route_name_mentioned_in_the_question():
    got = select_rows("why did Mumbai-Pune get pricier?", ROWS)
    assert {r["route"] for r in got} == {"Mumbai-Pune"}


def test_filters_by_month_name_mentioned_in_the_question():
    got = select_rows("what happened in October 2025?", ROWS)
    assert {r["week_of"] for r in got} == {"2025-10-06"}


def test_no_match_returns_everything_rather_than_nothing():
    assert len(select_rows("how are things?", ROWS)) == len(ROWS)
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/test_ask.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ftanomaly.ask'`

- [ ] **Step 3: Write `src/ftanomaly/ask.py`**

```python
"""Answer plain-English questions using only the generated report.

The model may quote rows and nothing else; it has no access to the raw
shipment data, so it cannot invent a number that is not in the report.
"""
from __future__ import annotations

import calendar
import csv
import re
import sys
from pathlib import Path

from . import config
from .llm import LLMClient

ANSWER_PROMPT = """Answer the question using ONLY the report rows below.
If the rows do not contain the answer, say so plainly. Never invent routes,
weeks, numbers, or note ids that do not appear here.

Question: {question}

Report rows:
{rows}

Answer:"""


def select_rows(question: str, rows: list[dict]) -> list[dict]:
    q = question.lower()

    routes = {r["route"] for r in rows if r["route"].lower() in q}
    months = {
        f"{i:02d}"
        for i in range(1, 13)
        if calendar.month_name[i].lower() in q
    }
    years = set(re.findall(r"\b(20\d{2})\b", question))

    picked = [
        r
        for r in rows
        if (not routes or r["route"] in routes)
        and (not months or r["week_of"][5:7] in months)
        and (not years or r["week_of"][:4] in years)
    ]
    return picked or rows


def answer(question: str, rows: list[dict], client: LLMClient) -> str:
    picked = select_rows(question, rows)
    rendered = "\n".join(
        f"- {r['route']} week of {r['week_of']}: {r['cost_per_tonne_km']} per tonne-km, "
        f"{r['vs_own_history']}, {r['vs_similar_routes']}, flagged={r['flagged']}, "
        f"note={r['matched_note_id'] or 'none'}. {r['reason']}"
        for r in picked
    )
    prompt = ANSWER_PROMPT.format(question=question, rows=rendered)
    return client.chat(prompt, cache_key=client.key_for("ask", question, rendered))


def main() -> int:
    if len(sys.argv) < 2:
        print('usage: python -m ftanomaly.ask "your question"', file=sys.stderr)
        return 1
    path = Path(config.PATHS.output_csv)
    if not path.exists():
        print("run `python -m ftanomaly.cli` first", file=sys.stderr)
        return 1
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(answer(" ".join(sys.argv[1:]), rows, LLMClient()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_ask.py -v`
Expected: 3 passed

- [ ] **Step 5: Try it**

```bash
python -m ftanomaly.ask "why did Mumbai-Pune get pricier in October 2025?"
python -m ftanomaly.ask "which routes had a justified cost rise?"
```

Expected: answers that cite only rows present in the report.

- [ ] **Step 6: Commit**

```bash
git add src/ftanomaly/ask.py tests/test_ask.py cache/
git commit -m "feat: plain-English Q&A over the generated findings"
```
