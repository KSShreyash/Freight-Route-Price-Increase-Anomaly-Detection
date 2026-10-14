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
    # A week where nothing moved any distance has no definable rate; drop the
    # observation rather than divide by zero or invent a number for it.
    cpk = {
        (rw.route, rw.week_of): rw.total_cost / rw.total_tonne_km
        for rw in route_weeks
        if rw.total_tonne_km > 0
    }
    route_weeks = [rw for rw in route_weeks if (rw.route, rw.week_of) in cpk]
    route_type = {rw.route: rw.route_type for rw in route_weeks}
    routes = sorted({rw.route for rw in route_weeks})

    by_type: dict[str, list[str]] = defaultdict(list)
    for r in routes:
        by_type[route_type[r]].append(r)

    # A route is only evaluated on the weeks it actually shipped in. Real data
    # has gaps, and indexing a route x week grid would raise on the first one.
    weeks_of: dict[str, list[dt.date]] = defaultdict(list)
    for rw in route_weeks:
        weeks_of[rw.route].append(rw.week_of)
    for ws in weeks_of.values():
        ws.sort()

    out: list[WeekMetrics] = []
    for route in routes:
        own_weeks = weeks_of[route]
        for i, week in enumerate(own_weeks):
            current = cpk[(route, week)]
            # the trailing window counts observed weeks, not calendar weeks
            prior = [
                cpk[(route, own_weeks[j])]
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
                if other != route and (other, week) in cpk
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
