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
