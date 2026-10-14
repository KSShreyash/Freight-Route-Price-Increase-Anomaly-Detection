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
