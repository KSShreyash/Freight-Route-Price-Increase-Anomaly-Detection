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


def test_a_missing_column_names_the_file_and_the_column(tmp_path):
    """A bare KeyError('distance_km') tells someone swapping in their own file
    almost nothing about what to fix."""
    import pytest

    bad = tmp_path / "ships.csv"
    bad.write_text(
        "shipment_id,origin,destination,route_type,quantity_tonnes,shipment_date\n"
        "S1,A,B,Short,10,2025-01-06\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc:
        load_route_weeks(bad)
    msg = str(exc.value)
    assert "distance_km" in msg
    assert "freight_cost_inr" in msg
    assert bad.name in msg


def test_aggregates_are_sums():
    rws = {(r.route, r.week_of): r for r in load_route_weeks(config.PATHS.shipments)}
    wk = rws[("Mumbai-Pune", dt.date(2024, 1, 1))]
    assert wk.shipment_count >= 3
    assert wk.total_cost > 0
    assert wk.total_tonne_km > 0
