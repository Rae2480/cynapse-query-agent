import json
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from cynapse import orchestration as graph_module
from cynapse.helpers.database.database_manager import SG, seed
from cynapse.helpers.data_models import Camera, QueryPlan
from cynapse.helpers.data_models import Extraction
from cynapse.helpers.resolvers.time_resolver import resolve_date, resolve_time, resolve_weekday

NOW = datetime(2026, 10, 18, 12, tzinfo=SG)


@pytest.mark.parametrize("expression,start,end", [
    ("August", "2026-08-01", "2026-09-01"),
    ("Only today", "2026-10-18", "2026-10-19"),
    ("Only today.", "2026-10-18", "2026-10-19"),
    ("just for yesterday", "2026-10-17", "2026-10-18"),
    ("the whole of August", "2026-08-01", "2026-09-01"),
    ("December 2026", "2026-12-01", "2027-01-01"),
    ("last month", "2026-09-01", "2026-10-01"),
    ("last week", "2026-10-05", "2026-10-12"),
    ("this month", "2026-10-01", "2026-11-01"),
    ("2026-08-09", "2026-08-09", "2026-08-10"),
    ("9th August 2026", "2026-08-09", "2026-08-10"),
    ("August 9, 2026", "2026-08-09", "2026-08-10"),
    ("from the 15th to 18th of last month", "2026-09-15", "2026-09-19"),
    ("between 15 and 18 of September 2026", "2026-09-15", "2026-09-19"),
    ("September 15 to 18, 2026", "2026-09-15", "2026-09-19"),
    ("2026-09-30 to 2026-10-02", "2026-09-30", "2026-10-03"),
    ("December 31 2025 to January 2 2026", "2025-12-31", "2026-01-03"),
])
def test_date_resolution(expression, start, end):
    actual = resolve_date(expression, NOW)
    assert actual == tuple(datetime.fromisoformat(day).replace(tzinfo=SG) for day in (start, end))


def test_january_last_month_and_leap_year():
    assert resolve_date("last month", datetime(2026, 1, 1, tzinfo=SG)) == (
        datetime(2025, 12, 1, tzinfo=SG), datetime(2026, 1, 1, tzinfo=SG))
    assert resolve_date("February 29 2024", NOW)[1] == datetime(2024, 3, 1, tzinfo=SG)


@pytest.mark.parametrize("expression", [
    "2026-02-29", "April 31 2026", "September 18 to September 15, 2026",
    "15 to 18 of yesterday", "31 to 32 of August", "08/09/2026",
    "today except Tuesday", "August or September", "all dates; DROP TABLE frames",
])
def test_invalid_or_ambiguous_dates(expression):
    with pytest.raises(ValueError):
        resolve_date(expression, NOW)


@pytest.mark.parametrize("expression,expected", [
    ("between 8 AM and 10 AM", ("08:00", "10:00")),
    ("only between 8 AM and 10 AM", ("08:00", "10:00")),
    ("08:00-10:00", ("08:00", "10:00")),
    ("from 12 AM to 12 PM", ("00:00", "12:00")),
    ("from noon until midnight", ("12:00", "24:00")),
    ("23:00 to 24:00", ("23:00", "24:00")),
])
def test_time_resolution(expression, expected):
    assert resolve_time(expression) == expected


@pytest.mark.parametrize("expression", ["8 to 10", "8 AM to 10", "13 PM to 14 PM", "08:60 to 10:00", "10 PM to 2 AM", "8 AM to 8 AM"])
def test_ambiguous_invalid_or_overnight_times(expression):
    with pytest.raises(ValueError):
        resolve_time(expression)


@pytest.mark.parametrize("expression,expected", [("every Tuesday", 1), ("on Tuesdays", 1), ("each Sunday", 6)])
def test_weekday(expression, expected):
    assert resolve_weekday(expression) == expected


@pytest.mark.parametrize("expression", ["Tuesday and Thursday", "not Tuesday", "weekends"])
def test_unsupported_recurrence(expression):
    with pytest.raises(ValueError):
        resolve_weekday(expression)


def extracted(camera="CTE", date=None, time=None, recurrence=None):
    def update(value):
        return {"action": "set", "value": value} if value else {"action": "keep", "value": None}
    return Extraction.model_validate({
        "intent": "query", "reset": False, "continuation": False, "camera": update(camera),
        "date": update(date), "time": update(time), "recurrence": update(recurrence),
        "clarification_question": None,
    })


@pytest.fixture(scope="module")
def database(tmp_path_factory):
    path = tmp_path_factory.mktemp("temporal") / "frames.db"
    seed(path, NOW)
    return path


# Golden extracted fields isolate temporal correctness from LLM accuracy.
TEMPORAL_CASES = {
    "brief-august": extracted("TPE", "the whole of August"),
    "brief-tuesdays": extracted("MCE", recurrence="every Tuesday"),
    "brief-kranji-range": extracted("Kranji Highway", "from the 15th to 18th of last month"),
    "brief-pie-hours": extracted("PIE", "yesterday", "between 8 AM and 10 AM"),
    "last-week": extracted(date="last week"),
    "this-month": extracted("AYE", "this month"),
    "exact-date": extracted("ECP", "2026-08-09"),
    "recurring-hours": extracted("MCE", "September 2026", "from 8 AM to 10 AM", "every Tuesday"),
}
GOLDEN = json.loads(Path(__file__).with_name("evaluation_cases.json").read_text())


@pytest.mark.parametrize("case", [c for c in GOLDEN if c["id"] in TEMPORAL_CASES], ids=lambda c: c["id"])
def test_temporal_acceptance_records(database, case):
    extraction = TEMPORAL_CASES[case["id"]]
    graph = graph_module.build_graph(database, extractor=lambda *args: extraction)
    state = graph.invoke({"request": case["turns"][0], "now": NOW}, {"configurable": {"thread_id": case["id"]}})
    response = state["response"]
    assert "filters" in response, response
    filters = response["filters"]
    assert filters["camera"] == Camera[case["camera"]].value
    for key in ("start", "end"):
        assert datetime.fromisoformat(filters[key]) == datetime.fromisoformat(case[key])
    for key in ("weekday", "time_start", "time_end"):
        assert filters[key] == case.get(key)
    assert response["total"] == case["total"]
    assert len(response["frames"]) == 20
    assert response["truncated"]
    for frame in response["frames"]:
        instant = datetime.fromisoformat(frame["datetime"])
        assert frame["camera_name"] == filters["camera"]
        assert state["plan"].start <= instant < state["plan"].end
        if filters["weekday"] is not None:
            assert instant.weekday() == filters["weekday"]
        if filters["time_start"] is not None:
            assert filters["time_start"] <= instant.strftime("%H:%M") < filters["time_end"]


def test_local_weekday_at_midnight_and_followup_clear(database):
    cleared = extracted(camera=None, date="2026-09-02").model_dump()
    cleared["recurrence"] = {"action": "clear", "value": None}
    outputs = iter([
        extracted(date="September 2026", time="00:00 to 01:00", recurrence="every Tuesday"),
        extracted(camera=None, date="2026-09-02"),
        cleared,
    ])
    graph = graph_module.build_graph(database, extractor=lambda *args: next(outputs))
    config = {"configurable": {"thread_id": "local"}}
    def ask():
        return graph.invoke({"request": "fixture", "now": NOW}, config)["response"]
    first = ask()
    assert first["total"] == 60  # Five September Tuesdays, 12 frames each.
    assert first["frames"][0]["datetime"] == "2026-09-01T00:00:00+08:00"
    assert ask()["total"] == 0  # Wednesday with Tuesday filter retained.
    assert ask()["total"] == 12  # Explicitly clear weekday, retain daily window.


@pytest.mark.parametrize("date,start,end,total", [
    ("December 31 2025 to January 2 2026", "2026-01-01T00:00:00+08:00", "2026-01-03T00:00:00+08:00", 576),
    ("October 17 to October 20 2026", "2026-10-17T00:00:00+08:00", "2026-10-18T12:00:00.000001+08:00", 433),
])
def test_partial_windows_clip_to_dataset(database, date, start, end, total):
    graph = graph_module.build_graph(database, extractor=lambda *args: extracted(date=date))
    response = graph.invoke({"request": "fixture", "now": NOW}, {"configurable": {"thread_id": "clip"}})["response"]
    assert response["total"] == total
    assert response["filters"]["start"] == start
    assert response["filters"]["end"] == end


def test_clock_converted_to_singapore_before_relative_resolution():
    utc = datetime.fromisoformat("2026-10-17T18:00:00+00:00")
    start, end = resolve_date("today", utc)
    assert start == datetime(2026, 10, 18, tzinfo=SG)
    assert end == datetime(2026, 10, 19, tzinfo=SG)


@pytest.mark.parametrize("date", ["2026-12-01", "2025-12-31", "2026-02-29", "2026-09-18 to 2026-09-15"])
def test_bad_dates_never_execute(database, monkeypatch, date):
    def forbidden(*args):
        pytest.fail("Invalid date must not execute SQL")
    monkeypatch.setattr("cynapse.agents.database_reader.query", forbidden)
    graph = graph_module.build_graph(database, extractor=lambda *args: extracted(date=date))
    response = graph.invoke({"request": "fixture", "now": NOW}, {"configurable": {"thread_id": "invalid"}})["response"]
    assert response["intent"] == "clarify"


@pytest.mark.parametrize("fields", [
    {"weekday": 7}, {"weekday": True}, {"time_start": "08:00"},
    {"time_start": "25:00", "time_end": "26:00"},
    {"time_start": "20:00", "time_end": "08:00"},
])
def test_plan_rejects_invalid_predicates(fields):
    with pytest.raises(ValidationError):
        QueryPlan(camera=Camera.CTE, start=NOW, end=NOW.replace(day=19), **fields)
