"""Acceptance checks for the nonai version against every evaluation case."""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from cynapse import orchestration as graph_module
from cynapse.helpers.database.database_manager import SG, seed
from cynapse.helpers.data_models import Camera

NOW = datetime(2026, 10, 18, 12, tzinfo=SG)
CASES = json.loads(Path(__file__).with_name("evaluation_cases.json").read_text())


@pytest.fixture(scope="module")
def database(tmp_path_factory):
    path = tmp_path_factory.mktemp("evaluation") / "frames.db"
    seed(path, NOW)
    return path


def expected_instants(case):
    instant = datetime.fromisoformat(case["start"])
    end = datetime.fromisoformat(case["end"])
    while instant < end:
        clock = instant.strftime("%H:%M")
        if ("weekday" not in case or instant.weekday() == case["weekday"]) and (
            "time_start" not in case or case["time_start"] <= clock < case["time_end"]
        ):
            yield instant.isoformat()
        instant += timedelta(minutes=5)


def test_case_ids_are_unique():
    assert len({case["id"] for case in CASES}) == len(CASES)


@pytest.mark.parametrize("case", [c for c in CASES if "camera" in c], ids=lambda c: c["id"])
def test_expected_counts_are_consistent(case):
    # Independent enumeration checks hand-calculated expectations, even for TODOs.
    assert len(list(expected_instants(case))) == case["total"]



@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_acceptance(case, database, monkeypatch):
    calls = []
    from cynapse.agents.database_reader import query as real_query

    def tracked_query(*args):
        calls.append(args)
        return real_query(*args)

    monkeypatch.setattr("cynapse.agents.database_reader.query", tracked_query)
    agent = graph_module.build_graph(database)
    config = {"configurable": {"thread_id": case["id"]}}
    now = datetime.fromisoformat(case["now"]) if "now" in case else NOW
    for index, request in enumerate(case["turns"]):
        before = len(calls)
        state = agent.invoke({"request": request, "now": now}, config)
        if index in case.get("no_query_turns", []):
            assert len(calls) == before
            assert state["response"].get("message", "").strip()

    response = state["response"]
    if case.get("outcome") == "no_query":
        assert response.get("message", "").strip()
        # All final non-query turns must avoid the database, even with old context.
        assert len(calls) == before
        if case["category"] == "guardrails":
            assert response["intent"] == "reject"
        if case.get("clear_context"):
            assert state.get("plan") is None
        return

    assert "filters" in response, response
    assert response["filters"]["camera"] == Camera[case["camera"]].value
    assert datetime.fromisoformat(response["filters"]["start"]) == datetime.fromisoformat(case["start"])
    assert datetime.fromisoformat(response["filters"]["end"]) == datetime.fromisoformat(case["end"])
    assert response["total"] == case["total"]
    expected = list(expected_instants(case))
    assert [row["datetime"] for row in response["frames"]] == expected[:20]
    assert all(row["camera_name"] == Camera[case["camera"]].value for row in response["frames"])
    ids = [row["frame_id"] for row in response["frames"]]
    assert all(type(frame_id) is int for frame_id in ids)
    assert len(ids) == len(set(ids))
    assert response["truncated"] == (case["total"] > 20)
    # Check the resolved predicate itself once the recurrence fields exist.
    if "weekday" in case:
        assert response["filters"]["weekday"] == case["weekday"]
    if "time_start" in case:
        assert response["filters"]["time_start"] == case["time_start"]
        assert response["filters"]["time_end"] == case["time_end"]
