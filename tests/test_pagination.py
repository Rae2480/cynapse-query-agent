from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from cynapse.agents import database_reader as agents
from cynapse.database_manager import query, seed
from cynapse.data_models import Camera, QueryPlan
from cynapse.orchestration import build_graph
from cynapse.settings import SG

NOW = datetime(2026, 1, 3, 12, tzinfo=SG)


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "frames.db"
    seed(path, NOW)
    return path


def ask(graph, text, thread="a", now=NOW):
    return graph.invoke({"request": text, "now": now}, {"configurable": {"thread_id": thread}})


def test_all_pages_cover_day_once_in_order(database):
    graph = build_graph(database)
    first = ask(graph, "Show CTE yesterday")["response"]
    frames = list(first["frames"])
    current = first
    page = 1
    while current["pagination"]["has_next"]:
        current = ask(graph, "next")["response"]
        page += 1
        assert current["pagination"]["page"] == page
        assert current["filters"] == first["filters"]
        assert current["total"] == 288
        frames.extend(current["frames"])
    assert page == 15
    assert len(current["frames"]) == 8
    assert len(frames) == len({row["frame_id"] for row in frames}) == 288
    start = datetime(2026, 1, 2, tzinfo=SG)
    assert [row["datetime"] for row in frames] == [(start + timedelta(minutes=5 * i)).isoformat() for i in range(288)]


def test_navigation_freezes_filters_and_skips_model(database):
    extraction = {
        "intent": "query", "reset": False, "continuation": False,
        "camera": {"action": "set", "value": "CTE"},
        "date": {"action": "set", "value": "today"},
        "time": {"action": "keep", "value": None},
        "recurrence": {"action": "keep", "value": None}, "clarification_question": None,
    }
    model = Mock(return_value=extraction)
    graph = build_graph(database, extractor=model)
    first = ask(graph, "Show CTE today")["response"]
    second = ask(graph, "next page", now=NOW + timedelta(minutes=5))["response"]
    assert second["filters"] == first["filters"]
    assert second["frames"][0]["datetime"] == "2026-01-03T01:40:00+08:00"
    assert ask(graph, "previous")["response"]["frames"] == first["frames"]
    assert model.call_count == 1


def test_boundaries_and_reset_do_not_query(database, monkeypatch):
    graph = build_graph(database)
    ask(graph, "Show CTE yesterday")
    for _ in range(14):
        ask(graph, "next")
    forbidden = Mock(side_effect=AssertionError("No page to fetch"))
    monkeypatch.setattr(agents, "query", forbidden)
    assert ask(graph, "next")["response"]["message"] == "No more pages."
    ask(graph, "reset")
    assert "before" in ask(graph, "next")["response"]["message"]
    assert "before" in ask(graph, "previous", thread="new")["response"]["message"]
    forbidden.assert_not_called()


def test_new_query_restarts_first_page_and_threads_are_isolated(database):
    graph = build_graph(database)
    ask(graph, "Show CTE yesterday", "a")
    ask(graph, "next", "a")
    ask(graph, "Show PIE today", "b")
    assert ask(graph, "next", "a")["response"]["pagination"]["page"] == 3
    assert ask(graph, "next", "b")["response"]["pagination"]["page"] == 2
    response = ask(graph, "Only today", "a")["response"]
    assert response["pagination"]["page"] == 1
    assert response["frames"][0]["datetime"] == "2026-01-03T00:00:00+08:00"


def test_pending_clarification_blocks_page_but_preserves_position(database, monkeypatch):
    def extraction(camera, date):
        return {"intent": "query", "reset": False, "continuation": False,
                "camera": {"action": "set", "value": camera}, "date": {"action": "set", "value": date},
                "time": {"action": "keep", "value": None}, "recurrence": {"action": "keep", "value": None},
                "clarification_question": None}
    model = Mock(side_effect=[extraction("CTE", "yesterday"), extraction("CPE", "today")])
    graph = build_graph(database, extractor=model)
    ask(graph, "Show CTE yesterday")
    ask(graph, "next")
    pending = ask(graph, "Show CPE today")
    forbidden = Mock(side_effect=AssertionError("Resolve pending clarification first"))
    monkeypatch.setattr(agents, "query", forbidden)
    result = ask(graph, "next")
    assert result["pending"] == pending["pending"]
    assert result["page_offset"] == 20
    assert result["response"]["intent"] == "clarify"
    forbidden.assert_not_called()


@pytest.mark.parametrize("offset", [-1, True, "20"])
def test_offset_validation(database, offset):
    plan = QueryPlan(camera=Camera.CTE, start=NOW - timedelta(days=1), end=NOW)
    with pytest.raises(ValueError):
        query(database, plan, offset)


def test_zero_records_and_first_page_boundary(database, monkeypatch):
    graph = build_graph(database, parser=lambda *args: QueryPlan(camera=Camera.CTE, start=NOW - timedelta(days=1), end=NOW, weekday=0))
    assert ask(graph, "query")["response"]["frames"] == []
    monkeypatch.setattr(agents, "query", Mock(side_effect=AssertionError("No boundary SQL")))
    assert ask(graph, "next")["response"]["message"] == "No more pages."
    assert ask(graph, "previous")["response"]["message"] == "Already at the first page."
