import json
from datetime import datetime
from unittest.mock import Mock

import pytest

from cynapse import orchestration as module
from cynapse.database_manager import SG, seed
from cynapse.data_models import Camera
from cynapse.openai_client import ExtractionError, LLMExtractor

NOW = datetime(2026, 1, 3, 12, tzinfo=SG)


def turn(camera=None, date=None, time=None, recurrence=None, *, continuation=False, reset=False, intent="query", question=None):
    def update(value):
        if value == "CLEAR":
            return {"action": "clear", "value": None}
        return {"action": "set", "value": value} if value else {"action": "keep", "value": None}
    return {
        "intent": intent, "reset": reset, "continuation": continuation,
        "camera": update(camera), "date": update(date), "time": update(time),
        "recurrence": update(recurrence), "clarification_question": question,
    }


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "frames.db"
    seed(path, datetime(2026, 1, 5, tzinfo=SG))
    return path


def ask(graph, request="fixture", *, thread="a", now=NOW):
    return graph.invoke({"request": request, "now": now}, {"configurable": {"thread_id": thread}})


def test_missing_camera_answer_completes_pending_date(database):
    extractor = Mock(side_effect=[
        turn(date="yesterday", intent="clarify", question="Which camera?"),
        turn(camera="CTE", continuation=True),
    ])
    graph = module.build_graph(database, extractor=extractor)
    first = ask(graph, "Show frames yesterday")
    assert first.get("plan") is None
    assert first["pending"]["extraction"]["date"]["value"] == "yesterday"
    result = ask(graph, "CTE")
    assert result["response"]["total"] == 288
    assert result["plan"].start == datetime(2026, 1, 2, tzinfo=SG)
    assert result["pending"] is None
    assert extractor.call_args.args[3]["question"] == "Which camera?"


def test_multiple_clarifications_accumulate_only_explicit_changes(database):
    graph = module.build_graph(database, extractor=Mock(side_effect=[
        turn(camera="CPE", date="yesterday", time="8 to 10"),
        turn(camera="CTE", continuation=True),
        turn(time="8 AM to 10 AM", continuation=True),
    ]))
    assert ask(graph)["response"]["intent"] == "clarify"
    second = ask(graph)
    assert second["response"]["intent"] == "clarify"
    assert second.get("plan") is None
    result = ask(graph)
    assert result["response"]["total"] == 24
    assert result["plan"].camera == Camera.CTE
    assert result["pending"] is None


@pytest.mark.parametrize("interruption", [
    turn(intent="reject"), turn(intent="reject", reset=True),
    ExtractionError("Retry later"), {"sql": "DROP TABLE frames"},
])
def test_reject_and_model_failure_preserve_accepted_and_pending(database, monkeypatch, interruption):
    graph = module.build_graph(database, extractor=Mock(side_effect=[
        turn(camera="CTE", date="yesterday"),
        turn(camera="CPE", date="today"), interruption,
        turn(camera="PIE", continuation=True),
    ]))
    accepted = ask(graph)["plan"]
    before = ask(graph)
    assert before["plan"] == accepted
    from cynapse.agents.database_reader import query as real_query
    with monkeypatch.context() as patch:
        query = Mock(side_effect=AssertionError("No database execution allowed"))
        patch.setattr("cynapse.agents.database_reader.query", query)
        interrupted = ask(graph, "unrelated or failed turn")
        query.assert_not_called()
    from cynapse.agents.database_reader import query as restored_query
    assert restored_query is real_query
    assert interrupted["plan"] == accepted
    assert interrupted["pending"] == before["pending"]
    assert interrupted["candidate"] is None
    result = ask(graph, "PIE")
    assert result["plan"].camera == Camera.PIE
    assert result["response"]["total"] == 145


def test_independent_request_discards_pending_filters(database):
    graph = module.build_graph(database, extractor=Mock(side_effect=[
        turn(camera="CTE", date="yesterday"),
        turn(camera="CPE", date="today", time="8 AM to 10 AM"),
        turn(camera="PIE", date="yesterday", continuation=False),
    ]))
    ask(graph)
    ask(graph)
    result = ask(graph, "Show PIE frames yesterday")
    assert result["response"]["total"] == 288
    assert result["plan"].time_start is None
    assert result["pending"] is None


@pytest.mark.parametrize("command", ["Start over.", "reset", "New search!", " clear context "])
def test_exact_reset_skips_model_and_database_and_clears_pending(database, monkeypatch, command):
    extractor = Mock(side_effect=[turn(camera="CTE", date="yesterday"), turn(camera="CPE", date="today")])
    graph = module.build_graph(database, extractor=extractor)
    ask(graph)
    ask(graph)
    count = extractor.call_count
    query = Mock(side_effect=AssertionError("Reset must not query"))
    monkeypatch.setattr("cynapse.agents.database_reader.query", query)
    result = ask(graph, command)
    assert result["plan"] is None
    assert result["pending"] is None
    assert result["candidate"] is None
    assert result["extraction"] is None
    assert extractor.call_count == count
    query.assert_not_called()


def test_reset_with_new_filters_does_not_inherit_previous_time(database):
    graph = module.build_graph(database, extractor=Mock(side_effect=[
        turn(camera="CTE", date="yesterday", time="8 AM to 10 AM"),
        turn(camera="PIE", date="today", reset=True),
    ]))
    assert ask(graph)["response"]["total"] == 24
    result = ask(graph, "Start over. Show PIE today")
    assert result["response"]["total"] == 145
    assert result["plan"].time_start is None


def test_incomplete_new_search_cannot_resurrect_old_camera(database):
    graph = module.build_graph(database, extractor=Mock(side_effect=[
        turn(camera="CTE", date="yesterday"), turn(date="today", reset=True),
        turn(time="8 AM to 10 AM", continuation=True),
        turn(camera="PIE", continuation=True),
    ]))
    ask(graph)
    assert ask(graph, "New search, today")["plan"] is None
    assert ask(graph, "8 AM to 10 AM")["response"]["intent"] == "clarify"
    result = ask(graph, "PIE")
    assert result["plan"].camera == Camera.PIE
    assert result["response"]["total"] == 24


def test_threads_isolate_pending_and_accepted_filters(database):
    graph = module.build_graph(database, extractor=Mock(side_effect=[
        turn(date="yesterday"), turn(camera="PIE", date="today"),
        turn(camera="CTE", continuation=True), turn(date="yesterday"),
    ]))
    assert ask(graph, thread="a")["response"]["intent"] == "clarify"
    assert ask(graph, thread="b")["response"]["total"] == 145
    a = ask(graph, thread="a")
    b = ask(graph, thread="b")
    assert a["plan"].camera == Camera.CTE
    assert b["plan"].camera == Camera.PIE
    assert a["response"]["total"] == b["response"]["total"] == 288


@pytest.mark.parametrize("new_date,expected_day,total", [(None, 2, 288), ("today", 4, 2)])
def test_relative_date_anchor_survives_midnight_unless_replaced(database, new_date, expected_day, total):
    graph = module.build_graph(database, extractor=Mock(side_effect=[
        turn(date="yesterday"), turn(camera="CTE", date=new_date, continuation=True),
    ]))
    ask(graph, now=datetime(2026, 1, 3, 23, 55, tzinfo=SG))
    result = ask(graph, now=datetime(2026, 1, 4, 0, 5, tzinfo=SG))
    assert result["plan"].start.day == expected_day
    assert result["response"]["total"] == total


def test_explicit_clears_remove_only_target_filter(database):
    graph = module.build_graph(database, extractor=Mock(side_effect=[
        turn(camera="CTE", date="yesterday", time="8 AM to 10 AM"),
        turn(time="CLEAR"), turn(camera="CLEAR"),
        turn(camera="PIE", continuation=True),
    ]))
    assert ask(graph)["response"]["total"] == 24
    assert ask(graph)["response"]["total"] == 288
    assert ask(graph)["response"]["intent"] == "clarify"
    result = ask(graph)
    assert result["plan"].camera == Camera.PIE
    assert result["response"]["total"] == 288


def test_model_receives_pending_as_separate_context_data():
    class CaptureModel:
        def with_structured_output(self, *args, **kwargs):
            return self

        async def ainvoke(self, messages):
            self.messages = messages
            return turn(camera="CTE", continuation=True)

    model = CaptureModel()
    extractor = LLMExtractor(model)
    pending = {"question": "Which camera? [pending-context-test]", "extraction": turn(date="yesterday")}
    try:
        result = extractor("CTE", None, NOW, pending)
        assert result.continuation
        assert json.loads(model.messages[1][1])["pending_clarification"] == pending
        assert pending["question"] not in model.messages[0][1]
    finally:
        extractor.close()


def test_no_change_query_cannot_accidentally_execute_old_context(database, monkeypatch):
    graph = module.build_graph(database, extractor=Mock(side_effect=[
        turn(camera="CTE", date="yesterday"), turn(),
    ]))
    accepted = ask(graph)["plan"]
    query = Mock(side_effect=AssertionError("No filter changes must not execute old query"))
    monkeypatch.setattr("cynapse.agents.database_reader.query", query)
    result = ask(graph, "What is the weather?")
    assert result["response"]["intent"] == "clarify"
    assert result["plan"] == accepted
    query.assert_not_called()


@pytest.mark.parametrize("pending", [False, True])
def test_idle_context_expires_before_model_and_pagination(database, pending):
    extractor = Mock(side_effect=[
        turn(date="yesterday") if pending else turn(camera="CTE", date="yesterday"),
        turn(date="today"),
    ])
    graph = module.build_graph(database, extractor=extractor)
    ask(graph, "Show frames yesterday" if pending else "Show CTE yesterday")
    result = ask(graph, "Only today", now=NOW.replace(hour=13))
    assert extractor.call_args.args[1] is None
    assert extractor.call_args.args[3] is None
    assert result["response"]["intent"] == "clarify"
    assert result.get("plan") is None
    assert "before" in ask(graph, "next", now=NOW.replace(hour=13))["response"]["message"]


def test_oversized_request_cannot_call_model_or_query(database, monkeypatch):
    extractor = Mock(side_effect=AssertionError("Oversized input must not reach OpenAI"))
    query = Mock(side_effect=AssertionError("Oversized input must not query SQLite"))
    monkeypatch.setattr("cynapse.agents.database_reader.query", query)
    result = ask(module.build_graph(database, extractor=extractor), "x" * 1001)
    assert result["response"]["intent"] == "clarify"
    extractor.assert_not_called()
    query.assert_not_called()


def test_model_payload_does_not_grow_with_conversation(database):
    class CaptureModel:
        def __init__(self):
            self.payloads = []

        def with_structured_output(self, *args, **kwargs):
            return self

        async def ainvoke(self, messages):
            assert len(messages) == 2
            self.payloads.append(json.loads(messages[1][1]))
            return turn(camera="CTE", date="yesterday")

    model = CaptureModel()
    extractor = LLMExtractor(model)
    graph = module.build_graph(database, extractor=extractor)
    try:
        for _ in range(25):
            ask(graph, "Show CTE yesterday")
    finally:
        extractor.close()
    assert len({len(json.dumps(payload)) for payload in model.payloads[1:]}) == 1
    assert all(set(payload) == {"request", "now", "accepted_filters", "pending_clarification"}
               for payload in model.payloads)
    assert all("frames" not in payload and "messages" not in payload for payload in model.payloads)


@pytest.mark.parametrize("minutes,kept", [(29, True), (30, False)])
def test_context_expiry_boundary(database, minutes, kept):
    from datetime import timedelta

    extractor = Mock(side_effect=[turn(camera="CTE", date="yesterday"), turn(date="today")])
    graph = module.build_graph(database, extractor=extractor)
    ask(graph, "Show CTE yesterday")
    result = ask(graph, "Only today", now=NOW + timedelta(minutes=minutes))
    assert (extractor.call_args.args[1] is not None) is kept
    assert ("frames" in result["response"]) is kept
