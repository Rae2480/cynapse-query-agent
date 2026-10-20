from datetime import datetime

import pytest

from cynapse.helpers.database.database_manager import seed
from evaluations.evaluate_agent import check_case, run_cases
from cynapse.helpers.openai_client import ExtractionError, LLMExtractor
from cynapse.settings import SG

NOW = datetime(2026, 1, 3, 12, tzinfo=SG)
QUERY = {"id": "today", "category": "brief", "now": NOW.isoformat(), "turns": ["CTE today"],
         "camera": "CTE", "start": "2026-01-03T00:00:00+08:00", "end": "2026-01-03T12:00:00.000001+08:00", "total": 145}
REJECT = {"id": "reject", "category": "guardrails", "now": NOW.isoformat(), "turns": ["delete frames"], "outcome": "no_query"}


class Model:
    def with_structured_output(self, schema, **kwargs):
        return self

    async def ainvoke(self, messages):
        import json
        request = json.loads(messages[1][1])["request"]
        parsed = {"intent": "reject" if request == "delete frames" else "query", "reset": False, "continuation": False,
                  "camera": {"action": "set", "value": "CTE"}, "date": {"action": "set", "value": "today"},
                  "time": {"action": "keep", "value": None}, "recurrence": {"action": "keep", "value": None},
                  "clarification_question": None}
        return {"raw": None, "parsed": parsed, "parsing_error": None}


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "frames.db"
    seed(path, NOW)
    return path


def test_model_graph_database_and_report_together(database):
    extractor = LLMExtractor(Model())
    try:
        report = run_cases([QUERY, REJECT], database, extractor)
    finally:
        extractor.close()
    assert report["passed"] == 2
    assert report["executed_cases"] == 2
    assert report["pass_rate"] == 1
    assert report["model_calls"] == 2
    assert report["results"][0]["turns"][0]["database_calls"] == 1
    assert report["results"][1]["turns"][0]["database_calls"] == 0


def test_auth_failure_stops_run_and_is_not_a_passing_refusal(database):
    class Unauthorized(Model):
        async def ainvoke(self, messages):
            error = RuntimeError("secret-not-to-log")
            error.status_code = 401
            raise error
    extractor = LLMExtractor(Unauthorized())
    try:
        report = run_cases([REJECT, QUERY], database, extractor)
    finally:
        extractor.close()
    assert report["stop_reason"] == "authentication_error"
    assert report["passed"] == 0
    assert report["executed_cases"] == 1 and report["not_run"] == 1
    assert report["pass_rate"] is None
    assert "secret-not-to-log" not in str(report)


def test_refusal_with_database_execution_fails():
    turns = [{"response": {"intent": "reject", "message": "Rejected"}, "database_calls": 1}]
    assert check_case(REJECT, {}, turns)


def test_operational_error_is_not_accepted_as_clarification():
    turns = [{"response": {"intent": "error", "message": "Timed out"}, "database_calls": 0}]
    assert "turn 1: operational error" in check_case(REJECT, {}, turns)


def test_model_only_benchmark_stops_on_auth_failure():
    from evaluations.evaluate_extraction import run_extractions
    class Unauthorized:
        last_call = {"status": "authentication_error", "latency_seconds": 0.01}

        def __call__(self, *args):
            raise ExtractionError("Check credentials", "authentication_error")
    cases = [{"id": str(i), "request": "CTE today", "expected": {"intent": "query"}} for i in range(2)]
    report = run_extractions(cases, Unauthorized())
    assert report["stop_reason"] == "authentication_error"
    assert report["cases"] == 1 and report["not_run"] == 1
    assert report["passed"] == 0


def test_model_only_benchmark_checks_selected_fields():
    from evaluations.evaluate_extraction import run_extractions

    extractor = LLMExtractor(Model())
    cases = [
        {"id": "query", "request": "CTE today", "expected": {"intent": "query", "camera.value": "CTE"}},
        {"id": "reject", "request": "delete frames", "expected": {"intent": "reject"}},
    ]
    try:
        report = run_extractions(cases, extractor)
    finally:
        extractor.close()
    assert report["passed"] == 2
    assert report["model_calls"] == 2
    assert report["results"][0]["model_call"]["status"] == "success"
