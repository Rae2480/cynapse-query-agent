"""Check all AI acceptance/extraction cases plus the nonai assignment examples.

Export OPENAI_API_KEY and CYNAPSE_MODEL, then explicitly opt into paid calls:
    CYNAPSE_LIVE_TESTS=1 pytest -q tests/test_assignment_inputs.py

Ordinary pytest runs skip live calls. Live cases cover evaluation_cases.json and
extraction_cases.json;
the nonai version always runs without API calls. The clock is frozen at noon
Singapore time on October 18, 2026. Each case has independent conversation state.
"""
import json
import os
from pathlib import Path

import pytest

from cynapse.database_manager import seed, query
from evaluations.evaluate_agent import NOW, run_cases, check_case
from evaluations.evaluate_extraction import run_extractions
from cynapse.orchestration import build_graph
from cynapse.openai_client import configured_extractor

# Five verbatim query examples, the two-turn follow-up, and camera-name variants
# from the Agentic AI section of the assignment (not the Computer Vision option).
ASSIGNMENT_IDS = {
    "brief-cte-today", "brief-august", "brief-tuesdays", "brief-kranji-range",
    "brief-pie-hours", "brief-followup", "full-camera-name", "camera-typo",
}
ALL_CASES = json.loads(Path(__file__).with_name("evaluation_cases.json").read_text())
EXTRACTION_CASES = json.loads(Path(__file__).with_name("extraction_cases.json").read_text())
CASES = [
    case for case in ALL_CASES
    if case["id"] in ASSIGNMENT_IDS
]


def test_assignment_examples_are_present():
    """Fail if a required fixture disappears or an assignment case is duplicated."""
    assert {case["id"] for case in CASES} == ASSIGNMENT_IDS
    assert len(CASES) == len(ASSIGNMENT_IDS)


@pytest.fixture(scope="module")
def live_resources(tmp_path_factory):
    """Yield a frozen seeded database and real extractor; always close the client."""
    if os.getenv("CYNAPSE_LIVE_TESTS") != "1":
        pytest.skip("Paid OpenAI tests: set CYNAPSE_LIVE_TESTS=1 to enable")
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("CYNAPSE_MODEL"):
        pytest.fail("Export OPENAI_API_KEY and CYNAPSE_MODEL before running live tests")
    path = tmp_path_factory.mktemp("assignment-live") / "frames.db"
    seed(path, NOW)
    extractor = configured_extractor()
    try:
        yield path, extractor
    finally:
        extractor.close()


@pytest.mark.parametrize("case", ALL_CASES, ids=lambda case: case["id"])
def test_acceptance_with_openai(case, live_resources):
    """Check actual extraction, resolved filters, record counts and first-page data."""
    path, extractor = live_resources
    report = run_cases([case], path, extractor)
    result = report["results"][0]
    assert report["stop_reason"] is None, report["stop_reason"]
    assert result["passed"], json.dumps(result, indent=2)
    assert all(
        call["status"] == "success"
        for turn in result["turns"] for call in turn["model_calls"]
    ), "Every model call must succeed; fallback to the nonai version is not permitted"
    if case["id"] in ASSIGNMENT_IDS:
        assert sum(turn["database_calls"] for turn in result["turns"]) == len(case["turns"])


@pytest.mark.parametrize("case", EXTRACTION_CASES, ids=lambda case: case["id"])
def test_extraction_with_openai(case, live_resources):
    """Check every model-only extraction case using the configured real OpenAI API."""
    _, extractor = live_resources
    report = run_extractions([case], extractor)
    assert report["passed"] == 1, json.dumps(report["results"][0], indent=2)


@pytest.fixture(scope="module")
def nonai_database(tmp_path_factory):
    """Seed the assignment clock's records once for nonai-version checks."""
    path = tmp_path_factory.mktemp("assignment-nonai") / "frames.db"
    seed(path, NOW)
    return path


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_assignment_input_with_nonai(case, nonai_database, monkeypatch):
    """Check each assignment input and its actual database results without AI."""
    def forbidden(*args, **kwargs):
        pytest.fail("The nonai version must never initialize or call OpenAI")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("cynapse.openai_client.configured_extractor", forbidden)
    monkeypatch.setattr("cynapse.openai_client.LLMExtractor.__call__", forbidden)
    calls = []

    def tracked_query(*args):
        calls.append(args)
        return query(*args)

    agent = build_graph(nonai_database, query_fn=tracked_query)
    config = {"configurable": {"thread_id": case["id"]}}
    turns = []
    for request in case["turns"]:
        before = len(calls)
        state = agent.invoke({"request": request, "now": NOW}, config)
        turns.append({"response": state["response"], "database_calls": len(calls) - before})
    assert not check_case(case, state, turns), json.dumps(turns, indent=2)
    assert len(calls) == len(case["turns"])
