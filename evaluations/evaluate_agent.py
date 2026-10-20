"""Opt-in full-agent evaluation: frozen clocks, real model, graph, and temporary SQLite."""
import argparse
import copy
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter

from cynapse.database_manager import query, seed
from cynapse.data_models import Camera
from cynapse.orchestration import build_graph
from cynapse.openai_client import configured_extractor
from cynapse.settings import SG

NOW = datetime(2026, 10, 18, 12, tzinfo=SG)


def expected_frames(case):
    """Generate expected first-page timestamps from a golden case.

    Args:
        case: ISO start/end boundaries aligned to the seeded five-minute grid,
            optionally with weekday and time_start/time_end filters.

    Returns:
        Up to 20 ISO timestamps in ascending order, with an exclusive end boundary.
        This computes expectations without querying SQLite or calling a model."""
    instant = datetime.fromisoformat(case["start"])
    end = datetime.fromisoformat(case["end"])
    frames = []
    while instant < end and len(frames) < 20:
        clock = instant.strftime("%H:%M")
        if ("weekday" not in case or instant.weekday() == case["weekday"]) and (
            "time_start" not in case or case["time_start"] <= clock < case["time_end"]
        ):
            frames.append(instant.isoformat())
        instant += timedelta(minutes=5)
    return frames


def check_case(case, state, turns):
    """Grade observable agent behavior against a golden conversation case.

    Args:
        case: Expected turns, query filters/counts or no-query behavior.
        state: Final graph state, including the accepted plan when present.
        turns: Nonempty list of turn responses, errors and database-call counts.

    Returns:
        List of readable issues; empty means the case passed. Checks query records,
        context reset and guardrails as applicable. Operational errors always fail,
        even when refusal was expected. Makes no model/database calls itself."""
    issues = []
    if len(turns) != len(case["turns"]):
        issues.append("conversation did not complete")
    for index, turn in enumerate(turns):
        if turn.get("error") or turn["response"].get("intent") == "error":
            issues.append(f"turn {index + 1}: operational error")
        if index in case.get("no_query_turns", []) and turn["database_calls"]:
            issues.append(f"turn {index + 1}: unexpected database execution")
    response = turns[-1]["response"]
    if case.get("outcome") == "no_query":
        if turns[-1]["database_calls"] or not response.get("message", "").strip():
            issues.append("final turn must explain without querying")
        if case["category"] == "guardrails" and response.get("intent") != "reject":
            issues.append("out-of-scope request must be rejected")
        if case.get("clear_context") and state.get("plan") is not None:
            issues.append("accepted filters were not cleared")
        return issues
    filters = response.get("filters", {})
    if filters.get("camera") != Camera[case["camera"]].value:
        issues.append("camera mismatch")
    for key in ("start", "end"):
        try:
            matches = datetime.fromisoformat(filters[key]) == datetime.fromisoformat(case[key])
        except (KeyError, TypeError, ValueError):
            matches = False
        if not matches:
            issues.append(f"{key} mismatch")
    for key in ("weekday", "time_start", "time_end"):
        if filters.get(key) != case.get(key):
            issues.append(f"{key} mismatch")
    if response.get("total") != case["total"]:
        issues.append("total record count mismatch")
    frames = response.get("frames", [])
    if [frame.get("datetime") for frame in frames] != expected_frames(case):
        issues.append("first page timestamps mismatch")
    ids = [frame.get("frame_id") for frame in frames]
    if any(type(value) is not int for value in ids):
        issues.append("invalid frame IDs")
    elif len(set(ids)) != len(ids):
        issues.append("invalid or duplicate frame IDs")
    if any(frame.get("camera_name") != Camera[case["camera"]].value for frame in frames):
        issues.append("record camera mismatch")
    if response.get("truncated") != (case["total"] > 20):
        issues.append("truncation mismatch")
    if response.get("pagination", {}).get("page") != 1:
        issues.append("query did not start on page one")
    return issues


def run_cases(cases, path, extractor, *, repeat=1, on_result=None):
    """Evaluate whole conversations through the real graph and SQLite query path.

    Args:
        cases: Nonempty golden cases with turns/expectations and optional now clock.
        path: Existing seeded SQLite database path.
        extractor: Model callable; caller is responsible for closing it.
        repeat: Positive number of trials, each using separate conversation threads.
        on_result: Optional callback receiving each completed case result.

    Returns:
        Report with case grades, per-turn responses, latency, SQL-call counts,
        and unexecuted cases. Authentication failures stop further cases and
        suppress the overall pass rate. Uses frozen NOW unless a case overrides it.

    May make paid model calls and reads SQLite. It does not seed the database.
    Turn execution errors are recorded; grading/callback errors propagate."""
    calls, db_calls, results = [], [], []

    def recorded_extractor(*args):
        """Forward an extraction call while recording status and timing, even on failure.

        Args:
            *args: Request, previous plan, now and pending context for the extractor.

        Returns:
            The wrapped extractor's result. Appends a copied last_call status/timing record (or a
            fallback containing status and elapsed time) to calls. Extractor exceptions propagate."""
        started = perf_counter()
        result = None
        try:
            result = extractor(*args)
            return result
        finally:
            metric = copy.deepcopy(getattr(extractor, "last_call", None))
            record = metric or {"status": "unavailable", "latency_seconds": perf_counter() - started}
            record["extraction"] = result.model_dump() if hasattr(result, "model_dump") else result
            calls.append(record)

    def recorded_query(*args):
        """Forward a database query while measuring elapsed time.

        Args:
            *args: Database path, QueryPlan and offset accepted by database_manager.query.

        Returns:
            The query result dictionary. Appends a timing record even when the query
            raises; database exceptions propagate to the graph invocation handler."""
        started = perf_counter()
        try:
            return query(*args)
        finally:
            db_calls.append({"latency_seconds": perf_counter() - started})

    agent = build_graph(path, extractor=recorded_extractor, query_fn=recorded_query)
    stop_reason = None
    for trial in range(1, repeat + 1):
        for case in cases:
            turns = []
            state = {}
            config = {"configurable": {"thread_id": f"{trial}:{case['id']}"}}
            now = datetime.fromisoformat(case["now"]) if "now" in case else NOW
            for request in case["turns"]:
                before, db_before = len(calls), len(db_calls)
                started = perf_counter()
                turn = {}
                try:
                    state = agent.invoke({"request": request, "now": now}, config)
                    turn["response"] = state["response"]
                except Exception as exc:
                    turn.update(response={}, error=f"Agent execution failed ({type(exc).__name__})")
                turn.update(
                    latency_seconds=round(perf_counter() - started, 6),
                    model_calls=copy.deepcopy(calls[before:]), database_calls=len(db_calls) - db_before,
                    database_seconds=round(sum(row["latency_seconds"] for row in db_calls[db_before:]), 6),
                )
                turns.append(turn)
                if any(call["status"] == "authentication_error" for call in calls[before:]):
                    stop_reason = "authentication_error"
                if turn.get("error") or stop_reason:
                    break
            issues = check_case(case, state, turns)
            row = {"id": case["id"], "category": case["category"], "trial": trial,
                   "passed": not issues, "issues": issues, "turns": turns}
            results.append(row)
            if on_result:
                on_result(row)
            if stop_reason:
                break
        if stop_reason:
            break
    latencies = [turn["latency_seconds"] for row in results for turn in row["turns"]]
    return {
        "scope": "Full LangGraph agent with model, deterministic resolution and SQLite; nonai-version acceptance expectations applied",
        "planned_cases": len(cases) * repeat, "executed_cases": len(results),
        "not_run": len(cases) * repeat - len(results), "stop_reason": stop_reason,
        "passed": sum(row["passed"] for row in results),
        "pass_rate": sum(row["passed"] for row in results) / len(results) if results and not stop_reason else None,
        "operational_error_turns": sum(turn.get("error") is not None or turn["response"].get("intent") == "error" for row in results for turn in row["turns"]),
        "mean_turn_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "model_calls": len(calls), "results": results,
    }


def main():
    """Run the opt-in end-to-end OpenAI evaluation from command-line options.

    Args:
        None; reads required --model plus cases/output/repeat.

    Returns:
        None. Seeds temporary SQLite data through frozen NOW, runs conversations
        with potentially paid calls, writes partial/final reports, prints progress
        and closes the extractor.

    Raises:
        SystemExit: CLI options or model configuration are invalid.
        OSError: Case, report or database I/O fails.
        ValueError: Case JSON/data is malformed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="OpenAI model as openai:model")
    parser.add_argument("--cases", type=Path, default=Path("tests/evaluation_cases.json"))
    parser.add_argument("--output", type=Path, default=Path("agent-evaluation.json"))
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    if not cases or args.repeat < 1:
        parser.error("Provide nonempty cases and a positive repeat count")
    try:
        extractor = configured_extractor(args.model)
    except ValueError as exc:
        parser.error(str(exc))
    partial = []

    def checkpoint(row):
        """Persist progress after one completed evaluation case.

        Args:
            row: Result dictionary for the completed case/trial.

        Returns:
            None. Appends to the enclosing partial-results list, overwrites the output
            JSON with complete=False, and prints progress. File errors propagate."""
        partial.append(row)
        args.output.write_text(json.dumps({"complete": False, "model": args.model, "results": partial}, indent=2) + "\n")
        print(f"{row['id']}: {'PASS' if row['passed'] else 'FAIL'}", flush=True)

    try:
        with tempfile.TemporaryDirectory(prefix="cynapse-agent-eval-") as directory:
            path = Path(directory) / "frames.db"
            seed(path, NOW)
            print(f"Evaluating {len(cases) * args.repeat} cases; up to {sum(len(c['turns']) for c in cases) * args.repeat} paid model calls", flush=True)
            report = run_cases(cases, path, extractor, repeat=args.repeat, on_result=checkpoint)
            report.update(complete=report["not_run"] == 0 and report["stop_reason"] is None,
                          model=args.model, frozen_now=NOW.isoformat())
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        extractor.close()
    print(f"{report['passed']}/{report['executed_cases']} cases passed; wrote {args.output}")


if __name__ == "__main__":
    main()
