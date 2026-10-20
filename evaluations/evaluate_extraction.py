"""Opt-in live benchmark. Each case makes one paid model call per repeat."""
import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

from cynapse.helpers.data_models import QueryPlan
from cynapse.helpers.openai_client import ExtractionError, configured_extractor


def compare(actual: dict, expected: dict):
    """Compare only the extraction fields specified by a case.

    Args:
        actual: Extracted dictionary, possibly containing nested dictionaries.
        expected: Mapping of dot-separated field paths to expected values.

    Returns:
        A list of field/expected/actual mismatch dictionaries; empty means all
        selected fields match. Missing paths resolve to None. No SQL is checked."""
    mismatches = []
    for path, value in expected.items():
        found = actual
        for part in path.split("."):
            found = found.get(part) if isinstance(found, dict) else None
        if found != value:
            mismatches.append({"field": path, "expected": value, "actual": found})
    return mismatches


def run_extractions(cases, extractor, *, repeat=1, on_result=None):
    """Evaluate structured extraction with a frozen clock and record call status and timing.

    Args:
        cases: Cases containing id, request and expected fields, with optional
            previous QueryPlan data and pending clarification context.
        extractor: Callable returning Extraction; caller owns its cleanup.
        repeat: Positive trial count for each case.
        on_result: Optional callback receiving each completed result dictionary.

    Returns:
        Report dictionary with selected-field results, errors, latency and
        unexecuted-case counts. Authentication failures stop further calls.

    May make one paid model call per case/trial. Does not execute SQL. ExtractionError
    is recorded as a failed case; other unexpected errors propagate."""
    results = []
    now = datetime.fromisoformat("2026-10-18T12:00:00+08:00")
    stop_reason = None
    for trial in range(1, repeat + 1):
        for case in cases:
            previous = QueryPlan.model_validate(case["previous"]) if "previous" in case else None
            started = perf_counter()
            row = {"id": case["id"], "trial": trial}
            try:
                actual = extractor(case["request"], previous, now, case.get("pending")).model_dump()
                mismatches = compare(actual, case["expected"])
                row.update(actual=actual, mismatches=mismatches, passed=not mismatches)
            except ExtractionError as exc:
                row.update(error=str(exc), error_category=exc.category, passed=False)
                if exc.category == "authentication_error":
                    stop_reason = exc.category
            row["latency_seconds"] = round(perf_counter() - started, 6)
            metric = copy.deepcopy(getattr(extractor, "last_call", None))
            row["model_call"] = metric or {"status": "unavailable", "latency_seconds": row["latency_seconds"]}
            results.append(row)
            if on_result:
                on_result(row)
            if stop_reason:
                break
        if stop_reason:
            break
    return {
        "scope": "Model-only selected-field extraction checks; no SQL or full-agent correctness grading",
        "now": now.isoformat(), "planned_cases": len(cases) * repeat,
        "cases": len(results), "not_run": len(cases) * repeat - len(results), "stop_reason": stop_reason,
        "passed": sum(row["passed"] for row in results),
        "operational_errors": sum("error" in row for row in results),
        "mean_latency_seconds": sum(row["latency_seconds"] for row in results) / len(results) if results else None,
        "model_calls": len(results), "results": results,
    }


def main():
    """Run the opt-in OpenAI extraction benchmark from command-line options.

    Args:
        None; reads required --model plus cases/output/repeat.

    Returns:
        None. Reads case JSON, makes potentially paid model calls, writes partial
        then final JSON reports, prints progress and closes the extractor.

    Raises:
        SystemExit: CLI options or model configuration are invalid.
        OSError: Input/output files cannot be read or written.
        ValueError: Case JSON/data is malformed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="OpenAI model as openai:model")
    parser.add_argument("--cases", type=Path, default=Path("tests/extraction_cases.json"))
    parser.add_argument("--output", type=Path, default=Path("extraction-report.json"))
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
        print(f"{row['id']}: {'PASS' if row['passed'] else 'FAIL'} ({row['latency_seconds']}s)", flush=True)

    try:
        print(f"Running up to {len(cases) * args.repeat} model calls with {args.model}", flush=True)
        report = run_extractions(cases, extractor, repeat=args.repeat, on_result=checkpoint)
        report.update(model=args.model,
                      complete=report["not_run"] == 0 and report["stop_reason"] is None)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        extractor.close()
    print(f"{report['passed']}/{report['cases']} passed; wrote {args.output}")


if __name__ == "__main__":
    main()
