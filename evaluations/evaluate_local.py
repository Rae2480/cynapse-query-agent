"""Reproducible deterministic benchmark; no model calls or API credentials needed."""
import argparse
import json
import platform
import statistics
import tempfile
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from cynapse.helpers.database.database_manager import query, seed
from cynapse.helpers.data_models import Extraction
from cynapse.helpers.resolvers.query_planner import resolve_extraction
from cynapse.settings import SG

NOW = datetime(2026, 10, 18, 12, tzinfo=SG)
CASES = [
    ("cte-today", "CTE", "today", None, None, 145),
    ("tampines-august", "Tampines Expressway", "the whole of August", None, None, 8928),
    ("mce-tuesdays", "MCE", None, None, "every Tuesday", 11808),
    ("kranji-range", "Kranji Highway", "from the 15th to 18th of last month", None, None, 1152),
    ("pie-hours", "PIE", "yesterday", "between 8 AM and 10 AM", None, 24),
]


def main():
    """Run the deterministic benchmark using CLI options and a frozen clock.

    Args:
        None; reads --output and positive --repeat from the command line.

    Returns:
        None. Seeds temporary SQLite data, checks five golden queries, writes a JSON
        timing/environment report and prints results. Does not call a model; timing
        excludes model inference and graph/checkpoint overhead.

    Raises:
        AssertionError: A query count differs from its expected result.
        SystemExit: Command-line arguments are invalid.
        OSError: Report/database I/O fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("local-evaluation.json"))
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    results = []
    with tempfile.TemporaryDirectory(prefix="cynapse-benchmark-") as directory:
        path = Path(directory) / "frames.db"
        start = perf_counter()
        seed(path, NOW)
        seed_seconds = perf_counter() - start
        size = path.stat().st_size
        for name, camera, date, time, recurrence, expected in CASES:
            def update(value):
                """Convert a fixture's optional filter value into an extraction update.

                Args:
                    value: Fixture expression/name, or None when unspecified.

                Returns:
                    A set action with the value when truthy, otherwise a keep action with None."""
                return {"action": "set", "value": value} if value else {"action": "keep", "value": None}
            extraction = Extraction.model_validate({
                "intent": "query", "reset": False, "continuation": False,
                "camera": update(camera), "date": update(date), "time": update(time),
                "recurrence": update(recurrence), "clarification_question": None,
            })
            samples = []
            for _ in range(args.repeat):
                started = perf_counter()
                plan = resolve_extraction(extraction, None, NOW)
                # Same availability intersection as the graph validation stage.
                plan = plan.model_copy(update={"end": min(plan.end, NOW.replace(microsecond=1))})
                result = query(path, plan)
                samples.append((perf_counter() - started) * 1000)
                if result["total"] != expected:
                    raise AssertionError(f"{name}: expected {expected}, got {result['total']}")
            results.append({
                "id": name, "total": result["total"], "returned": len(result["frames"]),
                "filters": plan.model_dump(mode="json"),
                "median_ms": round(statistics.median(samples), 3),
                "min_ms": round(min(samples), 3), "max_ms": round(max(samples), 3),
            })
    report = {
        "scope": "Golden filter extraction -> deterministic resolution -> SQLite; excludes model and graph/checkpoint time",
        "frozen_now": NOW.isoformat(), "python": platform.python_version(), "platform": platform.system(),
        "langgraph": version("langgraph"), "repeat": args.repeat,
        "seed_seconds": round(seed_seconds, 3), "database_bytes": size,
        "model_calls": 0, "live_model_evaluated": False, "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in results:
        print(f"{row['id']}: {row['total']} matches, median {row['median_ms']} ms")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
