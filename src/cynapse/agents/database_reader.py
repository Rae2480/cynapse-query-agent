"""Database reader role in the query workflow."""
from pathlib import Path

from cynapse.helpers.database.database_manager import query
from cynapse.helpers.data_models import QueryPlan
from cynapse.state import State


def result_page(plan: QueryPlan, offset: int, *, path: Path, query_fn=None):
    """Fetch a database page and accept its filters as conversation context.

    Args:
        plan: Validated QueryPlan containing filters and page size.
        offset: Number of matching records to skip (zero for the first page).
        path: SQLite database path.
        query_fn: Optional database query callable; defaults to database_manager.query.

    Returns:
        State updates containing the accepted plan, cleared pending query, page
        offset/total and response with frames, filters and pagination information.

    Reads SQLite through query_fn or database_manager.query; database errors propagate."""
    result = (query_fn or query)(path, plan, offset)
    return {"plan": plan, "pending": None, "page_offset": offset, "page_total": result["total"], "response": {
        **result, "filters": plan.model_dump(mode="json"),
        "truncated": result["total"] > len(result["frames"]),
        "pagination": {"page": offset // plan.limit + 1, "page_size": plan.limit,
                       "has_next": offset + len(result["frames"]) < result["total"],
                       "has_previous": offset > 0},
    }}


def database_reader(state: State, *, path: Path, query_fn=None):
    """Execute the first page of a validated query candidate.

    Args:
        state: State containing candidate as a QueryPlan.
        path: SQLite database path.
        query_fn: Optional replacement for database_manager.query(path, plan, offset).

    Returns:
        State updates from result_page(candidate, 0). Reads the database and
        propagates query errors."""
    return result_page(state["candidate"], 0, path=path, query_fn=query_fn)
