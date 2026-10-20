"""Page navigator role in the query workflow."""
from pathlib import Path

from cynapse.agents.database_reader import result_page
from cynapse.state import State


def page_navigator(state: State, *, path: Path, query_fn=None):
    """Move forward or backward through the last accepted query.

    Args:
        state: Pagination request, accepted plan, pending state and page counters.
        path: SQLite database path.
        query_fn: Optional replacement for database_manager.query(path, plan, offset).

    Returns:
        Next/previous page updates, or a clarification response if context is
        incomplete or the requested page is outside the results.

    Reads SQLite only for a valid page; reuses the accepted date window and filters
    without calling the model. Query errors propagate."""
    if not state.get("plan") or state.get("pending"):
        message = "Complete a query or clarification before requesting another page."
        return {"response": {"intent": "clarify", "message": message}}
    plan = state["plan"]
    backwards = state["request"].strip().rstrip(".!?").casefold() in {"previous", "previous page", "back"}
    offset = state.get("page_offset", 0) + (-plan.limit if backwards else plan.limit)
    if offset < 0 or offset >= state.get("page_total", 0):
        return {"response": {"intent": "clarify", "message": "Already at the first page." if backwards else "No more pages."}}
    return result_page(plan, offset, path=path, query_fn=query_fn)
