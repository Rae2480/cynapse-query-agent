"""Filter resolver role in the query workflow."""
from cynapse.context_manager import pending_query
from cynapse.data_models import Extraction, QueryPlan
from cynapse.query_planner import resolve_extraction
from cynapse.state import State


def filter_resolver(state: State):
    """Turn extracted filter changes into a concrete query candidate.

    Args:
        state: Validated extraction, resolution_base, now and date_now clock.

    Returns:
        A candidate QueryPlan update, or clarification updates preserving the
        unresolved query as pending when deterministic resolution raises ValueError.

    No model or database call is made."""
    try:
        extraction = Extraction.model_validate(state["extraction"])
        base = QueryPlan.model_validate(state["resolution_base"]) if state.get("resolution_base") else None
        plan = resolve_extraction(extraction, base, state["now"], date_now=state["date_now"])
        return {"candidate": plan}
    except ValueError as exc:
        return {"error": str(exc), "outcome": "clarify", "pending": pending_query(state, str(exc))}
