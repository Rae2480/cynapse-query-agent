"""Query validator role in the query workflow."""
from datetime import datetime, timedelta

from cynapse.context_manager import pending_query
from cynapse.data_models import QueryPlan
from cynapse.settings import SG
from cynapse.state import State


def query_validator(state: State, *, keep_pending=True):
    """Restrict a candidate to the supported dataset window and current time.

    Args:
        state: Candidate QueryPlan, aware now and pending context.
        keep_pending: Save unresolved filters for clarification; False for custom
            callables that return plans directly without extraction context.

    Returns:
        A candidate clipped to 2026 and at most now plus one microsecond, or
        clarification updates if the resulting window is invalid. When keep_pending is True, also
        saves the unresolved query for a follow-up. No SQL is executed here."""
    try:
        plan = QueryPlan.model_validate(state["candidate"])
        now = state["now"].astimezone(SG)
        start = max(plan.start.astimezone(SG), datetime(2026, 1, 1, tzinfo=SG))
        end = min(plan.end.astimezone(SG), now + timedelta(microseconds=1), datetime(2027, 1, 1, tzinfo=SG))
        plan = QueryPlan.model_validate({**plan.model_dump(), "start": start, "end": end})
        return {"candidate": plan}
    except ValueError:
        message = "No supported date window: frames cover 2026 up to now."
        update = {"error": message, "outcome": "clarify"}
        if keep_pending:
            update["pending"] = pending_query(state, message)
        return update
