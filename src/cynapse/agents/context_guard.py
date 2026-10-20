"""Bound request size and expire idle conversation context before routing."""
from datetime import timedelta

from cynapse.context_manager import reset_context
from cynapse.settings import CONTEXT_IDLE_MINUTES, MAX_REQUEST_CHARS
from cynapse.state import State


def context_guard(state: State):
    """Prepare one turn without accumulating a transcript.

    Args:
        state: Current request/aware now and optional last context_at timestamp.

    Returns:
        Partial state updates clearing old errors, recording this turn's clock,
        and clearing filters/pending pages after 30 idle minutes. Requests above
        MAX_REQUEST_CHARS receive clarification without a model or SQL call.
    """
    now = state["now"]
    last = state.get("context_at")
    update = reset_context() if last and now - last >= timedelta(minutes=CONTEXT_IDLE_MINUTES) else {}
    update.update(error=None, response={}, context_at=now)
    if len(state["request"]) > MAX_REQUEST_CHARS:
        update.update(candidate=None, outcome="clarify",
                      error=f"Please keep your request within {MAX_REQUEST_CHARS} characters.")
    return update
