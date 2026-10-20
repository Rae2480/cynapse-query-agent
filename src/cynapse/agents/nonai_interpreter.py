"""Nonai interpreter role in the query workflow."""
from cynapse.state import State


def nonai_interpreter(state: State, *, parser):
    """Interpret a request through an injected custom nonai callable.

    Args:
        state: Request text, optional accepted plan and aware now clock.
        parser: Custom callable (request, previous_plan, now) -> QueryPlan.

    Returns:
        Candidate plan and cleared error/response updates on success, or a
        clarification update if the custom nonai callable raises ValueError."""
    try:
        candidate = parser(state["request"], state.get("plan"), state["now"])
        return {"candidate": candidate, "error": None, "response": {}, "outcome": "query"}
    except ValueError as exc:
        return {"candidate": None, "error": str(exc), "response": {}, "outcome": "clarify"}
