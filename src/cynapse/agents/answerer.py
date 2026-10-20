"""Answerer role in the query workflow."""
from cynapse.state import State


def answerer(state: State):
    """Format a clarification, rejection or operational error for the user.

    Args:
        state: outcome string and error message produced by an earlier stage.

    Returns:
        A response update with intent and message; no model or database call."""
    return {"response": {"intent": state["outcome"], "message": state["error"]}}
