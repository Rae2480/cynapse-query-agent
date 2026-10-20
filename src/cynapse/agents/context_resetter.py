"""Context resetter role in the query workflow."""
from cynapse.helpers.context_manager import reset_context
from cynapse.state import State


def context_resetter(state: State):
    """Clear accepted filters, pending clarification and pagination.

    Args:
        state: Current conversation state; its values are not needed here.

    Returns:
        Partial state updates from reset_context(), including a reset response."""
    return reset_context()
