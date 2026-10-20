"""Intent extractor role in the query workflow."""
from cynapse.helpers.context_manager import prepare_query
from cynapse.helpers.data_models import Extraction
from cynapse.helpers.openai_client import ExtractionError
from cynapse.state import State


def intent_extractor(state: State, *, extractor):
    """Ask the extractor for explicit filter changes and prepare their context.

    Args:
        state: Current request, accepted plan, aware now and optional pending query.
        extractor: AI or nonai callable returning Extraction or a dictionary.

    Returns:
        Updates from prepare_query(), or an error update if extraction/validation
        raises ValueError. Previously accepted filters are retained on failure.

    This can make a paid model call when the configured extractor uses OpenAI."""
    try:
        result = extractor(state["request"], state.get("plan"), state["now"], state.get("pending"))
        data = result.model_dump() if isinstance(result, Extraction) else result
        return prepare_query(Extraction.model_validate(data), state)
    except ValueError as exc:
        message = str(exc) if isinstance(exc, ExtractionError) else "The language model returned malformed filters. Please retry."
        return {"extraction": None, "candidate": None, "error": message, "outcome": "error", "response": {}}
