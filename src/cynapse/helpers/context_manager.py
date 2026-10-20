"""Pure conversation updates, independent of model clients and database access."""
import re
from datetime import datetime

from cynapse.helpers.data_models import Extraction
from cynapse.state import State


def is_reset_request(request: str):
    """Check whether the entire message is a standalone reset command.

    Args:
        request: User text, such as "Reset!" or "start over".

    Returns:
        True for reset/start over/new search/clear context, ignoring case,
        surrounding whitespace and ending punctuation; otherwise False.
        A combined request such as "reset and show CTE" is not matched.
    """
    return re.fullmatch(r"\s*(?:start over|reset|new search|clear context)[.!?]*\s*", request, re.IGNORECASE) is not None


def reset_context():
    """Build updates that clear the conversation and pagination.

    Args:
        None.

    Returns:
        A state-update dictionary clearing accepted, pending and temporary
        filters, resetting page counters, and asking for a new camera/date.
        The caller applies these updates; this function does not mutate state.
    """
    return {
        "plan": None, "pending": None, "candidate": None, "extraction": None,
        "resolution_base": None, "date_now": None, "error": None, "outcome": "clarify",
        "page_offset": 0, "page_total": 0, "context_at": None,
        "response": {"intent": "clarify", "message": "Started a new search. Which camera and date window should I use?"},
    }


def pending_query(state: State, question: str):
    """Package an unfinished request so a later answer can complete it.

    Args:
        state: Current extraction, accepted base plan and date-resolution clock.
            date_now must be a datetime, not None.
        question: Clarification message to keep with the unfinished filters.

    Returns:
        A dictionary containing extraction, base_plan, the clock as an ISO
        string, and question. The saved clock keeps relative dates stable.
    """
    return {
        "extraction": state["extraction"], "base_plan": state.get("resolution_base"),
        "date_now": state["date_now"].isoformat(), "question": question,
    }


def prepare_query(extraction: Extraction, state: State):
    """Combine this turn's filter updates with accepted or pending context.

    Args:
        extraction: Validated model intent, filter actions and reset/continuation flags.
        state: Existing accepted plan, optional pending request and current clock.

    Returns:
        A state-update dictionary with the effective extraction, resolution
        base, date clock and optional clarification. Continuations merge pending
        filters; independent requests use accepted filters unless reset is set.
        Rejected requests leave accepted/pending filters untouched. A new plan
        is accepted only after later resolution, validation and execution.
        The supplied state is not mutated.
    """
    update = {"extraction": extraction.model_dump(), "candidate": None, "response": {}, "error": None, "outcome": extraction.intent}
    if extraction.intent == "reject":
        update["error"] = "I can only retrieve CCTV frame metadata by camera and date/time."
        return update
    pending = state.get("pending")
    if extraction.intent == "query" and all(
        getattr(extraction, field).action == "keep" for field in ("camera", "date", "time", "recurrence")
    ) and not (extraction.continuation and pending):
        update.update(error="Please specify a camera or date/time filter.", outcome="clarify")
        if extraction.reset:
            update.update(plan=None, pending=None, page_offset=0, page_total=0)
        return update
    if extraction.continuation and pending:
        merged = extraction.model_dump()
        for field in ("camera", "date", "time", "recurrence"):
            if merged[field]["action"] == "keep":
                merged[field] = pending["extraction"][field]
        base = pending["base_plan"]
        date_now = datetime.fromisoformat(pending["date_now"]) if extraction.date.action == "keep" else state["now"]
        extraction = Extraction.model_validate(merged)
    else:
        base = state["plan"].model_dump(mode="json") if state.get("plan") and not extraction.reset else None
        date_now = state["now"]
    update.update(extraction=extraction.model_dump(), resolution_base=base, date_now=date_now, pending=None)
    if extraction.reset:
        update["plan"] = None
        update.update(page_offset=0, page_total=0)
    if extraction.intent == "clarify":
        update["error"] = extraction.clarification_question
        update["pending"] = pending_query(update, extraction.clarification_question)
    return update
