"""Shared LangGraph state; accepted and pending filters stay separate."""
from datetime import datetime
from typing import TypedDict

from cynapse.data_models import QueryPlan


class State(TypedDict, total=False):
    """Dictionary fields shared by LangGraph stages for one conversation.

    Attributes:
        request: Current user message.
        now: Current turn's timezone-aware clock.
        plan: Last accepted QueryPlan, or None.
        candidate: Proposed plan awaiting validation/execution, or None.
        response: Public result page or explanatory message.
        error: Current explanatory/error message, or None.
        extraction: Current model extraction as a plain dictionary.
        outcome: Current intent or operational-error category.
        pending: Unfinished filter updates and clarification question.
        resolution_base: Serialized accepted plan used as the merge starting point.
        date_now: Clock anchor used to interpret the current date expression.
        page_offset: Number of matching records skipped for the current page.
        page_total: Last known total count for the accepted query.
        context_at: Last turn clock, used to expire idle filters and pending context.

    Fields may be absent initially. Nodes return partial updates which LangGraph
    applies; this TypedDict documents the shape without validating it at runtime.
    """
    request: str
    now: datetime
    plan: QueryPlan | None
    candidate: QueryPlan | None
    response: dict
    error: str | None
    extraction: dict | None
    outcome: str
    pending: dict | None
    resolution_base: dict | None
    date_now: datetime | None
    page_offset: int
    page_total: int
    context_at: datetime | None
