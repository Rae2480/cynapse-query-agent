"""LangGraph wiring: state, nodes, routes, and checkpoint configuration."""
from functools import partial
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from cynapse.helpers.context_manager import is_reset_request
from cynapse.agents.nonai_extractor import extract_nonai
from cynapse.agents.context_guard import context_guard
from cynapse.agents.context_resetter import context_resetter
from cynapse.agents.intent_extractor import intent_extractor
from cynapse.agents.filter_resolver import filter_resolver
from cynapse.agents.nonai_interpreter import nonai_interpreter
from cynapse.agents.query_validator import query_validator
from cynapse.agents.database_reader import database_reader
from cynapse.agents.page_navigator import page_navigator
from cynapse.agents.answerer import answerer
from cynapse.state import State


def build_graph(path: Path, parser=None, *, extractor=None, query_fn=None):
    """Build the conversational workflow and its in-memory checkpoint store.

    Args:
        path: SQLite database path used when a query is executed.
        parser: Optional custom nonai callable (request, previous_plan, now) -> QueryPlan.
        extractor: Optional AI or nonai callable (request, previous_plan, now, pending)
            -> Extraction or equivalent dictionary. Cannot be combined with parser.
        query_fn: Optional replacement for database_manager.query(path, plan, offset).

    Returns:
        A compiled LangGraph. Invoke with request and timezone-aware now in the
        input, and configurable.thread_id in the config to retain conversation state.

    Raises:
        ValueError: Both parser and extractor were supplied.

    Building the graph does not call a model or query SQLite. With neither dependency supplied,
    the nonai version is used. Checkpoints last only for this graph's lifetime."""
    if parser is not None and extractor is not None:
        raise ValueError("Supply a parser or an extractor, not both")
    if parser is None and extractor is None:
        extractor = extract_nonai
    graph = StateGraph(State)
    graph.add_node("context_guard", context_guard)
    graph.add_node("context_resetter", context_resetter)
    graph.add_node("query_validator", partial(query_validator, keep_pending=extractor is not None))
    graph.add_node("database_reader", partial(database_reader, path=path, query_fn=query_fn))
    graph.add_node("answerer", answerer)
    graph.add_node("page_navigator", partial(page_navigator, path=path, query_fn=query_fn))

    def router(state: State):
        """Choose the first workflow stage for the incoming request.

        Args:
            state: Conversation state containing the current request.

        Returns:
            context_resetter, page_navigator, intent_extractor (AI or nonai version),
            or nonai_interpreter (injected custom callable).
            Standalone reset and pagination commands bypass the model."""
        if state.get("error"):
            return "answerer"
        if is_reset_request(state["request"]):
            return "context_resetter"
        if state["request"].strip().rstrip(".!?").casefold() in {"next", "next page", "more", "previous", "previous page", "back"}:
            return "page_navigator"
        return "intent_extractor" if extractor is not None else "nonai_interpreter"

    if extractor is not None:
        graph.add_node("intent_extractor", partial(intent_extractor, extractor=extractor))
        graph.add_node("filter_resolver", filter_resolver)
        graph.add_conditional_edges("intent_extractor", lambda s: "answerer" if s.get("error") else "filter_resolver")
        graph.add_conditional_edges("filter_resolver", lambda s: "answerer" if s.get("error") else "query_validator")
    else:
        graph.add_node("nonai_interpreter", partial(nonai_interpreter, parser=parser))
        graph.add_conditional_edges("nonai_interpreter", lambda s: "answerer" if s.get("error") else "query_validator")
    graph.add_edge(START, "context_guard")
    graph.add_conditional_edges("context_guard", router)
    graph.add_conditional_edges("query_validator", lambda s: "answerer" if s.get("error") else "database_reader")
    for name in ("context_resetter", "database_reader", "answerer", "page_navigator"):
        graph.add_edge(name, END)
    serde = JsonPlusSerializer(allowed_msgpack_modules=[
        ("cynapse.helpers.data_models", "Camera"), ("cynapse.helpers.data_models", "QueryPlan"),
    ])
    return graph.compile(checkpointer=InMemorySaver(serde=serde))
