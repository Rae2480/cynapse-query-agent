# CYNAPSE: NATURAL-LANGUAGE CCTV METADATA QUERY AGENT

## Run

Python 3.11+. Create/activate a virtual environment and install:

```sh
pip install -e '.[dev,openai]'
```

Export OPENAI_API_KEY and CYNAPSE_MODEL=openai:gpt-5-mini in your terminal.

```sh
cynapse seed
cynapse chat --ai
```

  OR

```sh
cynapse chat --nonai
```

Commands: next, previous, reset, exit.

## Design and choices

- `app.py` selects the version: nonai or AI.
- `orchestration.py` connects agents with specific responsibilities.
- GPT-5-mini interprets intent and explicit filter changes into a strict Pydantic
  schema. It suits the bounded extraction role; the live tests checked its handling
  of the assignment requests. It does not generate SQL.
- Python agents resolve camera names, calendar windows, daily times and weekdays
  using helper functions, then validate a query plan. Only interpretation in AI
  mode calls OpenAI; the other stages use deterministic Python logic.
- LangGraph was chosen to make routing, conditional clarification and conversation
  state explicit and testable.
- SQLite fits the local simulated dataset without requiring a separate database
  server. Queries use fixed, parameterized SELECT statements through a read-only
  connection, with a camera/time index supporting lookups.
- Both versions share resolution, validation and database execution. This combines
  flexible AI interpretation with deterministic query behavior.

The workflow is split into agents with clearly defined responsibilities, each in
its own file, and orchestrated through LangGraph. This supports:

- **Modularity:** Individual agents can be tested, improved or replaced. Separating
  roles also makes bottlenecks and failure points easier to locate.
- **Extensibility:** New features can be added without reworking the whole workflow,
  and clear boundaries make it easier to identify where changes belong.
- **Maintainability:** Descriptive agent names and shared helper folders make the
  codebase easier for future developers to navigate and maintain.

## Dataset and behavior

Seeding creates one frame every five minutes for all ten specified cameras
from January 1, 2026 to the current time. Records contain integer frame_id,
ISO datetime and canonical enum camera_name. Run `cynapse seed` once to create the
database; chat automatically refreshes the records at startup. Calendar reasoning
uses Singapore time, Monday-start weeks, inclusive start/exclusive end and 2026
when the year is omitted. Results have exact totals and 20-record pages.

For example, a conversation can be:

```text
Show me frames from CTE.
How about only those from this week?
```

The first request searches available CTE records. The second retains CTE and
narrows the date window to this week, from Monday through the current time.

## Context and guardrails

Only the current request, accepted filters and one pending clarification reach
the model. Full chat history and returned frames are never sent. Context expires
after 30 idle minutes and explicit reset clears it
immediately. Requests are capped at 1,000 characters, filter expressions at 160
and clarification questions at 240.

Follow-ups preserve omitted filters; explicit changes replace them and clear
removes restrictions. Rejections and failures preserve valid context. Independent
requests do not merge unfinished filters.

The model also rejects unrelated, sensitive, image-content, injection and
database-write requests. Schema/date/name validation precedes SQL. Model timeouts
and malformed output produce safe errors, not queries.

The application handles explicit AI provider refusals separately. Seeding is the
only write path; natural-language requests cannot run UPDATE, DELETE or DROP database commands.

## Verification (As of 2026-10-20)

301 local tests passed. Paid tests skip unless CYNAPSE_LIVE_TESTS=1.
Live gpt-5-mini: all 46 full-agent cases and 18 extraction cases passed after fixes
and targeted reruns. These results are from one evaluation pass with targeted
reruns, rather than a statistical accuracy estimate for arbitrary requests.
Cases cover all assignment examples, all ten cameras, typos, aliases,
boundaries, invalid/future dates, recurring and combined filters, guardrails,
clarification answers, resets and rejected turns.
Separate tests confirmed that model payload size remains bounded across 25
conversation turns. They also check 30-minute idle expiry, handling of model
timeouts and malformed output, conversation isolation, and complete pagination
without duplicate IDs.

```sh
pytest -q
CYNAPSE_LIVE_TESTS=1 pytest -q tests/test_assignment_inputs.py -k openai
```

## Current Limitations

These requests are not supported as written and need clarification or rephrasing:

| Limitation | Example request | How to rephrase |
|---|---|---|
| Multiple cameras | `Show CTE and PIE frames yesterday.` | Query each camera separately. |
| Overnight daily windows | `Show CTE frames yesterday between 10 PM and 2 AM.` | Query 10 PM–midnight on the first date, then midnight–2 AM on the next date. |
| Multiple weekdays | `Show MCE frames every Tuesday and Thursday in September 2026.` | Query each weekday separately. |
| Exclusions | `Show CTE frames this week except Monday.` | Specify the dates you want to include. |
| Ambiguous numeric dates | `Show CTE frames on 08/09/2026.` | Use `2026-08-09` or `2026-09-08`, depending on the intended date. |
| Image analysis | `Show CTE frames containing a red car yesterday.` | Only camera and date/time metadata can be searched; image-content requests are rejected. |

- **Temporary context:** After restarting the app, `Only today.` cannot reuse a camera from the previous session. Start with a complete request such as `Show CTE frames today.` Context also expires after 30 idle minutes.

## Code walkthrough

### Application

Shared support code lives in `helpers/`, with resolver and database subfolders.

- `src/cynapse/app.py`: Reads terminal commands, selects AI/nonai mode and starts chat or seeding.
- `src/cynapse/orchestration.py`: Connects agents with LangGraph and keeps each conversation separate.
- `src/cynapse/state.py`: Defines the state fields shared by the agents.
- `src/cynapse/helpers/data_models.py`: Defines supported cameras and validates extraction updates and query plans.
- `src/cynapse/settings.py`: Stores the timezone, timeout, context expiry and input-size limits.
- `src/cynapse/prompt.py`: Tells the model how to interpret requests and reject unsupported ones.
- `src/cynapse/helpers/openai_client.py`: Calls OpenAI for structured extraction and handles failures and refusals.
- `src/cynapse/helpers/context_manager.py`: Merges follow-ups, saves unfinished queries and clears search context.
- `src/cynapse/helpers/resolvers/camera_resolver.py`: Matches camera acronyms, full names, aliases and small typos.
- `src/cynapse/helpers/resolvers/time_resolver.py`: Converts date, time and weekday expressions into precise boundaries.
- `src/cynapse/helpers/resolvers/query_planner.py`: Combines extracted changes and inherited filters into a query plan.
- `src/cynapse/helpers/database/database_manager.py`: Generates frame records and runs parameterized, read-only queries.
- `src/cynapse/helpers/__init__.py`: Marks shared helpers as a Python package.
- `src/cynapse/helpers/resolvers/__init__.py`: Marks resolution helpers as a Python package.
- `src/cynapse/helpers/database/__init__.py`: Marks database helpers as a Python package.
- `src/cynapse/agents/__init__.py`: Marks the agents folder as a Python package.
- `src/cynapse/agents/context_guard.py`: Expires idle context and blocks oversized requests before interpretation.
- `src/cynapse/agents/context_resetter.py`: Clears filters, pending questions and pagination for a new search.
- `src/cynapse/agents/intent_extractor.py`: Runs the selected extractor and prepares this turn for resolution.
- `src/cynapse/agents/nonai_extractor.py`: Interprets supported requests with Python rules, without API calls.
- `src/cynapse/agents/filter_resolver.py`: Builds a candidate plan or preserves unresolved filters for clarification.
- `src/cynapse/agents/query_validator.py`: Clips the candidate to available dates and prevents invalid queries.
- `src/cynapse/agents/database_reader.py`: Fetches a result page and formats records, filters and page information.
- `src/cynapse/agents/page_navigator.py`: Handles next/previous commands using the accepted filters.
- `src/cynapse/agents/answerer.py`: Returns clarification, rejection and error messages.
- `src/cynapse/agents/nonai_interpreter.py`: Adapts an optional custom callable that produces query plans directly.

### Evaluation tools

- `evaluations/__init__.py`: Marks the evaluation tools as a Python package.
- `evaluations/evaluate_local.py`: Measures deterministic resolution and SQLite performance without OpenAI.
- `evaluations/evaluate_agent.py`: Runs full conversations and checks returned filters, records and guardrails.
- `evaluations/evaluate_extraction.py`: Checks selected fields in model output without running database queries.

### Tests

- `tests/test_agent.py`: Checks basic queries, context, result limits and unsupported requests.
- `tests/test_agent_evaluation.py`: Checks evaluation grading and stopping on authentication failures.
- `tests/test_assignment_inputs.py`: Runs assignment examples without AI and all opt-in live AI cases.
- `tests/test_camera_resolver.py`: Checks camera matching, aliases, typos and ambiguity.
- `tests/test_chat_regressions.py`: Checks CLI mode selection, database refresh and checkpoint restoration.
- `tests/test_conversation.py`: Checks follow-ups, clarification, resets, isolation and bounded context.
- `tests/test_evaluation.py`: Runs the shared acceptance cases against the nonai version.
- `tests/test_openai_client.py`: Checks model configuration, structured output, timeouts and safe failures.
- `tests/test_pagination.py`: Checks page navigation, boundaries and complete retrieval without duplicates.
- `tests/test_time_resolver.py`: Checks dates, times, weekdays, invalid inputs and availability clipping.

## Codex declaration

I used OpenAI Codex to help develop and refactor the code, explain its behavior, and
create tests and documentation. I used it again to fix some bugs in date parsing,
guardrails and context handling.
