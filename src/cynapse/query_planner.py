"""Resolve extracted filters into a validated query plan."""
from datetime import datetime, timedelta

from cynapse.camera_resolver import resolve_camera
from cynapse.settings import SG
from cynapse.data_models import Extraction, QueryPlan
from cynapse.time_resolver import resolve_date, resolve_time, resolve_weekday


def resolve_extraction(extraction: Extraction, previous: QueryPlan | None, now: datetime, *, date_now: datetime | None = None):
    """Turn raw filter updates and accepted context into a validated query plan.

    Args:
        extraction: Validated Extraction with keep/set/clear actions. The graph
            calls this resolver for query intent after merging pending updates.
        previous: Accepted base plan to inherit omitted filters from, or None.
        now: Timezone-aware clock for defaults; converted to Singapore time.
        date_now: Optional earlier clock anchor for a pending relative date.

    Returns:
        QueryPlan with canonical camera, calendar bounds and optional daily/
        weekday predicates. Reset ignores previous; clear removes a restriction.
        With no date context, uses 2026 through now. The graph separately clips
        the resulting window to available data. No SQL is executed here.

    Raises:
        ValueError: If the clock is naive, camera is missing/ambiguous, or an
            expression or resulting plan is invalid or unsupported.
    """
    if now.tzinfo is None:
        raise ValueError("A timezone-aware clock is required.")
    previous = None if extraction.reset else previous
    if extraction.camera.action == "set":
        camera = resolve_camera(extraction.camera.value)
    elif extraction.camera.action == "keep" and previous:
        camera = previous.camera
    else:
        raise ValueError("Which camera should I use?")
    now = now.astimezone(SG)
    if extraction.date.action == "set":
        start, end = resolve_date(extraction.date.value, date_now or now)
    elif extraction.date.action == "keep" and previous:
        start, end = previous.start, previous.end
    else:
        start, end = datetime(2026, 1, 1, tzinfo=SG), now + timedelta(microseconds=1)
    if extraction.time.action == "set":
        time_start, time_end = resolve_time(extraction.time.value)
    elif extraction.time.action == "keep" and previous:
        time_start, time_end = previous.time_start, previous.time_end
    else:
        time_start = time_end = None
    if extraction.recurrence.action == "set":
        weekday = resolve_weekday(extraction.recurrence.value)
    elif extraction.recurrence.action == "keep" and previous:
        weekday = previous.weekday
    else:
        weekday = None
    return QueryPlan(camera=camera, start=start, end=end, time_start=time_start, time_end=time_end, weekday=weekday)
