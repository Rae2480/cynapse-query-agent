"""The nonai version: bounded language rules using shared camera/date resolvers.

Produces the same Extraction contract as the AI version. Unknown words and leftover
constraints require clarification or rejection instead of being silently ignored.
No model client, API key or network access is used.
"""
import re
from datetime import datetime

from cynapse.helpers.resolvers.camera_resolver import resolve_camera
from cynapse.helpers.data_models import Extraction, QueryPlan
from cynapse.helpers.resolvers.time_resolver import normalize, resolve_date

CLOCK = r"(?:\d{1,2}(?::\d{2})?\s*[ap]\.?m\.?|\d{2}:\d{2}|noon|midnight)"
TIME_RANGE = re.compile(rf"\b(?:(?:between|from)\s+)?{CLOCK}\s*(?:to|and|until|-)\s*{CLOCK}\b")
WEEKDAY = re.compile(r"\b(?:on\s+)?(?:(?:every|each|only)\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b")
FILLER = re.compile(r"\b(?:show|me|frames|framed|from|for|the|how|about|only|those|on|in|please|retrieve|get|instead|and)\b")
UNSUPPORTED = re.compile(
    r"\b(?:ignore|instructions|prompt|secret|reveal|delete|drop|update|insert|alter|"
    r"truncate|select|sqlite_master|containing|identify|people|plates?|weather)\b|;|--"
)


def take_expression(text, resolver):
    """Remove the longest contiguous phrase accepted by a deterministic resolver.

    Args:
        text: Remaining normalized request text.
        resolver: Callable validating one expression; raises ValueError on failure.

    Returns:
        (expression, remaining_text), or (None, text) if nothing matches. Searches
        at most 16 words per phrase. Unmatched words remain for the caller's check.
    """
    words = list(re.finditer(r"\S+", text))
    best = None
    for index, first in enumerate(words):
        for last in words[index:index + 16]:
            expression = text[first.start():last.end()]
            try:
                resolver(expression)
            except ValueError:
                continue
            if best is None or len(expression) > best[1] - best[0]:
                best = first.start(), last.end()
    if best is None:
        return None, text
    start, end = best
    return text[start:end], text[:start] + " " + text[end:]


def extract_nonai(text: str, previous: QueryPlan | None, now: datetime, pending: dict | None = None):
    """Interpret supported metadata requests in the nonai version.

    Args:
        text: Request or follow-up in the bounded natural-language grammar.
        previous: Accepted plan (merged later by shared conversation logic).
        now: Aware reference clock for validating candidate date expressions.
        pending: Optional unfinished query used to recognize short answers.

    Returns:
        Extraction with query/clarify/reject intent, raw filter updates and reset/
        continuation flags. Supports shared camera aliases/typos, calendar ranges,
        daily times and one recurring weekday. Unknown wording never runs SQL.

    Raises:
        ValueError: The reference clock is naive.

    Does not query the database or call an AI API. Final resolution and validation
    happen in the same graph nodes used by the AI version.
    """
    if now.tzinfo is None:
        raise ValueError("A timezone-aware clock is required.")
    text = normalize(text).strip().rstrip(".!?")
    data = {
        "intent": "query", "reset": False, "continuation": False,
        **{field: {"action": "keep", "value": None} for field in ("camera", "date", "time", "recurrence")},
        "clarification_question": None,
    }
    if len(text) > 500 or UNSUPPORTED.search(text):
        data["intent"] = "reject"
        return Extraction.model_validate(data)
    text, resets = re.subn(r"^(?:start over|reset|new search|clear context)[.,!]?\s*", "", text)
    data["reset"] = bool(resets)
    data["continuation"] = bool(pending) and not resets and not re.search(r"\b(?:show|retrieve|get)\b", text)

    for field, target in (("time", "time"), ("weekday", "recurrence"), ("date", "date"), ("camera", "camera")):
        text, count = re.subn(r"\b(?:remove|clear) (?:the )?" + field + r" restriction\b", " ", text)
        if count:
            data[target] = {"action": "clear", "value": None}

    camera, text = take_expression(text, resolve_camera)
    other_camera, _ = take_expression(text, resolve_camera)
    if camera:
        data["camera"] = {"action": "set", "value": camera}
    if other_camera:
        data.update(intent="clarify", clarification_question="Please select one camera.")
        return Extraction.model_validate(data)

    time = TIME_RANGE.search(text) or re.search(r"\bbetween \d{1,2} and \d{1,2}\b(?! of)", text)
    if time:
        data["time"] = {"action": "set", "value": time[0]}
        text = text[:time.start()] + " " + text[time.end():]
    recurrence = WEEKDAY.search(text)
    if recurrence:
        data["recurrence"] = {"action": "set", "value": recurrence[0]}
        text = text[:recurrence.start()] + " " + text[recurrence.end():]
    date, text = take_expression(text, lambda expression: resolve_date(expression, now))
    if date:
        data["date"] = {"action": "set", "value": date}
    remainder = FILLER.sub(" ", text).strip()
    if remainder:
        data.update(intent="clarify", clarification_question=
                    "Please specify one camera, a supported calendar window and an optional daily time range or weekday.")
        # Unsupported fragments must not become accepted filters on a later turn.
        data["continuation"] = False
        for field in ("camera", "date", "time", "recurrence"):
            data[field] = {"action": "keep", "value": None}
    return Extraction.model_validate(data)
