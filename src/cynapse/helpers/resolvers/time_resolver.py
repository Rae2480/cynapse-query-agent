"""Bounded deterministic grammar for extracted date/time expressions."""
import calendar
import re
from datetime import datetime, timedelta
from cynapse.settings import SG

MONTHS = {name.lower(): index for index, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.lower(): index for index, name in enumerate(calendar.month_abbr) if name})
WEEKDAYS = {name.lower(): index for index, name in enumerate(calendar.day_name)}


def normalize(text: str):
    """Normalize wording before deterministic date/time parsing.

    Args:
        text: Raw date, time or weekday expression.

    Returns:
        Lowercase text with numeric ordinal suffixes removed, commas replaced by
        spaces, surrounding sentence punctuation trimmed and whitespace collapsed.
        Does not validate dates."""
    text = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", text.casefold())
    return " ".join(text.replace(",", " ").split()).strip(" .!?\"\'")


def month_window(year: int, month: int):
    """Find the boundaries of a calendar month, including December rollover.

    Args:
        year: Calendar year.
        month: Month number from 1 through 12.

    Returns:
        Singapore-aware (start, end) datetimes: first day of this month and first
        day of the next. Start is inclusive; end is exclusive.

    Raises:
        ValueError: A resulting date falls outside valid calendar values."""
    start = datetime(year, month, 1, tzinfo=SG)
    end = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=SG)
    return start, end


def exact_date(text: str, default_year=2026):
    """Parse one explicit calendar date at Singapore midnight.

    Args:
        text: Normalized ISO date or named-month date, e.g. '9 august 2026'.
        default_year: Year to use when a named-month date omits it; defaults to 2026.

    Returns:
        A timezone-aware datetime at the beginning of the requested day.

    Raises:
        ValueError: Wording is unsupported or the date is invalid, e.g. February 30."""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=SG)
    match = re.fullmatch(r"([a-z]+) (\d{1,2})(?: (\d{4}))?", text)
    if match and match[1] in MONTHS:
        return datetime(int(match[3] or default_year), MONTHS[match[1]], int(match[2]), tzinfo=SG)
    match = re.fullmatch(r"(\d{1,2}) ([a-z]+)(?: (\d{4}))?", text)
    if match and match[2] in MONTHS:
        return datetime(int(match[3] or default_year), MONTHS[match[2]], int(match[1]), tzinfo=SG)
    raise ValueError("Use an ISO date or a named month, such as 2026-08-09 or August 9, 2026.")


def resolve_date(expression: str, now: datetime):
    """Resolve a supported date expression into a Singapore calendar window.

    Args:
        expression: Relative day/week/month, named month, explicit date or date range.
        now: Timezone-aware reference clock; relative expressions use Singapore time.

    Returns:
        Inclusive-start, exclusive-end datetime pair. Weeks begin Monday. Explicit
        final dates include that whole day. Named dates/months without a year default
        to 2026; relative expressions use now. Dataset clipping happens elsewhere.

    Raises:
        ValueError: Clock is naive, dates are invalid/reversed, or wording is
            unsupported or does not identify the required calendar month."""
    if now.tzinfo is None:
        raise ValueError("A timezone-aware clock is required.")
    text = normalize(expression)
    text = re.sub(r"^(?:(?:only|just|for|on|in)\s+)+", "", text)
    text = re.sub(r"^(?:the )?(?:whole of |month of )", "", text)
    now = now.astimezone(SG)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if text in ("today", "yesterday", "tomorrow"):
        start = midnight + timedelta(days={"today": 0, "yesterday": -1, "tomorrow": 1}[text])
        return start, start + timedelta(days=1)
    if text in ("this week", "last week"):
        start = midnight - timedelta(days=midnight.weekday() + (7 if text == "last week" else 0))
        return start, start + timedelta(days=7)
    if text in ("this month", "last month"):
        year, month = now.year, now.month
        if text == "last month":
            year, month = (year - 1, 12) if month == 1 else (year, month - 1)
        return month_window(year, month)
    # Shared month: 'from the 15th to 18th of last month'.
    match = re.fullmatch(r"(?:from |between )?(?:the )?(\d{1,2}) (?:to|through|and) (?:the )?(\d{1,2}) (?:of |in )(.+)", text)
    if match:
        month_start, month_end = resolve_date(match[3], now)
        if month_start.day != 1 or month_end != month_window(month_start.year, month_start.month)[1]:
            raise ValueError("The day range must name one calendar month.")
        start = month_start.replace(day=int(match[1]))
        last = month_start.replace(day=int(match[2]))
        return checked_range(start, last)
    # Explicit endpoints, with optional shared year/month on the right.
    match = re.fullmatch(r"(?:from |between )?(.+?) (?:to|through|and) (.+)", text)
    if match:
        left, right = match[1], match[2]
        years = set(re.findall(r"\b\d{4}\b", text))
        default_year = int(next(iter(years))) if len(years) == 1 else 2026
        if re.fullmatch(r"\d{1,2}(?: \d{4})?", right):
            left_month = re.search(r"\b(" + "|".join(MONTHS) + r")\b", left)
            if left_month:
                right = left_month[0] + " " + right
        start = exact_date(left, default_year)
        last = exact_date(right, default_year)
        return checked_range(start, last)
    match = re.fullmatch(r"([a-z]+)(?: (\d{4}))?", text)
    if match and match[1] in MONTHS:
        return month_window(int(match[2] or 2026), MONTHS[match[1]])
    start = exact_date(text)
    return start, start + timedelta(days=1)


def checked_range(start: datetime, last: datetime):
    """Convert inclusive calendar-day endpoints into a half-open interval.

    Args:
        start: Beginning of the first day.
        last: Beginning of the final included day, using the same timezone convention.

    Returns:
        (start, last plus one day), with an exclusive end.

    Raises:
        ValueError: last precedes start."""
    if last < start:
        raise ValueError("The end date must not precede the start date.")
    return start, last + timedelta(days=1)


def resolve_clock(text: str, *, end=False):
    """Convert one normalized clock expression into a sortable clock string.

    Args:
        text: Lowercase AM/PM time, HH:MM, noon or midnight.
        end: Allow 24:00 and interpret midnight as 24:00 for an interval's end.

    Returns:
        Zero-padded HH:MM string; 24:00 is allowed only for an end boundary.

    Raises:
        ValueError: Time is invalid or lacks an unambiguous supported format."""
    text = text.strip().replace(".", "")
    if text in ("noon", "midnight"):
        return "12:00" if text == "noon" else "24:00" if end else "00:00"
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", text)
    if match:
        hour, minute = int(match[1]), int(match[2] or 0)
        if not 1 <= hour <= 12 or not 0 <= minute <= 59:
            raise ValueError("Invalid clock time.")
        hour = hour % 12 + (12 if match[3] == "pm" else 0)
        return f"{hour:02}:{minute:02}"
    match = re.fullmatch(r"([01]\d|2[0-3]):([0-5]\d)", text)
    if match:
        return text
    if end and text == "24:00":
        return text
    raise ValueError("Specify AM/PM on each time, or use HH:MM in 24-hour time.")


def resolve_time(expression: str):
    """Parse an increasing daily time range, such as '8 AM to 10 AM'.

    Args:
        expression: Two clock expressions separated by to, and, until or a hyphen;
            optional leading 'from' or 'between' is accepted.

    Returns:
        (start, end) HH:MM strings for an inclusive start and exclusive end.

    Raises:
        ValueError: Format is unsupported, clocks are invalid/ambiguous, or the
            range is equal-ended, decreasing or overnight."""
    text = re.sub(r"^(?:only |just )", "", normalize(expression))
    match = re.fullmatch(r"(?:between |from )?(.+?)\s*(?:\bto\b|\band\b|\buntil\b|-)\s*(.+)", text)
    if not match:
        raise ValueError("Specify a time range, such as between 8 AM and 10 AM.")
    start, end = resolve_clock(match[1]), resolve_clock(match[2], end=True)
    if end <= start:
        raise ValueError("Use an increasing daily time range; overnight ranges need clarification.")
    return start, end


def resolve_weekday(expression: str):
    """Resolve a single weekday expression such as 'every Tuesday'.

    Args:
        expression: Weekday name, optionally plural or preceded by on/every/each/only.

    Returns:
        Integer weekday: Monday is 0 and Sunday is 6.

    Raises:
        ValueError: Expression does not identify one supported weekday."""
    text = normalize(expression)
    text = re.sub(r"^(?:on )?(?:every |each |only )?", "", text)
    text = text.removesuffix("s")
    if text not in WEEKDAYS:
        raise ValueError("Specify one weekday, such as every Tuesday.")
    return WEEKDAYS[text]
