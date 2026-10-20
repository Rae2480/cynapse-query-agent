"""Create simulated frame records and search them in a local SQLite database."""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from cynapse.data_models import Camera, QueryPlan
from cynapse.settings import SG


def seed(path: Path, now: datetime):
    """Create the database if needed and fill it with simulated frame records.

    Generates one record per camera every five minutes, starting at midnight
    on January 1, 2026. Stops at ``now`` or the end of 2026, whichever comes
    first. Existing camera/timestamp pairs are skipped, so rerunning is safe.
    Also creates an index to speed up camera-and-time searches.

    Args:
        path: Location of the SQLite file, for example Path("frames.db").
        now: Timezone-aware cutoff time; converted to Singapore time.

    Returns:
        None. Creates or updates the database file instead of returning data.
    """
    now = now.astimezone(SG)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS frames (frame_id INTEGER PRIMARY KEY, datetime TEXT NOT NULL, camera_name TEXT NOT NULL, UNIQUE(datetime, camera_name))")
        conn.execute("CREATE INDEX IF NOT EXISTS frames_camera_time ON frames(camera_name, datetime)")

        def rows():
            """Produce records one at a time instead of building a large list.

            Args:
                None. Uses the enclosing seed() function's ``now`` cutoff
                and the ten camera names defined in Camera.

            Yields:
                A (datetime, camera_name) tuple for each simulated capture.
                Both values are strings; datetime uses ISO 8601 Singapore time.
                SQLite assigns the frame ID when the tuple is inserted.
            """
            instant = datetime(2026, 1, 1, tzinfo=SG)
            while instant <= now and instant.year == 2026:
                for camera in Camera:
                    yield instant.isoformat(), camera.value
                instant += timedelta(minutes=5)

        conn.executemany("INSERT OR IGNORE INTO frames(datetime, camera_name) VALUES (?, ?)", rows())


def query(path: Path, plan: QueryPlan, offset: int = 0):
    """Count matching records and return one page without changing the database.

    Filters by camera and a date window: start is included, end is excluded.
    Optionally filters by weekday and a daily time window. Returns records in
    timestamp/frame-ID order, using parameterized SQL and a read-only connection.

    Args:
        path: Location of an existing SQLite file containing the frames table.
        plan: Validated QueryPlan with camera, date bounds, optional weekday
            and daily time bounds, and the maximum records per page (limit).
        offset: Number of matching records to skip; 0 starts at the first page.

    Returns:
        A dictionary with:
            total: Number of all matching records, regardless of page size.
            frames: List of up to plan.limit dictionaries, each containing
                frame_id (int), datetime (ISO string), and camera_name (string).
                An empty list means no matching records remain at this offset.

    Raises:
        ValueError: If offset is not a nonnegative integer (bool is rejected).
        sqlite3.Error: If the database cannot be read or the query fails.
    """
    if type(offset) is not int or offset < 0:
        raise ValueError("Page offset must be a nonnegative integer")
    # The only execution path uses fixed SQL and a read-only connection.
    uri = path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.execute("PRAGMA query_only = ON")
        conn.row_factory = sqlite3.Row
        params = [plan.camera.value, plan.start.astimezone(SG).isoformat(), plan.end.astimezone(SG).isoformat()]
        where = "FROM frames WHERE camera_name = ? AND datetime >= ? AND datetime < ?"
        if plan.weekday is not None:
            # Extract the local date before strftime: full ISO timestamps convert to UTC.
            where += " AND CAST(strftime('%w', substr(datetime, 1, 10)) AS INTEGER) = ?"
            params.append((plan.weekday + 1) % 7)
        if plan.time_start is not None:
            where += " AND substr(datetime, 12, 5) >= ? AND substr(datetime, 12, 5) < ?"
            params.extend([plan.time_start, plan.time_end])
        total = conn.execute("SELECT COUNT(*) " + where, params).fetchone()[0]
        rows = conn.execute("SELECT frame_id, datetime, camera_name " + where + " ORDER BY datetime, frame_id LIMIT ? OFFSET ?", (*params, plan.limit, offset)).fetchall()
        return {"total": total, "frames": [dict(row) for row in rows]}
