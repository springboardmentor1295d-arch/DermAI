"""
Indian Standard Time (IST - Asia/Kolkata, UTC+05:30) centralized timezone handling for DermAI.
Strictly adheres to Python 3.9+ zoneinfo standards without manual arithmetic or double conversions.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional
from sqlalchemy.types import TypeDecorator, DateTime

IST = ZoneInfo("Asia/Kolkata")


class UTCDateTime(TypeDecorator):
    """
    SQLAlchemy TypeDecorator that stores UTC instants in SQLite/SQL and always
    restores timezone-aware UTC datetimes on retrieval.
    Prevents timezone-naive ISO serialization issues in FastAPI/Pydantic.
    """
    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: Optional[datetime], dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value: Optional[datetime], dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


def format_utc_iso(dt: Optional[datetime]) -> Optional[str]:
    """
    Serializes a datetime as an ISO 8601 UTC string ending with 'Z'.
    Example output: '2026-11-25T10:51:47.123456Z'
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def now_ist() -> datetime:
    """Returns current timezone-aware timestamp in Asia/Kolkata (IST)."""
    return datetime.now(IST)


def now_utc() -> datetime:
    """Returns current timezone-aware timestamp in UTC (standard for storage)."""
    return datetime.now(timezone.utc)


def to_ist(dt: Optional[datetime]) -> Optional[datetime]:
    """
    Safely converts any datetime (naive or timezone-aware) to Asia/Kolkata (IST).
    If naive, assumes it was stored as UTC instant (SQLite standard convention).
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST)


def format_ist(dt: Optional[datetime], fmt: str = "%d %b %Y, %I:%M %p IST") -> str:
    """
    Formats a datetime object as a human-readable Indian Standard Time string.
    Example output: '21 Nov 2026, 04:30 PM IST'
    """
    ist_dt = to_ist(dt)
    if ist_dt is None:
        return "N/A"
    return ist_dt.strftime(fmt)


def format_ist_iso(dt: Optional[datetime]) -> Optional[str]:
    """
    Returns an ISO 8601 string representation with the +05:30 IST offset.
    Example output: '2026-11-21T16:30:00+05:30'
    """
    ist_dt = to_ist(dt)
    if ist_dt is None:
        return None
    return ist_dt.isoformat()
