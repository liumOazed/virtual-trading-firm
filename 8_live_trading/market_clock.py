"""
market_clock.py
===============
One source for "what trading day is it?" across the live stack.

The machine runs at UTC+6 while the market runs on America/New_York, so from
local midnight until ~10:00 the local date is already TOMORROW in New York
(e.g. 02:00 local on Oct 1 = 16:00 ET on Sep 30). Every date written to the
live logs must be the New York date, never date.today().
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")


def ny_now() -> datetime:
    return datetime.now(NY)


def ny_today() -> date:
    return ny_now().date()


def trading_date() -> str:
    """Current New York date as YYYY-MM-DD."""
    return ny_today().isoformat()
