"""
market_day.py
=============
Is the New York trading session that the scheduled run covers a real NYSE
trading day?

The workflow fires at 03:15 Asia/Dhaka (= 21:15 UTC the previous calendar day).
At that moment it is 16:15 (EST) or 17:15 (EDT) in New York, i.e. just after
the close of the SAME New York date. So "the session we are covering" is simply
today's New York date, which is also what run_live.py (ny_today) and the
ARIA-Growth logger (UTC date, identical to the NY date at 21:15 UTC) stamp
their rows with.

Uses the offline exchange_calendars XNYS calendar (weekends + NYSE holidays,
no API keys needed).

Usage:
  python scripts/market_day.py                # check today's NY date
  python scripts/market_day.py --date 2026-11-26

Prints a one-line verdict and, when running inside GitHub Actions, writes
`open=true|false` and `ny_date=YYYY-MM-DD` to $GITHUB_OUTPUT. Always exits 0
unless something is actually broken.
"""

import argparse
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")


def is_nyse_session(day: str) -> bool:
    import exchange_calendars as xcals
    import pandas as pd

    cal = xcals.get_calendar("XNYS")
    return bool(cal.is_session(pd.Timestamp(day)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="NY date to check (YYYY-MM-DD); default = today in New York")
    args = ap.parse_args()

    day = args.date or datetime.now(NY).date().isoformat()
    is_open = is_nyse_session(day)
    print(f"NY date {day}: {'NYSE trading day' if is_open else 'market CLOSED (weekend/holiday)'}")

    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"open={'true' if is_open else 'false'}\n")
            f.write(f"ny_date={day}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
