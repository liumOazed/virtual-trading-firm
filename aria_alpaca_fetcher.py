"""
ARIA-MR — Alpaca Hourly Bar Fetcher  (intraday data layer)
===========================================================
Pulls 1-hour bars from Alpaca's market-data API for the ARIA-MR universe,
2019 → present, into a local cache the rest of the pipeline reads.

Why hourly: daily large-cap pairs reversion is ~zero-EV after costs (six tests
proved it). Intraday MAY contain idiosyncratic dislocations not yet arbitraged
away. Hourly is the retail-survivable compromise (minute needs tick infra).

Honest caveats baked in:
  - Free tier = IEX feed only. Volume is understated vs consolidated tape and
    some mid-caps (PSX, EMR, BX) may have thin/gappy hours. This DOES NOT much
    affect us: our OU/tightness/confirmation engine uses log-price ratios, not
    volume. But the fetcher FLAGS thin bars so we see data quality honestly.
  - Set ALPACA_DATA_FEED=sip if you upgrade to the $49 pro tier (cleaner tape).
  - Cointegration MUST be re-tested at hourly resolution — daily cointegration
    does not imply hourly cointegration (different noise structure).

Reads keys from env: ALPACA_API_KEY / ALPACA_SECRET_KEY (same as ARIA .env).

Output: 9_aria_mr/data_hourly/<TICKER>.csv  (timestamp, o,h,l,c,v,n,vwap)
        9_aria_mr/data_hourly/_quality_report.csv

Install: pip install requests pandas
"""

import os
import time
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
START      = "2019-01-01T00:00:00Z"
END        = None                     # None = up to now
TIMEFRAME  = "1Hour"
FEED       = os.environ.get("ALPACA_DATA_FEED", "iex")   # 'iex' free, 'sip' pro
ADJUSTMENT = "all"                    # split + dividend adjusted
# Note: Alpaca returns regular-session hourly bars by default. Extended/overnight
# would require a separate request; we keep regular hours to start.

DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"
PAGE_LIMIT = 10_000                   # max bars per page

OUT_DIR = Path("9_aria_mr/data_hourly")

# Full Phase-0 candidate universe: 74 tickers across 41 sector-grouped pairs.
# We pull all of them and re-run cointegration at HOURLY resolution — hourly
# winners may differ from daily, so we don't pre-restrict to the daily-validated 10.
# ('K'/Kellogg is dead post-2023 split and will fail cleanly, like in Phase 0.)
TICKERS = sorted({
    "AAPL", "ABBV", "ADI", "AEP", "AMAT", "AMD", "AVGO", "BAC", "BLK", "BMY",
    "BX", "C", "CAT", "CHTR", "CL", "CMCSA", "COP", "COST", "CVS", "CVX",
    "D", "DE", "DIS", "DUK", "EMR", "EOG", "ETN", "EXC", "FDX", "GE",
    "GIS", "GOOGL", "GS", "HAL", "HD", "HON", "HSY", "JNJ", "JPM", "K",
    "KMB", "KMI", "KO", "LLY", "LOW", "LRCX", "MA", "MDLZ", "MPC", "MRK",
    "MS", "MSFT", "NFLX", "NVDA", "PEP", "PFE", "PG", "PSX", "QCOM", "ROST",
    "SLB", "SO", "T", "TGT", "TJX", "TXN", "UNH", "UPS", "V", "VLO",
    "VZ", "WMB", "WMT", "XOM",
})

# Expected hourly bars per regular trading day (09:30-16:00 → ~7 hourly buckets)
EXPECTED_BARS_PER_DAY = 7


# ══════════════════════════════════════════════════════════════════════════════
# FETCH
# ══════════════════════════════════════════════════════════════════════════════
def get_keys() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY")
    sec = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        # Fallback: try to read a sibling .env (same pattern as ARIA)
        for envpath in [Path(".env"), Path("../.env"), Path("8_live_trading/.env")]:
            if envpath.exists():
                for line in envpath.read_text().splitlines():
                    if line.startswith("ALPACA_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if line.startswith("ALPACA_SECRET_KEY="):
                        sec = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key or not sec:
        raise SystemExit(
            "Missing Alpaca keys. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
            "env vars (or put them in a .env file)."
        )
    return key, sec


def fetch_ticker(symbol: str, headers: dict) -> pd.DataFrame:
    """Fetch all hourly bars for one symbol, following pagination."""
    rows = []
    page_token = None
    while True:
        params = {
            "symbols": symbol,
            "timeframe": TIMEFRAME,
            "start": START,
            "adjustment": ADJUSTMENT,
            "feed": FEED,
            "limit": PAGE_LIMIT,
            "sort": "asc",
        }
        if END:
            params["end"] = END
        if page_token:
            params["page_token"] = page_token

        for attempt in range(5):
            resp = requests.get(DATA_URL, headers=headers, params=params, timeout=30)
            if resp.status_code == 429:           # rate limited → back off
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            break
        else:
            raise RuntimeError(f"{symbol}: repeated rate-limit/timeout")

        data = resp.json()
        bars = data.get("bars", {}).get(symbol, [])
        rows.extend(bars)
        page_token = data.get("next_page_token")
        if not page_token:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Alpaca fields: t,o,h,l,c,v,n,vw
    df = df.rename(columns={
        "t": "timestamp", "o": "open", "h": "high", "l": "low",
        "c": "close", "v": "volume", "n": "trade_count", "vw": "vwap",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY
# ══════════════════════════════════════════════════════════════════════════════
def quality_stats(symbol: str, df: pd.DataFrame) -> dict:
    """Summarise coverage + thin-bar flags so we see IEX data quality honestly."""
    if df.empty:
        return {"ticker": symbol, "bars": 0, "status": "NO DATA"}
    span_days = (df.index[-1] - df.index[0]).days
    trading_days = df.index.normalize().nunique()
    bars_per_day = len(df) / max(trading_days, 1)
    # thin bars: zero or near-zero volume / no trades
    thin = int(((df["volume"] <= 0) | (df.get("trade_count", 1) <= 0)).sum())
    return {
        "ticker": symbol,
        "bars": len(df),
        "first": df.index[0].strftime("%Y-%m-%d"),
        "last": df.index[-1].strftime("%Y-%m-%d"),
        "trading_days": trading_days,
        "bars_per_day": round(bars_per_day, 2),
        "thin_bars": thin,
        "thin_pct": round(100 * thin / len(df), 2),
        "status": "OK" if bars_per_day >= EXPECTED_BARS_PER_DAY * 0.6 else "SPARSE",
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 68)
    print("  ARIA-MR — Alpaca Hourly Fetcher")
    print(f"  {len(TICKERS)} tickers | {TIMEFRAME} | {START[:10]} → now | feed={FEED}")
    print("=" * 68)

    key, sec = get_keys()
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    quality = []
    for i, sym in enumerate(TICKERS, 1):
        cache = OUT_DIR / f"{sym}.csv"
        if cache.exists():
            df = pd.read_csv(cache, parse_dates=["timestamp"]).set_index("timestamp")
            print(f"  [{i:>2}/{len(TICKERS)}] {sym:<6} cached ({len(df)} bars)")
        else:
            try:
                df = fetch_ticker(sym, headers)
            except Exception as e:
                print(f"  [{i:>2}/{len(TICKERS)}] {sym:<6} ERROR: {e}")
                quality.append({"ticker": sym, "bars": 0, "status": f"ERROR: {e}"})
                continue
            if not df.empty:
                df.to_csv(cache)
            print(f"  [{i:>2}/{len(TICKERS)}] {sym:<6} fetched {len(df)} bars")

        quality.append(quality_stats(sym, df))

    qdf = pd.DataFrame(quality)
    qdf.to_csv(OUT_DIR / "_quality_report.csv", index=False)

    print("\n" + "=" * 68)
    print("  DATA QUALITY REPORT")
    print("=" * 68)
    if not qdf.empty and "bars_per_day" in qdf.columns:
        for _, r in qdf.iterrows():
            flag = "" if r.get("status") == "OK" else f"  ⚠ {r.get('status')}"
            print(f"  {r['ticker']:<6} {int(r['bars']):>6} bars | "
                  f"{r.get('bars_per_day','?')}/day | thin {r.get('thin_pct','?')}%"
                  f" | {r.get('first','?')}→{r.get('last','?')}{flag}")
        sparse = qdf[qdf["status"] != "OK"]["ticker"].tolist() if "status" in qdf else []
        if sparse:
            print(f"\n  ⚠ Sparse/low-quality tickers: {sparse}")
            print("    These may need the $49 pro (SIP) feed: set ALPACA_DATA_FEED=sip")
        else:
            print("\n  ✓ All tickers have adequate hourly coverage on this feed.")
    print(f"\n  💾 Cache: {OUT_DIR}/  (re-runs read from disk; delete a CSV to refetch)")
    print("  Next: re-run Phase 0 cointegration on these HOURLY bars.\n")


if __name__ == "__main__":
    main()