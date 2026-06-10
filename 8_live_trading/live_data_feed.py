"""
live_data_feed.py
=================
Layer 8 — Live price data feed for the Virtual Trading Firm.

Fetches real OHLCV data from Alpaca for all engine tickers and builds
the same price_data.pkl structure that backtest_engine_v2.py already reads.

The engine calls:
    with open("5_backtesting/results/price_data.pkl", "rb") as f:
        price_data = pickle.load(f)

This module produces EXACTLY that structure from live Alpaca data.
Zero engine changes needed — just swap the pkl file before each run.

Structure:
    price_data = {
        "NVDA": {
            "close":  pd.Series(index=DatetimeIndex),
            "open":   pd.Series(index=DatetimeIndex),
            "high":   pd.Series(index=DatetimeIndex),
            "low":    pd.Series(index=DatetimeIndex),
            "volume": pd.Series(index=DatetimeIndex),
        },
        ...
    }

Usage:
    python 8_live_trading/live_data_feed.py          # full refresh (252d)
    python 8_live_trading/live_data_feed.py --days 5 # top-up last 5 days
"""

import os
import sys
import pickle
import argparse
import time
from datetime import date, datetime
from typing import Dict, Optional

import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "8_live_trading"))

from alpaca_client import AlpacaClient, TICKERS

RESULTS_DIR  = os.path.join(ROOT, "5_backtesting", "results")
LIVE_DIR     = os.path.join(ROOT, "8_live_trading", "data")
os.makedirs(LIVE_DIR, exist_ok=True)

LIVE_PKL     = os.path.join(LIVE_DIR,    "live_price_data.pkl")
BACKTEST_PKL = os.path.join(RESULTS_DIR, "price_data.pkl")

# SPY and IEF needed for HMM regime detection
HMM_TICKERS  = ["SPY", "IEF"]
ALL_TICKERS  = list(dict.fromkeys(TICKERS + HMM_TICKERS))  # dedup, order preserved


# ══════════════════════════════════════════════════════════════════════════
# DATA FEED
# ══════════════════════════════════════════════════════════════════════════

class LiveDataFeed:
    """
    Fetches live OHLCV data from Alpaca and builds price_data structure.

    Two modes:
      full_refresh(days=252)  — fetches full history, overwrites pkl
      top_up(days=5)          — extends existing pkl with latest bars
    """

    def __init__(self, client: Optional[AlpacaClient] = None):
        self.client = client or AlpacaClient()

    def _bars_to_series(
        self,
        bars: list,
    ) -> Dict[str, pd.Series]:
        """
        Convert list of bar dicts to OHLCV pd.Series with DatetimeIndex.
        Exactly matches the structure in price_data.pkl from the backtest.
        """
        if not bars:
            return {}

        dates   = pd.to_datetime([b["date"] for b in bars])
        closes  = [b["close"]  for b in bars]
        opens   = [b["open"]   for b in bars]
        highs   = [b["high"]   for b in bars]
        lows    = [b["low"]    for b in bars]
        volumes = [b["volume"] for b in bars]

        return {
            "close":  pd.Series(closes,  index=dates, name="close"),
            "open":   pd.Series(opens,   index=dates, name="open"),
            "high":   pd.Series(highs,   index=dates, name="high"),
            "low":    pd.Series(lows,    index=dates, name="low"),
            "volume": pd.Series(volumes, index=dates, name="volume"),
        }

    def full_refresh(self, days: int = 252) -> Dict:
        """
        Fetch full price history for all tickers.
        Builds complete price_data dict and saves to pkl.
        Takes ~30-60 seconds for 18 tickers.
        """
        print(f"\n  Full refresh — {len(ALL_TICKERS)} tickers × {days} days")
        print(f"  Tickers: {', '.join(ALL_TICKERS)}")
        price_data = {}
        failed     = []

        # Batch fetch (more efficient than one-by-one)
        print(f"\n  Fetching bars in batches...")
        # Alpaca multi-bar endpoint handles up to 100 symbols
        batch_result = self.client.get_bars_multi(ALL_TICKERS, days=days)

        for tk in ALL_TICKERS:
            bars = batch_result.get(tk, [])
            if not bars:
                print(f"  ⚠  {tk}: no bars returned — trying single fetch")
                bars = self.client.get_bars(tk, days=days)
                time.sleep(0.3)

            if bars:
                price_data[tk] = self._bars_to_series(bars)
                last_date = bars[-1]["date"]
                last_close = bars[-1]["close"]
                print(f"  ✓  {tk:5}: {len(bars):>3} bars | "
                      f"last {last_date} close=${last_close:.2f}")
            else:
                failed.append(tk)
                print(f"  ✗  {tk}: failed to fetch")

        if failed:
            print(f"\n  ⚠  Failed tickers: {failed}")
            # Try to fill from existing backtest pkl
            price_data = self._fill_from_backtest(price_data, failed)

        self._save(price_data)
        print(f"\n  ✓  price_data built: {len(price_data)} tickers")
        print(f"  ✓  Saved → {LIVE_PKL}")
        return price_data

    def top_up(self, days: int = 5) -> Dict:
        """
        Extend existing pkl with the latest N days of bars.
        Much faster than full_refresh — use for daily updates.
        """
        print(f"\n  Top-up — last {days} days for {len(ALL_TICKERS)} tickers")

        # Load existing
        price_data = self._load_existing()
        if not price_data:
            print("  No existing data found — running full refresh")
            return self.full_refresh()

        # Fetch recent bars
        batch_result = self.client.get_bars_multi(ALL_TICKERS, days=days + 3)

        updated = 0
        for tk in ALL_TICKERS:
            new_bars = batch_result.get(tk, [])
            if not new_bars:
                continue

            new_series = self._bars_to_series(new_bars)
            if not new_series:
                continue

            if tk in price_data:
                # Merge: concat existing + new, drop duplicates
                for field in ["close", "open", "high", "low", "volume"]:
                    if field in price_data[tk] and field in new_series:
                        merged = pd.concat([
                            price_data[tk][field],
                            new_series[field],
                        ]).sort_index()
                        merged = merged[~merged.index.duplicated(keep="last")]
                        price_data[tk][field] = merged
            else:
                price_data[tk] = new_series

            last_date  = price_data[tk]["close"].index[-1].strftime("%Y-%m-%d")
            last_close = price_data[tk]["close"].iloc[-1]
            print(f"  ↑  {tk:5}: updated → last {last_date} close=${last_close:.2f}")
            updated += 1

        self._save(price_data)
        print(f"\n  ✓  Top-up complete: {updated} tickers updated")
        print(f"  ✓  Saved → {LIVE_PKL}")
        return price_data

    def _fill_from_backtest(
        self,
        price_data: Dict,
        failed: list,
    ) -> Dict:
        """Fill missing tickers from backtest pkl if available."""
        if not os.path.exists(BACKTEST_PKL):
            return price_data
        try:
            with open(BACKTEST_PKL, "rb") as f:
                bt = pickle.load(f)
            for tk in failed:
                if tk in bt:
                    price_data[tk] = bt[tk]
                    print(f"  ↩  {tk}: filled from backtest pkl")
        except Exception as e:
            print(f"  ⚠  Could not read backtest pkl: {e}")
        return price_data

    def _load_existing(self) -> Dict:
        """Load existing live pkl if it exists."""
        if os.path.exists(LIVE_PKL):
            try:
                with open(LIVE_PKL, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"  ⚠  Could not load existing pkl: {e}")
        return {}

    def _save(self, price_data: Dict):
        """Save price_data to live pkl."""
        with open(LIVE_PKL, "wb") as f:
            pickle.dump(price_data, f)

    def copy_to_engine(self):
        """
        Copy live pkl to where the engine expects it.
        CAUTION: overwrites 5_backtesting/results/price_data.pkl
        Only call this immediately before running the live engine.
        """
        import shutil
        if not os.path.exists(LIVE_PKL):
            raise FileNotFoundError(f"No live pkl found at {LIVE_PKL}")

        # Backup existing backtest pkl first
        backup = BACKTEST_PKL + ".backtest_backup"
        if os.path.exists(BACKTEST_PKL) and not os.path.exists(backup):
            shutil.copy2(BACKTEST_PKL, backup)
            print(f"  ✓  Backtest pkl backed up → {backup}")

        shutil.copy2(LIVE_PKL, BACKTEST_PKL)
        print(f"  ✓  Live pkl → {BACKTEST_PKL}")

    def restore_backtest_pkl(self):
        """Restore original backtest pkl after live run."""
        import shutil
        backup = BACKTEST_PKL + ".backtest_backup"
        if os.path.exists(backup):
            shutil.copy2(backup, BACKTEST_PKL)
            print(f"  ✓  Backtest pkl restored")

    def get_data_summary(self) -> Dict:
        """
        Returns summary of what's in the live pkl.
        """
        price_data = self._load_existing()
        if not price_data:
            return {"status": "empty", "tickers": []}

        summary = {"status": "ok", "tickers": {}}
        for tk, data in price_data.items():
            if "close" in data and len(data["close"]) > 0:
                close = data["close"]
                summary["tickers"][tk] = {
                    "bars":       len(close),
                    "first_date": close.index[0].strftime("%Y-%m-%d"),
                    "last_date":  close.index[-1].strftime("%Y-%m-%d"),
                    "last_close": round(float(close.iloc[-1]), 2),
                }
        return summary


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Data Feed")
    parser.add_argument("--days",    type=int, default=252,
                        help="Days of history to fetch (default 252)")
    parser.add_argument("--top-up",  action="store_true",
                        help="Top-up existing pkl with latest bars only")
    parser.add_argument("--summary", action="store_true",
                        help="Show summary of existing live pkl")
    args = parser.parse_args()

    print("\n" + "═"*55)
    print("  LIVE DATA FEED — Virtual Trading Firm")
    print("═"*55)

    client = AlpacaClient()
    feed   = LiveDataFeed(client)

    if args.summary:
        summary = feed.get_data_summary()
        if summary["status"] == "empty":
            print("\n  No live pkl found. Run without --summary to fetch.")
        else:
            print(f"\n  Live pkl: {len(summary['tickers'])} tickers\n")
            print(f"  {'Ticker':6} {'Bars':>5} {'First':>12} {'Last':>12} {'Close':>10}")
            print(f"  {'-'*50}")
            for tk, info in summary["tickers"].items():
                print(f"  {tk:6} {info['bars']:>5} "
                      f"{info['first_date']:>12} "
                      f"{info['last_date']:>12} "
                      f"${info['last_close']:>9.2f}")

    elif args.top_up:
        feed.top_up(days=5)

    else:
        feed.full_refresh(days=args.days)

    print(f"\n  Done. Next steps:")
    print(f"    python 8_live_trading/live_data_feed.py --summary")
    print(f"    python 8_live_trading/live_data_feed.py --top-up  (daily)")