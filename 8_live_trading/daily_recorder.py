"""
daily_recorder.py
=================
Daily record-keeper for ARIA. Takes NO trading action.

1. Prints today's LIVE snapshot (equity, P&L split, alpha so far) — for
   information only; it is NOT written to disk, because a mid-session number
   is not a daily value.
2. Syncs the live data files from Alpaca (alpaca_ledger.sync): trade log from
   Alpaca fills, daily_history.csv / live_equity_curve.csv from Alpaca's
   official end-of-day equity for every CLOSED trading day, with a
   books-vs-Alpaca reconciliation check. Idempotent — run it any time.

The engine calls record_today() at the end of every live run, so this
normally needs no manual run. Dates are always the New York trading date
(market_clock), never the local date.

Run:        python daily_recorder.py
Sync only:  python daily_recorder.py --sync     (alias: --backfill)
"""

import os, sys, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpaca_ledger as ledger
from market_clock import trading_date

INCEPTION_EQUITY = 100_000.0


def record_today(regime_arg=None, account=None, positions=None):
    """Print today's live snapshot, then sync the ledger from Alpaca."""
    try:
        acct = account if account is not None else ledger._get("/v2/account")
        if positions is None:
            positions = [{"symbol": p["symbol"], "market_value": p["market_value"],
                          "unrealized_pl": p["unrealized_pl"],
                          "unrealized_plpc": p["unrealized_plpc"]}
                         for p in ledger._get("/v2/positions")]
        _print_snapshot(acct, positions, regime_arg)
    except Exception as e:
        print(f"  ! Live snapshot unavailable ({e}) - syncing ledger anyway")
    return ledger.sync()


def _print_snapshot(acct, positions, regime_arg):
    equity = float(acct["equity"]); cash = float(acct["cash"])
    unreal = sum(float(p["unrealized_pl"]) for p in positions)
    deployed = (equity - cash) / equity * 100 if equity else 0
    regime = regime_arg
    if not regime:
        try: regime = json.load(open(ledger.STATE)).get("prev_regime", "Unknown")
        except Exception: regime = "Unknown"
    pos_str = " ".join(f"{p['symbol']}:{float(p['unrealized_plpc'])*100:+.1f}%" for p in positions)
    print("=" * 58)
    print(f"  LIVE SNAPSHOT - {trading_date()} (New York)   [not saved]")
    print("=" * 58)
    print(f"  Equity:        ${equity:,.2f}  "
          f"({(equity / INCEPTION_EQUITY - 1) * 100:+.2f}% since inception)")
    print(f"  Regime:        {regime}")
    print(f"  Deployed:      {deployed:.1f}%   Cash: ${cash:,.2f}")
    print(f"  Unrealized:    ${unreal:+,.2f}   {pos_str}")
    print("  (daily_history.csv only stores official closes - see sync below)\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default=None)
    ap.add_argument("--sync", action="store_true", help="only sync the ledger from Alpaca")
    ap.add_argument("--backfill", action="store_true", help="alias of --sync")
    args = ap.parse_args()
    if args.sync or args.backfill:
        ledger.sync()
    else:
        record_today(args.regime)
