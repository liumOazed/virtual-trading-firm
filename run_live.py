"""
run_live.py
===========
Daily entry point for Virtual Trading Firm — Live Paper Trading.

Run this every day after US market close (2:30am Dhaka time / 4:30pm EST).

Usage:
  python run_live.py                 # full daily run
  python run_live.py --status        # account + positions only
  python run_live.py --dry-run       # signals only, no real orders
  python run_live.py --explain       # run Groq explainer after trading
  python run_live.py --refresh       # full 252-day price refresh (weekly)
  python run_live.py --reset         # reset Alpaca paper account to $100k

What it does each day:
  1. Check account state + open positions
  2. Top-up price data (last 5 days from Alpaca)
  3. Generate signals (HMM regime + RSI + momentum)
  4. Execute paper orders via Alpaca API
  5. Log trades to 8_live_trading/data/live_trade_log.csv
  6. Log equity to 8_live_trading/data/live_equity_curve.csv
  7. Print daily summary
  8. (optional) Run Groq daily briefing
"""

import os
import sys
import time
import argparse
from datetime import date, datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "8_live_trading"))
sys.path.insert(0, os.path.join(ROOT, "7_explainer"))

LIVE_DIR   = os.path.join(ROOT, "8_live_trading", "data")
TRADE_LOG  = os.path.join(LIVE_DIR, "live_trade_log.csv")
EQUITY_LOG = os.path.join(LIVE_DIR, "live_equity_curve.csv")


# ── helpers ───────────────────────────────────────────────────────────────

def header(title: str):
    w = 60
    print(f"\n{'═'*w}")
    print(f"  {title}")
    print(f"{'═'*w}")

def step(n: int, msg: str):
    print(f"\n  [{n}] {msg}")

def ok(msg: str):    print(f"  ✓  {msg}")
def warn(msg: str):  print(f"  ⚠  {msg}")
def info(msg: str):  print(f"  →  {msg}")

def elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s:.1f}s"


# ── status command ────────────────────────────────────────────────────────

def cmd_status():
    import sys, json
    sys.path.insert(0, os.path.join(ROOT, "8_live_trading"))
    from status_report import render_status_report
    from alpaca_client import AlpacaClient

    regime = None
    _rs = os.path.join(ROOT, "8_live_trading", "data", "regime_state.json")
    if os.path.exists(_rs):
        try:
            regime = json.load(open(_rs)).get("prev_regime")
        except Exception:
            pass

    client = AlpacaClient()
    render_status_report(client, regime)


# ── full refresh command ──────────────────────────────────────────────────

def cmd_refresh():
    from alpaca_client import AlpacaClient
    from live_data_feed import LiveDataFeed

    header("PRICE DATA — FULL REFRESH (252 days)")
    client = AlpacaClient()
    feed   = LiveDataFeed(client)
    t0     = time.time()
    feed.full_refresh(days=252)
    ok(f"Full refresh complete ({elapsed(t0)})")


# ── reset command ─────────────────────────────────────────────────────────

def cmd_reset():
    import requests
    from alpaca_client import AlpacaClient
    client = AlpacaClient()

    header("RESET PAPER ACCOUNT")
    warn("This closes all positions and resets balance to $100,000")
    confirm = input("  Type 'yes' to confirm: ").strip().lower()
    if confirm != "yes":
        print("  Cancelled.")
        return

    # Close all positions first
    print("  Closing all positions...")
    client.close_all_positions()
    time.sleep(2)

    # Reset via Alpaca dashboard URL (no direct API endpoint)
    warn("To reset balance: go to app.alpaca.markets → Paper Trading → Reset")
    warn("Or cancel all orders and wait for positions to close.")
    ok("Positions closed. Reset balance manually in the Alpaca dashboard.")


# ── main daily run ────────────────────────────────────────────────────────

def cmd_run(dry_run: bool = False, explain: bool = False):
    from live_engine import LiveEngine

    header(f"VIRTUAL TRADING FIRM — {'DRY RUN' if dry_run else 'LIVE'} | {date.today()}")
    t0 = time.time()

    # Guard: skip on weekends / market holidays
    from alpaca_client import AlpacaClient
    client = AlpacaClient()
    if not client.is_trading_day() and not dry_run:
        print(f"\n  Market closed today (weekend/holiday) — skipping.")
        print(f"  Use --dry-run to force signal generation.")
        return

    # Run live engine
    engine  = LiveEngine(dry_run=dry_run, verbose=True)
    results = engine.run()

    # Optional Groq explainer
    if explain and not dry_run:
        step(7, "Groq daily briefing")
        try:
            from groq_explainer import GroqExplainer, DataLoader
            loader    = DataLoader(
                trade_file  = TRADE_LOG,
                equity_file = EQUITY_LOG,
            )
            explainer = GroqExplainer()
            today_str = date.today().strftime("%Y-%m-%d")
            text = explainer.daily_briefing(today_str, loader, verbose=True)
            explainer.save_report(text, f"live_daily_{today_str}.txt")
            ok(f"Groq briefing complete | {explainer.usage['total_tokens_session']} tokens")
        except Exception as e:
            warn(f"Groq explainer failed: {e}")

    # Final summary
    print(f"\n{'═'*60}")
    print(f"  DAILY RUN COMPLETE  ({elapsed(t0)})")
    print(f"  Date:     {results['date']}")
    print(f"  Signals:  {len(results['signals'])}")
    print(f"  Orders:   {len(results['orders'])}")
    print(f"  Equity:   ${results['equity']:,.2f}")
    print(f"  Regime:   {results['regime']}")
    if results["errors"]:
        print(f"  Errors:   {len(results['errors'])}")
        for e in results["errors"]:
            warn(e)
    print(f"{'═'*60}")
    print(f"\n  Logs:")
    if os.path.exists(TRADE_LOG):
        import pandas as pd
        tl = pd.read_csv(TRADE_LOG)
        print(f"    Trade log:   {TRADE_LOG} ({len(tl)} rows)")
    if os.path.exists(EQUITY_LOG):
        import pandas as pd
        eq = pd.read_csv(EQUITY_LOG)
        print(f"    Equity log:  {EQUITY_LOG} ({len(eq)} rows)")
    print()


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Virtual Trading Firm — Live Paper Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_live.py                  daily run (paper orders)
  python run_live.py --status         account + positions
  python run_live.py --dry-run        signals only, no orders
  python run_live.py --explain        run + Groq briefing
  python run_live.py --refresh        full 252-day data refresh
  python run_live.py --reset          reset paper account

Schedule (Dhaka time):
  Daily at 7:00pm BST → uses previous day close prices
  OR 2:30am BST → uses today's close prices (more accurate)
        """
    )
    parser.add_argument("--status",  action="store_true", help="Show account + positions")
    parser.add_argument("--dry-run", action="store_true", help="Signals only, no orders")
    parser.add_argument("--explain", action="store_true", help="Run Groq briefing after trading")
    parser.add_argument("--refresh", action="store_true", help="Full 252-day price refresh")
    parser.add_argument("--reset",   action="store_true", help="Reset paper account")
    args = parser.parse_args()

    if args.status:
        cmd_status()
    elif args.refresh:
        cmd_refresh()
    elif args.reset:
        cmd_reset()
    else:
        cmd_run(dry_run=args.dry_run, explain=args.explain)


if __name__ == "__main__":
    main()