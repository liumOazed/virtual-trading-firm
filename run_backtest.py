"""
run_backtest.py
===============
Main entry point for the Virtual Trading Firm pipeline.

Wires all layers together in one command:
  Layer 3: HMM regime detection     (inside backtest engine)
  Layer 4: XGBoost signal engine     (inside backtest engine)
  Layer 5: RL sizing rule            (inside backtest engine)
  Layer 6: Backtest execution        (backtest_engine_v2.py)
  Layer 7: Groq explainer            (groq_explainer.py)

Usage
-----
  python run_backtest.py                  # full run: backtest + metrics + groq
  python run_backtest.py --quick          # backtest + metrics only (no groq)
  python run_backtest.py --explain-only   # groq on existing results (skip backtest)
  python run_backtest.py -d 2024-10-04   # daily briefing for specific date
  python run_backtest.py -w 2024-10-07   # weekly summary for specific week
  python run_backtest.py -a              # audit all trades
  python run_backtest.py -a --limit 10  # audit first 10 trades
  python run_backtest.py --ticker NVDA TSLA  # filter explainer to tickers

Design
------
The backtest engine runs as a subprocess — this isolates its global
state, random seeds, and pkl caches from this process. All other
layers run in-process for clean error handling.
"""

import os
import sys
import time
import subprocess
import argparse
from datetime import date, timedelta
from typing import Optional, List

# ── project root ──────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "5_backtesting"))
sys.path.insert(0, os.path.join(ROOT, "7_explainer"))

RESULTS_DIR  = os.path.join(ROOT, "5_backtesting", "results")
EXPLAINER_DIR= os.path.join(ROOT, "7_explainer")
ENGINE_PATH  = os.path.join(ROOT, "5_backtesting", "backtest_engine_v2.py")
METRICS_PATH = os.path.join(ROOT, "5_backtesting", "metrics.py")
GROQ_PATH    = os.path.join(ROOT, "7_explainer",   "groq_explainer.py")


# ══════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════

def header(title: str):
    w = 60
    print(f"\n{'═'*w}")
    print(f"  {title}")
    print(f"{'═'*w}")

def step(msg: str):
    print(f"\n  ▶  {msg}")

def ok(msg: str):
    print(f"  ✓  {msg}")

def warn(msg: str):
    print(f"  ⚠  {msg}")

def elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s/60:.1f}min" if s >= 60 else f"{s:.1f}s"


# ══════════════════════════════════════════════════════════════════════════
# STEP 1: BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════

def run_engine() -> dict:
    """
    Runs backtest_engine_v2.py as a subprocess.
    Parses the final metrics from stdout.
    Returns dict with return_pct, sharpe, max_dd, trades.
    """
    step("Running backtest engine...")
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, ENGINE_PATH],
        cwd     = ROOT,
        capture_output = False,   # let stdout stream to terminal
        text    = True,
    )

    if result.returncode != 0:
        warn(f"Engine exited with code {result.returncode}")
        return {}

    ok(f"Engine complete ({elapsed(t0)})")

    # Parse metrics from the saved diagnostic CSV
    metrics = {}
    diag_path = os.path.join(RESULTS_DIR, "backtest_diagnostic.csv")
    if os.path.exists(diag_path):
        try:
            import pandas as pd
            diag = pd.read_csv(diag_path).iloc[-1]
            metrics = {
                "return_pct":  round(float(diag.get("total_return",    0)), 2),
                "sharpe":      round(float(diag.get("sharpe_ratio",     0)), 3),
                "max_dd":      round(float(diag.get("max_drawdown",     0)), 2),
                "trades":      int(diag.get("num_trades",   0)),
                "final_equity":round(float(diag.get("final_equity",     0)), 2),
            }
        except Exception as e:
            warn(f"Could not parse diagnostic CSV: {e}")

        # Read final equity from equity_curve.csv (more reliable than diagnostic)
        eq_path = os.path.join(RESULTS_DIR, "equity_curve.csv")
        if os.path.exists(eq_path):
            import pandas as pd
            eq = pd.read_csv(eq_path)
            metrics["final_equity"] = round(float(eq["equity"].iloc[-1]), 2)

    return metrics


# ══════════════════════════════════════════════════════════════════════════
# STEP 2: METRICS
# ══════════════════════════════════════════════════════════════════════════

def run_metrics() -> dict:
    """
    Runs metrics.py to generate the full tearsheet.
    Returns the metrics dict.
    """
    step("Running metrics...")
    t0 = time.time()

    if not os.path.exists(METRICS_PATH):
        warn("metrics.py not found — skipping")
        return {}

    result = subprocess.run(
        [sys.executable, METRICS_PATH],
        cwd            = ROOT,
        capture_output = False,
        text           = True,
    )

    if result.returncode != 0:
        warn(f"Metrics exited with code {result.returncode}")
        return {}

    ok(f"Metrics complete ({elapsed(t0)})")
    return {}


# ══════════════════════════════════════════════════════════════════════════
# STEP 3: GROQ EXPLAINER
# ══════════════════════════════════════════════════════════════════════════

def run_groq_daily(date_str: str, tickers: Optional[List[str]] = None):
    """Run daily briefing for a specific date."""
    step(f"Groq daily briefing — {date_str}")
    t0 = time.time()
    try:
        from groq_explainer import GroqExplainer, DataLoader
        loader    = DataLoader(
            trade_file  = os.path.join(RESULTS_DIR, "trade_log.csv"),
            equity_file = os.path.join(RESULTS_DIR, "equity_curve.csv"),
            events_file = os.path.join(ROOT, "6_rl_agent", "results",
                                       "signal_events_enriched.csv"),
        )
        explainer = GroqExplainer()
        text = explainer.daily_briefing(
            date_str, loader,
            verbose       = True,
            tickers_filter = tickers,
        )
        explainer.save_report(text, f"daily_{date_str}.txt")
        ok(f"Daily briefing complete ({elapsed(t0)}) | "
           f"{explainer.usage['total_tokens_session']} tokens")
        return explainer.usage
    except Exception as e:
        warn(f"Groq daily failed: {e}")
        return {}


def run_groq_weekly(week_start: str, tickers: Optional[List[str]] = None):
    """Run weekly summary for a specific week."""
    step(f"Groq weekly summary — {week_start}")
    t0 = time.time()
    try:
        from groq_explainer import GroqExplainer, DataLoader
        loader    = DataLoader(
            trade_file  = os.path.join(RESULTS_DIR, "trade_log.csv"),
            equity_file = os.path.join(RESULTS_DIR, "equity_curve.csv"),
            events_file = os.path.join(ROOT, "6_rl_agent", "results",
                                       "signal_events_enriched.csv"),
        )
        explainer = GroqExplainer()
        text = explainer.weekly_summary(week_start, loader, verbose=True)
        explainer.save_report(text, f"weekly_{week_start}.txt")
        ok(f"Weekly summary complete ({elapsed(t0)}) | "
           f"{explainer.usage['total_tokens_session']} tokens")
        return explainer.usage
    except Exception as e:
        warn(f"Groq weekly failed: {e}")
        return {}


def run_groq_audit(
    limit:   Optional[int]       = None,
    tickers: Optional[List[str]] = None,
):
    """Run trade audit batch."""
    step(f"Groq trade audit — "
         f"{'all trades' if not limit else f'first {limit} trades'}"
         f"{f' | tickers: {tickers}' if tickers else ''}")
    t0 = time.time()
    try:
        from groq_explainer import GroqExplainer, DataLoader
        loader    = DataLoader(
            trade_file  = os.path.join(RESULTS_DIR, "trade_log.csv"),
            equity_file = os.path.join(RESULTS_DIR, "equity_curve.csv"),
            events_file = os.path.join(ROOT, "6_rl_agent", "results",
                                       "signal_events_enriched.csv"),
        )
        trips = loader.trips
        if tickers:
            trips = [t for t in trips if t["ticker"] in tickers]
        if limit:
            trips = trips[:limit]

        explainer = GroqExplainer()
        # Temporarily replace loader.trips for the batch call
        original_trips = loader._trips
        loader._trips  = trips
        explainer.audit_batch(loader, max_trades=limit, verbose=True)
        loader._trips  = original_trips

        ok(f"Audit complete ({elapsed(t0)}) | "
           f"{explainer.usage['total_tokens_session']} tokens")
        return explainer.usage
    except Exception as e:
        warn(f"Groq audit failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def run_groq_auto(tickers: Optional[List[str]] = None):
    """
    Auto mode: runs daily briefing for last trading day
    and weekly summary for last completed week.
    Called by --full mode after backtest.
    """
    import pandas as pd
    eq_path = os.path.join(RESULTS_DIR, "equity_curve.csv")
    if not os.path.exists(eq_path):
        warn("equity_curve.csv not found — skipping Groq")
        return

    eq = pd.read_csv(eq_path, parse_dates=["date"]).sort_values("date")
    last_date = eq["date"].iloc[-1].strftime("%Y-%m-%d")

    # Last Monday for weekly
    last_dt   = eq["date"].iloc[-1]
    last_mon  = (last_dt - timedelta(days=last_dt.weekday())).strftime("%Y-%m-%d")

    usage_d = run_groq_daily(last_date, tickers)
    usage_w = run_groq_weekly(last_mon, tickers)

    total = (usage_d.get("total_tokens_session", 0) +
             usage_w.get("total_tokens_session", 0))
    ok(f"Groq auto complete — {total} tokens total")


# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════

def print_summary(metrics: dict, t_total: float):
    w = 60
    print(f"\n{'═'*w}")
    print(f"  PIPELINE COMPLETE  ({elapsed(t_total)})")
    print(f"{'═'*w}")

    if metrics:
        rows = [
            ("Total Return",  f"{metrics.get('return_pct', 0):+.2f}%"),
            ("Sharpe Ratio",  f"{metrics.get('sharpe', 0):.3f}"),
            ("Max Drawdown",  f"{metrics.get('max_dd', 0):.2f}%"),
            ("Total Trades",  f"{metrics.get('trades', 0)}"),
            ("Final Equity",  f"${metrics.get('final_equity', 0):,.0f}"),
        ]
        for label, val in rows:
            print(f"  {label:<20} {val:>12}")

    print(f"\n  Outputs:")
    outputs = [
        os.path.join(RESULTS_DIR, "equity_curve.csv"),
        os.path.join(RESULTS_DIR, "trade_log.csv"),
        os.path.join(RESULTS_DIR, "backtest_diagnostic.csv"),
        os.path.join(EXPLAINER_DIR, "reports"),
        os.path.join(EXPLAINER_DIR, "cache"),
    ]
    for path in outputs:
        if os.path.exists(path):
            print(f"    ✓ {os.path.relpath(path, ROOT)}")

    print(f"{'═'*w}\n")


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Virtual Trading Firm — Pipeline Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_backtest.py                    full run (backtest + metrics + groq auto)
  python run_backtest.py --quick            backtest + metrics only
  python run_backtest.py --explain-only     groq on existing results
  python run_backtest.py -d 2024-10-04     daily briefing
  python run_backtest.py -w 2024-10-07     weekly summary
  python run_backtest.py -a                audit all trades
  python run_backtest.py -a --limit 10     audit first 10 trades
  python run_backtest.py --ticker NVDA TSLA  filter to specific tickers
        """
    )

    # Run modes
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick",        action="store_true",
                      help="Backtest + metrics only (no Groq)")
    mode.add_argument("--explain-only", action="store_true",
                      help="Groq on existing results (skip backtest)")
    mode.add_argument("-d", "--daily",  metavar="DATE",
                      help="Daily briefing for DATE (YYYY-MM-DD)")
    mode.add_argument("-w", "--weekly", metavar="WEEK",
                      help="Weekly summary for week starting WEEK (YYYY-MM-DD)")
    mode.add_argument("-a", "--audit",  action="store_true",
                      help="Audit trades (use --limit and --ticker to filter)")

    # Shared options
    parser.add_argument("--limit",  type=int,   default=None,
                        help="Max trades for audit mode")
    parser.add_argument("--ticker", nargs="+",  default=None,
                        help="Filter to specific ticker(s)")
    parser.add_argument("--no-groq", action="store_true",
                        help="Skip all Groq calls")

    args   = parser.parse_args()
    t_total= time.time()
    metrics= {}

    header("VIRTUAL TRADING FIRM — PIPELINE")
    print(f"  Date: {date.today()}")
    print(f"  Root: {ROOT}")

    # ── Explainer-only modes (no backtest) ────────────────────────────────
    if args.daily:
        run_groq_daily(args.daily, args.ticker)
        print_summary({}, t_total)
        return

    if args.weekly:
        run_groq_weekly(args.weekly, args.ticker)
        print_summary({}, t_total)
        return

    if args.audit:
        run_groq_audit(args.limit, args.ticker)
        print_summary({}, t_total)
        return

    if args.explain_only:
        run_groq_auto(args.ticker)
        print_summary({}, t_total)
        return

    # ── Full pipeline ─────────────────────────────────────────────────────
    # Step 1: Backtest engine
    metrics = run_engine()

    # Step 2: Metrics tearsheet
    if os.path.exists(METRICS_PATH):
        run_metrics()
    else:
        warn("metrics.py not found — skipping tearsheet")

    # Step 3: Groq (skipped with --quick or --no-groq)
    if not args.quick and not args.no_groq:
        run_groq_auto(args.ticker)
    elif args.quick:
        ok("--quick flag set — skipping Groq explainer")

    print_summary(metrics, t_total)


if __name__ == "__main__":
    main()