"""
ARIA-Growth — Paper-Trade Executor
===================================
Syncs your Alpaca PAPER account to the current regime's growth basket.
Detects the regime, builds the target portfolio (reusing the allocator), and
computes the buy/sell/hold plan to move your account toward it.

★ DRY-RUN BY DEFAULT. Nothing is sent until you pass --execute. Always read
  the printed plan first.
★ PAPER ACCOUNT ONLY. Uses the Alpaca paper endpoint.
★ Long-only growth book. Equal-weight across N names, small cash buffer.

Honest framing (unchanged): this is a forward, rules-based portfolio, NOT a
validated edge. Paper-trading it tells you whether the regime calls and
rotations behave sensibly in real time — it does not prove it beats SPY.

Reads Alpaca keys from env: ALPACA_API_KEY / ALPACA_SECRET_KEY (ARIA's .env).
Reuses regime logic from aria_growth_regime_allocator.py (same folder).

Usage:
  python aria_growth_executor.py --status              # show account + positions + regime
  python aria_growth_executor.py                       # DRY-RUN: print the sync plan
  python aria_growth_executor.py --execute             # actually place paper orders
  python aria_growth_executor.py --regime RISK_OFF     # force a regime (inspection)

Install: pip install requests pandas numpy yfinance openpyxl
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd

# Reuse the validated allocator logic (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from aria_growth_regime_allocator import (
        detect_regime, regime_fit, build_portfolio, REGIME_STYLE,
    )
except ImportError:
    raise SystemExit("Put aria_growth_regime_allocator.py in the same folder as this file.")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
PAPER_BASE   = "https://paper-api.alpaca.markets"
SCREEN_PATH  = "growth_screen_results.csv"   # your screener's native output
OUT_DIR      = Path("9_aria_growth")
STATE_FILE   = OUT_DIR / "executor_state.json"
TRADE_LOG    = OUT_DIR / "growth_trade_log.csv"

# ── HARD SAFETY GUARD ─────────────────────────────────────────────────────────
# The growth book MUST run on the Zed2 paper account, never ARIA's. The executor
# refuses to do anything if the connected account number isn't this. Set to ""
# to disable the guard (not recommended). Find yours via --status.
EXPECTED_ACCOUNT = "PA3NOJK22RWE"   # Zed2 (growth).  ARIA = PA304AGCZBF4

DEPLOY_FRACTION   = 0.95     # fraction of equity to deploy (5% cash buffer)
N_HOLDINGS        = 20
MAX_PER_SECTOR    = 4
REBALANCE_DRIFT   = 0.30     # only adjust an existing holding if it drifts >30%
                            # from target (avoids churn on small moves)

# ── STOP-LOSS ─────────────────────────────────────────────────────────────────
# Close a holding once it's down this much FROM ENTRY (not from a trailing peak —
# entry-based avoids churning winners that merely pull back from a high). The
# freed cash is redistributed EVENLY across survivors (basket shrinks 20→19, each
# survivor's equal-weight slice grows). Stopped names are blocklisted so they
# aren't instantly rebought; clear the blocklist after a monthly re-screen with
# --reset-stops (or it clears automatically on a regime flip).
#
# NOTE: 0.10 (10%) is TIGHT for high-beta growth (NVDA/SMCI/PLTR routinely dip
# 10%+ inside uptrends → frequent whipsaw). 0.15 is the more defensible floor for
# the RISK_ON basket. Change here:
STOP_LOSS_PCT     = 0.15


# ══════════════════════════════════════════════════════════════════════════════
# ALPACA REST
# ══════════════════════════════════════════════════════════════════════════════
def _keys():
    # Growth book uses its OWN account keys, isolated from ARIA. Resolution order:
    #   1. GROWTH_ALPACA_* env vars (explicit, never collide with ARIA)
    #   2. 9_aria_growth/.env located RELATIVE TO THIS SCRIPT (so the launch
    #      directory can't change which file we read — the bug that kept flipping
    #      us onto ARIA's account)
    # We deliberately do NOT fall back to a bare ALPACA_API_KEY shell var or
    # ARIA's 8_live_trading/.env, so the two books can never share an account.
    k = os.environ.get("GROWTH_ALPACA_API_KEY")
    s = os.environ.get("GROWTH_ALPACA_SECRET_KEY")
    if not k or not s:
        here = Path(__file__).resolve().parent          # the 9_aria_growth folder
        for p in [here / ".env", Path("9_aria_growth/.env"), Path(".env")]:
            if p.exists():
                kk = ss = None
                for line in p.read_text().splitlines():
                    if line.startswith("ALPACA_API_KEY="):
                        kk = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if line.startswith("ALPACA_SECRET_KEY="):
                        ss = line.split("=", 1)[1].strip().strip('"').strip("'")
                if kk and ss:
                    k, s = kk, ss
                    break
    if not k or not s:
        raise SystemExit("Missing ALPACA keys. Put the Zed2 key/secret in "
                         "9_aria_growth/.env or set GROWTH_ALPACA_API_KEY / "
                         "GROWTH_ALPACA_SECRET_KEY.")
    return {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}


def get_account(h):
    r = requests.get(f"{PAPER_BASE}/v2/account", headers=h, timeout=20)
    r.raise_for_status()
    return r.json()


def get_positions(h):
    r = requests.get(f"{PAPER_BASE}/v2/positions", headers=h, timeout=20)
    r.raise_for_status()
    return {p["symbol"]: p for p in r.json()}


def place_order(h, symbol, notional, side):
    body = {"symbol": symbol, "notional": round(notional, 2), "side": side,
            "type": "market", "time_in_force": "day"}   # fractional → DAY tif
    r = requests.post(f"{PAPER_BASE}/v2/orders", headers=h, json=body, timeout=20)
    r.raise_for_status()
    return r.json()


def close_position(h, symbol):
    r = requests.delete(f"{PAPER_BASE}/v2/positions/{symbol}", headers=h, timeout=20)
    r.raise_for_status()
    return r.json()


# ══════════════════════════════════════════════════════════════════════════════
# PLAN COMPUTATION  (pure function — unit-testable, no network)
# ══════════════════════════════════════════════════════════════════════════════
def compute_plan(target_dollars: dict, current_mv: dict, drift_thresh: float,
                 stopped: set = None, redistribute: bool = False,
                 close_reasons: dict = None):
    """
    target_dollars : {symbol: desired $ exposure}  (survivors only — stopped &
                     blocklisted names already excluded by the caller)
    current_mv     : {symbol: current market value $}
    stopped        : symbols to force-close + redistribute this cycle (stops/drops)
    close_reasons  : optional {symbol: reason} override for the close label
                     (e.g. manual drop vs stop-loss), for clean trade-log records
    redistribute   : if True (a removal fired), top survivors UP to target even on
                     small drift (this spreads freed cash evenly); winners ABOVE
                     target are still never trimmed below drift_thresh.
    Returns list of actions: (symbol, side, notional, reason).
    """
    stopped = stopped or set()
    close_reasons = close_reasons or {}
    actions = []
    # 1) Forced closes first (stops + manual drops)
    for sym in stopped:
        if sym in current_mv:
            reason = close_reasons.get(sym, f"STOP-LOSS (-{STOP_LOSS_PCT:.0%} from entry)")
            actions.append((sym, "close", current_mv[sym], reason))
    # 2) Exits: held, not in target, not already stopped
    for sym, mv in current_mv.items():
        if sym not in target_dollars and sym not in stopped:
            actions.append((sym, "close", mv, "not in regime basket"))
    # 3) Entries / adjustments for target names
    buy_thresh = 0.0 if redistribute else drift_thresh    # redistribute → fill to target
    for sym, tgt in target_dollars.items():
        if sym in stopped:
            continue
        cur = current_mv.get(sym, 0.0)
        if cur == 0.0:
            actions.append((sym, "buy", tgt, "new entry"))
        else:
            drift = (cur - tgt) / tgt if tgt else 0
            if drift < -buy_thresh and tgt - cur > 1:         # underweight → top up
                reason = ("redistribute (stop freed cash)" if redistribute
                          else f"top-up ({drift:+.0%})")
                actions.append((sym, "buy", tgt - cur, reason))
            elif drift > drift_thresh:                        # overweight → trim
                actions.append((sym, "sell", cur - tgt, f"trim ({drift:+.0%})"))
            # else: within tolerance → hold (no action)
    return actions


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def load_target(regime, screen_path, n, max_sec, deploy_dollars, exclude=None,
                backfill=False):
    """
    exclude  : tickers to drop (stopped/blocklisted).
    backfill : if False (default), the basket SHRINKS — excluded names are NOT
               replaced, so freed cash concentrates into the survivors (your
               'distribute among the remaining' intent). If True, pull the next
               best-ranked names to keep N holdings.
    """
    exclude = exclude or set()
    # Resolve the screen file from likely locations (run-dir, project root, script dir)
    p = Path(screen_path)
    if not p.exists():
        here = Path(__file__).resolve().parent
        for cand in [Path.cwd() / p.name, here / p.name, here.parent / p.name,
                     Path.cwd() / "growth_screen_results.csv",
                     Path.cwd() / "growth_screen_results.xlsx"]:
            if cand.exists():
                p = cand
                break
    if not p.exists():
        raise SystemExit(
            f"Screen file not found: {screen_path}\n"
            f"   Pass --screen with the full path to your growth_screen_results.csv/.xlsx")
    df = pd.read_excel(p) if p.suffix == ".xlsx" else pd.read_csv(p)
    ranked = regime_fit(df, regime)
    if backfill:
        ranked = ranked[~ranked["ticker"].isin(exclude)]
        port = build_portfolio(ranked, n, max_sec)
    else:
        # Build the full N-name basket FIRST, then drop excluded → shrinks to survivors.
        port = build_portfolio(ranked, n, max_sec)
        port = port[~port["ticker"].isin(exclude)]
    if port.empty:
        return {}, port
    per = deploy_dollars / len(port)
    target = {row["ticker"]: per for _, row in port.iterrows()}
    return target, port


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="actually place paper orders")
    ap.add_argument("--status", action="store_true", help="show account + positions + regime")
    ap.add_argument("--regime", choices=["RISK_ON", "RISK_OFF", "NEUTRAL"])
    ap.add_argument("--n", type=int, default=N_HOLDINGS)
    ap.add_argument("--max-per-sector", type=int, default=MAX_PER_SECTOR)
    ap.add_argument("--deploy", type=float, default=DEPLOY_FRACTION)
    ap.add_argument("--screen", default=SCREEN_PATH)
    ap.add_argument("--reset-stops", action="store_true",
                    help="clear the stop-loss blocklist (run after a monthly re-screen)")
    ap.add_argument("--stop-pct", type=float, default=STOP_LOSS_PCT,
                    help=f"stop-loss threshold from entry (default {STOP_LOSS_PCT})")
    ap.add_argument("--drop", nargs="+", metavar="TICKER", default=None,
                    help="manually close & blocklist one or more held names, "
                         "redistributing their cash evenly to survivors "
                         "(e.g. --drop LLY  or  --drop LLY NVDA). Stays out until "
                         "--reset-stops after a re-screen.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    h = _keys()
    acct = get_account(h)
    acct_num = acct.get("account_number", "?")

    # ── HARD GUARD: refuse to run on the wrong account (protects ARIA) ──
    if EXPECTED_ACCOUNT and acct_num != EXPECTED_ACCOUNT:
        print("=" * 70)
        print("  ⛔ ACCOUNT GUARD TRIPPED — refusing to run.")
        print(f"     Connected to account {acct_num}, but this book must run on")
        print(f"     {EXPECTED_ACCOUNT} (Zed2/growth). Your keys are pointing at the")
        print(f"     WRONG account — likely ARIA's. No orders were touched.")
        print(f"     Fix: ensure 9_aria_growth/.env has the Zed2 keys, and that no")
        print(f"     ALPACA_API_KEY is exported in this shell overriding it.")
        print("=" * 70)
        sys.exit(1)

    equity = float(acct["equity"])
    positions = get_positions(h)
    cur_mv = {s: float(p["market_value"]) for s, p in positions.items()}

    print("=" * 70)
    print("  ARIA-Growth — Paper Executor" + ("  [STATUS]" if args.status else
          "  [EXECUTE]" if args.execute else "  [DRY-RUN]"))
    print("=" * 70)
    print(f"  Account {acct_num} | equity ${equity:,.0f} | cash ${float(acct['cash']):,.0f} | "
          f"{len(positions)} positions")

    # ---- STATUS mode: show holdings + P&L, then exit ----
    if args.status:
        if positions:
            print(f"\n  {'SYMBOL':<8}{'MKT VAL':>12}{'UNREAL P&L':>14}{'P&L %':>9}")
            print("  " + "-" * 42)
            for s, p in sorted(positions.items()):
                pl = float(p["unrealized_pl"]); plpc = float(p["unrealized_plpc"]) * 100
                print(f"  {s:<8}{float(p['market_value']):>12,.0f}{pl:>14,.0f}{plpc:>8.1f}%")
        state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
        if state:
            print(f"\n  Last regime: {state.get('regime','?')} "
                  f"(set {state.get('updated','?')[:10]})")
        return

    # ---- Detect regime ----
    if args.regime:
        regime = args.regime
        print(f"\n  Regime: {regime} (forced)")
    else:
        info = detect_regime()
        regime = info["regime"]
        print(f"\n  📡 SPY {info['spy']} | trend {info['trend_vs_200dma_pct']:+.1f}% | "
              f"vol {info['realized_vol_pct']}% | dd {info['drawdown_from_high_pct']:+.1f}% "
              f"→ {regime}")
    print(f"  Style: {REGIME_STYLE[regime]}")

    # ---- Regime-flip detection ----
    prev = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    flipped = prev.get("regime") and prev["regime"] != regime
    if flipped:
        print(f"\n  🔄 REGIME FLIP: {prev['regime']} → {regime}  (full rotation)")

    # ---- Stop-loss detection + blocklist ----
    # A held name down ≥ stop_pct from entry (Alpaca's unrealized_plpc) stops out.
    # Blocklist persists so stopped names aren't rebought; cleared on a regime
    # flip or via --reset-stops (run that after a monthly re-screen).
    blocklist = set() if (flipped or args.reset_stops) else set(prev.get("blocklist", []))
    if args.reset_stops:
        print(f"\n  ♻ Stop-loss blocklist cleared (--reset-stops).")

    stopped = set()
    for s, p in positions.items():
        try:
            plpc = float(p["unrealized_plpc"])
        except (KeyError, ValueError):
            continue
        if plpc <= -abs(args.stop_pct):
            stopped.add(s)
    if stopped:
        print(f"\n  🛑 STOP-LOSS triggered ({args.stop_pct:.0%} from entry): "
              f"{', '.join(sorted(stopped))}")

    # Manual drops: treated like stop-outs (close + blocklist + even redistribute).
    # Only act on names actually held; warn on the rest.
    dropped = set()
    if args.drop:
        req = {t.strip().upper() for t in args.drop}
        dropped = {t for t in req if t in positions}
        not_held = req - dropped
        if dropped:
            print(f"\n  ✋ MANUAL DROP: {', '.join(sorted(dropped))} "
                  f"(closed, blocklisted, cash redistributed evenly)")
        if not_held:
            print(f"  ⚠ --drop ignored (not currently held): {', '.join(sorted(not_held))}")

    removed = stopped | dropped          # both close + redistribute the same way
    blocklist |= removed                 # and both join the blocklist

    # ---- Build target (survivors only: exclude blocklisted/stopped/dropped) ----
    deploy_dollars = args.deploy * equity
    target, port = load_target(regime, args.screen, args.n, args.max_per_sector,
                               deploy_dollars, exclude=blocklist)
    if not target:
        print("\n  ✗ No stocks passed the regime filter.")
        return
    per = deploy_dollars / len(port)
    extra = ""
    if blocklist:
        extra = f" | {len(blocklist)} blocklisted: {', '.join(sorted(blocklist))}"
    print(f"\n  Target: {len(port)} survivors, ${per:,.0f} each "
          f"(deploy {args.deploy:.0%} of equity){extra}")
    if removed:
        print(f"  → Freed cash from {len(removed)} removal(s) redistributed EVENLY "
              f"across {len(port)} survivors.")

    close_reasons = {t: "MANUAL DROP" for t in dropped}
    plan = compute_plan(target, cur_mv, REBALANCE_DRIFT,
                        stopped=removed, redistribute=bool(removed),
                        close_reasons=close_reasons)
    if not plan:
        print("\n  ✓ Account already matches target — no action needed.")
        _save_state(regime, target, blocklist)
        return

    print(f"\n  PLAN ({len(plan)} actions):")
    print(f"  {'ACTION':<7}{'SYMBOL':<8}{'NOTIONAL':>12}  REASON")
    print("  " + "-" * 50)
    for sym, side, notional, reason in sorted(plan, key=lambda x: (x[1] != "close", x[0])):
        print(f"  {side.upper():<7}{sym:<8}{notional:>12,.0f}  {reason}")

    if not args.execute:
        print(f"\n  [DRY-RUN] Nothing sent. Re-run with --execute to place these orders.")
        return

    # ---- Execute ----
    print(f"\n  Placing orders (paper)...")
    log_rows = []
    # Closes first (frees cash for redistribution top-ups), then buys/sells.
    ordered = sorted(plan, key=lambda x: 0 if x[1] == "close" else 1)
    for sym, side, notional, reason in ordered:
        try:
            if side == "close":
                close_position(h, sym)
                res = "closed"
            else:
                place_order(h, sym, notional, side)
                res = "ok"
            print(f"    ✓ {side.upper()} {sym} ${notional:,.0f}")
        except Exception as e:
            res = f"ERROR: {e}"
            print(f"    ✗ {side.upper()} {sym}: {e}")
        log_rows.append({"timestamp": datetime.now(timezone.utc).isoformat(),
                         "regime": regime, "symbol": sym, "side": side,
                         "notional": round(notional, 2), "reason": reason, "result": res})

    # Append trade log
    new = pd.DataFrame(log_rows)
    if TRADE_LOG.exists():
        new = pd.concat([pd.read_csv(TRADE_LOG), new], ignore_index=True)
    new.to_csv(TRADE_LOG, index=False)
    _save_state(regime, target, blocklist)
    print(f"\n  ✓ Logged {len(log_rows)} actions → {TRADE_LOG}")


def _save_state(regime, target, blocklist=None):
    STATE_FILE.write_text(json.dumps({
        "regime": regime, "updated": datetime.now(timezone.utc).isoformat(),
        "holdings": list(target.keys()),
        "blocklist": sorted(blocklist) if blocklist else [],
    }, indent=2))


if __name__ == "__main__":
    main()