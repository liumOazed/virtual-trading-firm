"""
ARIA-Growth — Daily Snapshot Logger  (READ-ONLY)
=================================================
Appends one dated snapshot per day for pattern analysis. NEVER places or
modifies orders — pure read. Run once a day AFTER the US close (≈2am Bangladesh)
so prices are true closing prices and day-over-day comparisons are consistent.

Two append-only CSVs, every row stamped with its own `date` (so days group
cleanly for analysis; no blank separator rows — those break pandas):

  9_aria_growth/daily_positions.csv   — one row PER TICKER per day
     date, timestamp, ticker, sector, entry_price, current_price, qty,
     market_value, day_change_pct, total_pnl_usd, total_pnl_pct,
     days_held, room_to_stop_pct, regime

  9_aria_growth/daily_portfolio.csv   — one row per day (the summary)
     date, timestamp, regime, spy_trend_pct, spy_vol_pct, spy_drawdown_pct,
     equity, cash, total_pnl_pct, day_change_pct,
     spy_day_pct, spy_since_pct, qqq_day_pct, qqq_since_pct,
     n_positions, n_sectors

Benchmarks (SPY, QQQ) and the book are indexed to your FIRST log run
("since go-live") so "me vs QQQ" is apples-to-apples. Idempotent: re-running on
the same date overwrites that date's rows rather than duplicating them.

Usage:
  python aria_growth_daily_log.py          # log today's snapshot + print summary

Install: pip install requests pandas yfinance
"""

import sys
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Reuse validated, read-only helpers + config from the executor (no orders here)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aria_growth_executor import (
    _keys, get_account, get_positions, PAPER_BASE, EXPECTED_ACCOUNT,
    STOP_LOSS_PCT, OUT_DIR, TRADE_LOG,
)
from aria_growth_regime_allocator import detect_regime

POSITIONS_CSV = OUT_DIR / "daily_positions.csv"
PORTFOLIO_CSV = OUT_DIR / "daily_portfolio.csv"
LOG_STATE     = OUT_DIR / "daily_log_state.json"


# ── benchmark prices (SPY + QQQ): last close, prev close ──────────────────────
def fetch_benchmarks():
    import yfinance as yf
    data = yf.download(["SPY", "QQQ"], period="7d", auto_adjust=True, progress=False)
    close = data["Close"]
    out = {}
    for sym in ("SPY", "QQQ"):
        s = close[sym].dropna()
        out[sym] = {"last": float(s.iloc[-1]), "prev": float(s.iloc[-2])}
    return out


# ── entry dates from the trade log (for days_held) ────────────────────────────
def entry_dates():
    if not TRADE_LOG.exists():
        return {}
    df = pd.read_csv(TRADE_LOG)
    buys = df[df["side"] == "buy"]
    out = {}
    for sym, g in buys.groupby("symbol"):
        out[sym] = pd.to_datetime(g["timestamp"]).min()
    return out


def append_idempotent(path: Path, new_df: pd.DataFrame, date_str: str):
    """Overwrite any existing rows for date_str, then append the new ones."""
    if path.exists():
        old = pd.read_csv(path)
        if "date" in old.columns:
            old = old[old["date"].astype(str) != date_str]
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(path, index=False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    h = _keys()
    acct = get_account(h)
    acct_num = acct.get("account_number", "?")

    # Read-only, but we still refuse to log the WRONG account (data integrity)
    if EXPECTED_ACCOUNT and acct_num != EXPECTED_ACCOUNT:
        print(f"⛔ Connected to {acct_num}, expected {EXPECTED_ACCOUNT} (Zed2). "
              f"Refusing to log — fix keys so the growth log stays clean.")
        sys.exit(1)

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    ts = now.isoformat()

    equity = float(acct["equity"])
    cash = float(acct["cash"])
    last_equity = float(acct.get("last_equity", equity))    # prev day's close equity
    positions = get_positions(h)

    # Regime + SPY signals (read-only)
    info = detect_regime()
    regime = info["regime"]
    bm = fetch_benchmarks()
    spy_day = bm["SPY"]["last"] / bm["SPY"]["prev"] - 1
    qqq_day = bm["QQQ"]["last"] / bm["QQQ"]["prev"] - 1

    # Inception baseline (since go-live) — set on first run, reused after
    if LOG_STATE.exists():
        base = json.loads(LOG_STATE.read_text())
    else:
        base = {"inception_date": date_str, "equity_base": equity,
                "spy_base": bm["SPY"]["last"], "qqq_base": bm["QQQ"]["last"]}
        LOG_STATE.write_text(json.dumps(base, indent=2))

    book_since = equity / base["equity_base"] - 1
    spy_since  = bm["SPY"]["last"] / base["spy_base"] - 1
    qqq_since  = bm["QQQ"]["last"] / base["qqq_base"] - 1
    book_day   = equity / last_equity - 1 if last_equity else 0.0

    entries = entry_dates()

    # ── per-position rows ──
    rows = []
    for sym, p in sorted(positions.items()):
        entry = float(p.get("avg_entry_price", 0) or 0)
        curr = float(p.get("current_price", 0) or 0)
        total_plpc = float(p.get("unrealized_plpc", 0)) * 100
        day_pct = float(p.get("change_today", 0)) * 100          # today's price move
        if sym in entries and pd.notna(entries[sym]):
            ts_entry = entries[sym]
            ts_utc = ts_entry.tz_convert("UTC") if ts_entry.tzinfo is not None else ts_entry.tz_localize("UTC")
            held = (now - ts_utc).days
        else:
            held = None
        rows.append({
            "date": date_str, "timestamp": ts, "ticker": sym,
            "entry_price": round(entry, 2), "current_price": round(curr, 2),
            "qty": float(p.get("qty", 0)),
            "market_value": round(float(p.get("market_value", 0)), 2),
            "day_change_pct": round(day_pct, 2),
            "total_pnl_usd": round(float(p.get("unrealized_pl", 0)), 2),
            "total_pnl_pct": round(total_plpc, 2),
            "days_held": held,
            # how many pts of P&L until the -STOP_LOSS_PCT stop triggers
            "room_to_stop_pct": round(total_plpc + STOP_LOSS_PCT * 100, 2),
            "regime": regime,
        })
    pos_df = pd.DataFrame(rows)

    # ── portfolio summary row ──
    port_row = pd.DataFrame([{
        "date": date_str, "timestamp": ts, "regime": regime,
        "spy_trend_pct": info["trend_vs_200dma_pct"],
        "spy_vol_pct": info["realized_vol_pct"],
        "spy_drawdown_pct": info["drawdown_from_high_pct"],
        "equity": round(equity, 2), "cash": round(cash, 2),
        "total_pnl_pct": round(book_since * 100, 2),
        "day_change_pct": round(book_day * 100, 2),
        "spy_day_pct": round(spy_day * 100, 2),
        "spy_since_pct": round(spy_since * 100, 2),
        "qqq_day_pct": round(qqq_day * 100, 2),
        "qqq_since_pct": round(qqq_since * 100, 2),
        "n_positions": len(positions),
        "n_sectors": None,   # sector needs the screen; left blank unless wired in
    }])

    append_idempotent(POSITIONS_CSV, pos_df, date_str)
    append_idempotent(PORTFOLIO_CSV, port_row, date_str)

    # ── console summary ──
    print("=" * 70)
    print(f"  ARIA-Growth - Daily Log  {date_str}  (account {acct_num})")
    print("=" * 70)
    print(f"  Regime {regime} | SPY trend {info['trend_vs_200dma_pct']:+.1f}% "
          f"vol {info['realized_vol_pct']}% dd {info['drawdown_from_high_pct']:+.1f}%")
    print(f"  Book: ${equity:,.0f} | today {book_day*100:+.2f}% | since go-live {book_since*100:+.2f}%")
    print(f"  vs SPY: today {spy_day*100:+.2f}% / since {spy_since*100:+.2f}%   "
          f"|  vs QQQ: today {qqq_day*100:+.2f}% / since {qqq_since*100:+.2f}%")
    edge = (book_since - qqq_since) * 100
    print(f"  -> Book vs QQQ since go-live: {edge:+.2f} pts "
          f"({'ahead' if edge >= 0 else 'behind'})")
    if not pos_df.empty:
        print(f"\n  {'TICKER':<7}{'DAY%':>7}{'TOTAL%':>9}{'$P&L':>9}{'STOP ROOM':>11}")
        print("  " + "-" * 43)
        for _, r in pos_df.sort_values("total_pnl_pct").iterrows():
            print(f"  {r['ticker']:<7}{r['day_change_pct']:>6.1f}%{r['total_pnl_pct']:>8.1f}%"
                  f"{r['total_pnl_usd']:>9,.0f}{r['room_to_stop_pct']:>9.1f}pt")
    print(f"\n  [saved] {POSITIONS_CSV}  (+{len(pos_df)} rows)")
    print(f"  [saved] {PORTFOLIO_CSV}  (+1 row)")
    if date_str == base["inception_date"]:
        print(f"\n  [*] Baseline set today -- 'since go-live' starts from here.")


if __name__ == "__main__":
    main()