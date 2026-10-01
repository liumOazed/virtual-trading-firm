"""
ARIA-Momentum — Month-End Quant Review  (READ-ONLY)
====================================================
Institutional-style periodic review of the live ARIA momentum book.
Reflects everything from inception (2026-06-15) through the end of the
selected month (default: the latest month with a closed trading day).

PRIMARY SOURCE   8_live_trading/data/daily_history.csv
  One row per CLOSED trading day, rebuilt from Alpaca's own records by
  alpaca_ledger.py (engine run / daily_recorder.py): equity = Alpaca official
  end-of-day equity, cash/realized/fees from Alpaca activities, SPY/QQQ at
  official closes, since-inception returns from the inception-day open.
SECONDARY        8_live_trading/data/live_trade_log.csv   (Alpaca fills)
CROSS-CHECK      8_live_trading/data/live_equity_curve.csv

INFLATION        official CPI-U (BLS, SA) from FRED CPIAUCSL — the ONE network
                 call; cached to data/cpi_cpiaucsl.csv and used offline.

No Alpaca calls, no yfinance — everything else is computed from the synced
CSVs (run daily_recorder.py --sync first if the engine hasn't run since the
month closed). READ-ONLY: never places orders, never modifies trading data.

Signature sections (what makes this ARIA-momentum's report):
  1. REGIME ATTRIBUTION  — performance grouped by HMM regime
  2. BACKTEST PARITY     — live-so-far vs the locked 95.55% backtest's
                           expectation for the SAME regimes

Output:  8_live_trading/month_end/<YYYY-MM>/
  report_<YYYY-MM>.md            written analysis
  equity_vs_benchmarks.png       indexed equity vs SPY/QQQ
  drawdown.png                   underwater plot
  regime_attribution.png         perf + exposure by regime   (signature)
  backtest_parity.png            live vs backtest expectation (signature)
  returns_and_capture.png        daily return dist + up/down capture
  positions_pnl.png              open positions + closed round trips
  deployment_pnl_split.png       deployed % + realized/unrealized P&L
  month_by_month.png             per-calendar-month return: ARIA vs SPY/QQQ
  inflation_real_value.png       nominal vs CPI-adjusted real value (base 2026-06-16)
  inflation_real_value.csv       per-day series behind that chart

Usage:
  python aria_momentum_month_end.py                  # latest month
  python aria_momentum_month_end.py --month 2026-09  # a specific month

Honest note baked into the report: a few weeks of live data verifies the
MACHINERY and observes behavior. It cannot prove or disprove the edge that
was validated over 5.5 backtest years. All risk metrics on this sample are
indicative, not conclusive — especially while the book has traded in only
one regime.
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from market_clock import ny_today   # New York date, never local
DATA_DIR   = _HERE / "data" if (_HERE / "data").exists() else _HERE
HIST_FILE  = DATA_DIR / "daily_history.csv"
TRADE_FILE = DATA_DIR / "live_trade_log.csv"
EQ_FILE    = DATA_DIR / "live_equity_curve.csv"

# ── inflation (official BLS CPI-U, seasonally adjusted, via FRED) ─────────────
CPI_SERIES     = "CPIAUCSL"
CPI_URL        = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={CPI_SERIES}&cosd=2025-01-01"
CPI_CACHE      = DATA_DIR / "cpi_cpiaucsl.csv"
INFLATION_BASE = pd.Timestamp("2026-06-16")      # go-live; nominal = real = CPI = 100

# ── style ─────────────────────────────────────────────────────────────────────
GREEN, RED, BLUE, AMBER = "#1D9E75", "#E24B4A", "#378ADD", "#EF9F27"
NAVY, GREY, MGREY       = "#1A3A5C", "#888780", "#D1D5DB"
PURPLE                  = "#8E6BB8"
REGIME_COLORS = {"Bull-Trending": GREEN, "Bull-Stable": BLUE,
                 "Bear-Stable": AMBER, "Bear-Stress": RED, "Unknown": MGREY}
TRADING_DAYS = 252

# ── locked backtest reference (the 95.55% engine) ─────────────────────────────
# Source: locked tearsheet + per-regime breakdown at engine lock (2026-06).
# If you re-lock the engine, update these from the new metrics.json/tearsheet.
BACKTEST = {
    "label":      "Locked backtest (95.55%)",
    "total_ret":  95.55, "ann_ret": 12.81, "ann_vol": 8.61,
    "sharpe":     1.442, "sortino": 1.320, "calmar": 1.849,
    "max_dd":     -6.82, "win_rate": 66.57, "avg_hold_days": 67.3,
    "regimes": {   # ann_ret %, sharpe, max_dd %  (from locked tearsheet)
        "Bull-Trending": {"ann_ret": 89.96, "sharpe": 3.82, "max_dd": -4.92},
        "Bull-Stable":   {"ann_ret": 26.18, "sharpe": 1.33, "max_dd": -6.72},
        "Bear-Stable":   {"ann_ret": 29.60, "sharpe": 1.79, "max_dd": -2.97},
        "Bear-Stress":   {"ann_ret": -3.61, "sharpe": -0.35, "max_dd": -1.95},
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
def load():
    hist = pd.read_csv(HIST_FILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    trades = pd.read_csv(TRADE_FILE, parse_dates=["date"]).sort_values("date", kind="stable").reset_index(drop=True)
    eq = None
    if EQ_FILE.exists():
        eq = pd.read_csv(EQ_FILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return hist, trades, eq


def cross_check(hist, eq):
    """Flag divergence between daily_history equity and live_equity_curve."""
    if eq is None:
        return []
    merged = pd.merge(hist[["date", "equity"]], eq[["date", "equity"]],
                      on="date", suffixes=("_hist", "_curve"))
    merged["diff"] = (merged["equity_hist"] - merged["equity_curve"]).abs()
    bad = merged[merged["diff"] > 1.0]  # > $1 difference = worth flagging
    return [f"{r['date'].date()}: history ${r['equity_hist']:,.2f} vs curve "
            f"${r['equity_curve']:,.2f} (Δ ${r['diff']:,.2f})" for _, r in bad.iterrows()]


# ══════════════════════════════════════════════════════════════════════════════
# QUANT METRICS  (from daily_history.csv)
# ══════════════════════════════════════════════════════════════════════════════
def quant_metrics(hist: pd.DataFrame) -> dict:
    m = {}
    m["start"], m["end"] = hist["date"].iloc[0], hist["date"].iloc[-1]
    m["n_days"] = len(hist)

    r = hist["daily_pnl_pct"].values / 100.0
    m["r"] = r

    # benchmark daily returns from logged prices
    spy_r = hist["spy_price"].pct_change().values
    qqq_r = hist["qqq_price"].pct_change().values
    m["spy_r"], m["qqq_r"] = spy_r, qqq_r

    # headline (already logged — use the file's own numbers)
    m["total_ret"] = hist["total_return_pct"].iloc[-1]
    m["spy_since"] = hist["spy_ret_since_incept"].iloc[-1]
    m["qqq_since"] = hist["qqq_ret_since_incept"].iloc[-1]
    m["alpha_spy"] = hist["alpha_vs_spy"].iloc[-1]
    m["alpha_qqq"] = hist["alpha_vs_qqq"].iloc[-1]

    mu, sd = np.nanmean(r), np.nanstd(r, ddof=1)
    dn_dev = np.nanstd(np.minimum(r, 0), ddof=1)
    m["ann_ret"] = mu * TRADING_DAYS * 100
    m["ann_vol"] = sd * np.sqrt(TRADING_DAYS) * 100 if sd > 0 else np.nan
    m["sharpe"]  = mu / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan
    m["sortino"] = mu / dn_dev * np.sqrt(TRADING_DAYS) if dn_dev > 0 else np.nan

    eqv  = hist["equity"].values
    peak = np.maximum.accumulate(eqv)
    dd   = eqv / peak - 1
    m["dd_series"] = dd * 100
    m["max_dd"]    = dd.min() * 100
    m["calmar"]    = m["ann_ret"] / abs(m["max_dd"]) if m["max_dd"] < 0 else np.nan

    mask = ~np.isnan(r) & ~np.isnan(spy_r)
    if mask.sum() > 2 and np.nanstd(spy_r[mask]) > 0:
        m["beta_spy"] = np.cov(r[mask], spy_r[mask])[0, 1] / np.var(spy_r[mask])
        m["corr_spy"] = np.corrcoef(r[mask], spy_r[mask])[0, 1]
    else:
        m["beta_spy"] = m["corr_spy"] = np.nan

    up, dn = spy_r > 0, spy_r < 0
    m["up_capture"]   = (np.nanmean(r[up]) / np.nanmean(spy_r[up]) * 100) if up.sum() and np.nanmean(spy_r[up]) else np.nan
    m["down_capture"] = (np.nanmean(r[dn]) / np.nanmean(spy_r[dn]) * 100) if dn.sum() and np.nanmean(spy_r[dn]) else np.nan

    m["win_days"]  = int((r > 0).sum())
    m["lose_days"] = int((r < 0).sum())
    m["hit_rate"]  = m["win_days"] / max(m["win_days"] + m["lose_days"], 1) * 100
    m["best_day"], m["worst_day"] = np.nanmax(r) * 100, np.nanmin(r) * 100

    # indexed curves for the chart
    # from the since-inception columns, so all three share the headline's
    # base (the $100k at the inception-day open) and end on the table values
    m["idx_book"] = 100 + hist["total_return_pct"].values
    m["idx_spy"]  = 100 + hist["spy_ret_since_incept"].values
    m["idx_qqq"]  = 100 + hist["qqq_ret_since_incept"].values

    # deployment + P&L split (straight from the file)
    m["deployed"]   = hist["deployed_pct"].values
    m["realized"]   = hist["realized_pnl"].values
    m["unrealized"] = hist["unrealized_pnl"].values
    m["avg_deployed"] = float(np.nanmean(m["deployed"]))
    return m


# ══════════════════════════════════════════════════════════════════════════════
# REGIME ATTRIBUTION   (signature section 1)
# ══════════════════════════════════════════════════════════════════════════════
def regime_attribution(hist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, g in hist.groupby("regime"):
        r = g["daily_pnl_pct"].values / 100.0
        n = len(g)
        mu, sd = np.nanmean(r), np.nanstd(r, ddof=1) if n > 1 else np.nan
        eqv = g["equity"].values
        dd = (eqv / np.maximum.accumulate(eqv) - 1).min() * 100 if n > 1 else 0.0
        rows.append({
            "regime": regime, "days": n,
            "period_ret": (np.prod(1 + np.nan_to_num(r)) - 1) * 100,
            "ann_ret": mu * TRADING_DAYS * 100,
            "sharpe": (mu / sd * np.sqrt(TRADING_DAYS)) if sd and sd > 0 else np.nan,
            "max_dd": dd,
            "hit_rate": (r > 0).sum() / max((r != 0).sum(), 1) * 100,
        })
    out = pd.DataFrame(rows).sort_values("days", ascending=False).reset_index(drop=True)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST PARITY   (signature section 2)
# ══════════════════════════════════════════════════════════════════════════════
def backtest_parity(reg_attr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in reg_attr.iterrows():
        bt = BACKTEST["regimes"].get(r["regime"])
        if bt is None:
            continue
        rows.append({
            "regime": r["regime"], "days_live": r["days"],
            "live_ann": r["ann_ret"],  "bt_ann": bt["ann_ret"],
            "live_sharpe": r["sharpe"], "bt_sharpe": bt["sharpe"],
            "live_dd": r["max_dd"],     "bt_dd": bt["max_dd"],
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# TRADE ANALYSIS  (round trips via per-ticker FIFO on the trade log)
# ══════════════════════════════════════════════════════════════════════════════
def round_trips(trades: pd.DataFrame) -> pd.DataFrame:
    """Match BUYs to SELLs per ticker (FIFO). Returns one row per closed trip."""
    trips = []
    for tkr, g in trades.groupby("ticker"):
        g = g.sort_values("date", kind="stable")
        open_lots = []  # [ {date, price, shares, regime} ]
        for _, t in g.iterrows():
            if str(t["action"]).upper() == "BUY":
                open_lots.append({"date": t["date"], "price": float(t["price"]),
                                  "shares": float(t["shares"]),
                                  "regime": t.get("hmm_regime", "?")})
            elif str(t["action"]).upper() == "SELL":
                sell_sh, sell_px = float(t["shares"]), float(t["price"])
                while sell_sh > 1e-9 and open_lots:
                    lot = open_lots[0]
                    take = min(sell_sh, lot["shares"])
                    pnl_usd = take * (sell_px - lot["price"])
                    pnl_pct = (sell_px / lot["price"] - 1) * 100 if lot["price"] else np.nan
                    trips.append({
                        "ticker": tkr,
                        "entry": lot["date"].date(), "exit": t["date"].date(),
                        "hold_days": (t["date"] - lot["date"]).days,
                        "entry_px": lot["price"], "exit_px": sell_px,
                        "shares": take, "pnl_usd": pnl_usd, "pnl_pct": pnl_pct,
                        "entry_regime": lot["regime"],
                        "exit_regime": t.get("hmm_regime", "?"),
                        "exit_reason": t.get("reason", "?"),
                    })
                    lot["shares"] -= take
                    sell_sh -= take
                    if lot["shares"] <= 1e-9:
                        open_lots.pop(0)
    df = pd.DataFrame(trips)
    if len(df):
        df = df[df["shares"] > 1e-3].reset_index(drop=True)  # drop fractional dust
    return df


def open_positions(trades: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct open lots (unmatched BUY shares) + parse latest positions
    column for current P&L% if available."""
    lots = []
    for tkr, g in trades.groupby("ticker"):
        g = g.sort_values("date", kind="stable")
        stack = []
        for _, t in g.iterrows():
            if str(t["action"]).upper() == "BUY":
                stack.append({"date": t["date"], "price": float(t["price"]),
                              "shares": float(t["shares"])})
            elif str(t["action"]).upper() == "SELL":
                s = float(t["shares"])
                while s > 1e-9 and stack:
                    take = min(s, stack[0]["shares"])
                    stack[0]["shares"] -= take
                    s -= take
                    if stack[0]["shares"] <= 1e-9:
                        stack.pop(0)
        for lot in stack:
            if lot["shares"] > 1e-3 and (lot["price"] * lot["shares"]) > 1.0:
                lots.append({"ticker": tkr, "entry": lot["date"].date(),
                             "entry_px": lot["price"], "shares": lot["shares"],
                             "cost": lot["price"] * lot["shares"]})
    op = pd.DataFrame(lots)
    if op.empty:
        return op

    # current P&L% from the latest daily_history 'positions' string, e.g.
    # "QQQ:+0.5%" or "AAPL:+1.2%,MSFT:-0.3%,..."  (defensive parse)
    cur = {}
    try:
        latest_str = str(hist["positions"].iloc[-1])
        for token in latest_str.replace(";", " ").replace(",", " ").split():
            if ":" in token:
                k, v = token.split(":", 1)
                cur[k.strip().upper()] = float(v.strip().rstrip("%").replace("+", ""))
    except Exception:
        pass
    op["cur_pnl_pct"] = op["ticker"].map(cur)
    return op.sort_values("cur_pnl_pct", na_position="last")


def month_by_month(hist: pd.DataFrame) -> pd.DataFrame:
    """Per-calendar-month return for the book vs SPY vs QQQ, since go-live.

    Chained from the SINCE-INCEPTION columns written by alpaca_ledger:
        month_ret = (1 + cum_at_month_end) / (1 + cum_at_prev_month_end) − 1
    Book and benchmarks therefore share one base (the $100k at the inception
    open), each month starts exactly where the previous one ended, and the
    months telescope back to the since-inception return exactly.
    A month is partial if it is the first (mid-month go-live) or is still
    running in New York.
    """
    hist = hist.sort_values("date").reset_index(drop=True)
    ym = hist["date"].dt.to_period("M")
    months = sorted(ym.unique())
    current_month = pd.Period(ny_today(), "M")
    cols = {"aria": "total_return_pct", "spy": "spy_ret_since_incept",
            "qqq": "qqq_ret_since_incept"}

    rows, prev = [], {k: 0.0 for k in cols}
    for i, mo in enumerate(months):
        g = hist[ym == mo]
        end = {k: g[c].iloc[-1] for k, c in cols.items()}
        ret = {k: ((1 + end[k] / 100) / (1 + prev[k] / 100) - 1) * 100 for k in cols}
        prev = end
        aria_ret, spy_ret, qqq_ret = ret["aria"], ret["spy"], ret["qqq"]
        partial = (i == 0) or (mo == current_month)

        rows.append({
            "month": mo, "label": mo.strftime("%b %Y"), "days": len(g),
            "aria": aria_ret, "spy": spy_ret, "qqq": qqq_ret,
            "vs_spy": aria_ret - spy_ret, "vs_qqq": aria_ret - qqq_ret,
            "partial": partial,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# INFLATION — nominal vs real (purchasing-power) fund value
# ══════════════════════════════════════════════════════════════════════════════
def fetch_cpi() -> pd.Series | None:
    """Monthly CPI-U (BLS, seasonally adjusted — FRED series CPIAUCSL).

    SA rather than NSA: over a window of a few months the NSA series swings
    with seasonal price patterns that are not real inflation.
    This is the script's ONLY network call. It refreshes the local cache
    (data/cpi_cpiaucsl.csv) and falls back to that cache when offline, so
    the report still builds without network. Returns None if neither works.
    """
    from io import StringIO
    from urllib.request import urlopen
    try:
        with urlopen(CPI_URL, timeout=15) as resp:
            text = resp.read().decode("utf-8")
        cpi = pd.read_csv(StringIO(text), parse_dates=["observation_date"])
        cpi.to_csv(CPI_CACHE, index=False)
    except Exception as e:
        if not CPI_CACHE.exists():
            print(f"  ! CPI unavailable ({e}) and no cache — skipping inflation chart")
            return None
        print(f"  ! CPI fetch failed ({e}) — using cached {CPI_CACHE.name}")
        cpi = pd.read_csv(CPI_CACHE, parse_dates=["observation_date"])
    cpi = cpi.dropna()
    return cpi.set_index("observation_date")[CPI_SERIES].astype(float)


def daily_cpi(cpi_m: pd.Series, dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Align monthly CPI to daily fund dates. Returns (cpi_level, is_estimate).

    Method:
      1. Each monthly CPI print is an AVERAGE price level over that month, so
         it is anchored at the 15th of its month (not the 1st, which would
         credit a full month of inflation two weeks early).
      2. Between anchors, CPI is interpolated GEOMETRICALLY (linear in log
         CPI by calendar day) — i.e. a constant daily inflation rate within
         each month-to-month step.
      3. After the latest published print (BLS publishes ~2 weeks after month
         end, so the current month is always missing) CPI is EXTRAPOLATED at
         the trailing 3-month average monthly rate. Those days are flagged
         is_estimate=True, shaded on the chart and revised automatically
         when the next print lands. Forward-filling instead would assume
         zero inflation since the last print and overstate the real value.
    """
    anchors = cpi_m.copy()
    anchors.index = anchors.index + pd.Timedelta(days=14)          # 1st → 15th
    t_anchor = anchors.index.values.astype("datetime64[D]").astype(float)
    log_cpi  = np.log(anchors.values)

    t = dates.values.astype("datetime64[D]").astype(float)
    out = np.interp(t, t_anchor, log_cpi)                           # log-linear

    last_t = t_anchor[-1]
    est = t > last_t
    if est.any():
        n = min(3, len(log_cpi) - 1)
        per_day = (log_cpi[-1] - log_cpi[-1 - n]) / (last_t - t_anchor[-1 - n])
        out[est] = log_cpi[-1] + per_day * (t[est] - last_t)
    return np.exp(out), est


def real_value(nominal_idx, inflation_factor):
    """Exact real value: nominal / cumulative inflation factor.
    (NOT nominal − inflation — that is only a first-order approximation.)
    Equivalent return form: real_ret = (1 + nominal_ret) / (1 + inflation) − 1.
    """
    return np.asarray(nominal_idx, dtype=float) / np.asarray(inflation_factor, dtype=float)


def _check_real_value_math():
    """Worked example: fund 100 → 115 while cumulative inflation is 5%."""
    nominal = np.array([100.0, 115.0])
    factor  = np.array([1.00, 1.05])
    real = real_value(nominal, factor)
    assert np.isclose(real[0], 100.0)
    assert round(real[1], 2) == 109.52, real[1]
    assert round((real[1] / 100 - 1) * 100, 2) == 9.52       # real gain %
    assert np.isclose((1 + 0.15) / (1 + 0.05) - 1, real[1] / 100 - 1)


def inflation_real(hist: pd.DataFrame) -> dict | None:
    """Nominal vs real fund value, both indexed to 100 on INFLATION_BASE.

    Nominal fund = logged equity / equity on the base date (same equity series
    month_by_month uses — see its docstring for why not daily_pnl_pct).
    CPI index    = daily CPI / daily CPI on the base date (× 100).
    Real fund    = nominal / (CPI index / 100).
    """
    cpi_m = fetch_cpi()
    if cpi_m is None or cpi_m.empty:
        return None
    h = hist[hist["date"] >= INFLATION_BASE].reset_index(drop=True)
    if h.empty:
        return None

    cpi_d, est = daily_cpi(cpi_m, h["date"])
    factor   = cpi_d / cpi_d[0]                        # cumulative inflation factor
    nominal  = h["equity"].values / h["equity"].iloc[0] * 100
    real     = real_value(nominal, factor)

    df = pd.DataFrame({
        "date":            h["date"].dt.date,
        "equity":          h["equity"].values,
        "nominal_index":   nominal,
        "cpi_level":       cpi_d,
        "cpi_index":       factor * 100,
        "real_index":      real,
        "cum_inflation_pct": (factor - 1) * 100,
        "nominal_ret_pct": nominal - 100,
        "real_ret_pct":    real - 100,
        # purchasing power the book has lost to inflation, in today's dollars:
        # equity minus what that equity is worth in base-date dollars
        "inflation_drag_usd": h["equity"].values - h["equity"].values / factor,
        "cpi_is_estimate": est,
    })
    last = df.iloc[-1]
    return {
        "df": df, "base": h["date"].iloc[0], "end": h["date"].iloc[-1],
        "latest_print": cpi_m.index[-1],
        "nominal": last["nominal_index"], "real": last["real_index"],
        "nominal_ret": last["nominal_ret_pct"], "real_ret": last["real_ret_pct"],
        "infl": last["cum_inflation_pct"], "drag_usd": last["inflation_drag_usd"],
        "ann_infl": ((factor[-1]) ** (365.25 / max((h["date"].iloc[-1] - h["date"].iloc[0]).days, 1)) - 1) * 100,
        "any_est": bool(est.any()),
    }


def hold_duration_note(trips: pd.DataFrame) -> str:
    """Momentum-specific behavior read: losers fast, winners held (min-hold)."""
    if trips.empty:
        return "No closed round trips yet."
    losers  = trips[trips["pnl_usd"] <= 0]
    winners = trips[trips["pnl_usd"] > 0]
    parts = [f"{len(trips)} closed round trips: {len(winners)}W / {len(losers)}L."]
    if len(losers):
        parts.append(f"Losers avg hold {losers['hold_days'].mean():.1f}d "
                     f"(fast loser exits = min-hold design working).")
    if len(winners):
        parts.append(f"Winners avg hold {winners['hold_days'].mean():.1f}d.")
    parts.append("Note: min-hold HOLD events (skipped sells on profitable young "
                 "positions) print to console but are not CSV-logged, so churn "
                 "AVOIDED is not directly countable here.")
    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
def _style(ax, title, ylabel=None):
    ax.set_title(title, fontsize=11, color=NAVY, weight="bold", loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=GREY)
    ax.tick_params(colors=GREY, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25)


def make_charts(hist, qm, reg_attr, parity, trips, opos, mbm, outdir: Path, infl=None):
    dates = hist["date"].dt.strftime("%b %d")

    # 1 — equity vs benchmarks, regime-shaded background
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    prev = 0
    for i in range(1, len(hist) + 1):
        if i == len(hist) or hist["regime"].iloc[i] != hist["regime"].iloc[prev]:
            ax.axvspan(prev - 0.5, i - 0.5, alpha=0.07,
                       color=REGIME_COLORS.get(hist["regime"].iloc[prev], MGREY))
            prev = i
    ax.plot(dates, qm["idx_book"], color=NAVY, lw=2.2, label="ARIA momentum")
    ax.plot(dates, qm["idx_spy"], color=BLUE, lw=1.3, ls="--", label="SPY")
    ax.plot(dates, qm["idx_qqq"], color=AMBER, lw=1.3, ls="--", label="QQQ")
    ax.axhline(100, color=MGREY, lw=0.8)
    ax.legend(fontsize=8)
    _style(ax, "Equity vs benchmarks (indexed to 100, background = regime)")
    plt.xticks(rotation=45); plt.tight_layout()
    fig.savefig(outdir / "equity_vs_benchmarks.png", dpi=150); plt.close(fig)

    # 2 — drawdown
    fig, ax = plt.subplots(figsize=(9.5, 3))
    ax.fill_between(dates, qm["dd_series"], 0, color=RED, alpha=0.35)
    ax.plot(dates, qm["dd_series"], color=RED, lw=1.2)
    ax.axhline(BACKTEST["max_dd"], color=NAVY, lw=1, ls=":",
               label=f"backtest max DD {BACKTEST['max_dd']}%")
    ax.legend(fontsize=8)
    _style(ax, f"Drawdown (live max {qm['max_dd']:.2f}%)", "%")
    plt.xticks(rotation=45); plt.tight_layout()
    fig.savefig(outdir / "drawdown.png", dpi=150); plt.close(fig)

    # 3 — regime attribution (signature)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    cols = [REGIME_COLORS.get(r, MGREY) for r in reg_attr["regime"]]
    a1.bar(reg_attr["regime"], reg_attr["days"], color=cols)
    _style(a1, "Days traded per regime", "days")
    a1.tick_params(axis="x", rotation=20)
    a2.bar(reg_attr["regime"], reg_attr["period_ret"], color=cols)
    a2.axhline(0, color=MGREY, lw=0.8)
    _style(a2, "Period return by regime", "%")
    a2.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig.savefig(outdir / "regime_attribution.png", dpi=150); plt.close(fig)

    # 4 — backtest parity (signature)
    if len(parity):
        fig, axes = plt.subplots(1, len(parity), figsize=(4.6 * len(parity), 4),
                                 squeeze=False)
        for ax, (_, row) in zip(axes[0], parity.iterrows()):
            labels = ["Ann ret %", "Sharpe", "Max DD %"]
            live = [row["live_ann"], row["live_sharpe"], row["live_dd"]]
            bt   = [row["bt_ann"], row["bt_sharpe"], row["bt_dd"]]
            x = np.arange(3); w = 0.38
            ax.bar(x - w / 2, live, w, color=NAVY, label=f"Live ({int(row['days_live'])}d)")
            ax.bar(x + w / 2, bt, w, color=MGREY, label="Backtest (5.5y)")
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
            ax.axhline(0, color=MGREY, lw=0.8)
            ax.legend(fontsize=7)
            _style(ax, f"{row['regime']}")
        plt.suptitle("Live vs locked-backtest expectation, per regime "
                     "(tiny live sample — directional only)",
                     fontsize=10, color=GREY)
        plt.tight_layout()
        fig.savefig(outdir / "backtest_parity.png", dpi=150); plt.close(fig)

    # 5 — return distribution + capture
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.hist(qm["r"] * 100, bins=min(12, max(5, qm["n_days"] // 2)),
            color=BLUE, alpha=0.75, edgecolor="white")
    a1.axvline(0, color=MGREY, lw=1)
    _style(a1, "Daily return distribution", "days")
    a1.set_xlabel("Daily %", fontsize=9, color=GREY)
    caps = [qm["up_capture"], qm["down_capture"]]
    if not any(np.isnan(caps)):
        a2.bar(["Up-capture", "Down-capture"], caps,
               color=[GREEN if caps[0] >= 100 else AMBER,
                      GREEN if caps[1] <= 100 else RED])
        a2.axhline(100, color=MGREY, ls="--", lw=1)
    _style(a2, "Capture vs SPY (100 = matches)", "%")
    plt.tight_layout()
    fig.savefig(outdir / "returns_and_capture.png", dpi=150); plt.close(fig)

    # 6 — positions: open book + closed round trips
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, max(3.5, 0.4 * max(len(opos), len(trips), 4))))
    if len(opos):
        vals = opos["cur_pnl_pct"].fillna(0)
        a1.barh(opos["ticker"], vals,
                color=[GREEN if v > 0 else RED for v in vals], height=0.55)
        a1.axvline(0, color=MGREY, lw=0.8)
    _style(a1, "Open positions — current P&L % (latest snapshot)")
    if len(trips):
        lbl = trips["ticker"] + " " + trips["exit"].astype(str)
        a2.barh(lbl, trips["pnl_pct"],
                color=[GREEN if v > 0 else RED for v in trips["pnl_pct"]], height=0.55)
        a2.axvline(0, color=MGREY, lw=0.8)
    _style(a2, "Closed round trips — realized %")
    plt.tight_layout()
    fig.savefig(outdir / "positions_pnl.png", dpi=150); plt.close(fig)

    # 7 — deployment + realized/unrealized split
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.5, 5.4), sharex=True)
    a1.fill_between(dates, qm["deployed"], color=PURPLE, alpha=0.35)
    a1.plot(dates, qm["deployed"], color=PURPLE, lw=1.4)
    _style(a1, f"Capital deployed (avg {qm['avg_deployed']:.0f}%)", "%")
    a2.plot(dates, qm["realized"], color=NAVY, lw=1.6, label="Realized P&L $")
    a2.plot(dates, qm["unrealized"], color=AMBER, lw=1.6, label="Unrealized P&L $")
    a2.axhline(0, color=MGREY, lw=0.8); a2.legend(fontsize=8)
    _style(a2, "Realized vs unrealized P&L", "$")
    plt.xticks(rotation=45); plt.tight_layout()
    fig.savefig(outdir / "deployment_pnl_split.png", dpi=150); plt.close(fig)

    # 8 — month-by-month return comparison
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    labels = [r["label"].split()[0][:3] + ("*" if r["partial"] else "")
              for _, r in mbm.iterrows()]
    x = np.arange(len(mbm)); w = 0.26
    bars = [
        (x - w, mbm["aria"], NAVY, "ARIA"),
        (x,     mbm["spy"],  BLUE, "SPY"),
        (x + w, mbm["qqq"],  AMBER, "QQQ"),
    ]
    for pos, vals, color, name in bars:
        b = ax.bar(pos, vals, w, color=color, label=name)
        ax.bar_label(b, labels=[f"{v:+.1f}%" for v in vals], fontsize=7,
                     color=GREY, padding=2)
    ax.axhline(0, color=MGREY, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(fontsize=8)
    _style(ax, "Month-by-month return — ARIA vs SPY vs QQQ", "%")
    fig.text(0.01, 0.01, "* partial month", fontsize=7, color=GREY)
    plt.tight_layout()
    fig.savefig(outdir / "month_by_month.png", dpi=150); plt.close(fig)

    # 9 — nominal vs inflation-adjusted real fund value
    if infl is not None:
        d = infl["df"]
        x = np.arange(len(d))
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        if infl["any_est"]:
            i0 = int(np.argmax(d["cpi_is_estimate"].values))
            ax.axvspan(i0 - 0.5, len(d) - 0.5, color=MGREY, alpha=0.25, lw=0)
            ax.text(i0, 0.97, f" CPI estimated after {infl['latest_print']:%b %Y} print",
                    transform=ax.get_xaxis_transform(), fontsize=7, color=GREY, va="top")
        # the gap between the two lines is what inflation has eaten
        ax.fill_between(x, d["real_index"], d["nominal_index"], color=RED, alpha=0.12,
                        lw=0, label="Eaten by inflation")
        ax.plot(x, d["nominal_index"], color=NAVY, lw=2.2, label="Nominal Fund Value")
        ax.plot(x, d["real_index"], color=GREEN, lw=2.2, label="Real Fund Value (CPI-adjusted)")
        ax.plot(x, d["cpi_index"], color=GREY, lw=1.1, ls=":",
                label="CPI index (break-even to keep purchasing power)")
        ax.axhline(100, color=MGREY, lw=0.8)
        for col, color in (("nominal_index", NAVY), ("real_index", GREEN)):
            v = d[col].iloc[-1]
            ax.annotate(f"{v:.2f}", (x[-1], v), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=color, weight="bold", va="center")
        step = max(1, len(d) // 14)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(pd.to_datetime(d["date"]).dt.strftime("%b %d").iloc[::step],
                           rotation=45)
        ax.set_xlim(-0.5, len(d) + 2.5)
        ax.legend(fontsize=8, loc="upper left")
        _style(ax, "Fund Value vs Inflation — Nominal vs Real", "Indexed value (base = 100)")
        ax.text(0, 1.01, f"Since {infl['base']:%B %d, %Y} · Base = 100 · "
                f"CPI-U (BLS, SA) through {infl['latest_print']:%b %Y}",
                transform=ax.transAxes, fontsize=8, color=GREY, va="bottom")
        ax.set_title("Fund Value vs Inflation — Nominal vs Real", fontsize=11,
                     color=NAVY, weight="bold", loc="left", pad=16)
        fig.text(0.5, 0.012,
                 f"Nominal Fund: {infl['nominal']:.2f}  ·  Real Fund: {infl['real']:.2f}  ·  "
                 f"Nominal Gain: {infl['nominal_ret']:+.2f}%  ·  "
                 f"Cumulative Inflation: {infl['infl']:+.2f}%  ·  "
                 f"Real Gain: {infl['real_ret']:+.2f}%\n"
                 f"Inflation has eaten ${infl['drag_usd']:,.0f} of purchasing power "
                 f"since {infl['base']:%b %d} (in today's dollars)",
                 fontsize=8, color=NAVY, weight="bold", ha="center", linespacing=1.6)
        plt.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(outdir / "inflation_real_value.png", dpi=150); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════
def write_report(qm, reg_attr, parity, trips, opos, mbm, hold_note, xcheck,
                 outdir: Path, tag: str, infl=None):
    L, A = [], None
    A = L.append
    A(f"# ARIA-Momentum — Month-End Review {tag}\n")
    A(f"_Generated {datetime.now():%Y-%m-%d %H:%M} · inception "
      f"{qm['start']:%Y-%m-%d} → {qm['end']:%Y-%m-%d} · "
      f"{qm['n_days']} trading days · read-only analysis._\n")
    A("> **Sample-size caveat:** a few weeks of live data verifies the "
      "machinery and shows behavior; it cannot prove or disprove the edge "
      "validated over 5.5 backtest years. Every number below is indicative, "
      "not conclusive — especially while the book has traded in only "
      f"{len(reg_attr)} regime(s).\n")

    A("## Headline")
    A("| Metric | Book | SPY | QQQ |")
    A("|---|---|---|---|")
    A(f"| Return since inception | **{qm['total_ret']:+.2f}%** | "
      f"{qm['spy_since']:+.2f}% | {qm['qqq_since']:+.2f}% |")
    A(f"| Alpha | — | {qm['alpha_spy']:+.2f}pp | {qm['alpha_qqq']:+.2f}pp |\n")

    A("## Month-by-month")
    A("| Month | Days | ARIA | SPY | QQQ | vs SPY | vs QQQ |")
    A("|---|---|---|---|---|---|---|")
    for _, r in mbm.iterrows():
        label = r["label"] + ("*" if r["partial"] else "")
        A(f"| {label} | {int(r['days'])} | {r['aria']:+.2f}% | {r['spy']:+.2f}% | "
          f"{r['qqq']:+.2f}% | {r['vs_spy']:+.2f}pp | {r['vs_qqq']:+.2f}pp |")
    A(f"\n_* partial month (first month starts at the {qm['start']:%Y-%m-%d} "
      "go-live; a month still running in New York is partial through its "
      "latest closed trading day)._\n")

    if infl is not None:
        d = infl["df"]
        A("## Inflation — nominal vs real")
        A(f"_Base {infl['base']:%Y-%m-%d} = 100 for fund, real fund and CPI. "
          f"Real = nominal ÷ cumulative CPI factor (exact, not nominal − inflation). "
          f"CPI-U (BLS, seasonally adjusted, FRED {CPI_SERIES}) through the "
          f"{infl['latest_print']:%b %Y} print; monthly prints anchored mid-month and "
          f"interpolated geometrically"
          + (", later days extrapolated at the trailing 3-month pace (estimate)"
             if infl["any_est"] else "") + "._\n")
        A("| Nominal Fund | Real Fund | Nominal Gain | Cumulative Inflation | Real Gain | Inflation drag |")
        A("|---|---|---|---|---|---|")
        A(f"| **{infl['nominal']:.2f}** | **{infl['real']:.2f}** | {infl['nominal_ret']:+.2f}% | "
          f"{infl['infl']:+.2f}% ({infl['ann_infl']:.1f}% annualized) | "
          f"**{infl['real_ret']:+.2f}%** | ${infl['drag_usd']:,.2f} |\n")
        A("Month-end snapshots (full per-day series in `inflation_real_value.csv`):\n")
        A("| Date | Nominal | Real | Cum. inflation | Nominal ret | Real ret | CPI |")
        A("|---|---|---|---|---|---|---|")
        dd = pd.to_datetime(d["date"])
        snaps = d[dd.dt.to_period("M") != dd.shift(-1).dt.to_period("M")]
        for _, r in snaps.iterrows():
            A(f"| {r['date']} | {r['nominal_index']:.2f} | {r['real_index']:.2f} | "
              f"{r['cum_inflation_pct']:+.2f}% | {r['nominal_ret_pct']:+.2f}% | "
              f"{r['real_ret_pct']:+.2f}% | {'est.' if r['cpi_is_estimate'] else 'actual'} |")
        A("")

    A("## Risk-adjusted (annualized from daily — small sample!)")
    A(f"- Sharpe **{qm['sharpe']:.2f}** · Sortino **{qm['sortino']:.2f}** · "
      f"Calmar {qm['calmar']:.2f}")
    A(f"- Ann. return {qm['ann_ret']:+.1f}% · ann. vol {qm['ann_vol']:.1f}% · "
      f"max drawdown **{qm['max_dd']:.2f}%** "
      f"(backtest budget {BACKTEST['max_dd']}%)")
    A(f"- Beta vs SPY {qm['beta_spy']:.2f} (corr {qm['corr_spy']:.2f}) · "
      f"up-capture {qm['up_capture']:.0f}% · down-capture {qm['down_capture']:.0f}%")
    A(f"- Hit rate {qm['hit_rate']:.0f}% ({qm['win_days']}W/{qm['lose_days']}L) · "
      f"best day {qm['best_day']:+.2f}% · worst {qm['worst_day']:+.2f}%")
    A(f"- Avg capital deployed {qm['avg_deployed']:.0f}%\n")

    A("## Regime attribution  *(signature)*")
    A("| Regime | Days | Period ret | Ann ret | Sharpe | Max DD | Hit |")
    A("|---|---|---|---|---|---|---|")
    for _, r in reg_attr.iterrows():
        A(f"| {r['regime']} | {int(r['days'])} | {r['period_ret']:+.2f}% | "
          f"{r['ann_ret']:+.1f}% | {r['sharpe']:.2f} | {r['max_dd']:.2f}% | "
          f"{r['hit_rate']:.0f}% |")
    A("")

    A("## Backtest parity  *(signature)*")
    A("_Live per-regime vs the locked 95.55% backtest's expectation for the "
      "same regime. With days this few, read direction, not magnitude._\n")
    if len(parity):
        A("| Regime | Live days | Live ann | BT ann | Live Sharpe | BT Sharpe | Live DD | BT DD |")
        A("|---|---|---|---|---|---|---|---|")
        for _, r in parity.iterrows():
            A(f"| {r['regime']} | {int(r['days_live'])} | {r['live_ann']:+.1f}% | "
              f"{r['bt_ann']:+.1f}% | {r['live_sharpe']:.2f} | {r['bt_sharpe']:.2f} | "
              f"{r['live_dd']:.2f}% | {r['bt_dd']:.2f}% |")
    else:
        A("No overlapping regimes yet.")
    A("")

    A("## Trades")
    A(hold_note + "\n")
    if len(trips):
        A("| Ticker | Entry | Exit | Hold | Realized % | Realized $ | Exit regime |")
        A("|---|---|---|---|---|---|---|")
        for _, t in trips.iterrows():
            A(f"| {t['ticker']} | {t['entry']} | {t['exit']} | {t['hold_days']}d | "
              f"{t['pnl_pct']:+.2f}% | ${t['pnl_usd']:+,.2f} | {t['exit_regime']} |")
        A("")

    A("## Open book")
    if len(opos):
        A("| Ticker | Entry | Entry px | Shares | Cost | Current P&L % |")
        A("|---|---|---|---|---|---|")
        for _, p in opos.iterrows():
            cur = f"{p['cur_pnl_pct']:+.1f}%" if pd.notna(p["cur_pnl_pct"]) else "n/a"
            A(f"| {p['ticker']} | {p['entry']} | ${p['entry_px']:.2f} | "
              f"{p['shares']:.4f} | ${p['cost']:,.0f} | {cur} |")
    else:
        A("Flat — no open positions.")
    A("")

    if xcheck:
        A("## ⚠ Data cross-check flags")
        A("daily_history vs live_equity_curve equity mismatches (> $1):")
        for x in xcheck:
            A(f"- {x}")
        A("")

    A("## Charts")
    for f in ["equity_vs_benchmarks.png", "drawdown.png", "regime_attribution.png",
              "backtest_parity.png", "returns_and_capture.png",
              "positions_pnl.png", "deployment_pnl_split.png", "month_by_month.png",
              "inflation_real_value.png"]:
        if (outdir / f).exists():
            A(f"![{f}]({f})")
    A("")

    (outdir / f"report_{tag}.md").write_text("\n".join(L), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", help="YYYY-MM to review (default: latest month in the data)")
    args = ap.parse_args()

    hist, trades, eq = load()
    month = pd.Period(args.month, "M") if args.month else hist["date"].max().to_period("M")
    cutoff = month.end_time.normalize()
    hist   = hist[hist["date"] <= cutoff].reset_index(drop=True)
    trades = trades[trades["date"] <= cutoff].reset_index(drop=True)
    if eq is not None:
        eq = eq[eq["date"] <= cutoff].reset_index(drop=True)
    if hist.empty:
        raise SystemExit(f"No closed trading days on or before {month}.")
    tag = str(month)
    outdir = _HERE / "month_end" / tag
    outdir.mkdir(parents=True, exist_ok=True)

    qm       = quant_metrics(hist)
    reg_attr = regime_attribution(hist)
    parity   = backtest_parity(reg_attr)
    trips    = round_trips(trades)
    opos     = open_positions(trades, hist)
    mbm      = month_by_month(hist)
    hnote    = hold_duration_note(trips)
    xcheck   = cross_check(hist, eq)
    _check_real_value_math()
    infl     = inflation_real(hist)
    if infl is not None:
        infl["df"].to_csv(outdir / "inflation_real_value.csv", index=False)

    make_charts(hist, qm, reg_attr, parity, trips, opos, mbm, outdir, infl)
    write_report(qm, reg_attr, parity, trips, opos, mbm, hnote, xcheck, outdir, tag, infl)

    # console summary
    print("=" * 68)
    print(f"  ARIA-Momentum — Month-End Review {tag}"
          f"  ({qm['n_days']} trading days since {qm['start']:%Y-%m-%d})")
    print("=" * 68)
    print(f"  Return   : {qm['total_ret']:+.2f}%   "
          f"(SPY {qm['spy_since']:+.2f}%, QQQ {qm['qqq_since']:+.2f}%)")
    print(f"  Alpha    : vs SPY {qm['alpha_spy']:+.2f}pp | vs QQQ {qm['alpha_qqq']:+.2f}pp")
    print(f"  Sharpe   : {qm['sharpe']:.2f}   Sortino {qm['sortino']:.2f}   "
          f"MaxDD {qm['max_dd']:.2f}%  (backtest budget {BACKTEST['max_dd']}%)")
    print(f"  Beta     : {qm['beta_spy']:.2f}   up-cap {qm['up_capture']:.0f}%  "
          f"down-cap {qm['down_capture']:.0f}%")
    print(f"  Hit rate : {qm['hit_rate']:.0f}%  ({qm['win_days']}W/{qm['lose_days']}L)"
          f"   deployed avg {qm['avg_deployed']:.0f}%")
    print(f"  Regimes  : " + ", ".join(f"{r['regime']} {int(r['days'])}d "
          f"({r['period_ret']:+.2f}%)" for _, r in reg_attr.iterrows()))
    if len(trips):
        w = (trips['pnl_usd'] > 0).sum()
        print(f"  Trips    : {len(trips)} closed ({w}W/{len(trips)-w}L, "
              f"net ${trips['pnl_usd'].sum():+,.2f})")
    if len(opos):
        print(f"  Open     : {len(opos)} positions, ${opos['cost'].sum():,.0f} at cost")
    print("  Months   : " + " | ".join(
        f"{r['label']}{'*' if r['partial'] else ''} ARIA {r['aria']:+.2f}% "
        f"(SPY {r['spy']:+.2f}%, QQQ {r['qqq']:+.2f}%)" for _, r in mbm.iterrows()))
    if infl is not None:
        print(f"  Real     : nominal {infl['nominal']:.2f} -> real {infl['real']:.2f}  "
              f"(nominal {infl['nominal_ret']:+.2f}%, CPI {infl['infl']:+.2f}%, "
              f"real {infl['real_ret']:+.2f}%, drag ${infl['drag_usd']:,.0f})")
    if xcheck:
        print(f"  ! Cross-check: {len(xcheck)} equity mismatches vs live_equity_curve "
              f"— see report")
    n_charts = len(list(outdir.glob("*.png")))
    print(f"\n  Report + {n_charts} charts -> {outdir}")
    print("  ! Weeks of data = machinery check, not an edge verdict.")


if __name__ == "__main__":
    main()