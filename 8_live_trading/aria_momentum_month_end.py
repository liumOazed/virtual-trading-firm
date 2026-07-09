"""
ARIA-Momentum — Month-End Quant Review  (READ-ONLY)
====================================================
Institutional-style periodic review of the live ARIA momentum book.
Reflects everything from inception (2026-06-15) up to the moment you run it.

PRIMARY SOURCE   8_live_trading/data/daily_history.csv
  (equity, cash, deployed_pct, n_positions, daily/total P&L, realized/
   unrealized split, regime, SPY/QQQ prices + since-inception returns +
   alpha — all logged daily by daily_recorder.py)
SECONDARY        8_live_trading/data/live_trade_log.csv   (trade events)
CROSS-CHECK      8_live_trading/data/live_equity_curve.csv

No network calls, no Alpaca calls, no yfinance — everything is computed
from the logged CSVs. READ-ONLY: never places orders, never modifies data.

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

Usage:
  python aria_momentum_month_end.py

Honest note baked into the report: a few weeks of live data verifies the
MACHINERY and observes behavior. It cannot prove or disprove the edge that
was validated over 5.5 backtest years. All risk metrics on this sample are
indicative, not conclusive — especially while the book has traded in only
one regime.
"""

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
DATA_DIR   = _HERE / "data" if (_HERE / "data").exists() else _HERE
HIST_FILE  = DATA_DIR / "daily_history.csv"
TRADE_FILE = DATA_DIR / "live_trade_log.csv"
EQ_FILE    = DATA_DIR / "live_equity_curve.csv"

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
    trades = pd.read_csv(TRADE_FILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
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
    m["idx_book"] = 100 * (1 + pd.Series(r).fillna(0)).cumprod().values
    m["idx_spy"]  = 100 * (1 + pd.Series(spy_r).fillna(0)).cumprod().values
    m["idx_qqq"]  = 100 * (1 + pd.Series(qqq_r).fillna(0)).cumprod().values

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
        g = g.sort_values("date")
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
        g = g.sort_values("date")
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
            if lot["shares"] > 1e-6:
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
        for token in latest_str.replace(";", ",").split(","):
            if ":" in token:
                k, v = token.split(":", 1)
                cur[k.strip().upper()] = float(v.strip().rstrip("%").replace("+", ""))
    except Exception:
        pass
    op["cur_pnl_pct"] = op["ticker"].map(cur)
    return op.sort_values("cur_pnl_pct", na_position="last")


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


def make_charts(hist, qm, reg_attr, parity, trips, opos, outdir: Path):
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


# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════
def write_report(qm, reg_attr, parity, trips, opos, hold_note, xcheck,
                 outdir: Path, tag: str):
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
              "positions_pnl.png", "deployment_pnl_split.png"]:
        if (outdir / f).exists():
            A(f"![{f}]({f})")
    A("")

    (outdir / f"report_{tag}.md").write_text("\n".join(L), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    hist, trades, eq = load()
    tag = hist["date"].max().strftime("%Y-%m")
    outdir = _HERE / "month_end" / tag
    outdir.mkdir(parents=True, exist_ok=True)

    qm       = quant_metrics(hist)
    reg_attr = regime_attribution(hist)
    parity   = backtest_parity(reg_attr)
    trips    = round_trips(trades)
    opos     = open_positions(trades, hist)
    hnote    = hold_duration_note(trips)
    xcheck   = cross_check(hist, eq)

    make_charts(hist, qm, reg_attr, parity, trips, opos, outdir)
    write_report(qm, reg_attr, parity, trips, opos, hnote, xcheck, outdir, tag)

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
    if xcheck:
        print(f"  ⚠ Cross-check: {len(xcheck)} equity mismatches vs live_equity_curve "
              f"— see report")
    print(f"\n  📁 Report + 7 charts → {outdir}")
    print("  ⚠ Weeks of data = machinery check, not an edge verdict.")


if __name__ == "__main__":
    main()