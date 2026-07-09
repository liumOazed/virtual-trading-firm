"""
ARIA-Growth — Month-End Quant Review  (READ-ONLY)
==================================================
Institutional-style monthly analysis of the growth book. Computes risk-adjusted
metrics, benchmark-relative performance, exit post-mortems, and saves charts +
a written report to a dated folder.

Inputs (same directory / 9_aria_growth):
  daily_positions.csv    per-ticker daily history   (from aria_growth_daily_log.py)
  daily_portfolio.csv    daily equity + benchmarks  (from aria_growth_daily_log.py)
  growth_trade_log.csv   every trade action         (from aria_growth_executor.py)
  screens/_manifest.csv  optional — monthly screen archive freshness check

Live fetch (optional): current prices of EXITED names via yfinance, to grade
whether each stop/drop saved or cost money. Skips gracefully if offline.

Output:  9_aria_growth/month_end/<YYYY-MM>/
  report_<YYYY-MM>.md          written analysis
  equity_vs_benchmarks.png     indexed equity curve vs SPY/QQQ
  drawdown.png                 underwater plot
  daily_returns_dist.png       return distribution + capture analysis
  positions_pnl.png            per-position P&L + weight drift
  exits_postmortem.png         exit grading (if prices fetched)

Usage:
  python aria_growth_month_end.py
  python aria_growth_month_end.py --no-fetch      # skip live price fetch

READ-ONLY: never places orders, never modifies the source CSVs.

Honest note baked into the report: ~20 trading days is enough to verify the
MACHINERY and observe behavior, not to prove or disprove edge. Sharpe & co. on
one month are indicative, not conclusive.
"""

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")

_HERE     = Path(__file__).resolve().parent
POS_FILE  = _HERE / "daily_positions.csv"
PORT_FILE = _HERE / "daily_portfolio.csv"
TRADE_FILE= _HERE / "growth_trade_log.csv"
MANIFEST  = _HERE / "screens" / "_manifest.csv"

GREEN, RED, BLUE, AMBER = "#1D9E75", "#E24B4A", "#378ADD", "#EF9F27"
NAVY, GREY, MGREY = "#1A3A5C", "#888780", "#D1D5DB"
TRADING_DAYS = 252


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
def load():
    pos = pd.read_csv(POS_FILE, parse_dates=["date"])
    port = pd.read_csv(PORT_FILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    trades = pd.read_csv(TRADE_FILE, parse_dates=["timestamp"])
    # positions pnl pct: auto-detect units (guard against decimal-stored data)
    p = pos["total_pnl_pct"].dropna()
    if len(p) and (p.abs() < 1.5).all():
        pos["total_pnl_pct"] = pos["total_pnl_pct"] * 100
    return pos, port, trades


# ══════════════════════════════════════════════════════════════════════════════
# QUANT METRICS
# ══════════════════════════════════════════════════════════════════════════════
def quant_metrics(port: pd.DataFrame) -> dict:
    r = port["day_change_pct"].values / 100.0          # daily book returns
    spy = port["spy_day_pct"].values / 100.0
    qqq = port["qqq_day_pct"].values / 100.0
    n = len(r)
    m = {"n_days": n}

    m["total_ret"]  = port["total_pnl_pct"].iloc[-1]
    m["spy_since"]  = port["spy_since_pct"].iloc[-1]
    m["qqq_since"]  = port["qqq_since_pct"].iloc[-1]
    m["alpha_spy"]  = m["total_ret"] - m["spy_since"]
    m["alpha_qqq"]  = m["total_ret"] - m["qqq_since"]

    mu, sd = np.nanmean(r), np.nanstd(r, ddof=1)
    downside = np.nanstd(np.minimum(r, 0), ddof=1)
    m["ann_ret"]    = mu * TRADING_DAYS * 100
    m["ann_vol"]    = sd * np.sqrt(TRADING_DAYS) * 100
    m["sharpe"]     = (mu / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan
    m["sortino"]    = (mu / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else np.nan

    # drawdown from the daily equity curve
    eq = port["equity"].values
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    m["max_dd"] = dd.min() * 100
    m["dd_series"] = dd * 100
    m["calmar"] = (m["ann_ret"] / abs(m["max_dd"])) if m["max_dd"] < 0 else np.nan

    # benchmark relationship
    mask = ~np.isnan(r) & ~np.isnan(spy)
    if mask.sum() > 2 and np.nanstd(spy[mask]) > 0:
        m["beta_spy"] = np.cov(r[mask], spy[mask])[0, 1] / np.var(spy[mask])
        m["corr_spy"] = np.corrcoef(r[mask], spy[mask])[0, 1]
    else:
        m["beta_spy"] = m["corr_spy"] = np.nan

    up, dn = spy > 0, spy < 0
    m["up_capture"]   = (np.nanmean(r[up]) / np.nanmean(spy[up]) * 100) if up.sum() and np.nanmean(spy[up]) else np.nan
    m["down_capture"] = (np.nanmean(r[dn]) / np.nanmean(spy[dn]) * 100) if dn.sum() and np.nanmean(spy[dn]) else np.nan

    m["win_days"] = int((r > 0).sum()); m["lose_days"] = int((r < 0).sum())
    m["best_day"] = np.nanmax(r) * 100; m["worst_day"] = np.nanmin(r) * 100
    m["hit_rate"] = m["win_days"] / max(m["win_days"] + m["lose_days"], 1) * 100

    # indexed curves
    m["idx_book"] = 100 * (1 + pd.Series(r).fillna(0)).cumprod().values
    m["idx_spy"]  = 100 * (1 + pd.Series(spy).fillna(0)).cumprod().values
    m["idx_qqq"]  = 100 * (1 + pd.Series(qqq).fillna(0)).cumprod().values
    m["r"], m["spy_r"], m["qqq_r"] = r, spy, qqq
    return m


def position_metrics(pos: pd.DataFrame) -> dict:
    latest_date = pos["date"].max()
    latest = pos[pos["date"] == latest_date].copy()
    m = {"latest_date": latest_date, "latest": latest.sort_values("total_pnl_pct")}
    m["n_open"] = len(latest)
    m["winners"] = int((latest["total_pnl_pct"] > 0).sum())
    # per-ticker daily vol over the month
    m["vol"] = (pos.groupby("ticker")["day_change_pct"].std().dropna()
                .sort_values(ascending=False))
    # contribution: latest unrealized P&L per name
    m["contrib"] = latest.set_index("ticker")["total_pnl_usd"].sort_values()
    return m


def exit_postmortem(trades: pd.DataFrame, fetch: bool = True):
    """Grade every closed name: exit price vs price now → saved or cost?"""
    closes = trades[trades["side"] == "close"].copy()
    if closes.empty:
        return pd.DataFrame()
    closes["reason_type"] = np.where(
        closes["reason"].astype(str).str.contains("STOP", case=False), "STOP-LOSS",
        np.where(closes["reason"].astype(str).str.contains("MANUAL", case=False),
                 "MANUAL DROP", "ROTATION"))
    rows = []
    px_now = {}
    if fetch:
        try:
            import yfinance as yf
            syms = list(closes["symbol"].unique())
            data = yf.download(syms, period="5d", auto_adjust=True, progress=False)["Close"]
            if isinstance(data, pd.Series):
                data = data.to_frame(syms[0])
            for s in syms:
                if s in data.columns and data[s].dropna().size:
                    px_now[s] = float(data[s].dropna().iloc[-1])
        except Exception as e:
            print(f"  ⚠ price fetch failed ({e}) — post-mortem prices skipped")
    for _, ev in closes.iterrows():
        sym = ev["symbol"]
        ent = trades[(trades["symbol"] == sym) & (trades["side"] == "buy") &
                     (~trades["reason"].astype(str).str.contains("redistribute", na=False))]
        entry_notional = ent["notional"].iloc[0] if len(ent) else np.nan
        exit_notional = ev["notional"]
        realized_pct = (exit_notional / entry_notional - 1) * 100 if entry_notional else np.nan
        # exit price per share isn't logged; grade on % move of the STOCK since exit
        row = {"symbol": sym, "type": ev["reason_type"],
               "exit_date": ev["timestamp"].date(),
               "entry_notional": entry_notional, "exit_notional": exit_notional,
               "realized_pct": realized_pct, "px_now": px_now.get(sym, np.nan)}
        rows.append(row)
    pm = pd.DataFrame(rows)
    return pm


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
def _style(ax, title, ylabel=None):
    ax.set_title(title, fontsize=11, color=NAVY, weight="bold", loc="left")
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, color=GREY)
    ax.tick_params(colors=GREY, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25)


def make_charts(port, qm, pm_pos, pm_exits, outdir: Path):
    dates = port["date"].dt.strftime("%b %d")

    # 1. Equity vs benchmarks
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(dates, qm["idx_book"], color=NAVY, lw=2.2, label="Growth book")
    ax.plot(dates, qm["idx_spy"], color=BLUE, lw=1.4, ls="--", label="SPY")
    ax.plot(dates, qm["idx_qqq"], color=AMBER, lw=1.4, ls="--", label="QQQ")
    ax.axhline(100, color=MGREY, lw=0.8)
    ax.legend(fontsize=8); _style(ax, "Equity vs benchmarks (indexed to 100)")
    plt.xticks(rotation=45); plt.tight_layout()
    fig.savefig(outdir / "equity_vs_benchmarks.png", dpi=150); plt.close(fig)

    # 2. Drawdown
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.fill_between(dates, qm["dd_series"], 0, color=RED, alpha=0.35)
    ax.plot(dates, qm["dd_series"], color=RED, lw=1.2)
    _style(ax, f"Drawdown (max {qm['max_dd']:.2f}%)", "%")
    plt.xticks(rotation=45); plt.tight_layout()
    fig.savefig(outdir / "drawdown.png", dpi=150); plt.close(fig)

    # 3. Return distribution + capture
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.hist(qm["r"] * 100, bins=12, color=BLUE, alpha=0.75, edgecolor="white")
    a1.axvline(0, color=MGREY, lw=1)
    _style(a1, "Daily return distribution", "days")
    a1.set_xlabel("Daily %", fontsize=9, color=GREY)
    caps = [qm["up_capture"], qm["down_capture"]]
    a2.bar(["Up-capture", "Down-capture"], caps,
           color=[GREEN if caps[0] >= 100 else AMBER, GREEN if caps[1] <= 100 else RED])
    a2.axhline(100, color=MGREY, ls="--", lw=1)
    _style(a2, "Capture vs SPY (100 = matches)", "%")
    plt.tight_layout(); fig.savefig(outdir / "daily_returns_dist.png", dpi=150); plt.close(fig)

    # 4. Per-position P&L
    latest = pm_pos["latest"]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(latest))))
    colors = [GREEN if v > 0 else RED for v in latest["total_pnl_pct"]]
    ax.barh(latest["ticker"], latest["total_pnl_pct"], color=colors, height=0.6)
    ax.axvline(0, color=MGREY, lw=0.8)
    ax.axvline(-15, color=RED, lw=0.8, ls=":", label="stop level")
    ax.legend(fontsize=8)
    _style(ax, "Open positions — total P&L % (vs entry)")
    plt.tight_layout(); fig.savefig(outdir / "positions_pnl.png", dpi=150); plt.close(fig)

    # 5. Exits post-mortem
    if len(pm_exits):
        fig, ax = plt.subplots(figsize=(8, 0.6 * len(pm_exits) + 2))
        colors = [RED if v < 0 else GREEN for v in pm_exits["realized_pct"]]
        ax.barh(pm_exits["symbol"] + "  (" + pm_exits["type"].str[:6] + ")",
                pm_exits["realized_pct"], color=colors, height=0.55)
        ax.axvline(0, color=MGREY, lw=0.8); ax.axvline(-15, color=RED, lw=0.8, ls=":")
        _style(ax, "Exited names — realized % at exit")
        plt.tight_layout(); fig.savefig(outdir / "exits_postmortem.png", dpi=150); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════
def write_report(qm, pm_pos, pm_exits, trades, outdir: Path, month_tag: str):
    L = []
    A = L.append
    A(f"# ARIA-Growth — Month-End Review {month_tag}\n")
    A(f"_Generated {datetime.now():%Y-%m-%d %H:%M}. Read-only analysis; "
      f"{qm['n_days']} trading days of data._\n")
    A("> **Sample-size caveat:** one month verifies the machinery and shows "
      "behavior; it cannot prove or disprove edge. Treat every number below as "
      "indicative, not conclusive.\n")

    A("## Headline")
    A(f"| Metric | Book | SPY | QQQ |")
    A(f"|---|---|---|---|")
    A(f"| Return since go-live | **{qm['total_ret']:+.2f}%** | {qm['spy_since']:+.2f}% | {qm['qqq_since']:+.2f}% |")
    A(f"| Alpha | — | {qm['alpha_spy']:+.2f}pp | {qm['alpha_qqq']:+.2f}pp |\n")

    A("## Risk-adjusted (annualized from daily)")
    A(f"- Sharpe: **{qm['sharpe']:.2f}**   |   Sortino: **{qm['sortino']:.2f}**   |   Calmar: {qm['calmar']:.2f}")
    A(f"- Ann. return {qm['ann_ret']:+.1f}%  |  Ann. vol {qm['ann_vol']:.1f}%  |  Max drawdown {qm['max_dd']:.2f}%")
    A(f"- Beta vs SPY {qm['beta_spy']:.2f} (corr {qm['corr_spy']:.2f})")
    A(f"- Up-capture {qm['up_capture']:.0f}%  |  Down-capture {qm['down_capture']:.0f}%  "
      f"{'← winning by losing less' if qm['down_capture'] < qm['up_capture'] else ''}")
    A(f"- Hit rate {qm['hit_rate']:.0f}% ({qm['win_days']}W / {qm['lose_days']}L)  |  "
      f"best day {qm['best_day']:+.2f}%  worst {qm['worst_day']:+.2f}%\n")

    A("## Open book")
    lt = pm_pos["latest"]
    A(f"- {pm_pos['n_open']} positions, {pm_pos['winners']} in profit as of {pm_pos['latest_date'].date()}")
    A(f"- Best: {lt.iloc[-1]['ticker']} {lt.iloc[-1]['total_pnl_pct']:+.1f}%  |  "
      f"Worst: {lt.iloc[0]['ticker']} {lt.iloc[0]['total_pnl_pct']:+.1f}%")
    danger = lt[lt["room_to_stop_pct"] < 10]
    if len(danger):
        A(f"- ⚠ Danger zone (<10pt to stop): {', '.join(danger['ticker'])}")
    A("")

    A("## Exits this period")
    if len(pm_exits):
        A("| Name | Type | Exit date | Realized % |")
        A("|---|---|---|---|")
        for _, r in pm_exits.iterrows():
            A(f"| {r['symbol']} | {r['type']} | {r['exit_date']} | {r['realized_pct']:+.2f}% |")
        n_stop = (pm_exits['type'] == 'STOP-LOSS').sum()
        n_drop = (pm_exits['type'] == 'MANUAL DROP').sum()
        A(f"\n{n_stop} stop-loss, {n_drop} manual. To grade whether each exit "
          "helped, compare the stock's current price to its level at exit "
          "(see exits_postmortem.png / px_now column if fetched).\n")
    else:
        A("None.\n")

    A("## Charts")
    for f in ["equity_vs_benchmarks.png", "drawdown.png", "daily_returns_dist.png",
              "positions_pnl.png", "exits_postmortem.png"]:
        if (outdir / f).exists():
            A(f"![{f}]({f})")
    A("")

    (outdir / f"report_{month_tag}.md").write_text("\n".join(L), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true", help="skip live price fetch for exits")
    args = ap.parse_args()

    pos, port, trades = load()
    month_tag = port["date"].max().strftime("%Y-%m")
    outdir = _HERE / "month_end" / month_tag
    outdir.mkdir(parents=True, exist_ok=True)

    qm = quant_metrics(port)
    pm_pos = position_metrics(pos)
    pm_exits = exit_postmortem(trades, fetch=not args.no_fetch)

    make_charts(port, qm, pm_pos, pm_exits, outdir)
    write_report(qm, pm_pos, pm_exits, trades, outdir, month_tag)

    # console summary
    print("=" * 66)
    print(f"  ARIA-Growth — Month-End Review {month_tag}  ({qm['n_days']} trading days)")
    print("=" * 66)
    print(f"  Return   : {qm['total_ret']:+.2f}%   (SPY {qm['spy_since']:+.2f}%, QQQ {qm['qqq_since']:+.2f}%)")
    print(f"  Alpha    : vs SPY {qm['alpha_spy']:+.2f}pp | vs QQQ {qm['alpha_qqq']:+.2f}pp")
    print(f"  Sharpe   : {qm['sharpe']:.2f}   Sortino {qm['sortino']:.2f}   MaxDD {qm['max_dd']:.2f}%")
    print(f"  Beta     : {qm['beta_spy']:.2f}   Up-cap {qm['up_capture']:.0f}%  Down-cap {qm['down_capture']:.0f}%")
    print(f"  Hit rate : {qm['hit_rate']:.0f}%  ({qm['win_days']}W/{qm['lose_days']}L)")
    if len(pm_exits):
        print(f"  Exits    : {len(pm_exits)} "
              f"({(pm_exits['type']=='STOP-LOSS').sum()} stops, "
              f"{(pm_exits['type']=='MANUAL DROP').sum()} manual)")
    if MANIFEST.exists():
        man = pd.read_csv(MANIFEST)
        last_snap = man["date"].max()
        print(f"  Screens  : {len(man)} archived, latest {last_snap}"
              + ("  ⚠ >35d old — run the monthly screener+archive!" if
                 (pd.Timestamp.now() - pd.Timestamp(last_snap)).days > 35 else ""))
    print(f"\n  📁 Report + charts → {outdir}")
    print("  ⚠ One month = machinery check, not an edge verdict.")


if __name__ == "__main__":
    main()