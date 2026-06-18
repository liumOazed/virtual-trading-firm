"""
Growth Portfolio Report — Full Metrics
=======================================
Reads 3 CSV files and prints all metrics to console + saves a chart.

Requirements:
    pip install pandas numpy matplotlib

Usage:
    python portfolio_report.py

Expected files in same directory:
    daily_positions.csv   — daily snapshot of each open position
    daily_portfolio.csv   — daily portfolio-level summary + benchmark data
    growth_trade_log.csv  — every trade action (entry, stop-loss, redistribute)

Expected columns:
    daily_positions  : date, timestamp, ticker, entry_price, current_price, qty,
                       market_value, day_change_pct, total_pnl_usd, total_pnl_pct,
                       days_held, room_to_stop_pct, regime
    daily_portfolio  : date, timestamp, regime, spy_trend_pct, spy_vol_pct,
                       spy_drawdown_pct, equity, cash, total_pnl_pct, day_change_pct,
                       spy_day_pct, spy_since_pct, qqq_day_pct, qqq_since_pct,
                       n_positions, n_sectors
    growth_trade_log : timestamp, regime, symbol, side, notional, reason, result
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent
POS_FILE   = str(_HERE / "daily_positions.csv")
PORT_FILE  = str(_HERE / "daily_portfolio.csv")
TRADE_FILE = str(_HERE / "growth_trade_log.csv")
_REPORT_DIR = _HERE / "report"
_REPORT_DIR.mkdir(exist_ok=True)
OUTPUT_PNG = str(_REPORT_DIR / "portfolio_report.png")
STYLE      = "seaborn-v0_8-whitegrid"

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN  = "#1D9E75"
AMBER  = "#EF9F27"
RED    = "#E24B4A"
BLUE   = "#378ADD"
NAVY   = "#1A3A5C"
GREY   = "#888780"
LGREY  = "#F4F6F9"
MGREY  = "#D1D5DB"
MUTC   = "#6B7280"


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD & CLEAN
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    pos   = pd.read_csv(POS_FILE)
    port  = pd.read_csv(PORT_FILE)
    trades= pd.read_csv(TRADE_FILE)

    pos["date"]       = pd.to_datetime(pos["date"])
    port["date"]      = pd.to_datetime(port["date"])
    trades["timestamp"] = pd.to_datetime(trades["timestamp"])

    # pnl_pct in positions is stored as a decimal (e.g. 0.0297 = 2.97%)
    pos["total_pnl_pct_real"] = pos["total_pnl_pct"] * 100

    return pos, port, trades


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  ALL METRIC CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calc_all_metrics(pos, port, trades):
    m = {}

    # Latest date
    latest_date = pos["date"].max()
    latest      = pos[pos["date"] == latest_date].copy()
    latest_port = port[port["date"] == latest_date].iloc[0]
    m["latest_date"] = latest_date

    # ── 2.1 Portfolio-level summary ──────────────────────────────────────────
    m["equity"]          = latest_port["equity"]
    m["cash"]            = latest_port["cash"]
    m["cash_pct"]        = latest_port["cash"] / latest_port["equity"] * 100
    m["total_pnl_pct"]   = latest_port["total_pnl_pct"]
    m["regime"]          = latest_port["regime"]
    m["spy_trend_pct"]   = latest_port["spy_trend_pct"]
    m["spy_drawdown_pct"]= latest_port["spy_drawdown_pct"]
    m["n_positions"]     = latest_port["n_positions"]
    m["spy_since_pct"]   = latest_port["spy_since_pct"]
    m["qqq_since_pct"]   = latest_port["qqq_since_pct"]

    # ── 2.2 Daily portfolio time series ─────────────────────────────────────
    m["daily"] = port[["date","equity","cash","total_pnl_pct","day_change_pct",
                        "spy_day_pct","qqq_day_pct","spy_since_pct","qqq_since_pct",
                        "regime","spy_trend_pct","spy_vol_pct","spy_drawdown_pct",
                        "n_positions"]].copy()

    # ── 2.3 Benchmark comparison ─────────────────────────────────────────────
    # Index to 100 on day 0 (entry day, equity before first market day)
    starting_equity = port["equity"].iloc[0] / (1 + port["day_change_pct"].iloc[0] / 100)
    m["indexed_equity"] = (port["equity"] / starting_equity * 100).values
    # SPY / QQQ indexed: cumulative product of daily returns
    m["indexed_spy"]  = [100.0]
    m["indexed_qqq"]  = [100.0]
    for _, row in port.iterrows():
        m["indexed_spy"].append(m["indexed_spy"][-1] * (1 + row["spy_day_pct"] / 100))
        m["indexed_qqq"].append(m["indexed_qqq"][-1] * (1 + row["qqq_day_pct"] / 100))
    # trim to same length as port rows
    m["indexed_spy"] = m["indexed_spy"][1:]
    m["indexed_qqq"] = m["indexed_qqq"][1:]

    # Alpha vs each benchmark (cumulative)
    m["alpha_vs_spy"] = m["total_pnl_pct"] - m["spy_since_pct"]
    m["alpha_vs_qqq"] = m["total_pnl_pct"] - m["qqq_since_pct"]

    # ── 2.4 Daily win/loss count ─────────────────────────────────────────────
    m["daily_wl"] = (
        pos.groupby("date")
        .apply(lambda g: pd.Series({
            "winners": (g["total_pnl_usd"] > 0).sum(),
            "losers":  (g["total_pnl_usd"] < 0).sum(),
            "total":   len(g),
            "net_pnl": g["total_pnl_usd"].sum(),
        }))
        .reset_index()
    )

    # ── 2.5 Latest positions snapshot ────────────────────────────────────────
    m["latest_positions"] = latest.sort_values("total_pnl_usd", ascending=False)

    # ── 2.6 Position sizing stats ─────────────────────────────────────────────
    latest["weight_pct"] = latest["market_value"] / m["equity"] * 100
    m["weight_mean"]  = latest["weight_pct"].mean()
    m["weight_std"]   = latest["weight_pct"].std()
    m["weight_min"]   = latest["weight_pct"].min()
    m["weight_max"]   = latest["weight_pct"].max()
    m["heaviest_pos"] = latest.loc[latest["weight_pct"].idxmax(), "ticker"]
    m["lightest_pos"] = latest.loc[latest["weight_pct"].idxmin(), "ticker"]
    m["latest_positions"]["weight_pct"] = latest["weight_pct"]

    # ── 2.7 P&L distribution in latest snapshot ──────────────────────────────
    m["n_winners"]    = (latest["total_pnl_usd"] > 0).sum()
    m["n_losers"]     = (latest["total_pnl_usd"] < 0).sum()
    m["best_ticker"]  = latest.loc[latest["total_pnl_pct_real"].idxmax(), "ticker"]
    m["worst_ticker"] = latest.loc[latest["total_pnl_pct_real"].idxmin(), "ticker"]
    m["best_pct"]     = latest["total_pnl_pct_real"].max()
    m["worst_pct"]    = latest["total_pnl_pct_real"].min()
    m["total_unrealized_pnl"] = latest["total_pnl_usd"].sum()

    # ── 2.8 Stop-loss proximity ──────────────────────────────────────────────
    m["stop_proximity"] = latest.sort_values("room_to_stop_pct")[
        ["ticker","room_to_stop_pct","total_pnl_pct_real","total_pnl_usd","current_price"]
    ]
    m["danger_zone"]   = latest[latest["room_to_stop_pct"] < 10]   # <10% room
    m["caution_zone"]  = latest[(latest["room_to_stop_pct"] >= 10) &
                                 (latest["room_to_stop_pct"] < 15)]

    # ── 2.9 Per-position volatility (std of daily day_change_pct) ────────────
    m["position_vol"] = (
        pos.groupby("ticker")["day_change_pct"]
        .std()
        .sort_values(ascending=False)
        .rename("daily_vol_std")
    )

    # ── 2.10 Per-position timeline (every ticker × date) ────────────────────
    m["position_timeline"] = pos.sort_values(["ticker","date"])[
        ["ticker","date","entry_price","current_price","total_pnl_usd",
         "total_pnl_pct","day_change_pct","room_to_stop_pct","days_held"]
    ]

    # ── 2.11 Trade log analysis ──────────────────────────────────────────────
    m["trades_all"]    = trades
    m["entries"]       = trades[trades["side"] == "buy"]
    m["stops"]         = trades[trades["side"] == "close"]
    m["redistributions"] = trades[trades["reason"].str.contains("redistribute", na=False)]

    m["n_entries"]     = len(m["entries"][~m["entries"]["reason"].str.contains("redistribute", na=False)])
    m["n_stops"]       = len(m["stops"])
    m["n_redist"]      = len(m["redistributions"])
    m["total_deployed"]= m["entries"][~m["entries"]["reason"].str.contains("redistribute", na=False)]["notional"].sum()
    m["stop_loss_recovered"] = m["stops"]["notional"].sum()

    # ── 2.12 Redistribution detail ───────────────────────────────────────────
    m["redist_detail"] = (
        m["redistributions"]
        .groupby("symbol")["notional"]
        .sum()
        .sort_values(ascending=False)
    )
    m["redist_total"]  = m["redistributions"]["notional"].sum()

    # ── 2.13 SMCI stop-loss event detail ─────────────────────────────────────
    smci_entry  = trades[(trades["symbol"]=="SMCI") & (trades["side"]=="buy")].iloc[0]
    smci_exit   = trades[(trades["symbol"]=="SMCI") & (trades["side"]=="close")].iloc[0]
    m["smci_entry_notional"] = smci_entry["notional"]
    m["smci_exit_notional"]  = smci_exit["notional"]
    m["smci_realized_loss"]  = smci_exit["notional"] - smci_entry["notional"]
    m["smci_realized_pct"]   = m["smci_realized_loss"] / smci_entry["notional"] * 100
    m["smci_entry_date"]     = smci_entry["timestamp"]
    m["smci_exit_date"]      = smci_exit["timestamp"]

    # ── 2.14 APP deep dive (worst current position) ──────────────────────────
    m["app_timeline"] = pos[pos["ticker"]=="APP"].sort_values("date")[
        ["date","entry_price","current_price","total_pnl_usd",
         "total_pnl_pct","day_change_pct","room_to_stop_pct"]
    ]
    m["app_min_room"]  = m["app_timeline"]["room_to_stop_pct"].min()
    m["app_entry"]     = m["app_timeline"].iloc[0]["entry_price"]

    # ── 2.15 Beta-like: daily corr with SPY ──────────────────────────────────
    merged = port[["day_change_pct","spy_day_pct"]].dropna()
    if len(merged) > 1:
        m["beta_vs_spy"]  = np.polyfit(merged["spy_day_pct"],
                                        merged["day_change_pct"], 1)[0]
        m["corr_vs_spy"]  = merged["spy_day_pct"].corr(merged["day_change_pct"])
    else:
        m["beta_vs_spy"]  = np.nan
        m["corr_vs_spy"]  = np.nan

    # ── 2.16 Up-day vs down-day asymmetry ───────────────────────────────────
    up_days   = port[port["spy_day_pct"] > 0]
    down_days = port[port["spy_day_pct"] < 0]
    m["up_day_avg_port"]  = up_days["day_change_pct"].mean() if len(up_days)   else np.nan
    m["up_day_avg_spy"]   = up_days["spy_day_pct"].mean()    if len(up_days)   else np.nan
    m["dn_day_avg_port"]  = down_days["day_change_pct"].mean() if len(down_days) else np.nan
    m["dn_day_avg_spy"]   = down_days["spy_day_pct"].mean()    if len(down_days) else np.nan

    return m


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PRINT ALL METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(m):
    sep = "─" * 60
    print(f"\n{sep}")
    print("  GROWTH PORTFOLIO — FULL METRICS REPORT")
    print(f"  Snapshot date: {m['latest_date'].date()}")
    print(sep)

    print("\n【PORTFOLIO SUMMARY】")
    print(f"  Equity             : ${m['equity']:,.2f}")
    print(f"  Cash               : ${m['cash']:,.2f}  ({m['cash_pct']:.1f}%)")
    print(f"  Total return       : {m['total_pnl_pct']:.2f}%")
    print(f"  vs SPY since entry : {m['alpha_vs_spy']:+.2f}pp  (SPY {m['spy_since_pct']:+.2f}%)")
    print(f"  vs QQQ since entry : {m['alpha_vs_qqq']:+.2f}pp  (QQQ {m['qqq_since_pct']:+.2f}%)")
    print(f"  Regime             : {m['regime']}")
    print(f"  SPY trend          : {m['spy_trend_pct']:.1f}%")
    print(f"  SPY drawdown       : {m['spy_drawdown_pct']:.2f}%")
    print(f"  Open positions     : {int(m['n_positions'])}")
    print(f"  Unrealized P&L     : ${m['total_unrealized_pnl']:,.2f}")

    print("\n【DAILY PERFORMANCE】")
    d = m["daily"].copy()
    d["port_vs_spy"] = d["day_change_pct"] - d["spy_day_pct"]
    print(d[["date","day_change_pct","spy_day_pct","qqq_day_pct","port_vs_spy",
              "total_pnl_pct","equity"]].round(2).to_string(index=False))

    print("\n【DAILY WIN / LOSS COUNT】")
    print(m["daily_wl"].round(2).to_string(index=False))

    print("\n【BENCHMARK COMPARISON (daily returns)】")
    print(f"  Beta vs SPY       : {m['beta_vs_spy']:.3f}  (correlation {m['corr_vs_spy']:.3f})")
    print(f"  Up-day (SPY>0):  portfolio avg {m['up_day_avg_port']:+.2f}%  |  SPY avg {m['up_day_avg_spy']:+.2f}%")
    print(f"  Down-day (SPY<0): portfolio avg {m['dn_day_avg_port']:+.2f}%  |  SPY avg {m['dn_day_avg_spy']:+.2f}%")

    print("\n【POSITION SIZING】")
    print(f"  Target weight     : 5.00%")
    print(f"  Actual range      : {m['weight_min']:.2f}% – {m['weight_max']:.2f}%")
    print(f"  Std dev of weights: {m['weight_std']:.2f}pp")
    print(f"  Heaviest position : {m['heaviest_pos']} ({m['weight_max']:.2f}%)")
    print(f"  Lightest position : {m['lightest_pos']} ({m['weight_min']:.2f}%)")

    print("\n【LATEST POSITIONS SNAPSHOT】")
    cols = ["ticker","entry_price","current_price","market_value","weight_pct",
            "total_pnl_usd","total_pnl_pct_real","room_to_stop_pct","days_held","day_change_pct"]
    print(m["latest_positions"][cols].round(2).to_string(index=False))

    print("\n【P&L SUMMARY】")
    print(f"  Winners     : {m['n_winners']} / {m['n_winners']+m['n_losers']}")
    print(f"  Losers      : {m['n_losers']} / {m['n_winners']+m['n_losers']}")
    print(f"  Best ticker : {m['best_ticker']}  ({m['best_pct']:+.2f}%)")
    print(f"  Worst ticker: {m['worst_ticker']}  ({m['worst_pct']:+.2f}%)")

    print("\n【STOP-LOSS PROXIMITY (sorted closest first)】")
    print(m["stop_proximity"].round(2).to_string(index=False))
    print(f"\n  Danger zone (<10% room): {list(m['danger_zone']['ticker'])}")
    print(f"  Caution zone (10-15%):   {list(m['caution_zone']['ticker'])}")

    print("\n【POSITION VOLATILITY (std of daily % change)】")
    print(m["position_vol"].round(3).to_string())

    print("\n【PER-POSITION TIMELINE】")
    print(m["position_timeline"].round(3).to_string(index=False))

    print("\n【APP DEEP DIVE (worst current position)】")
    print(f"  Entry price: ${m['app_entry']:.2f}")
    print(f"  Min room to stop ever: {m['app_min_room']:.2f}%")
    print(m["app_timeline"].round(3).to_string(index=False))

    print("\n【TRADE LOG SUMMARY】")
    print(f"  Initial entries   : {m['n_entries']} trades  (${m['total_deployed']:,.2f} deployed)")
    print(f"  Stop-loss exits   : {m['n_stops']}")
    print(f"  Redistribute buys : {m['n_redist']}")

    print("\n【SMCI STOP-LOSS EVENT】")
    print(f"  Entry : {m['smci_entry_date'].date()}  ${m['smci_entry_notional']:,.2f}")
    print(f"  Exit  : {m['smci_exit_date'].date()}   ${m['smci_exit_notional']:,.2f}")
    print(f"  Realized loss: ${m['smci_realized_loss']:,.2f}  ({m['smci_realized_pct']:.2f}%)")
    print(f"  Proceeds redistributed: ${m['redist_total']:,.2f}")

    print("\n【REDISTRIBUTION ALLOCATION】")
    print(m["redist_detail"].round(2).to_string())

    print(f"\n{sep}")
    print("  END OF REPORT")
    print(sep + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def make_plots(m):
    plt.style.use(STYLE)
    fig = plt.figure(figsize=(20, 24), facecolor="white")
    fig.suptitle("Growth Portfolio — 4-Day Performance Report",
                  fontsize=18, fontweight="bold", color=NAVY, y=0.99)

    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.55, wspace=0.38,
                            left=0.06, right=0.97, top=0.96, bottom=0.04)

    def style_ax(ax, title, xlabel="", ylabel=""):
        ax.set_title(title, fontsize=11, fontweight="bold", color=NAVY, pad=8)
        ax.set_xlabel(xlabel, fontsize=9, color=MUTC)
        ax.set_ylabel(ylabel, fontsize=9, color=MUTC)
        ax.tick_params(colors=MUTC, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(MGREY)

    # ── ROW 0: KPI CARDS ────────────────────────────────────────────────────
    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.set_facecolor(LGREY)
    ax_kpi.set_xlim(0,1); ax_kpi.set_ylim(0,1); ax_kpi.axis("off")
    ax_kpi.set_title("Summary", fontsize=11, fontweight="bold", color=NAVY, pad=6)

    kpis = [
        ("Equity",       f"${m['equity']:,.0f}",     NAVY),
        ("Total return", f"{m['total_pnl_pct']:.2f}%",
             GREEN if m["total_pnl_pct"] >= 0 else RED),
        ("vs SPY",       f"{m['alpha_vs_spy']:+.2f}pp",
             GREEN if m["alpha_vs_spy"] >= 0 else RED),
        ("vs QQQ",       f"{m['alpha_vs_qqq']:+.2f}pp",
             GREEN if m["alpha_vs_qqq"] >= 0 else RED),
        ("Positions",    str(int(m["n_positions"])),  NAVY),
        ("Cash",         f"${m['cash']:,.0f}",        NAVY),
        ("Win/Loss",     f"{m['n_winners']} / {m['n_losers']}", AMBER),
        ("Regime",       m["regime"],                  GREEN),
    ]
    xs = np.linspace(0.05, 0.95, len(kpis))
    for x, (lbl, val, col) in zip(xs, kpis):
        ax_kpi.text(x, 0.72, val,  ha="center", va="center",
                    fontsize=13, fontweight="bold", color=col)
        ax_kpi.text(x, 0.28, lbl, ha="center", va="center",
                    fontsize=8, color=MUTC)

    # ── ROW 1: Equity curve | Win-loss daily | Stop proximity ─────────────
    dates = m["daily"]["date"].dt.strftime("%b %d").tolist()

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(dates, m["indexed_equity"],  label="Portfolio", color=BLUE,  linewidth=2,  marker="o", markersize=4)
    ax1.plot(dates, m["indexed_spy"],     label="SPY",       color=GREY,  linewidth=1.5, linestyle="--", marker="s", markersize=3)
    ax1.plot(dates, m["indexed_qqq"],     label="QQQ",       color=AMBER, linewidth=1.5, linestyle=":",  marker="^", markersize=3)
    ax1.legend(fontsize=8, loc="lower left")
    ax1.axhline(100, color=MGREY, linewidth=0.8, linestyle="--")
    style_ax(ax1, "Equity vs SPY / QQQ (indexed to 100)", ylabel="Indexed value")

    ax2 = fig.add_subplot(gs[1, 1])
    wl = m["daily_wl"]
    x2 = np.arange(len(wl))
    ax2.bar(x2, wl["winners"], label="Winners", color=GREEN, width=0.5)
    ax2.bar(x2, wl["losers"],  label="Losers",  color=RED,   width=0.5, bottom=wl["winners"])
    ax2.set_xticks(x2)
    ax2.set_xticklabels(wl["date"].dt.strftime("%b %d"), rotation=20, fontsize=7)
    ax2.legend(fontsize=8)
    style_ax(ax2, "Daily win / loss count", ylabel="Positions")

    ax3 = fig.add_subplot(gs[1, 2])
    sp = m["stop_proximity"].reset_index(drop=True)
    colors3 = [RED if v < 10 else AMBER if v < 15 else GREEN
                for v in sp["room_to_stop_pct"]]
    ax3.barh(sp["ticker"], sp["room_to_stop_pct"], color=colors3, height=0.6)
    ax3.axvline(10, color=RED,   linewidth=0.8, linestyle="--")
    ax3.axvline(15, color=AMBER, linewidth=0.8, linestyle="--")
    ax3.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:.0f}%"))
    style_ax(ax3, "Room to stop-loss (% below current price)", xlabel="Room %")

    # ── ROW 2: P&L by ticker | Position weights | Daily returns bar ────────
    ax4 = fig.add_subplot(gs[2, 0])
    lp = m["latest_positions"].sort_values("total_pnl_pct_real")
    colors4 = [GREEN if v > 0 else RED for v in lp["total_pnl_pct_real"]]
    ax4.barh(lp["ticker"], lp["total_pnl_pct_real"], color=colors4, height=0.6)
    ax4.axvline(0, color=MGREY, linewidth=0.8)
    ax4.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:.1f}%"))
    style_ax(ax4, "Unrealized P&L % (latest)", xlabel="P&L %")

    ax5 = fig.add_subplot(gs[2, 1])
    lp2 = m["latest_positions"].sort_values("weight_pct", ascending=False)
    ax5.barh(lp2["ticker"], lp2["weight_pct"], color=BLUE, height=0.6)
    ax5.axvline(5.0, color=AMBER, linewidth=1, linestyle="--", label="5% target")
    ax5.legend(fontsize=8)
    ax5.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:.1f}%"))
    style_ax(ax5, "Position weight (% of equity)", xlabel="Weight %")

    ax6 = fig.add_subplot(gs[2, 2])
    daily = m["daily"]
    x6 = np.arange(len(daily))
    w6 = 0.25
    ax6.bar(x6 - w6, daily["day_change_pct"], width=w6, color=BLUE,  label="Portfolio")
    ax6.bar(x6,      daily["spy_day_pct"],    width=w6, color=GREY,  label="SPY")
    ax6.bar(x6 + w6, daily["qqq_day_pct"],   width=w6, color=AMBER, label="QQQ")
    ax6.axhline(0, color=MGREY, linewidth=0.8)
    ax6.set_xticks(x6)
    ax6.set_xticklabels(daily["date"].dt.strftime("%b %d"), rotation=20, fontsize=7)
    ax6.legend(fontsize=8)
    ax6.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:.1f}%"))
    style_ax(ax6, "Daily return vs benchmarks", ylabel="Daily return %")

    # ── ROW 3: Position vol | APP timeline | Redistribution ───────────────
    ax7 = fig.add_subplot(gs[3, 0])
    vol = m["position_vol"].dropna()
    colors7 = [RED if v > 4 else AMBER if v > 2 else GREEN for v in vol.values]
    ax7.barh(vol.index, vol.values, color=colors7, height=0.6)
    ax7.axvline(vol.mean(), color=NAVY, linewidth=0.8, linestyle="--", label=f"Mean {vol.mean():.2f}%")
    ax7.legend(fontsize=8)
    style_ax(ax7, "Intra-portfolio daily volatility (std %)", xlabel="Std dev of daily moves")

    ax8 = fig.add_subplot(gs[3, 1])
    app = m["app_timeline"].copy()
    app["pnl_pct_real"] = app["total_pnl_pct"] * 100
    ax8.plot(app["date"].dt.strftime("%b %d"), app["pnl_pct_real"],
             color=RED, linewidth=2, marker="o", markersize=5, label="P&L %")
    ax8b = ax8.twinx()
    ax8b.plot(app["date"].dt.strftime("%b %d"), app["room_to_stop_pct"],
              color=AMBER, linewidth=1.5, linestyle="--", marker="s", markersize=4,
              label="Room to stop %")
    ax8b.axhline(0, color=RED, linewidth=0.6, linestyle=":")
    ax8b.tick_params(colors=MUTC, labelsize=8)
    ax8b.set_ylabel("Room to stop %", fontsize=9, color=AMBER)
    ax8.axhline(0, color=MGREY, linewidth=0.8)
    ax8.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:.1f}%"))
    ax8.legend(fontsize=8, loc="lower left")
    ax8b.legend(fontsize=8, loc="lower right")
    style_ax(ax8, "APP — P&L & room to stop over time", ylabel="P&L %")

    ax9 = fig.add_subplot(gs[3, 2])
    rd = m["redist_detail"].sort_values(ascending=True)
    colors9 = [RED if v > 400 else AMBER if v > 200 else BLUE for v in rd.values]
    ax9.barh(rd.index, rd.values, color=colors9, height=0.6)
    ax9.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f"${v:.0f}"))
    style_ax(ax9, f"SMCI proceeds redistribution (${m['redist_total']:,.0f} total)", xlabel="Amount $")

    # ── ROW 4: Equity $ | Cumulative since % | Regime context ─────────────
    ax10 = fig.add_subplot(gs[4, 0])
    daily = m["daily"]
    ax10.plot(daily["date"].dt.strftime("%b %d"), daily["equity"],
              color=BLUE, linewidth=2, marker="o", markersize=4)
    ax10.fill_between(daily["date"].dt.strftime("%b %d"), daily["equity"],
                       daily["equity"].max(), alpha=0.1, color=RED)
    ax10.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"${v/1000:.0f}K"))
    style_ax(ax10, "Portfolio equity ($)", ylabel="Equity")

    ax11 = fig.add_subplot(gs[4, 1])
    ax11.plot(daily["date"].dt.strftime("%b %d"), daily["total_pnl_pct"],
              label="Portfolio", color=BLUE, linewidth=2, marker="o", markersize=4)
    ax11.plot(daily["date"].dt.strftime("%b %d"), daily["spy_since_pct"],
              label="SPY",       color=GREY, linewidth=1.5, linestyle="--", marker="s", markersize=3)
    ax11.plot(daily["date"].dt.strftime("%b %d"), daily["qqq_since_pct"],
              label="QQQ",       color=AMBER, linewidth=1.5, linestyle=":", marker="^", markersize=3)
    ax11.axhline(0, color=MGREY, linewidth=0.8)
    ax11.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:.1f}%"))
    ax11.legend(fontsize=8)
    style_ax(ax11, "Cumulative since-entry return %", ylabel="Cumulative %")

    ax12 = fig.add_subplot(gs[4, 2])
    ax12.plot(daily["date"].dt.strftime("%b %d"), daily["spy_trend_pct"],
              label="SPY trend %", color=GREEN, linewidth=2, marker="o", markersize=4)
    ax12.plot(daily["date"].dt.strftime("%b %d"), daily["spy_vol_pct"],
              label="SPY vol %",   color=AMBER, linewidth=1.5, linestyle="--", marker="s", markersize=3)
    ax12b = ax12.twinx()
    ax12b.plot(daily["date"].dt.strftime("%b %d"), daily["spy_drawdown_pct"],
               label="SPY DD %",  color=RED, linewidth=1.5, linestyle=":", marker="^", markersize=3)
    ax12b.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:.1f}%"))
    ax12b.tick_params(colors=MUTC, labelsize=8)
    ax12.legend(fontsize=8, loc="upper left")
    ax12b.legend(fontsize=8, loc="upper right")
    style_ax(ax12, "Market regime context", ylabel="%")

    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✓  Chart saved → {OUTPUT_PNG}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading data ...")
    pos, port, trades = load_data()
    print(f"  Positions : {len(pos)} rows  ({pos['date'].nunique()} dates, {pos['ticker'].nunique()} tickers)")
    print(f"  Portfolio : {len(port)} daily rows")
    print(f"  Trades    : {len(trades)} events\n")

    metrics = calc_all_metrics(pos, port, trades)
    print_report(metrics)
    make_plots(metrics)