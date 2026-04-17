"""
metrics.py
==========
Professional quant tearsheet for VIRTUAL_TRADING_FIRM.
Reads equity_curve_v2.csv + trade_log_v2.csv and produces:
  - Full performance metrics (Sharpe, Sortino, Calmar, Omega, VaR, CVaR)
  - Per-regime breakdown
  - Trade-level P&L (computed from BUY/SELL pairs, no pnl_pct column needed)
  - Alpha decay summary (optional, reads alpha_decay.csv if present)
  - Clean console tearsheet  +  PNG chart
  - Returns a dict so RL agent can consume everything directly
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
from datetime import datetime
from typing import Dict, Optional, Tuple

warnings.filterwarnings("ignore")

RESULTS_DIR = "5_backtesting/results"
OUT_DIR     = "5_backtesting/results"


# ════════════════════════════════════════════════════════════════════════════
# 1.  TRADE P&L BUILDER  (reconstructs pnl from BUY/SELL pairs)
# ════════════════════════════════════════════════════════════════════════════

def build_trade_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Match BUY → SELL pairs per ticker (FIFO).
    Returns enriched DataFrame with columns:
        ticker, entry_date, exit_date, entry_price, exit_price,
        shares, gross_pnl, pnl_pct, days_held, regime
    """
    records = []
    # stack per ticker
    for ticker, grp in trades.groupby("ticker"):
        grp     = grp.sort_values("date").reset_index(drop=True)
        buy_q   = []   # queue of open lots

        for _, row in grp.iterrows():
            if row["action"] == "BUY":
                buy_q.append(row)
            elif row["action"] == "SELL" and buy_q:
                buy_row = buy_q.pop(0)          # FIFO
                entry   = float(buy_row["price"])
                exit_   = float(row["price"])
                shares  = float(buy_row["shares"])
                gross   = (exit_ - entry) * shares
                pnl_pct = (exit_ - entry) / entry if entry > 0 else 0.0
                d_held  = (pd.to_datetime(row["date"]) -
                           pd.to_datetime(buy_row["date"])).days

                records.append({
                    "ticker":       ticker,
                    "entry_date":   buy_row["date"],
                    "exit_date":    row["date"],
                    "entry_price":  round(entry,  4),
                    "exit_price":   round(exit_,  4),
                    "shares":       round(shares, 4),
                    "gross_pnl":    round(gross,  4),
                    "pnl_pct":      round(pnl_pct, 6),
                    "days_held":    d_held,
                    "regime":       buy_row.get("regime", "Unknown"),
                    "weight":       buy_row.get("weight",  0.0),
                })

    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════════════════
# 2.  CORE METRICS
# ════════════════════════════════════════════════════════════════════════════

def compute_core_metrics(eq: pd.DataFrame,
                         rf_annual: float = 0.05) -> Dict:
    """All metrics from equity curve. rf_annual = risk-free rate (5%)."""
    eq      = eq.copy().sort_values("date").reset_index(drop=True)
    equity  = eq["equity"].values
    rets    = pd.Series(equity).pct_change().fillna(0).values

    # ── returns ──────────────────────────────────────────────────────────
    total_ret   = (equity[-1] - equity[0]) / equity[0]
    days        = max((pd.to_datetime(eq["date"].iloc[-1]) -
                       pd.to_datetime(eq["date"].iloc[0])).days, 1)
    ann_ret     = (1 + total_ret) ** (365.25 / days) - 1
    ann_vol     = rets.std() * np.sqrt(252)
    rf_daily    = (1 + rf_annual) ** (1/252) - 1

    # ── Sharpe ───────────────────────────────────────────────────────────
    excess      = rets - rf_daily
    sharpe      = (excess.mean() / rets.std() * np.sqrt(252)
                   if rets.std() > 1e-9 else 0.0)

    # ── Sortino ──────────────────────────────────────────────────────────
    down        = rets[rets < rf_daily]
    down_std    = down.std() * np.sqrt(252) if len(down) > 1 else 1e-9
    sortino     = (ann_ret - rf_annual) / down_std if down_std > 1e-9 else 0.0

    # ── Drawdown ─────────────────────────────────────────────────────────
    peak        = np.maximum.accumulate(equity)
    dd          = (equity - peak) / np.where(peak > 0, peak, 1)
    max_dd      = dd.min()
    calmar      = ann_ret / abs(max_dd) if max_dd < 0 else 0.0

    # Avg drawdown duration (bars in drawdown)
    in_dd       = dd < 0
    dd_dur      = []
    cur         = 0
    for v in in_dd:
        if v:
            cur += 1
        elif cur > 0:
            dd_dur.append(cur); cur = 0
    avg_dd_dur  = np.mean(dd_dur) if dd_dur else 0

    # ── Omega ────────────────────────────────────────────────────────────
    gains       = rets[rets > 0].sum()
    losses      = abs(rets[rets < 0].sum())
    omega       = gains / losses if losses > 1e-9 else np.inf

    # ── VaR / CVaR  (95%) ───────────────────────────────────────────────
    var_95      = float(np.percentile(rets, 5))
    cvar_95     = float(rets[rets <= var_95].mean()) if (rets <= var_95).any() else var_95

    # ── Rolling 30-day Sharpe ────────────────────────────────────────────
    r           = pd.Series(rets)
    roll_sh     = (r.rolling(30).mean() / r.rolling(30).std()) * np.sqrt(252)
    worst_30d   = float(roll_sh.min())
    best_30d    = float(roll_sh.max())

    # ── Hit rate (daily) ─────────────────────────────────────────────────
    daily_hit   = float((rets > 0).mean())

    return {
        "total_return":      round(total_ret * 100, 2),
        "annualized_return": round(ann_ret   * 100, 2),
        "annualized_vol":    round(ann_vol   * 100, 2),
        "sharpe":            round(sharpe,   3),
        "sortino":           round(sortino,  3),
        "calmar":            round(calmar,   3),
        "omega":             round(omega,    3),
        "max_drawdown":      round(max_dd   * 100, 2),
        "avg_dd_duration":   round(avg_dd_dur, 1),
        "var_95":            round(var_95   * 100, 4),
        "cvar_95":           round(cvar_95  * 100, 4),
        "worst_30d_sharpe":  round(worst_30d, 3),
        "best_30d_sharpe":   round(best_30d,  3),
        "daily_hit_rate":    round(daily_hit * 100, 2),
        "days_traded":       days,
    }


# ════════════════════════════════════════════════════════════════════════════
# 3.  TRADE METRICS
# ════════════════════════════════════════════════════════════════════════════

def compute_trade_metrics(pnl_df: pd.DataFrame) -> Dict:
    if pnl_df.empty:
        return {}

    wins    = pnl_df[pnl_df["pnl_pct"] > 0]
    losses  = pnl_df[pnl_df["pnl_pct"] <= 0]

    win_rate        = len(wins) / len(pnl_df) if len(pnl_df) > 0 else 0
    avg_win         = wins["pnl_pct"].mean()    if not wins.empty   else 0
    avg_loss        = losses["pnl_pct"].mean()  if not losses.empty else 0
    payoff_ratio    = abs(avg_win / avg_loss)   if avg_loss != 0    else np.inf

    gross_profit    = wins["gross_pnl"].sum()
    gross_loss      = abs(losses["gross_pnl"].sum())
    profit_factor   = gross_profit / gross_loss if gross_loss > 0 else np.inf

    avg_hold        = pnl_df["days_held"].mean()
    best_trade      = pnl_df["pnl_pct"].max()
    worst_trade     = pnl_df["pnl_pct"].min()
    total_gross_pnl = pnl_df["gross_pnl"].sum()

    # Expectancy per trade
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    return {
        "round_trips":      len(pnl_df),
        "win_rate":         round(win_rate * 100, 2),
        "avg_win_pct":      round(avg_win  * 100, 4),
        "avg_loss_pct":     round(avg_loss * 100, 4),
        "payoff_ratio":     round(payoff_ratio, 3),
        "profit_factor":    round(profit_factor, 3),
        "expectancy_pct":   round(expectancy * 100, 4),
        "avg_hold_days":    round(avg_hold, 1),
        "best_trade_pct":   round(best_trade  * 100, 2),
        "worst_trade_pct":  round(worst_trade * 100, 2),
        "total_gross_pnl":  round(total_gross_pnl, 2),
    }


# ════════════════════════════════════════════════════════════════════════════
# 4.  PER-REGIME BREAKDOWN
# ════════════════════════════════════════════════════════════════════════════

def compute_regime_metrics(eq: pd.DataFrame,
                           pnl_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, grp in eq.groupby("regime"):
        rets    = grp["equity"].pct_change().dropna()
        ann_ret = rets.mean() * 252 * 100
        sharpe  = (rets.mean() / rets.std() * np.sqrt(252)
                   if rets.std() > 1e-9 else 0)
        peak    = grp["equity"].cummax()
        dd      = ((grp["equity"] - peak) / peak).min() * 100

        # trade win rate in this regime
        regime_trades = pnl_df[pnl_df["regime"] == regime] if not pnl_df.empty else pd.DataFrame()
        wr = (len(regime_trades[regime_trades["pnl_pct"] > 0]) /
              len(regime_trades) * 100) if len(regime_trades) > 0 else 0

        rows.append({
            "regime":       regime,
            "bars":         len(grp),
            "ann_ret_%":    round(ann_ret, 2),
            "sharpe":       round(sharpe, 3),
            "max_dd_%":     round(dd,     2),
            "trades":       len(regime_trades),
            "win_rate_%":   round(wr,     2),
        })
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


# ════════════════════════════════════════════════════════════════════════════
# 5.  BENCHMARK COMPARISON
# ════════════════════════════════════════════════════════════════════════════

def fetch_benchmark(start: str, end: str,
                    ticker: str = "SPY") -> Optional[pd.Series]:
    try:
        df = yf.download(ticker, start=start, end=end,
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        close = df["Close"].squeeze()
        return (close / close.iloc[0] * 100)
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════════
# 6.  CHART  (4-panel professional tearsheet)
# ════════════════════════════════════════════════════════════════════════════

REGIME_COLORS = {
    "Bull-Trending":  "#2ecc71",
    "Bull-MeanRev":   "#27ae60",
    "Bear-Trending":  "#e74c3c",
    "Bear-MeanRev":   "#c0392b",
}

def plot_tearsheet(eq: pd.DataFrame,
                   pnl_df: pd.DataFrame,
                   core: Dict,
                   regime_df: pd.DataFrame,
                   benchmark: Optional[pd.Series] = None,
                   save_path: str = f"{OUT_DIR}/tearsheet.png"):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dates   = pd.to_datetime(eq["date"])
    equity  = eq["equity"].values
    rets    = pd.Series(equity).pct_change().fillna(0)

    fig     = plt.figure(figsize=(16, 11), facecolor="#0d1117")
    gs      = gridspec.GridSpec(3, 3, figure=fig,
                                hspace=0.42, wspace=0.35)

    ax_eq   = fig.add_subplot(gs[0, :])     # equity curve (full width)
    ax_dd   = fig.add_subplot(gs[1, :2])    # drawdown
    ax_ret  = fig.add_subplot(gs[1, 2])     # return distribution
    ax_reg  = fig.add_subplot(gs[2, :2])    # regime bar chart
    ax_tbl  = fig.add_subplot(gs[2, 2])     # metrics table

    TEXT    = "#e0e0e0"
    GRID    = "#1f2937"

    for ax in [ax_eq, ax_dd, ax_ret, ax_reg, ax_tbl]:
        ax.set_facecolor("#111827")
        ax.tick_params(colors=TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)

    # ── Panel 1: Equity curve ─────────────────────────────────────────────
    eq_norm = equity / equity[0] * 100
    ax_eq.plot(dates, eq_norm, color="#3b82f6", lw=1.5, label="Strategy")

    if benchmark is not None:
        bench_dates = benchmark.index
        ax_eq.plot(bench_dates, benchmark.values,
                   color="#6b7280", lw=1.0, ls="--", alpha=0.7, label="SPY")

    # shade regimes
    if "regime" in eq.columns:
        prev_regime = eq["regime"].iloc[0]
        seg_start   = dates.iloc[0]
        for i in range(1, len(eq)):
            r = eq["regime"].iloc[i]
            if r != prev_regime or i == len(eq) - 1:
                color = REGIME_COLORS.get(prev_regime, "#374151")
                ax_eq.axvspan(seg_start, dates.iloc[i],
                              alpha=0.08, color=color, lw=0)
                prev_regime = r
                seg_start   = dates.iloc[i]

    ax_eq.set_title("Equity Curve vs Benchmark", color=TEXT, fontsize=10, pad=8)
    ax_eq.set_ylabel("Normalised (base 100)", color=TEXT, fontsize=8)
    ax_eq.legend(fontsize=8, facecolor="#1f2937", labelcolor=TEXT,
                 framealpha=0.8, loc="upper left")
    ax_eq.grid(True, color=GRID, lw=0.5)
    ax_eq.tick_params(axis="x", labelrotation=30)

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / np.where(peak > 0, peak, 1) * 100
    ax_dd.fill_between(dates, dd, 0, color="#ef4444", alpha=0.6)
    ax_dd.plot(dates, dd, color="#ef4444", lw=0.8)
    ax_dd.set_title("Drawdown %", color=TEXT, fontsize=10, pad=8)
    ax_dd.set_ylabel("%", color=TEXT, fontsize=8)
    ax_dd.grid(True, color=GRID, lw=0.5)
    ax_dd.tick_params(axis="x", labelrotation=30)

    # ── Panel 3: Return distribution ─────────────────────────────────────
    ax_ret.hist(rets * 100, bins=50, color="#8b5cf6", alpha=0.8,
                edgecolor="none", density=True)
    ax_ret.axvline(rets.mean() * 100, color="#facc15", lw=1.2,
                   ls="--", label=f"Mean {rets.mean()*100:.3f}%")
    ax_ret.axvline(core["var_95"], color="#f87171", lw=1.2,
                   ls=":", label=f"VaR95 {core['var_95']:.2f}%")
    ax_ret.set_title("Daily Return Distribution", color=TEXT, fontsize=10, pad=8)
    ax_ret.set_xlabel("%", color=TEXT, fontsize=8)
    ax_ret.legend(fontsize=7, facecolor="#1f2937", labelcolor=TEXT)
    ax_ret.grid(True, color=GRID, lw=0.5)

    # ── Panel 4: Per-regime Sharpe bar ───────────────────────────────────
    if not regime_df.empty:
        labels  = regime_df["regime"].values
        sharpes = regime_df["sharpe"].values
        colors  = [REGIME_COLORS.get(r, "#4b5563") for r in labels]
        bars    = ax_reg.barh(labels, sharpes, color=colors, height=0.55)
        ax_reg.axvline(0, color=TEXT, lw=0.8, alpha=0.5)
        for bar, val in zip(bars, sharpes):
            ax_reg.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                        f"{val:.2f}", va="center", color=TEXT, fontsize=7)
        ax_reg.set_title("Sharpe by Regime", color=TEXT, fontsize=10, pad=8)
        ax_reg.set_xlabel("Sharpe", color=TEXT, fontsize=8)
        ax_reg.grid(True, color=GRID, lw=0.5, axis="x")

    # ── Panel 5: Metrics table ────────────────────────────────────────────
    ax_tbl.axis("off")
    metrics_display = [
        ("Total Return",    f"{core['total_return']}%"),
        ("Ann. Return",     f"{core['annualized_return']}%"),
        ("Ann. Vol",        f"{core['annualized_vol']}%"),
        ("Sharpe",          f"{core['sharpe']}"),
        ("Sortino",         f"{core['sortino']}"),
        ("Calmar",          f"{core['calmar']}"),
        ("Omega",           f"{core['omega']}"),
        ("Max DD",          f"{core['max_drawdown']}%"),
        ("VaR 95",          f"{core['var_95']}%"),
        ("CVaR 95",         f"{core['cvar_95']}%"),
        ("Worst 30d Sharpe",f"{core['worst_30d_sharpe']}"),
        ("Daily Hit Rate",  f"{core['daily_hit_rate']}%"),
    ]
    for i, (label, val) in enumerate(metrics_display):
        y = 0.97 - i * 0.077
        ax_tbl.text(0.02, y, label, transform=ax_tbl.transAxes,
                    color="#9ca3af", fontsize=8, va="top")
        ax_tbl.text(0.98, y, val,   transform=ax_tbl.transAxes,
                    color=TEXT,      fontsize=8, va="top", ha="right",
                    fontweight="bold")

    fig.suptitle("VIRTUAL TRADING FIRM — Strategy Tearsheet",
                 color=TEXT, fontsize=13, fontweight="bold", y=0.99)

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📊 Tearsheet saved → {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# 7.  CONSOLE TEARSHEET
# ════════════════════════════════════════════════════════════════════════════

def print_tearsheet(core: Dict, trade: Dict, regime_df: pd.DataFrame):
    W = 48
    def line(k, v): print(f"  {k:<26}{str(v):>18}")
    def sep():       print("  " + "─" * W)

    print("\n" + "═" * (W + 4))
    print(f"   {'VIRTUAL TRADING FIRM — TEARSHEET':^{W}}")
    print("═" * (W + 4))

    print("  PERFORMANCE")
    sep()
    line("Total Return",      f"{core['total_return']}%")
    line("Annualised Return",  f"{core['annualized_return']}%")
    line("Annualised Vol",     f"{core['annualized_vol']}%")
    line("Days Traded",        core["days_traded"])
    sep()
    print("  RISK-ADJUSTED")
    sep()
    line("Sharpe",             core["sharpe"])
    line("Sortino",            core["sortino"])
    line("Calmar",             core["calmar"])
    line("Omega",              core["omega"])
    line("Worst 30d Sharpe",   core["worst_30d_sharpe"])
    line("Best 30d Sharpe",    core["best_30d_sharpe"])
    sep()
    print("  RISK")
    sep()
    line("Max Drawdown",       f"{core['max_drawdown']}%")
    line("Avg DD Duration",    f"{core['avg_dd_duration']} bars")
    line("VaR 95%",            f"{core['var_95']}%")
    line("CVaR 95%",           f"{core['cvar_95']}%")
    line("Daily Hit Rate",     f"{core['daily_hit_rate']}%")

    if trade:
        sep()
        print("  TRADE ANALYSIS")
        sep()
        line("Round Trips",       trade["round_trips"])
        line("Win Rate",          f"{trade['win_rate']}%")
        line("Avg Win",           f"{trade['avg_win_pct']}%")
        line("Avg Loss",          f"{trade['avg_loss_pct']}%")
        line("Payoff Ratio",      trade["payoff_ratio"])
        line("Profit Factor",     trade["profit_factor"])
        line("Expectancy",        f"{trade['expectancy_pct']}%")
        line("Avg Hold Days",     trade["avg_hold_days"])
        line("Best Trade",        f"{trade['best_trade_pct']}%")
        line("Worst Trade",       f"{trade['worst_trade_pct']}%")
        line("Total Gross P&L",   f"${trade['total_gross_pnl']:,.2f}")

    if not regime_df.empty:
        sep()
        print("  REGIME BREAKDOWN")
        sep()
        print(f"  {'Regime':<22}{'Bars':>5}{'AnnRet%':>9}"
              f"{'Sharpe':>8}{'MaxDD%':>8}{'WinR%':>7}")
        sep()
        for _, r in regime_df.iterrows():
            print(f"  {r['regime']:<22}{int(r['bars']):>5}"
                  f"{r['ann_ret_%']:>9.2f}{r['sharpe']:>8.3f}"
                  f"{r['max_dd_%']:>8.2f}{r['win_rate_%']:>7.2f}")

    print("═" * (W + 4) + "\n")


# ════════════════════════════════════════════════════════════════════════════
# 8.  RL-READY FEATURE DICT
# ════════════════════════════════════════════════════════════════════════════

def build_rl_features(core: Dict, trade: Dict,
                      regime_df: pd.DataFrame) -> Dict:
    """
    Returns a flat dict the RL agent can consume directly as state features.
    All values are floats in roughly [-1, 1] or [0, 1] ranges.
    """
    feats = {
        # Normalised performance
        "f_total_return":      core["total_return"] / 100,
        "f_ann_return":        core["annualized_return"] / 100,
        "f_ann_vol":           core["annualized_vol"] / 100,

        # Risk-adjusted (clip to reasonable range)
        "f_sharpe":            np.clip(core["sharpe"],  -3, 3) / 3,
        "f_sortino":           np.clip(core["sortino"], -3, 3) / 3,
        "f_calmar":            np.clip(core["calmar"],  -5, 5) / 5,
        "f_omega":             np.clip(core["omega"],    0, 5) / 5,

        # Risk
        "f_max_dd":            core["max_drawdown"] / 100,
        "f_var95":             core["var_95"] / 100,
        "f_cvar95":            core["cvar_95"] / 100,
        "f_worst30d_sharpe":   np.clip(core["worst_30d_sharpe"], -3, 3) / 3,
        "f_daily_hit":         core["daily_hit_rate"] / 100,
    }

    if trade:
        feats.update({
            "f_win_rate":      trade["win_rate"] / 100,
            "f_payoff_ratio":  np.clip(trade["payoff_ratio"], 0, 5) / 5,
            "f_profit_factor": np.clip(trade["profit_factor"], 0, 5) / 5,
            "f_expectancy":    np.clip(trade["expectancy_pct"], -5, 5) / 5,
            "f_avg_hold":      np.clip(trade["avg_hold_days"], 0, 30) / 30,
        })

    if not regime_df.empty:
        for _, r in regime_df.iterrows():
            key = r["regime"].replace("-", "_").replace(" ", "_").lower()
            feats[f"f_regime_{key}_sharpe"] = np.clip(r["sharpe"], -3, 3) / 3
            feats[f"f_regime_{key}_winrate"] = r["win_rate_%"] / 100

    return feats


# ════════════════════════════════════════════════════════════════════════════
# 9.  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def run_metrics(
    equity_file: str = f"{RESULTS_DIR}/equity_curve_v2.csv",
    trade_file:  str = f"{RESULTS_DIR}/trade_log_v2.csv",
    benchmark:   str = "SPY",
    save_chart:  bool = True,
) -> Dict:
    """
    Full tearsheet pipeline. Returns dict with:
        core, trade, regime, rl_features, pnl_df
    For direct use by RL agent.
    """
    print("\n📂 Loading data …")

    # ── load ──────────────────────────────────────────────────────────────
    if not os.path.exists(equity_file):
        raise FileNotFoundError(f"❌ Not found: {equity_file}")
    if not os.path.exists(trade_file):
        raise FileNotFoundError(f"❌ Not found: {trade_file}")

    eq     = pd.read_csv(equity_file)
    trades = pd.read_csv(trade_file)

    # ensure date col is string for consistency
    eq["date"]     = eq["date"].astype(str)
    trades["date"] = trades["date"].astype(str)

    # ── compute ───────────────────────────────────────────────────────────
    print("🔢 Computing metrics …")
    pnl_df     = build_trade_pnl(trades)
    core       = compute_core_metrics(eq)
    trade      = compute_trade_metrics(pnl_df)
    regime_df  = compute_regime_metrics(eq, pnl_df) if "regime" in eq.columns \
                 else pd.DataFrame()

    # ── RL features ───────────────────────────────────────────────────────
    rl_feats   = build_rl_features(core, trade, regime_df)

    # ── benchmark ─────────────────────────────────────────────────────────
    bench = None
    if benchmark:
        print(f"📡 Fetching {benchmark} benchmark …")
        bench = fetch_benchmark(eq["date"].iloc[0], eq["date"].iloc[-1], benchmark)

    # ── console tearsheet ─────────────────────────────────────────────────
    print_tearsheet(core, trade, regime_df)

    # ── chart ─────────────────────────────────────────────────────────────
    if save_chart:
        plot_tearsheet(eq, pnl_df, core, regime_df, bench)

    # ── save enriched pnl ─────────────────────────────────────────────────
    if not pnl_df.empty:
        out_path = f"{RESULTS_DIR}/trade_pnl.csv"
        pnl_df.to_csv(out_path, index=False)
        print(f"  💾 Trade P&L saved → {out_path}")

    return {
        "core":        core,
        "trade":       trade,
        "regime":      regime_df,
        "rl_features": rl_feats,
        "pnl_df":      pnl_df,
    }


# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_metrics(
        equity_file = f"{RESULTS_DIR}/equity_curve.csv",
        trade_file  = f"{RESULTS_DIR}/trade_log.csv",
        benchmark   = "SPY",
        save_chart  = True,
    )
    print(f"\n🤖 RL feature vector ({len(results['rl_features'])} features):")
    for k, v in results["rl_features"].items():
        print(f"   {k:<40} {v:>8.4f}")