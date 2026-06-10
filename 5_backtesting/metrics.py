"""
metrics.py  (v3 — window-aware, multi-benchmark)
================================================
Professional quant tearsheet for VIRTUAL_TRADING_FIRM.
Reads equity_curve.csv + trade_log.csv and produces:
  - Full performance metrics (Sharpe [raw + rf-adj], Sortino, Calmar, Omega, VaR, CVaR)
  - Per-regime breakdown
  - Per-window breakdown (walk-forward) when a 'window' column exists
  - Trade-level P&L (FIFO BUY/SELL pairs) — closed round trips only
  - Multi-benchmark comparison: SPY + QQQ buy & hold on identical OOS dates
  - Clean console tearsheet + PNG chart
  - Returns dict for RL agent consumption

v3 changes:
  - Hardened yfinance download (handles multiindex columns / auto_adjust)
  - SPY *and* QQQ benchmark comparison with alpha, beta, correlation, tracking error
  - 'Q' -> 'QE' resample (pandas 2.2+)
  - Graceful handling when 'window' column is absent (no fake window=0)
  - Dual Sharpe: raw (matches engine) + rf-adjusted
  - Trade metrics explicitly labelled "closed round trips only"
  - Benchmark series aligned to the strategy's own trading dates before compare
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, Optional, List

try:
    import yfinance as yf
    _HAS_YF = True
except Exception:
    yf = None
    _HAS_YF = False
warnings.filterwarnings("ignore")

RESULTS_DIR = "5_backtesting/results"
OUT_DIR     = "5_backtesting/results"
RF_ANNUAL   = 0.05
TRADING_DAYS = 252


# ════════════════════════════════════════════════════════════════════════════
# 0.  ROBUST YFINANCE HELPER
# ════════════════════════════════════════════════════════════════════════════

def _download_close(ticker: str, start: str, end: str) -> Optional[pd.Series]:
    """
    Robustly fetch an adjusted close series for a single ticker.
    Handles yfinance multiindex columns, single-level columns, and
    the auto_adjust default. Returns a clean float Series indexed by date,
    or None on failure.
    """
    try:
        if not _HAS_YF:
            print(f"  ⚠️  yfinance not installed — skipping {ticker} benchmark")
            return None
        # pad end by 1 day — yfinance end is exclusive
        end_pad = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.download(
            ticker, start=start, end=end_pad,
            progress=False, auto_adjust=True, threads=False,
        )
        if df is None or df.empty:
            return None

        # Case 1: multiindex columns e.g. ('Close','SPY')
        if isinstance(df.columns, pd.MultiIndex):
            if ("Close", ticker) in df.columns:
                close = df[("Close", ticker)]
            else:
                # take the Close level, first sub-column
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
        # Case 2: flat columns
        else:
            if "Close" in df.columns:
                close = df["Close"]
            elif "Adj Close" in df.columns:
                close = df["Adj Close"]
            else:
                return None
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

        close = pd.to_numeric(close, errors="coerce").dropna()
        close.index = pd.to_datetime(close.index)
        return close if len(close) > 1 else None
    except Exception as e:
        print(f"  ⚠️  {ticker} download failed: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# 1.  TRADE P&L BUILDER  (FIFO, closed round trips only)
# ════════════════════════════════════════════════════════════════════════════

def build_trade_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    """Match BUY → SELL pairs per ticker (FIFO). Preserves window if present."""
    has_window = "window" in trades.columns
    records = []
    for ticker, grp in trades.groupby("ticker"):
        grp   = grp.sort_values("date").reset_index(drop=True)
        buy_q: List[pd.Series] = []
        for _, row in grp.iterrows():
            act = str(row["action"]).upper()
            if act == "BUY":
                buy_q.append(row)
            elif act == "SELL" and buy_q:
                buy_row = buy_q.pop(0)
                entry   = float(buy_row["price"])
                exit_   = float(row["price"])
                shares  = float(buy_row["shares"])
                gross   = (exit_ - entry) * shares
                pnl_pct = (exit_ - entry) / entry if entry > 0 else 0.0
                d_held  = (pd.to_datetime(row["date"]) -
                           pd.to_datetime(buy_row["date"])).days
                rec = {
                    "ticker":      ticker,
                    "entry_date":  buy_row["date"],
                    "exit_date":   row["date"],
                    "entry_price": round(entry,  4),
                    "exit_price":  round(exit_,  4),
                    "shares":      round(shares, 4),
                    "gross_pnl":   round(gross,  4),
                    "pnl_pct":     round(pnl_pct, 6),
                    "days_held":   d_held,
                    "regime":      buy_row.get("regime", "Unknown"),
                    "weight":      buy_row.get("weight",  0.0),
                }
                if has_window:
                    rec["window"] = int(buy_row.get("window", 0))
                records.append(rec)
    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════════════════
# 2.  CORE METRICS
# ════════════════════════════════════════════════════════════════════════════

def compute_core_metrics(eq: pd.DataFrame,
                         rf_annual: float = RF_ANNUAL) -> Dict:
    eq     = eq.copy().sort_values("date").reset_index(drop=True)
    equity = eq["equity"].values.astype(float)
    rets   = pd.Series(equity).pct_change().fillna(0).values

    total_ret  = (equity[-1] - equity[0]) / equity[0]
    days       = max((pd.to_datetime(eq["date"].iloc[-1]) -
                      pd.to_datetime(eq["date"].iloc[0])).days, 1)
    ann_ret    = (1 + total_ret) ** (365.25 / days) - 1
    ann_vol    = rets.std() * np.sqrt(TRADING_DAYS)
    rf_daily   = (1 + rf_annual) ** (1/TRADING_DAYS) - 1

    std        = rets.std()
    # raw Sharpe (matches engine convention — no rf subtraction)
    sharpe_raw = (rets.mean() / std * np.sqrt(TRADING_DAYS)
                  if std > 1e-9 else 0.0)
    # rf-adjusted Sharpe
    excess     = rets - rf_daily
    sharpe_rf  = (excess.mean() / std * np.sqrt(TRADING_DAYS)
                  if std > 1e-9 else 0.0)

    down       = rets[rets < rf_daily]
    down_std   = down.std() * np.sqrt(TRADING_DAYS) if len(down) > 1 else 1e-9
    sortino    = (ann_ret - rf_annual) / down_std if down_std > 1e-9 else 0.0

    peak       = np.maximum.accumulate(equity)
    dd         = (equity - peak) / np.where(peak > 0, peak, 1)
    max_dd     = dd.min()
    calmar     = ann_ret / abs(max_dd) if max_dd < 0 else 0.0

    in_dd      = dd < 0
    dd_dur, cur = [], 0
    for v in in_dd:
        if v:   cur += 1
        elif cur > 0: dd_dur.append(cur); cur = 0
    avg_dd_dur = np.mean(dd_dur) if dd_dur else 0

    gains      = rets[rets > 0].sum()
    losses     = abs(rets[rets < 0].sum())
    omega      = gains / losses if losses > 1e-9 else np.inf

    var_95     = float(np.percentile(rets, 5))
    cvar_95    = float(rets[rets <= var_95].mean()) if (rets <= var_95).any() else var_95

    r          = pd.Series(rets)
    roll_sh    = (r.rolling(30).mean() / r.rolling(30).std()) * np.sqrt(TRADING_DAYS)
    worst_30d  = float(roll_sh.min())
    best_30d   = float(roll_sh.max())
    daily_hit  = float((rets > 0).mean())

    return {
        "total_return":      round(total_ret * 100, 2),
        "annualized_return": round(ann_ret   * 100, 2),
        "annualized_vol":    round(ann_vol   * 100, 2),
        "sharpe":            round(sharpe_raw, 3),   # canonical = raw (engine match)
        "sharpe_rf_adj":     round(sharpe_rf,  3),
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
# 3.  TRADE METRICS  (closed round trips only)
# ════════════════════════════════════════════════════════════════════════════

def compute_trade_metrics(pnl_df: pd.DataFrame) -> Dict:
    if pnl_df.empty:
        return {}
    wins   = pnl_df[pnl_df["pnl_pct"] > 0]
    losses = pnl_df[pnl_df["pnl_pct"] <= 0]

    win_rate      = len(wins) / len(pnl_df) if len(pnl_df) > 0 else 0
    avg_win       = wins["pnl_pct"].mean()   if not wins.empty   else 0
    avg_loss      = losses["pnl_pct"].mean() if not losses.empty else 0
    payoff_ratio  = abs(avg_win / avg_loss)  if avg_loss != 0    else np.inf
    gross_profit  = wins["gross_pnl"].sum()
    gross_loss    = abs(losses["gross_pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    expectancy    = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    return {
        "round_trips":     len(pnl_df),
        "win_rate":        round(win_rate * 100, 2),
        "avg_win_pct":     round(avg_win  * 100, 4),
        "avg_loss_pct":    round(avg_loss * 100, 4),
        "payoff_ratio":    round(payoff_ratio, 3),
        "profit_factor":   round(profit_factor, 3),
        "expectancy_pct":  round(expectancy * 100, 4),
        "avg_hold_days":   round(pnl_df["days_held"].mean(), 1),
        "best_trade_pct":  round(pnl_df["pnl_pct"].max() * 100, 2),
        "worst_trade_pct": round(pnl_df["pnl_pct"].min() * 100, 2),
        "total_gross_pnl": round(pnl_df["gross_pnl"].sum(), 2),
    }


# ════════════════════════════════════════════════════════════════════════════
# 4.  PER-REGIME BREAKDOWN
# ════════════════════════════════════════════════════════════════════════════

def compute_regime_metrics(eq: pd.DataFrame,
                           pnl_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, grp in eq.groupby("regime"):
        rets   = grp["equity"].pct_change().dropna()
        ann_ret= rets.mean() * TRADING_DAYS * 100
        sharpe = (rets.mean() / rets.std() * np.sqrt(TRADING_DAYS)
                  if rets.std() > 1e-9 else 0)
        peak   = grp["equity"].cummax()
        dd     = ((grp["equity"] - peak) / peak).min() * 100
        rt     = (pnl_df[pnl_df["regime"] == regime]
                  if not pnl_df.empty else pd.DataFrame())
        wr     = (len(rt[rt["pnl_pct"] > 0]) / len(rt) * 100
                  if len(rt) > 0 else 0)
        rows.append({
            "regime":     regime,
            "bars":       len(grp),
            "ann_ret_%":  round(ann_ret, 2),
            "sharpe":     round(sharpe,  3),
            "max_dd_%":   round(dd,      2),
            "trades":     len(rt),
            "win_rate_%": round(wr,      2),
        })
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


# ════════════════════════════════════════════════════════════════════════════
# 5.  PER-WINDOW BREAKDOWN  (graceful if absent)
# ════════════════════════════════════════════════════════════════════════════

def compute_window_metrics(eq: pd.DataFrame,
                           pnl_df: pd.DataFrame) -> pd.DataFrame:
    if "window" not in eq.columns:
        return pd.DataFrame()

    has_window_pnl = (not pnl_df.empty) and ("window" in pnl_df.columns)
    rows = []
    for window, grp in eq.groupby("window"):
        grp    = grp.sort_values("date").reset_index(drop=True)
        equity = grp["equity"].values.astype(float)
        rets   = pd.Series(equity).pct_change().fillna(0).values

        total_ret = (equity[-1] - equity[0]) / equity[0] * 100
        sharpe    = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(TRADING_DAYS)
        peak      = np.maximum.accumulate(equity)
        max_dd    = ((equity - peak) / np.where(peak > 0, peak, 1)).min() * 100

        wt = (pnl_df[pnl_df["window"] == window]
              if has_window_pnl else pd.DataFrame())
        wr = (len(wt[wt["pnl_pct"] > 0]) / len(wt) * 100
              if len(wt) > 0 else 0)

        dom_regime = (grp["regime"].value_counts().index[0]
                      if "regime" in grp.columns and not grp.empty
                      else "Unknown")

        rows.append({
            "window":      int(window),
            "start":       grp["date"].iloc[0],
            "end":         grp["date"].iloc[-1],
            "bars":        len(grp),
            "total_ret_%": round(total_ret, 2),
            "sharpe":      round(sharpe,    3),
            "max_dd_%":    round(max_dd,    2),
            "trades":      len(wt),
            "win_rate_%":  round(wr,        2),
            "dom_regime":  dom_regime,
        })
    return pd.DataFrame(rows).sort_values("window")


# ════════════════════════════════════════════════════════════════════════════
# 6.  BENCHMARK COMPARISON  (multi-ticker, date-aligned)
# ════════════════════════════════════════════════════════════════════════════

def compute_benchmark_comparison(eq: pd.DataFrame,
                                 tickers: List[str]) -> Dict[str, Dict]:
    """
    For each benchmark ticker, compute buy & hold metrics over the SAME
    OOS dates as the strategy, plus alpha/beta/correlation/tracking error
    vs the strategy's daily returns. Benchmark is aligned to the strategy's
    own trading dates so the comparison is exactly apples-to-apples.
    """
    start = eq["date"].iloc[0]
    end   = eq["date"].iloc[-1]

    strat_eq = pd.Series(
    eq["equity"].values.astype(float),
    index=pd.to_datetime(eq["date"]),
    ).sort_index()

    # remove duplicate timestamps
    strat_eq = strat_eq[~strat_eq.index.duplicated(keep="last")]
    strat_ret = strat_eq.pct_change().dropna()

    out: Dict[str, Dict] = {}
    for tk in tickers:
        close = _download_close(tk, start, end)
        if close is None:
            out[tk] = {"error": "download_failed"}
            continue

        # align benchmark to the strategy's trading dates (forward-fill gaps)
        close = close[~close.index.duplicated(keep="last")]
        close = close.reindex(strat_eq.index, method="ffill").dropna()
        if len(close) < 2:
            out[tk] = {"error": "alignment_failed"}
            continue

        b_ret   = close.pct_change().dropna()
        b_total = (close.iloc[-1] / close.iloc[0] - 1) * 100
        b_sharpe= ((b_ret.mean() / b_ret.std()) * np.sqrt(TRADING_DAYS)
                   if b_ret.std() > 1e-9 else 0.0)
        b_cum   = (1 + b_ret).cumprod()
        b_dd    = ((b_cum / b_cum.cummax()) - 1).min() * 100

        # align the two return series on common dates for beta/alpha/corr
        # remove duplicate dates if any
        strat_ret = strat_ret[~strat_ret.index.duplicated(keep="last")]
        b_ret     = b_ret[~b_ret.index.duplicated(keep="last")]

        joined = pd.concat([strat_ret, b_ret], axis=1, join="inner").dropna()
        joined.columns = ["strat", "bench"]
        if len(joined) > 2 and joined["bench"].std() > 1e-9:
            cov   = np.cov(joined["strat"], joined["bench"])[0, 1]
            beta  = cov / joined["bench"].var()
            corr  = joined["strat"].corr(joined["bench"])
            # daily alpha annualised: strat - (rf + beta*(bench-rf)), rf~0 daily approx
            rf_d  = (1 + RF_ANNUAL) ** (1/TRADING_DAYS) - 1
            alpha_d = (joined["strat"].mean()
                       - (rf_d + beta * (joined["bench"].mean() - rf_d)))
            alpha_ann = ((1 + alpha_d) ** TRADING_DAYS - 1) * 100
            tracking  = (joined["strat"] - joined["bench"]).std() * np.sqrt(TRADING_DAYS) * 100
        else:
            beta = corr = alpha_ann = tracking = float("nan")

        # quarterly head-to-head (QE = pandas 2.2+ quarter-end)
        q_ours  = strat_eq.resample("QE").last().pct_change() * 100
        q_bench = close.resample("QE").last().pct_change() * 100
        quarterly = pd.DataFrame({"Ours": q_ours, tk: q_bench}).dropna()
        quarterly["Gap"]   = quarterly["Ours"] - quarterly[tk]
        quarterly["Beat?"] = quarterly["Gap"].apply(lambda x: "YES" if x > 0 else "NO")

        out[tk] = {
            "total_return": round(float(b_total),  2),
            "sharpe":       round(float(b_sharpe), 3),
            "max_dd":       round(float(b_dd),     2),
            "beta":         round(float(beta),     3) if beta == beta else None,
            "alpha_ann_%":  round(float(alpha_ann),2) if alpha_ann == alpha_ann else None,
            "correlation":  round(float(corr),     3) if corr == corr else None,
            "tracking_err": round(float(tracking), 2) if tracking == tracking else None,
            "quarterly":    quarterly,
            "norm_series":  (close / close.iloc[0] * 100),
        }
    return out


def print_benchmark_comparison(core: Dict, bench: Dict[str, Dict]):
    W = 78
    print("=" * W)
    print("  BENCHMARK COMPARISON  (buy & hold, identical OOS dates)")
    print("=" * W)
    print(f"  {'Metric':<20}{'Our System':>14}", end="")
    valid = [tk for tk, d in bench.items() if "error" not in d]
    for tk in valid:
        print(f"{tk + ' B&H':>14}", end="")
    print()
    print("  " + "-" * (20 + 14 + 14 * len(valid)))

    def row(label, our_val, key, fmt="{:>+13.2f}%", raw=False):
        print(f"  {label:<20}{our_val:>13}{'' if raw else ''}", end="")
        # our value already formatted by caller via our_val string
        for tk in valid:
            v = bench[tk].get(key)
            if v is None:
                print(f"{'n/a':>14}", end="")
            else:
                print(f"{v:>+13.2f}%" if '%' in fmt else f"{v:>14.3f}", end="")
        print()

    # Total return
    print(f"  {'Total Return':<20}{core['total_return']:>+13.2f}%", end="")
    for tk in valid:
        print(f"{bench[tk]['total_return']:>+13.2f}%", end="")
    print()
    # Sharpe
    print(f"  {'Sharpe (raw)':<20}{core['sharpe']:>14.3f}", end="")
    for tk in valid:
        print(f"{bench[tk]['sharpe']:>14.3f}", end="")
    print()
    # Max DD
    print(f"  {'Max Drawdown':<20}{core['max_drawdown']:>+13.2f}%", end="")
    for tk in valid:
        print(f"{bench[tk]['max_dd']:>+13.2f}%", end="")
    print()
    print()

    # alpha/beta block
    print(f"  {'Alpha/Beta vs each benchmark (strategy daily rets)':<60}")
    print(f"  {'':<20}", end="")
    for tk in valid:
        print(f"{tk:>14}", end="")
    print()
    for label, key in [("Alpha (ann)", "alpha_ann_%"),
                       ("Beta", "beta"),
                       ("Correlation", "correlation"),
                       ("Tracking Err", "tracking_err")]:
        print(f"  {label:<20}", end="")
        for tk in valid:
            v = bench[tk].get(key)
            if v is None:
                print(f"{'n/a':>14}", end="")
            elif key in ("alpha_ann_%", "tracking_err"):
                print(f"{v:>+13.2f}%", end="")
            else:
                print(f"{v:>14.3f}", end="")
        print()
    print()

    # verdict
    print("  VERDICT")
    print("  " + "-" * (W - 4))
    for tk in valid:
        gap = core["total_return"] - bench[tk]["total_return"]
        sh_gap = core["sharpe"] - bench[tk]["sharpe"]
        beat_ret = "BEATS" if gap > 0 else "TRAILS"
        beat_sh  = "better" if sh_gap > 0 else "worse"
        q = bench[tk]["quarterly"]
        qbeat = (q["Gap"] > 0).sum() if not q.empty else 0
        qtot  = len(q) if not q.empty else 0
        print(f"  vs {tk}: strategy {beat_ret} on return by "
              f"{gap:+.2f}%, Sharpe {beat_sh} by {sh_gap:+.3f}, "
              f"beat {qbeat}/{qtot} quarters")
    print("=" * W + "\n")

    # quarterly tables
    for tk in valid:
        q = bench[tk]["quarterly"]
        if not q.empty:
            print(f"  QUARTERLY HEAD-TO-HEAD vs {tk}")
            print(q.round(2).to_string())
            print()


# ════════════════════════════════════════════════════════════════════════════
# 7.  CHART  (5-panel tearsheet, multi-benchmark overlay)
# ════════════════════════════════════════════════════════════════════════════

REGIME_COLORS = {
    "Bull-Trending": "#2ecc71",
    "Bull-Stable":   "#27ae60",
    "Bull-MeanRev":  "#27ae60",
    "Bear-Trending": "#e74c3c",
    "Bear-Stable":   "#c0392b",
    "Bear-Stress":   "#a93226",
    "Bear-MeanRev":  "#c0392b",
}
BENCH_COLORS = {"SPY": "#9ca3af", "QQQ": "#f59e0b"}


def plot_tearsheet(eq:        pd.DataFrame,
                   pnl_df:    pd.DataFrame,
                   core:      Dict,
                   regime_df: pd.DataFrame,
                   window_df: pd.DataFrame,
                   bench:     Dict[str, Dict],
                   save_path: str = f"{OUT_DIR}/tearsheet.png"):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dates  = pd.to_datetime(eq["date"])
    equity = eq["equity"].values.astype(float)
    rets   = pd.Series(equity).pct_change().fillna(0)

    fig  = plt.figure(figsize=(18, 13), facecolor="#0d1117")
    gs   = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_eq  = fig.add_subplot(gs[0, :])
    ax_dd  = fig.add_subplot(gs[1, :2])
    ax_ret = fig.add_subplot(gs[1, 2])
    ax_reg = fig.add_subplot(gs[2, 0])
    ax_win = fig.add_subplot(gs[2, 1])
    ax_tbl = fig.add_subplot(gs[2, 2])

    TEXT, GRID = "#e0e0e0", "#1f2937"
    for ax in [ax_eq, ax_dd, ax_ret, ax_reg, ax_win, ax_tbl]:
        ax.set_facecolor("#111827")
        ax.tick_params(colors=TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)

    # equity curve + benchmarks
    eq_norm = equity / equity[0] * 100
    ax_eq.plot(dates, eq_norm, color="#3b82f6", lw=1.6, label="Strategy", zorder=5)
    for tk, d in bench.items():
        if "error" in d:
            continue
        s = d["norm_series"]
        ax_eq.plot(s.index, s.values, color=BENCH_COLORS.get(tk, "#6b7280"),
                   lw=1.0, ls="--", alpha=0.8, label=f"{tk} B&H")

    if "regime" in eq.columns:
        prev_r, seg_start = eq["regime"].iloc[0], dates.iloc[0]
        for i in range(1, len(eq)):
            r = eq["regime"].iloc[i]
            if r != prev_r or i == len(eq) - 1:
                ax_eq.axvspan(seg_start, dates.iloc[i], alpha=0.08,
                              color=REGIME_COLORS.get(prev_r, "#374151"), lw=0)
                prev_r, seg_start = r, dates.iloc[i]

    ax_eq.set_title("Equity Curve vs Benchmarks (regime shading)",
                    color=TEXT, fontsize=10, pad=8)
    ax_eq.set_ylabel("Normalised (base 100)", color=TEXT, fontsize=8)
    ax_eq.legend(fontsize=8, facecolor="#1f2937", labelcolor=TEXT,
                 framealpha=0.8, loc="upper left")
    ax_eq.grid(True, color=GRID, lw=0.5)
    ax_eq.tick_params(axis="x", labelrotation=30)

    # drawdown
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / np.where(peak > 0, peak, 1) * 100
    ax_dd.fill_between(dates, dd, 0, color="#ef4444", alpha=0.6)
    ax_dd.plot(dates, dd, color="#ef4444", lw=0.8)
    ax_dd.set_title("Drawdown %", color=TEXT, fontsize=10, pad=8)
    ax_dd.grid(True, color=GRID, lw=0.5)
    ax_dd.tick_params(axis="x", labelrotation=30)

    # return distribution
    ax_ret.hist(rets * 100, bins=50, color="#8b5cf6", alpha=0.8, density=True)
    ax_ret.axvline(rets.mean() * 100, color="#facc15", lw=1.2, ls="--",
                   label=f"Mean {rets.mean()*100:.3f}%")
    ax_ret.axvline(core["var_95"], color="#f87171", lw=1.2, ls=":",
                   label=f"VaR95 {core['var_95']:.2f}%")
    ax_ret.set_title("Daily Return Distribution", color=TEXT, fontsize=10, pad=8)
    ax_ret.legend(fontsize=7, facecolor="#1f2937", labelcolor=TEXT)
    ax_ret.grid(True, color=GRID, lw=0.5)

    # regime sharpe
    if not regime_df.empty:
        labels  = regime_df["regime"].values
        sharpes = regime_df["sharpe"].values
        colors  = [REGIME_COLORS.get(r, "#4b5563") for r in labels]
        bars    = ax_reg.barh(labels, sharpes, color=colors, height=0.5)
        ax_reg.axvline(0, color=TEXT, lw=0.8, alpha=0.5)
        for bar, val in zip(bars, sharpes):
            ax_reg.text(val, bar.get_y() + bar.get_height()/2,
                        f" {val:.2f}", va="center", color=TEXT, fontsize=7)
        ax_reg.set_title("Sharpe by Regime", color=TEXT, fontsize=10, pad=8)
        ax_reg.grid(True, color=GRID, lw=0.5, axis="x")

    # per-window or fallback message
    if not window_df.empty:
        w_labels = [f"W{int(w)}" for w in window_df["window"]]
        w_rets   = window_df["total_ret_%"].values
        w_colors = ["#22c55e" if r >= 0 else "#ef4444" for r in w_rets]
        ax_win.bar(w_labels, w_rets, color=w_colors, alpha=0.8)
        ax_win.axhline(0, color=TEXT, lw=0.8, alpha=0.5)
        ax_win.set_title("Return % per Window", color=TEXT, fontsize=10, pad=8)
        ax_win.tick_params(axis="x", labelrotation=45, labelsize=6)
        ax_win.grid(True, color=GRID, lw=0.5, axis="y")
    else:
        ax_win.axis("off")
        ax_win.text(0.5, 0.5, "No 'window' column\nin equity_curve.csv",
                    transform=ax_win.transAxes, ha="center", va="center",
                    color="#6b7280", fontsize=9)

    # metrics table
    ax_tbl.axis("off")
    metrics_display = [
        ("Total Return",     f"{core['total_return']}%"),
        ("Ann. Return",      f"{core['annualized_return']}%"),
        ("Ann. Vol",         f"{core['annualized_vol']}%"),
        ("Sharpe (raw)",     f"{core['sharpe']}"),
        ("Sharpe (rf-adj)",  f"{core['sharpe_rf_adj']}"),
        ("Sortino",          f"{core['sortino']}"),
        ("Calmar",           f"{core['calmar']}"),
        ("Max DD",           f"{core['max_drawdown']}%"),
        ("VaR 95",           f"{core['var_95']}%"),
        ("CVaR 95",          f"{core['cvar_95']}%"),
        ("Daily Hit Rate",   f"{core['daily_hit_rate']}%"),
    ]
    for i, (label, val) in enumerate(metrics_display):
        y = 0.97 - i * 0.085
        ax_tbl.text(0.02, y, label, transform=ax_tbl.transAxes,
                    color="#9ca3af", fontsize=8, va="top")
        ax_tbl.text(0.98, y, val, transform=ax_tbl.transAxes,
                    color=TEXT, fontsize=8, va="top", ha="right", fontweight="bold")

    fig.suptitle("VIRTUAL TRADING FIRM — Strategy Tearsheet",
                 color=TEXT, fontsize=13, fontweight="bold", y=0.99)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📊 Tearsheet saved → {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# 8.  CONSOLE TEARSHEET
# ════════════════════════════════════════════════════════════════════════════

def print_tearsheet(core, trade, regime_df, window_df):
    W = 48
    def line(k, v): print(f"  {k:<26}{str(v):>18}")
    def sep():       print("  " + "─" * W)

    print("\n" + "═" * (W + 4))
    print(f"   {'VIRTUAL TRADING FIRM — TEARSHEET':^{W}}")
    print("═" * (W + 4))

    print("  PERFORMANCE");  sep()
    line("Total Return",      f"{core['total_return']}%")
    line("Annualised Return", f"{core['annualized_return']}%")
    line("Annualised Vol",    f"{core['annualized_vol']}%")
    line("Days Traded",        core["days_traded"])
    sep(); print("  RISK-ADJUSTED"); sep()
    line("Sharpe (raw)",       core["sharpe"])
    line("Sharpe (rf-adj)",    core["sharpe_rf_adj"])
    line("Sortino",            core["sortino"])
    line("Calmar",             core["calmar"])
    line("Omega",              core["omega"])
    line("Worst 30d Sharpe",   core["worst_30d_sharpe"])
    line("Best 30d Sharpe",    core["best_30d_sharpe"])
    sep(); print("  RISK"); sep()
    line("Max Drawdown",       f"{core['max_drawdown']}%")
    line("Avg DD Duration",    f"{core['avg_dd_duration']} bars")
    line("VaR 95%",            f"{core['var_95']}%")
    line("CVaR 95%",           f"{core['cvar_95']}%")
    line("Daily Hit Rate",     f"{core['daily_hit_rate']}%")

    if trade:
        sep(); print("  TRADE ANALYSIS  (closed round trips only)"); sep()
        line("Round Trips",    trade["round_trips"])
        line("Win Rate",       f"{trade['win_rate']}%")
        line("Avg Win",        f"{trade['avg_win_pct']}%")
        line("Avg Loss",       f"{trade['avg_loss_pct']}%")
        line("Payoff Ratio",   trade["payoff_ratio"])
        line("Profit Factor",  trade["profit_factor"])
        line("Expectancy",     f"{trade['expectancy_pct']}%")
        line("Avg Hold Days",  trade["avg_hold_days"])
        line("Best Trade",     f"{trade['best_trade_pct']}%")
        line("Worst Trade",    f"{trade['worst_trade_pct']}%")
        line("Total Gross P&L",f"${trade['total_gross_pnl']:,.2f}")

    if not regime_df.empty:
        sep(); print("  REGIME BREAKDOWN"); sep()
        print(f"  {'Regime':<22}{'Bars':>5}{'AnnRet%':>9}"
              f"{'Sharpe':>8}{'MaxDD%':>8}{'WinR%':>7}")
        sep()
        for _, r in regime_df.reset_index(drop=True).iterrows():
            print(f"  {r['regime']:<22}{int(r['bars']):>5}"
                  f"{r['ann_ret_%']:>9.2f}{r['sharpe']:>8.3f}"
                  f"{r['max_dd_%']:>8.2f}{r['win_rate_%']:>7.2f}")

    if not window_df.empty:
        sep(); print("  WALK-FORWARD WINDOW BREAKDOWN"); sep()
        print(f"  {'W':<4}{'Start':<12}{'End':<12}{'Ret%':>7}"
              f"{'Sharpe':>8}{'DD%':>7}{'Trd':>5}{'WinR%':>7}  Regime")
        sep()
        best = window_df["sharpe"].max()
        for _, r in window_df.reset_index(drop=True).iterrows():
            flag = " ★" if r["sharpe"] == best else ""
            print(f"  W{int(r['window']):<3}{r['start']:<12}{r['end']:<12}"
                  f"{r['total_ret_%']:>7.2f}{r['sharpe']:>8.3f}"
                  f"{r['max_dd_%']:>7.2f}{int(r['trades']):>5}"
                  f"{r['win_rate_%']:>7.2f}  {r['dom_regime']}{flag}")

    print("═" * (W + 4) + "\n")


# ════════════════════════════════════════════════════════════════════════════
# 9.  RL-READY FEATURE DICT
# ════════════════════════════════════════════════════════════════════════════

def build_rl_features(core, trade, regime_df, window_df,
                      bench: Optional[Dict] = None) -> Dict:
    feats = {
        "f_total_return":    core["total_return"] / 100,
        "f_ann_return":      core["annualized_return"] / 100,
        "f_ann_vol":         core["annualized_vol"] / 100,
        "f_sharpe":          np.clip(core["sharpe"],  -3, 3) / 3,
        "f_sortino":         np.clip(core["sortino"], -3, 3) / 3,
        "f_calmar":          np.clip(core["calmar"],  -5, 5) / 5,
        "f_omega":           np.clip(core["omega"],    0, 5) / 5,
        "f_max_dd":          core["max_drawdown"] / 100,
        "f_var95":           core["var_95"] / 100,
        "f_cvar95":          core["cvar_95"] / 100,
        "f_worst30d_sharpe": np.clip(core["worst_30d_sharpe"], -3, 3) / 3,
        "f_daily_hit":       core["daily_hit_rate"] / 100,
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
            key = r["regime"].replace("-","_").replace(" ","_").lower()
            feats[f"f_regime_{key}_sharpe"]  = np.clip(r["sharpe"], -3, 3) / 3
            feats[f"f_regime_{key}_winrate"] = r["win_rate_%"] / 100
    if not window_df.empty:
        feats["f_best_window_sharpe"]  = np.clip(window_df["sharpe"].max(), -3, 3) / 3
        feats["f_worst_window_sharpe"] = np.clip(window_df["sharpe"].min(), -3, 3) / 3
        feats["f_window_sharpe_std"]   = np.clip(window_df["sharpe"].std(), 0, 3) / 3
    # benchmark-relative features (alpha vs SPY/QQQ) — useful RL signal
    if bench:
        for tk, d in bench.items():
            if "error" in d:
                continue
            a = d.get("alpha_ann_%")
            b = d.get("beta")
            if a is not None:
                feats[f"f_alpha_{tk.lower()}"] = np.clip(a / 100, -1, 1)
            if b is not None:
                feats[f"f_beta_{tk.lower()}"]  = np.clip(b, -2, 2) / 2
    return feats


# ════════════════════════════════════════════════════════════════════════════
# 10.  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def run_metrics(
    equity_file: str       = f"{RESULTS_DIR}/equity_curve.csv",
    trade_file:  str       = f"{RESULTS_DIR}/trade_log.csv",
    benchmarks:  List[str] = ("SPY", "QQQ"),
    save_chart:  bool      = True,
) -> Dict:
    print("\n📂 Loading data …")
    if not os.path.exists(equity_file):
        raise FileNotFoundError(f"❌ Not found: {equity_file}")
    if not os.path.exists(trade_file):
        raise FileNotFoundError(f"❌ Not found: {trade_file}")

    eq     = pd.read_csv(equity_file)
    trades = pd.read_csv(trade_file)
    eq["date"]     = eq["date"].astype(str)
    trades["date"] = trades["date"].astype(str)

    print("🔢 Computing metrics …")
    pnl_df    = build_trade_pnl(trades)
    core      = compute_core_metrics(eq)
    trade     = compute_trade_metrics(pnl_df)
    regime_df = (compute_regime_metrics(eq, pnl_df)
                 if "regime" in eq.columns else pd.DataFrame())
    window_df = compute_window_metrics(eq, pnl_df)

    bench = {}
    if benchmarks:
        print(f"📡 Fetching benchmarks {list(benchmarks)} "
              f"({eq['date'].iloc[0]} → {eq['date'].iloc[-1]}) …")
        bench = compute_benchmark_comparison(eq, list(benchmarks))

    rl_feats = build_rl_features(core, trade, regime_df, window_df, bench)

    print_tearsheet(core, trade, regime_df, window_df)
    if bench and any("error" not in d for d in bench.values()):
        print_benchmark_comparison(core, bench)

    if save_chart:
        plot_tearsheet(eq, pnl_df, core, regime_df, window_df, bench)

    if not pnl_df.empty:
        out = f"{RESULTS_DIR}/trade_pnl.csv"
        pnl_df.to_csv(out, index=False)
        print(f"  💾 Trade P&L → {out}")
    if not window_df.empty:
        out = f"{RESULTS_DIR}/window_breakdown.csv"
        window_df.to_csv(out, index=False)
        print(f"  💾 Window breakdown → {out}")

    return {
        "core": core, "trade": trade, "regime": regime_df,
        "window": window_df, "rl_features": rl_feats,
        "pnl_df": pnl_df, "benchmarks": bench,
    }


if __name__ == "__main__":
    results = run_metrics(
        equity_file = f"{RESULTS_DIR}/equity_curve.csv",
        trade_file  = f"{RESULTS_DIR}/trade_log.csv",
        benchmarks  = ("SPY", "QQQ"),
        save_chart  = True,
    )
    print(f"\n🤖 RL feature vector ({len(results['rl_features'])} features):")
    for k, v in results["rl_features"].items():
        print(f"   {k:<44} {v:>8.4f}")