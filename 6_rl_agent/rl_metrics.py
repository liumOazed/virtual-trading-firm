import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────
INPUT_PATH = "6_rl_agent/rl_equity_curve.csv"
OUT_DIR    = "6_rl_agent/results"
os.makedirs(OUT_DIR, exist_ok=True)

TRADING_DAYS = 252


# ────────────────────────────────────────────────────────────────
# LOAD DATA
# ────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ────────────────────────────────────────────────────────────────
# METRICS
# ────────────────────────────────────────────────────────────────
def compute_metrics(df: pd.DataFrame) -> dict:
    equity = df["equity"].values

    rets = np.diff(equity) / np.maximum(equity[:-1], 1e-9)
    rets = rets[np.abs(rets) > 1e-10]

    if len(rets) < 2:
        return {}

    total_return = equity[-1] / equity[0] - 1
    ann_return   = (1 + total_return) ** (TRADING_DAYS / len(df)) - 1
    ann_vol      = np.std(rets) * np.sqrt(TRADING_DAYS)

    sharpe = (np.mean(rets) / (np.std(rets) + 1e-9)) * np.sqrt(TRADING_DAYS)

    downside = rets[rets < 0]
    sortino  = (np.mean(rets) / (np.std(downside) + 1e-9)) * np.sqrt(TRADING_DAYS)

    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / peak
    max_dd = dd.min()

    calmar = ann_return / abs(max_dd + 1e-9)

    var_95  = np.percentile(rets, 5) * 100
    cvar_95 = rets[rets <= np.percentile(rets, 5)].mean() * 100

    hit_rate = (rets > 0).mean() * 100

    return {
        "total_return": round(total_return * 100, 2),
        "annualized_return": round(ann_return * 100, 2),
        "annualized_vol": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "var_95": round(var_95, 2),
        "cvar_95": round(cvar_95, 2),
        "daily_hit_rate": round(hit_rate, 2),
    }


# ────────────────────────────────────────────────────────────────
# TEARSHEET
# ────────────────────────────────────────────────────────────────
def plot_tearsheet(df: pd.DataFrame, metrics: dict):
    dates  = df["date"]
    equity = df["equity"].values

    rets = np.diff(equity) / np.maximum(equity[:-1], 1e-9)
    rets = np.insert(rets, 0, 0)

    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / peak * 100

    weight_cols = [c for c in df.columns if c.startswith("w_")]

    fig = plt.figure(figsize=(18, 14), facecolor="#0d1117")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_eq  = fig.add_subplot(gs[0, :])
    ax_dd  = fig.add_subplot(gs[1, :2])
    ax_ret = fig.add_subplot(gs[1, 2])
    ax_w   = fig.add_subplot(gs[2, :2])
    ax_tbl = fig.add_subplot(gs[2, 2])

    TEXT = "#e5e7eb"
    GRID = "#1f2937"

    for ax in [ax_eq, ax_dd, ax_ret, ax_w, ax_tbl]:
        ax.set_facecolor("#111827")
        ax.tick_params(colors=TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)

    # ── EQUITY ───────────────────────────────────────────────
    eq_norm = equity / equity[0] * 100
    ax_eq.plot(dates, eq_norm, lw=1.6)
    ax_eq.set_title("RL Equity Curve (Normalized)", color=TEXT)
    ax_eq.grid(True, color=GRID)

    # ── DRAWDOWN ─────────────────────────────────────────────
    ax_dd.fill_between(dates, dd, 0, alpha=0.7)
    ax_dd.set_title("Drawdown (%)", color=TEXT)
    ax_dd.grid(True, color=GRID)

    # ── RETURNS DIST ─────────────────────────────────────────
    ax_ret.hist(rets * 100, bins=50, alpha=0.8, density=True)
    ax_ret.axvline(np.mean(rets) * 100, linestyle="--")
    ax_ret.set_title("Return Distribution", color=TEXT)
    ax_ret.grid(True, color=GRID)

    # ── WEIGHTS ──────────────────────────────────────────────
    if weight_cols:
        for col in weight_cols:
            ax_w.plot(dates, df[col], lw=1.0, alpha=0.7, label=col.replace("w_", ""))

        ax_w.set_title("Portfolio Weights", color=TEXT)
        ax_w.legend(fontsize=7)
        ax_w.grid(True, color=GRID)

    # ── METRICS TABLE ────────────────────────────────────────
    ax_tbl.axis("off")

    y = 0.95
    for k, v in metrics.items():
        ax_tbl.text(0.02, y, k.replace("_", " ").title(),
                    color="#9ca3af", fontsize=9)
        ax_tbl.text(0.98, y, str(v),
                    color=TEXT, fontsize=9, ha="right", fontweight="bold")
        y -= 0.07

    fig.suptitle("RL AGENT — PERFORMANCE TEARSHEET",
                 color=TEXT, fontsize=14, fontweight="bold")

    out_path = f"{OUT_DIR}/rl_tearsheet.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    print(f"📊 Saved → {out_path}")


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("📂 Loading RL equity curve...")
    df = load_data(INPUT_PATH)

    print("🔢 Computing metrics...")
    metrics = compute_metrics(df)

    print("\n══════════════════════════════════════")
    print("        RL AGENT PERFORMANCE")
    print("══════════════════════════════════════")
    for k, v in metrics.items():
        print(f"{k:25s}: {v}")
    print("══════════════════════════════════════")

    print("📊 Plotting tearsheet...")
    plot_tearsheet(df, metrics)