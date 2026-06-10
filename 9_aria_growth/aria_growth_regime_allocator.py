"""
ARIA-Growth — Regime-Switching Growth Allocator  (FORWARD TOOL)
================================================================
Classifies today's market regime and builds the growth sub-portfolio whose
characteristics historically suit that regime, from your screen of 516 stocks.

★ WHAT THIS IS:  a disciplined, rules-based portfolio constructor you can
  paper-trade today. It rotates style as the regime flips.
★ WHAT THIS IS NOT:  a backtested strategy. The screen is a CURRENT snapshot
  of fundamentals, so we CANNOT prove this beat the market historically — that
  needs point-in-time fundamentals (the SEC-EDGAR backtest, a separate build).
  Treat this as a defensible rule, not a validated edge.

Regime logic (transparent, inspectable — adjust the thresholds to taste):
  Detect from SPY: trend (vs 200-day SMA) + realized volatility + drawdown.
    RISK_ON   (bull, calm)      → Aggressive Growth: high rev-growth / high score,
                                  unprofitable hypergrowth tolerated (but not broken).
    RISK_OFF  (bear or stressed)→ Quality Defensive: profitable, large-cap, sane
                                  valuation; the −100%-margin lottery tickets cut.
    NEUTRAL   (mixed signals)   → Balanced GARP: high Rule-of-40, profitable,
                                  reasonable multiple.

Each regime ranks the universe by a regime-fit score, applies a per-sector cap
so you're not 60% tech, and equal-weights the top N (robust default for a
forward tool — no weight-overfitting on a single snapshot).

Inputs : growth_screen_results.xlsx  (your screen)
Output : console portfolio + 9_aria_growth/portfolio_<REGIME>.csv

Usage:
  python aria_growth_regime_allocator.py
  python aria_growth_regime_allocator.py --regime RISK_OFF   # force, for inspection
  python aria_growth_regime_allocator.py --n 20 --max-per-sector 4

Install: pip install yfinance pandas numpy openpyxl
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
SCREEN_PATH   = Path("growth_screen_results.csv")
OUT_DIR       = Path("9_aria_growth")
DEFAULT_N     = 20
MAX_PER_SECTOR = 4

# Regime detection thresholds (on SPY)
VOL_CALM      = 0.18      # annualized realized vol below this = calm
VOL_STRESS    = 0.28      # above this = stressed
DRAWDOWN_RISK = -0.10     # >10% off the 1y high = risk-off trigger
TREND_BEAR    = -0.01     # SPY this far below 200dma = bear


# ══════════════════════════════════════════════════════════════════════════════
# REGIME DETECTION (from SPY)
# ══════════════════════════════════════════════════════════════════════════════
def detect_regime():
    """Classify regime from SPY trend + realized vol + drawdown. Returns dict."""
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("pip install yfinance (or pass --regime to skip detection)")

    spy = yf.download("SPY", period="2y", auto_adjust=True, progress=False)
    close = (spy["Close"]["SPY"] if isinstance(spy.columns, pd.MultiIndex)
             else spy["Close"]).dropna()
    if len(close) < 220:
        raise SystemExit("Not enough SPY history to compute 200-day trend.")

    sma200 = close.rolling(200).mean()
    last = float(close.iloc[-1])
    trend = last / float(sma200.iloc[-1]) - 1.0
    rets = close.pct_change().dropna()
    vol = float(rets.iloc[-21:].std() * np.sqrt(252))          # 21d annualized
    high_1y = float(close.iloc[-252:].max())
    drawdown = last / high_1y - 1.0

    if trend > 0 and vol < VOL_CALM and drawdown > DRAWDOWN_RISK:
        regime = "RISK_ON"
    elif trend < TREND_BEAR or vol > VOL_STRESS or drawdown < DRAWDOWN_RISK:
        regime = "RISK_OFF"
    else:
        regime = "NEUTRAL"

    return {"regime": regime, "spy": round(last, 2),
            "trend_vs_200dma_pct": round(trend * 100, 2),
            "realized_vol_pct": round(vol * 100, 1),
            "drawdown_from_high_pct": round(drawdown * 100, 2)}


# ══════════════════════════════════════════════════════════════════════════════
# REGIME-FIT SCORING
# ══════════════════════════════════════════════════════════════════════════════
def _pctile(s):
    return s.rank(pct=True) * 100


def regime_fit(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Filter to regime-appropriate candidates and add a 'fit' score (0-100)."""
    d = df.copy()
    mcap_p = _pctile(d["market_cap"])

    if regime == "RISK_ON":
        # Aggressive growth: real revenue growth, high composite, tolerate
        # unprofitability — but cut truly broken names (margin < -30%).
        d = d[(d["rev_growth_pct"] >= 20) & (d["growth_score"].notna())]
        d = d[(d["profit_margin_pct"].isna()) | (d["profit_margin_pct"] > -30)]
        d["fit"] = (0.6 * _pctile(d["growth_score"]) +
                    0.4 * _pctile(d["rev_growth_pct"]))

    elif regime == "RISK_OFF":
        # Quality defensive: solidly profitable, larger-cap, sane valuation.
        d = d[(d["profit_margin_pct"] >= 10) &
              (mcap_p >= 50) &                                   # top half by size
              (d["forward_pe"].between(0, 40))]
        d["fit"] = (0.45 * _pctile(d["profit_margin_pct"]) +
                    0.30 * _pctile(-d["forward_pe"]) +           # cheaper = better
                    0.25 * _pctile(d["market_cap"]))             # bigger = steadier

    else:  # NEUTRAL — balanced GARP
        # High Rule-of-40, profitable, not nosebleed valuation.
        d = d[(d["rule_of_40"] >= 40) &
              (d["profit_margin_pct"] > 0) &
              (d["forward_pe"].between(0, 60))]
        d["fit"] = (0.55 * _pctile(d["rule_of_40"]) +
                    0.25 * _pctile(d["profit_margin_pct"]) +
                    0.20 * _pctile(-d["ps_ratio"]))              # cheaper P/S better

    return d.dropna(subset=["fit"]).sort_values("fit", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO CONSTRUCTION (top-N, sector-capped, equal weight)
# ══════════════════════════════════════════════════════════════════════════════
def build_portfolio(ranked: pd.DataFrame, n: int, max_per_sector: int) -> pd.DataFrame:
    picks, sector_count = [], {}
    for _, row in ranked.iterrows():
        sec = row.get("sector", "Unknown")
        if sector_count.get(sec, 0) >= max_per_sector:
            continue
        picks.append(row)
        sector_count[sec] = sector_count.get(sec, 0) + 1
        if len(picks) >= n:
            break
    port = pd.DataFrame(picks)
    if len(port):
        port["weight_pct"] = round(100.0 / len(port), 2)       # equal weight
    return port


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
REGIME_STYLE = {
    "RISK_ON":  "Aggressive Growth (bull/calm) - high revenue growth, hypergrowth tolerated",
    "RISK_OFF": "Quality Defensive (bear/stress) - profitable, large-cap, sane valuation",
    "NEUTRAL":  "Balanced GARP (mixed) - high Rule-of-40, profitable, reasonable multiple",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["RISK_ON", "RISK_OFF", "NEUTRAL"],
                    help="force a regime (skip SPY detection) for inspection")
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--max-per-sector", type=int, default=MAX_PER_SECTOR)
    ap.add_argument("--screen", default=str(SCREEN_PATH))
    args = ap.parse_args()

    print("=" * 70)
    print("  ARIA-Growth - Regime-Switching Growth Allocator  (FORWARD TOOL)")
    print("  [!] Rules-based portfolio, NOT a backtested strategy (current snapshot)")
    print("=" * 70)

    p = Path(args.screen)
    df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_excel(p)
    print(f"\n  Loaded screen: {len(df)} stocks")

    if args.regime:
        regime = args.regime
        print(f"\n  Regime: {regime}  (forced via --regime)")
    else:
        info = detect_regime()
        regime = info["regime"]
        print(f"\n  [SPY] Regime detection:")
        print(f"     SPY {info['spy']} | trend vs 200dma {info['trend_vs_200dma_pct']:+.2f}% "
              f"| 21d vol {info['realized_vol_pct']}% | drawdown {info['drawdown_from_high_pct']:+.2f}%")
        print(f"     → REGIME: {regime}")

    print(f"\n  Style: {REGIME_STYLE[regime]}")

    ranked = regime_fit(df, regime)
    port = build_portfolio(ranked, args.n, args.max_per_sector)
    if port.empty:
        print("\n  [!] No stocks passed the regime filter. Loosen thresholds.")
        return

    print(f"\n  Candidates passing {regime} filter: {len(ranked)}  → selecting {len(port)}\n")
    print(f"  {'TICKER':<7}{'SECTOR':<22}{'REV%':>6}{'MARG%':>7}{'FwdPE':>7}{'R40':>6}{'WT%':>6}  NAME")
    print("  " + "-" * 92)
    for _, r in port.iterrows():
        rev = f"{r['rev_growth_pct']:.0f}" if pd.notna(r['rev_growth_pct']) else "-"
        mg  = f"{r['profit_margin_pct']:.0f}" if pd.notna(r['profit_margin_pct']) else "-"
        fpe = f"{r['forward_pe']:.0f}" if pd.notna(r['forward_pe']) else "-"
        r40 = f"{r['rule_of_40']:.0f}" if pd.notna(r['rule_of_40']) else "-"
        print(f"  {r['ticker']:<7}{str(r['sector'])[:21]:<22}{rev:>6}{mg:>7}{fpe:>7}{r40:>6}"
              f"{r['weight_pct']:>6}  {str(r['name'])[:26]}")

    # Sector breakdown
    print(f"\n  Sector allocation:")
    for sec, cnt in port["sector"].value_counts().items():
        print(f"    {sec:<24} {cnt} names ({cnt*port['weight_pct'].iloc[0]:.0f}%)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_cols = ["ticker", "name", "sector", "rev_growth_pct", "profit_margin_pct",
                "forward_pe", "rule_of_40", "growth_score", "market_cap", "fit", "weight_pct"]
    out = port[[c for c in out_cols if c in port.columns]]
    out.to_csv(OUT_DIR / f"portfolio_{regime}.csv", index=False)
    print(f"\n  [saved] {OUT_DIR}/portfolio_{regime}.csv")

    print("\n" + "=" * 70)
    print("  REBALANCE RULE")
    print("  • Re-run weekly (or monthly). If the regime label changes, rotate")
    print("    into the new regime's portfolio. If unchanged, hold.")
    print("  • Equal-weight; rebalance drift back to target on regime change.")
    print("  [!] HONEST LIMIT: not historically validated. To know if this beats")
    print("    SPY, we need the point-in-time EDGAR backtest (separate build).")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()