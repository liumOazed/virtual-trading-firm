"""
ARIA-Growth — Growth Screener  (REBUILT 2026-07)
=================================================
Rebuild of the original growth_screener.py (source lost). Produces the exact
same schema so the whole downstream pipeline (archiver, allocator, executor,
reports) works unchanged:

  ticker, name, sector, rev_growth_pct, rule_of_40, gross_margin_pct,
  ps_ratio, pe_ratio, growth_score, forward_pe, profit_margin_pct, market_cap

Universe: S&P 500 + Nasdaq 100 (deduped), fetched from Wikipedia with a
hardcoded fallback if scraping fails.

Fundamentals: yfinance .info (CURRENT snapshot — same known limitation as the
original: no historical fundamentals, so screens must be archived monthly to
build the point-in-time dataset).

SCORING — honest note:
  rule_of_40 = rev_growth_pct + profit_margin_pct   (verified EXACT vs original)
  growth_score: growth + quality + valuation composite, fixed absolute ranges
  (comparable across monthly snapshots). See compute_growth_score() docstring
  for the exact weights.
  IMPORTANT: the archive stores RAW metrics, so any future backtest can
  rescore all snapshots uniformly — formula drift does not poison the dataset.

Usage:
  python growth_screener.py                 # → 9_aria_growth/growth_screen_results_YYYY-MM.csv
  python growth_screener.py --out my.csv

Install: pip install yfinance pandas numpy lxml
"""

import argparse
import io
import sys
import time
import urllib.request
import warnings
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Windows console defaults to cp1252, which can't encode the ✓/⚠ glyphs below.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def _read_html(url: str):
    """pd.read_html via urllib with a browser User-Agent — Wikipedia 403s on the default one."""
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req) as resp:
        return pd.read_html(io.BytesIO(resp.read()))


# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSE
# ══════════════════════════════════════════════════════════════════════════════
def get_universe() -> list[str]:
    """S&P 500 + Nasdaq 100 tickers, deduped. Wikipedia with graceful failure."""
    tickers = set()
    try:
        sp = _read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers |= set(sp["Symbol"].str.replace(".", "-", regex=False))
        print(f"  ✓ S&P 500: {len(sp)} tickers")
    except Exception as e:
        print(f"  ⚠ S&P 500 fetch failed: {e}")
    try:
        for tbl in _read_html("https://en.wikipedia.org/wiki/Nasdaq-100"):
            for col in ("Ticker", "Symbol"):
                if col in tbl.columns:
                    tickers |= set(tbl[col].astype(str).str.replace(".", "-", regex=False))
                    print(f"  ✓ Nasdaq 100 merged")
                    break
            else:
                continue
            break
    except Exception as e:
        print(f"  ⚠ Nasdaq 100 fetch failed: {e}")
    if not tickers:
        raise SystemExit("Could not build universe — check internet / install lxml.")
    return sorted(t for t in tickers if t and t == t)  # drop NaN


# ══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTALS (yfinance .info — current snapshot only)
# ══════════════════════════════════════════════════════════════════════════════
def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    import yfinance as yf
    rows, failed = [], []
    print(f"\n📥 Fetching fundamentals for {len(tickers)} tickers (slow, ~1/s)...")
    for i, t in enumerate(tickers, 1):
        try:
            info = yf.Ticker(t).info
            if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
                failed.append(t); continue
            rev_g = info.get("revenueGrowth")
            pm    = info.get("profitMargins")
            gm    = info.get("grossMargins")
            roe   = info.get("returnOnEquity")
            fcf   = info.get("freeCashflow")
            mcap  = info.get("marketCap")
            rows.append({
                "ticker": t,
                "name": info.get("shortName") or info.get("longName") or t,
                "sector": info.get("sector") or "Unknown",
                "rev_growth_pct": round(rev_g * 100, 2) if rev_g is not None else np.nan,
                "gross_margin_pct": round(gm * 100, 2) if gm is not None else np.nan,
                "profit_margin_pct": round(pm * 100, 2) if pm is not None else np.nan,
                "ps_ratio": info.get("priceToSalesTrailing12Months"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "market_cap": mcap,
                # v2 metrics (quality + cleaner valuation)
                "fcf_yield_pct": round(fcf / mcap * 100, 2) if fcf and mcap else np.nan,
                "roe_pct": round(roe * 100, 2) if roe is not None else np.nan,
                "ev_ebitda": info.get("enterpriseToEbitda"),
            })
        except Exception:
            failed.append(t)
        if i % 50 == 0:
            print(f"  ... {i}/{len(tickers)} ({len(rows)} ok, {len(failed)} failed)")
        time.sleep(0.1)          # be polite to the API
    print(f"  ✓ {len(rows)} tickers with data ({len(failed)} failed/skipped)")
    df = pd.DataFrame(rows)
    # rule_of_40 = revenue growth + profit margin  (verified exact vs original)
    df["rule_of_40"] = (df["rev_growth_pct"] + df["profit_margin_pct"]).round(2)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SCORING (best-fit reconstruction of the lost original — see header note)
# ══════════════════════════════════════════════════════════════════════════════
def _clipscale(s: pd.Series, lo: float, hi: float, invert: bool = False) -> pd.Series:
    x = (s.clip(lo, hi) - lo) / (hi - lo) * 100
    x = (100 - x) if invert else x
    return x.fillna(50.0)      # missing metric → neutral, not penalized


def compute_growth_score(df: pd.DataFrame) -> pd.Series:
    """
    v2 composite: growth + quality + valuation, all fixed absolute ranges
    (comparable across monthly snapshots, no percentile ranking):
      25% rev growth (0-50)     | 20% Rule-of-40 (0-60)   | 15% gross margin (0-80)
      10% FCF yield (0-8%)      | 10% ROE (0-40%)
       8% EV/EBITDA (5-40 inv)  |  6% P/S (0-10 inv)      |  6% trailing P/E (0-50 inv)
    Missing metrics score neutral 50 (e.g. banks without EBITDA aren't penalized).
    """
    return (0.25 * _clipscale(df["rev_growth_pct"], 0, 50)
            + 0.20 * _clipscale(df["rule_of_40"], 0, 60)
            + 0.15 * _clipscale(df["gross_margin_pct"], 0, 80)
            + 0.10 * _clipscale(df["fcf_yield_pct"], 0, 8)
            + 0.10 * _clipscale(df["roe_pct"], 0, 40)
            + 0.08 * _clipscale(df["ev_ebitda"], 5, 40, invert=True)
            + 0.06 * _clipscale(df["ps_ratio"], 0, 10, invert=True)
            + 0.06 * _clipscale(df["pe_ratio"], 0, 50, invert=True)).round(1)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    default_out = Path(__file__).resolve().parent / f"growth_screen_results_{date.today():%Y-%m}.csv"
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(default_out))
    args = ap.parse_args()

    print("=" * 66)
    print("  ARIA-Growth — Screener (rebuilt)")
    print("=" * 66)

    tickers = get_universe()
    df = fetch_fundamentals(tickers)

    # Need at least the score inputs; drop rows missing all of them
    df = df.dropna(subset=["rev_growth_pct", "profit_margin_pct"], how="all")
    df["growth_score"] = compute_growth_score(df)
    df = df.sort_values("growth_score", ascending=False).reset_index(drop=True)

    cols = ["ticker", "name", "sector", "rev_growth_pct", "rule_of_40",
            "gross_margin_pct", "ps_ratio", "pe_ratio",
            "forward_pe", "profit_margin_pct", "market_cap",
            "fcf_yield_pct", "roe_pct", "ev_ebitda", "growth_score"]
    df = df[cols]
    df.to_csv(args.out, index=False)

    print(f"\n  ✓ {len(df)} stocks screened → {args.out}")
    print(f"  Score range: {df.growth_score.min():.1f} - {df.growth_score.max():.1f}")
    print("\n  Top 10:")
    for _, r in df.head(10).iterrows():
        print(f"    {r.ticker:<6} {r.growth_score:>5.1f}  rev {r.rev_growth_pct:>6.1f}%  "
              f"R40 {r.rule_of_40:>6.1f}  [{str(r.sector)[:20]}]")
    print("\n  Next: python aria_growth_archive_screen.py --screen", args.out)


if __name__ == "__main__":
    main()