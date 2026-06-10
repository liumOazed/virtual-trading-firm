"""
inflation_signals.py
====================
Inflation macro signal engine for regime-gated factor tilts.

Downloads (self-contained — does not depend on DataLoader):
  TIP  — TIPS ETF (inflation-protected treasuries; breakeven proxy)
  IEF  — 7-10 year Treasury (nominal yield proxy)
  DBC  — Commodity index ETF (broad commodity inflation)
  XLE  — Energy sector ETF (oil/energy inflation component)
  KRE  — Regional banks ETF (yield curve / inflation sensitivity)

Breakeven inflation proxy:
  IEF 60d annualised return − TIP 60d annualised return.
  Positive → nominal outperforms real → higher inflation expectations.

Inflation regimes (priority order):
  accelerating : breakeven > 3.0% AND momentum > +0.5%
  rising       : breakeven > 2.5% AND momentum > 0%
  deflationary : breakeven < 1.5% AND momentum < 0%
  disinflation : breakeven < 2.0% AND momentum < 0%
  stable       : all other cases

Growth-value tilts by regime (added to base factor weights):
  deflationary : growth=+0.5, value=-0.3  (growth thrives in low-rate world)
  disinflation : growth=+0.3, value=-0.1
  stable       : growth=0.0,  value=0.0
  rising       : growth=-0.2, value=+0.2  (value holds real assets)
  accelerating : growth=-0.5, value=+0.5
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional


INFLATION_TICKERS = ['TIP', 'IEF', 'DBC', 'XLE', 'KRE']
WARMUP_DAYS = 450   # ensures 2-year z-score windows are populated from start_date

GROWTH_VALUE_TILTS: Dict[str, Dict[str, float]] = {
    'deflationary': {'growth': +0.5, 'value': -0.3},
    'disinflation': {'growth': +0.3, 'value': -0.1},
    'stable':       {'growth':  0.0, 'value':  0.0},
    'rising':       {'growth': -0.2, 'value': +0.2},
    'accelerating': {'growth': -0.5, 'value': +0.5},
}


class InflationSignalEngine:
    """
    Computes inflation regime from TIPS/nominal spread, commodity momentum,
    and yield curve signals.  Self-contained: downloads its own data.
    All signals are causally correct (data up to and including `date` only).
    Per-date single-entry cache on get_inflation_regime avoids redundant
    computation when multiple callers ask for the same bar.
    """

    def __init__(self):
        self.data:           Dict[str, pd.DataFrame] = {}
        self._cache_date:    Optional[str]            = None
        self._cache_regime:  Optional[str]            = None

    # ── data ingestion ────────────────────────────────────────────────────────

    def load_data(self, start_date: str, end_date: str):
        """Download all inflation proxy tickers with warmup period."""
        start_dt  = datetime.strptime(start_date[:10], "%Y-%m-%d")
        ext_start = (start_dt - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")

        print("Loading inflation proxy data ...")
        for ticker in INFLATION_TICKERS:
            try:
                df = yf.download(ticker, start=ext_start, end=end_date,
                                 progress=False, auto_adjust=True)
                if df.empty:
                    print(f"  ! {ticker}: no data"); continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                self.data[ticker] = df[['close']].dropna()
                print(f"  + {ticker}: {len(df)} rows")
            except Exception as e:
                print(f"  ! {ticker} failed: {e}")

    # ── helper ────────────────────────────────────────────────────────────────

    def _close_up_to(self, ticker: str, date: str) -> Optional[pd.Series]:
        """Return close prices up to and including `date`."""
        if ticker not in self.data:
            return None
        ts = pd.Timestamp(date)
        s  = self.data[ticker]['close']
        s  = s[s.index <= ts]
        return s if len(s) >= 2 else None

    # ── public signal methods ─────────────────────────────────────────────────

    def get_breakeven_inflation(self, date: str) -> float:
        """
        Proxy breakeven: IEF 60d annualised return − TIP 60d annualised return.
        Positive ↔ nominal outperforms real → higher inflation expectations.
        Returns annualised %, e.g. 2.3 means 2.3%.
        """
        ief = self._close_up_to('IEF', date)
        tip = self._close_up_to('TIP', date)
        if ief is None or tip is None or len(ief) < 61 or len(tip) < 61:
            return 2.0   # neutral fallback
        ief_ret = (float(ief.iloc[-1]) / float(ief.iloc[-61]) - 1.0) * (252 / 60) * 100
        tip_ret = (float(tip.iloc[-1]) / float(tip.iloc[-61]) - 1.0) * (252 / 60) * 100
        return round(ief_ret - tip_ret, 4)

    def get_inflation_momentum(self, date: str, lookback: int = 60) -> float:
        """
        breakeven_today − breakeven_lookback_days_ago.
        Positive → expectations rising.  Returns difference in %.
        """
        ief = self.data.get('IEF')
        tip = self.data.get('TIP')
        if ief is None or tip is None:
            return 0.0
        ts     = pd.Timestamp(date)
        ief_up = ief['close'][ief.index <= ts]
        tip_up = tip['close'][tip.index <= ts]
        if len(ief_up) < lookback + 61 or len(tip_up) < lookback + 61:
            return 0.0

        def _be(close_s: pd.Series, offset: int) -> float:
            return (float(close_s.iloc[-1 - offset]) /
                    float(close_s.iloc[-61 - offset]) - 1.0) * (252 / 60) * 100

        be_today = _be(ief_up, 0) - _be(tip_up, 0)
        be_past  = _be(ief_up, lookback) - _be(tip_up, lookback)
        return round(be_today - be_past, 4)

    def get_real_rate_proxy(self, date: str) -> float:
        """
        Proxy real rate: nominal yield proxy − breakeven inflation.
        Nominal yield proxy = −1 × IEF 60d annualised price return
        (bond price up → yield down, so flip sign).
        Positive real rate = restrictive; negative = accommodative.
        """
        ief = self._close_up_to('IEF', date)
        if ief is None or len(ief) < 61:
            return 0.0
        ief_ret_ann      = (float(ief.iloc[-1]) / float(ief.iloc[-61]) - 1.0) * (252 / 60) * 100
        nominal_yield_px = -ief_ret_ann   # price up → yield down
        breakeven        = self.get_breakeven_inflation(date)
        return round(nominal_yield_px - breakeven, 4)

    def get_commodity_inflation_signal(self, date: str) -> float:
        """
        DBC 60-day return z-scored vs 2-year rolling history.
        Positive → commodity price pressure rising (inflationary).
        """
        dbc = self._close_up_to('DBC', date)
        if dbc is None or len(dbc) < 504:   # 2 years minimum for z-score
            return 0.0

        ret_60d = float(dbc.iloc[-1]) / float(dbc.iloc[-61]) - 1.0

        history = []
        for offset in range(0, min(504, len(dbc) - 61), 5):
            r = float(dbc.iloc[-1 - offset]) / float(dbc.iloc[-61 - offset]) - 1.0
            history.append(r)
        if len(history) < 10:
            return 0.0
        mu  = float(np.mean(history))
        std = float(np.std(history))
        return round((ret_60d - mu) / (std + 1e-9), 4)

    def get_yield_curve_slope(self, date: str) -> float:
        """
        IEF 60d return − KRE 60d return as slope proxy.
        Positive → IEF (long-end) outperforming → curve steepening (expansionary).
        """
        ief = self._close_up_to('IEF', date)
        kre = self._close_up_to('KRE', date)
        if ief is None or kre is None or len(ief) < 61 or len(kre) < 61:
            return 0.0
        ief_ret = float(ief.iloc[-1]) / float(ief.iloc[-61]) - 1.0
        kre_ret = float(kre.iloc[-1]) / float(kre.iloc[-61]) - 1.0
        return round(ief_ret - kre_ret, 4)

    def get_inflation_regime(self, date: str) -> str:
        """
        Classify inflation environment from breakeven + momentum.
        Priority order: accelerating > rising > deflationary > disinflation > stable.
        """
        if date == self._cache_date and self._cache_regime is not None:
            return self._cache_regime

        be  = self.get_breakeven_inflation(date)
        mom = self.get_inflation_momentum(date)

        if be > 3.0 and mom > 0.5:
            regime = 'accelerating'
        elif be > 2.5 and mom > 0.0:
            regime = 'rising'
        elif be < 1.5 and mom < 0.0:
            regime = 'deflationary'
        elif be < 2.0 and mom < 0.0:
            regime = 'disinflation'
        else:
            regime = 'stable'

        self._cache_date   = date
        self._cache_regime = regime
        return regime

    def get_growth_value_tilt(self, inflation_regime: str) -> Dict[str, float]:
        """
        Factor tilt deltas for current inflation regime.
        Returns {'growth': float, 'value': float} to add to base factor weights.
        """
        return dict(GROWTH_VALUE_TILTS.get(inflation_regime, {'growth': 0.0, 'value': 0.0}))


if __name__ == "__main__":
    print("=" * 55)
    print("  INFLATION SIGNAL ENGINE TEST")
    print("=" * 55)

    eng = InflationSignalEngine()
    eng.load_data("2022-01-01", "2026-04-30")

    for test_date in ["2022-06-15", "2024-01-10", "2025-03-01"]:
        print(f"\nDate: {test_date}")
        print(f"  Breakeven inflation : {eng.get_breakeven_inflation(test_date):.2f}%")
        print(f"  Inflation momentum  : {eng.get_inflation_momentum(test_date):.2f}%")
        print(f"  Real rate proxy     : {eng.get_real_rate_proxy(test_date):.2f}%")
        print(f"  Commodity signal    : {eng.get_commodity_inflation_signal(test_date):.3f} (z)")
        print(f"  Yield curve slope   : {eng.get_yield_curve_slope(test_date):.4f}")
        regime = eng.get_inflation_regime(test_date)
        print(f"  Inflation regime    : {regime}")
        tilt = eng.get_growth_value_tilt(regime)
        print(f"  Factor tilt         : growth={tilt['growth']:+.1f}  value={tilt['value']:+.1f}")
