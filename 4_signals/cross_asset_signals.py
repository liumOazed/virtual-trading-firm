"""
cross_asset_signals.py
======================
Cross-asset macro signal engine for regime-gated position sizing.

Downloads (self-contained — does not depend on DataLoader):
  SPY  — S&P 500 (used for cross-asset correlation computation)
  TLT  — 20+ year Treasury bonds
  IEF  — 7-10 year Treasury bonds (yield curve long-end proxy)
  SHY  — 1-3 year Treasury bonds  (yield curve short-end proxy)
  GLD  — Gold (risk-off / inflation indicator)
  UUP  — Dollar index ETF (dollar strength)
  HYG  — High-yield corporate bonds (credit spread proxy)
  LQD  — Investment-grade corporate bonds (credit quality anchor)
  USO  — Crude oil (growth / risk-on indicator)
  ^VIX — Attempted; falls back gracefully if unavailable

Risk-on/off composite signal ∈ [-1, +1]:
  +0.3 if stocks-bonds correlation < 0   (negative = flight-to-quality absent)
  -0.2 if stocks-dollar correlation > 0  (positive dollar co-movement = risk-off)
  -0.2 × credit_stress_score             (HYG underperforming LQD = stress)
  +0.2 if yield curve steepening         (long yields rising > short = expansion)
  +0.1 if gold falling                   (gold rallying = risk-off hedge demand)

Macro regime thresholds:
  risk_score > +0.30 → 'risk_on'
  risk_score < -0.30 → 'risk_off'
  else               → 'transition'
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional


CROSS_ASSET_TICKERS = ['SPY', 'TLT', 'IEF', 'SHY', 'GLD', 'UUP', 'HYG', 'LQD', 'USO']
WARMUP_DAYS = 450   # ensures 252d z-score windows are fully populated from start_date
CROSS_ASSET_MIN_START = "2019-01-01"  # floor: VIX substate needs pre-2020 history regardless of config window

MACRO_REGIME_THRESHOLDS = {'risk_on': 0.30, 'risk_off': -0.30}


class CrossAssetSignalEngine:
    """
    Computes macro risk-on/off regime from cross-asset price relationships.
    Self-contained: downloads its own data via load_data().
    All signal methods are causally correct (data up to and including `date` only).
    Per-date single-entry cache on get_risk_on_off_signal avoids redundant
    computation when multiple methods are called for the same bar.
    """

    def __init__(self):
        self.cross_asset_data: Dict[str, pd.DataFrame] = {}
        self._cache_date:      Optional[str]            = None
        self._cache_result:    Optional[Dict]           = None

    # ── data ingestion ────────────────────────────────────────────────────────

    def load_data(self, start_date: str, end_date: str):
        """Download all cross-asset tickers with warmup period."""
        start_dt     = datetime.strptime(start_date, "%Y-%m-%d")
        ext_start_dt = start_dt - timedelta(days=WARMUP_DAYS)
        floor_dt     = datetime.strptime(CROSS_ASSET_MIN_START, "%Y-%m-%d")
        ext_start    = min(ext_start_dt, floor_dt).strftime("%Y-%m-%d")

        print("Loading cross-asset data ...")
        for ticker in CROSS_ASSET_TICKERS:
            try:
                df = yf.download(ticker, start=ext_start, end=end_date,
                                 progress=False, auto_adjust=True)
                if df.empty:
                    print(f"  ! {ticker}: no data"); continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                self.cross_asset_data[ticker] = df[['close']].dropna()
                print(f"  + {ticker}: {len(df)} rows")
            except Exception as e:
                print(f"  ! {ticker} failed: {e}")

        try:
            vix_df = yf.download('^VIX', start=ext_start, end=end_date,
                                 progress=False, auto_adjust=True)
            if not vix_df.empty:
                if isinstance(vix_df.columns, pd.MultiIndex):
                    vix_df.columns = vix_df.columns.get_level_values(0)
                vix_df.columns = [c.lower() for c in vix_df.columns]
                self.cross_asset_data['^VIX'] = vix_df[['close']].dropna()
                print(f"  + ^VIX: {len(vix_df)} rows")
        except Exception:
            print("  ! ^VIX unavailable — SPY-derived vol proxy available via HYG/LQD spread")

    # ── internal helpers ──────────────────────────────────────────────────────

    def _close_up_to(self, ticker: str, date: str) -> Optional[pd.Series]:
        """Causal close price series up to and including `date`."""
        if ticker not in self.cross_asset_data:
            return None
        close = self.cross_asset_data[ticker]['close']
        ts    = pd.Timestamp(date)
        return close[close.index <= ts]

    def _rolling_corr(self, ticker1: str, ticker2: str, date: str,
                      lookback: int = 30) -> float:
        """Pearson correlation between daily returns of two tickers over last `lookback` bars."""
        s1 = self._close_up_to(ticker1, date)
        s2 = self._close_up_to(ticker2, date)
        if s1 is None or s2 is None:
            return 0.0
        if len(s1) < lookback + 1 or len(s2) < lookback + 1:
            return 0.0
        r1 = s1.iloc[-(lookback + 1):].pct_change().dropna()
        r2 = s2.iloc[-(lookback + 1):].pct_change().dropna()
        common = r1.index.intersection(r2.index)
        if len(common) < max(lookback // 2, 5):
            return 0.0
        r1c, r2c = r1.loc[common].values, r2.loc[common].values
        if r1c.std() < 1e-9 or r2c.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(r1c, r2c)[0, 1])

    def _z_score_1yr(self, series: pd.Series, date: str, lookback: int = 252) -> float:
        """Z-score of the last value vs. trailing `lookback`-bar history ending at `date`."""
        ts  = pd.Timestamp(date)
        up  = series[series.index <= ts]
        if len(up) < max(lookback // 4, 20):
            return 0.0
        window = up.iloc[-lookback:]
        mu     = float(window.mean())
        std    = float(window.std(ddof=1))
        if std < 1e-12:
            return 0.0
        return float((float(up.iloc[-1]) - mu) / std)

    def _recent_return(self, ticker: str, date: str, lookback: int = 30) -> float:
        """Total return of ticker over the last `lookback` bars ending at `date`."""
        cl = self._close_up_to(ticker, date)
        if cl is None or len(cl) < lookback + 1:
            return 0.0
        return float(cl.iloc[-1] / cl.iloc[-(lookback + 1)] - 1.0)

    # ── public scalar metrics ─────────────────────────────────────────────────

    def get_stocks_bonds_corr(self, date: str, lookback: int = 30) -> float:
        """30-day rolling Pearson correlation between SPY and TLT daily returns."""
        if self._cache_date == date and self._cache_result is not None:
            return self._cache_result['stocks_bonds_corr']
        return self._rolling_corr('SPY', 'TLT', date, lookback)

    def get_stocks_dollar_corr(self, date: str, lookback: int = 30) -> float:
        """30-day rolling correlation between SPY and UUP (dollar proxy) daily returns."""
        if self._cache_date == date and self._cache_result is not None:
            return self._cache_result['stocks_dollar_corr']
        return self._rolling_corr('SPY', 'UUP', date, lookback)

    def get_stocks_credit_corr(self, date: str, lookback: int = 30) -> float:
        """30-day rolling correlation between SPY and HYG daily returns."""
        return self._rolling_corr('SPY', 'HYG', date, lookback)

    def get_yield_curve_slope(self, date: str) -> float:
        """
        Yield curve slope proxy: SHY 30-day return minus IEF 30-day return.
        Positive = IEF falling more than SHY = long yields rising faster = steepening.
        Negative = IEF outperforming SHY = curve flattening.
        """
        if self._cache_date == date and self._cache_result is not None:
            return self._cache_result['yield_slope']
        ief_ret = self._recent_return('IEF', date, lookback=30)
        shy_ret = self._recent_return('SHY', date, lookback=30)
        return float(shy_ret - ief_ret)

    def get_dollar_strength_score(self, date: str) -> float:
        """UUP close price z-scored vs. trailing 1-year (252-bar) history."""
        cl = self._close_up_to('UUP', date)
        if cl is None:
            return 0.0
        return self._z_score_1yr(cl, date)

    def get_credit_stress_score(self, date: str) -> float:
        """
        HYG/LQD ratio z-scored vs. trailing 1-year history (inverted).
        High positive score = HYG underperforming LQD = credit stress elevated.
        """
        if self._cache_date == date and self._cache_result is not None:
            return self._cache_result['credit_stress_score']
        hyg = self._close_up_to('HYG', date)
        lqd = self._close_up_to('LQD', date)
        if hyg is None or lqd is None:
            return 0.0
        ts     = pd.Timestamp(date)
        common = (hyg.index[hyg.index <= ts]
                  .intersection(lqd.index[lqd.index <= ts]))
        if len(common) < 40:
            return 0.0
        hyg_c  = hyg.loc[common]
        lqd_c  = lqd.loc[common].replace(0, np.nan)
        spread = (hyg_c / lqd_c).dropna()
        if len(spread) < 40:
            return 0.0
        z = self._z_score_1yr(spread, date)
        return float(-z)   # invert: spread falling (stress) → high positive score

    def get_risk_on_off_signal(self, date: str) -> float:
        """
        Composite risk-on/off signal ∈ [-1, +1].
        Positive = risk-on conditions, negative = risk-off conditions.
        Result and all component values are cached per date.
        """
        if self._cache_date == date and self._cache_result is not None:
            return self._cache_result['risk_score']

        sbc   = self._rolling_corr('SPY', 'TLT', date, 30)
        sdc   = self._rolling_corr('SPY', 'UUP', date, 30)
        css   = self.get_credit_stress_score(date)
        slope = self.get_yield_curve_slope(date)
        gold  = self._recent_return('GLD', date, lookback=30)

        risk_score = (
            +0.3 * (1.0 if sbc < 0.0 else -1.0)
            -0.2 * (1.0 if sdc > 0.0 else 0.0)
            -0.2 * float(np.clip(css, -3.0, 3.0))
            +0.2 * (1.0 if slope > 0.0 else -1.0)
            +0.1 * (1.0 if gold < 0.0 else -1.0)
        )
        risk_score = float(np.clip(risk_score, -1.0, 1.0))

        self._cache_date   = date
        self._cache_result = {
            'risk_score':          risk_score,
            'stocks_bonds_corr':   sbc,
            'stocks_dollar_corr':  sdc,
            'credit_stress_score': css,
            'yield_slope':         slope,
            'gold_ret':            gold,
        }
        return risk_score

    def get_macro_regime(self, date: str) -> str:
        """'risk_on' / 'risk_off' / 'transition' based on composite risk score."""
        rs = self.get_risk_on_off_signal(date)
        if rs > MACRO_REGIME_THRESHOLDS['risk_on']:
            return 'risk_on'
        elif rs < MACRO_REGIME_THRESHOLDS['risk_off']:
            return 'risk_off'
        return 'transition'


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = CrossAssetSignalEngine()
    engine.load_data("2022-01-01", "2024-12-31")

    test_date = "2023-06-15"
    print(f"\nCross-asset signals on {test_date}:")
    print(f"  stocks-bonds corr : {engine.get_stocks_bonds_corr(test_date):+.3f}")
    print(f"  stocks-dollar corr: {engine.get_stocks_dollar_corr(test_date):+.3f}")
    print(f"  stocks-credit corr: {engine.get_stocks_credit_corr(test_date):+.3f}")
    print(f"  yield curve slope : {engine.get_yield_curve_slope(test_date):+.4f}")
    print(f"  dollar strength   : {engine.get_dollar_strength_score(test_date):+.3f}")
    print(f"  credit stress     : {engine.get_credit_stress_score(test_date):+.3f}")
    print(f"  risk_on/off signal: {engine.get_risk_on_off_signal(test_date):+.3f}")
    print(f"  macro regime      : {engine.get_macro_regime(test_date)}")
