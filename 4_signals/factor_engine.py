"""
factor_engine.py
================
Regime-conditioned factor scoring for long candidate ranking.
Replaces composite_pct-only sorting with ML × factor combined score.

5 factors computed cross-sectionally (z-scored across tickers each bar):
  momentum = price / price_252 - 1
  quality  = -1 * realized_vol_60d          (lower vol = higher quality)
  value    = -1 * (price / price_3yr - 1)   (cheaper rel. to 3yr = higher value)
  lowvol   = -1 * downside_deviation_30d
  growth   = (price / sma50) - 1

Regime factor weights:
  Bull-Trending: momentum=0.5, growth=0.3, quality=0.1, value=0.1, lowvol=0.0
  Bull-Stable:   momentum=0.3, growth=0.2, quality=0.3, value=0.1, lowvol=0.1
  Bear-Stable:   momentum=0.1, growth=0.0, quality=0.4, value=0.3, lowvol=0.2
  Bear-Stress:   momentum=0.0, growth=0.0, quality=0.3, value=0.2, lowvol=0.5
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


FACTOR_REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    'Bull-Trending': {'momentum': 0.5, 'growth': 0.3, 'quality': 0.1, 'value': 0.1, 'lowvol': 0.0},
    'Bull-Stable':   {'momentum': 0.3, 'growth': 0.2, 'quality': 0.3, 'value': 0.1, 'lowvol': 0.1},
    'Bear-Stable':   {'momentum': 0.1, 'growth': 0.0, 'quality': 0.4, 'value': 0.3, 'lowvol': 0.2},
    'Bear-Stress':   {'momentum': 0.0, 'growth': 0.0, 'quality': 0.3, 'value': 0.2, 'lowvol': 0.5},
}

FACTOR_NAMES = ('momentum', 'quality', 'value', 'lowvol', 'growth')


class FactorEngine:
    """
    Computes regime-conditioned factor scores for each ticker.
    Scores are z-standardised cross-sectionally each bar so that
    the combined score is comparable across dates.

    Cache: single-entry per-date cache avoids redundant computation
    when multiple callers ask for the same bar.
    """

    def __init__(self, tickers: List[str]):
        # Exclude inverse ETFs — not part of the long universe
        self.tickers = [t for t in tickers if t not in ('SH', 'PSQ')]
        self._price_data: Optional[Dict[str, pd.DataFrame]] = None

        # Single-entry cache
        self._cache_date:   Optional[str]                      = None
        self._cache_scores: Optional[Dict[str, Dict[str, float]]] = None

    # ── data ingestion ────────────────────────────────────────────────────────

    def update_price_data(self, price_data: Dict[str, pd.DataFrame]):
        """Store full price_data dict. Called once after DataLoader.load()."""
        self._price_data = price_data
        self._cache_date  = None   # invalidate cache

    # ── raw factor computation ────────────────────────────────────────────────

    def _compute_raw_scores(self, date: str) -> Dict[str, Dict[str, float]]:
        """
        Causal raw factor values for each ticker up to `date`.
        Returns {ticker: {factor_name: raw_value}}.
        """
        if self._price_data is None:
            return {}

        ts  = pd.Timestamp(date)
        raw: Dict[str, Dict[str, float]] = {}

        for ticker in self.tickers:
            if ticker not in self._price_data:
                continue
            close_all = self._price_data[ticker]['close']
            close_up  = close_all[close_all.index <= ts]
            n = len(close_up)
            if n < 30:
                continue

            vals: Dict[str, float] = {}
            price_now = float(close_up.iloc[-1])

            # ── Momentum: price / price_252 - 1 ──────────────────────────────
            if n >= 253:
                vals['momentum'] = price_now / float(close_up.iloc[-253]) - 1.0
            else:
                vals['momentum'] = price_now / float(close_up.iloc[0]) - 1.0

            # ── Quality: -1 × 60-day realised vol (annualised) ───────────────
            window60 = close_up.iloc[-min(n, 61):]
            rets60   = window60.pct_change().dropna()
            vals['quality'] = -(float(rets60.std()) * np.sqrt(252)) if len(rets60) > 1 else 0.0

            # ── Value: -1 × (price / price_3yr - 1) ──────────────────────────
            if n >= 757:
                vals['value'] = -(price_now / float(close_up.iloc[-757]) - 1.0)
            elif n >= 253:
                vals['value'] = -(price_now / float(close_up.iloc[-253]) - 1.0)
            else:
                vals['value'] = 0.0

            # ── Low-vol: -1 × downside deviation 30d ─────────────────────────
            rets30 = close_up.iloc[-min(n, 31):].pct_change().dropna()
            down   = rets30[rets30 < 0]
            vals['lowvol'] = -(float(down.std()) * np.sqrt(252)) if len(down) > 1 else 0.0

            # ── Growth: price / SMA_50 - 1 ───────────────────────────────────
            sma50 = float(close_up.iloc[-min(n, 50):].mean())
            vals['growth'] = price_now / max(sma50, 1e-9) - 1.0

            raw[ticker] = vals

        return raw

    def _z_score_cross_section(
        self, raw: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Z-standardise each factor cross-sectionally.
        Returns {ticker: {factor: z_score}}.
        """
        tickers = list(raw.keys())
        scores: Dict[str, Dict[str, float]] = {t: {} for t in tickers}

        if len(tickers) < 2:
            for t in tickers:
                for f in FACTOR_NAMES:
                    scores[t][f] = 0.0
            return scores

        for factor in FACTOR_NAMES:
            vals = np.array([raw[t].get(factor, np.nan) for t in tickers])
            valid = vals[np.isfinite(vals)]
            if len(valid) < 2:
                for t in tickers:
                    scores[t][factor] = 0.0
                continue
            mu  = float(valid.mean())
            std = float(valid.std(ddof=1))
            std = max(std, 1e-9)
            for i, t in enumerate(tickers):
                scores[t][factor] = float((vals[i] - mu) / std) if np.isfinite(vals[i]) else 0.0

        return scores

    # ── public interface ──────────────────────────────────────────────────────

    def compute_factor_scores(self, date: str) -> Dict[str, Dict[str, float]]:
        """
        Z-scored factor scores for all tickers on `date` (causal).
        Returns {ticker: {factor: z_score}}.
        Cached per-date: same call within one bar is free after first call.
        """
        if date == self._cache_date and self._cache_scores is not None:
            return self._cache_scores

        raw    = self._compute_raw_scores(date)
        scores = self._z_score_cross_section(raw)

        self._cache_date   = date
        self._cache_scores = scores
        return scores

    def get_factor_preference(self, regime: str,
                              inflation_tilt: Dict[str, float] = None) -> Dict[str, float]:
        """
        Regime-conditioned factor weights, optionally adjusted by inflation tilt.
        inflation_tilt: {'growth': delta, 'value': delta} from InflationSignalEngine.
        Tilts are added to base weights then renormalized to sum to 1.0.
        """
        base = dict(FACTOR_REGIME_WEIGHTS.get(regime, {f: 0.2 for f in FACTOR_NAMES}))
        if inflation_tilt:
            base['growth'] = base.get('growth', 0.0) + inflation_tilt.get('growth', 0.0)
            base['value']  = base.get('value',  0.0) + inflation_tilt.get('value',  0.0)
            total = sum(max(v, 0.0) for v in base.values())
            if total > 1e-9:
                base = {k: max(v, 0.0) / total for k, v in base.items()}
        return base

    def get_combined_factor_score(self, ticker: str, date: str, regime: str,
                                  inflation_tilt: Dict[str, float] = None) -> float:
        """
        Weighted sum of z-scored factor scores conditioned on `regime`.
        Returns a scalar in roughly [-2, +2]; 0.0 for unknown tickers.
        inflation_tilt: optional {'growth': delta, 'value': delta} applied before normalization.
        """
        scores = self.compute_factor_scores(date)
        if ticker not in scores:
            return 0.0
        weights = self.get_factor_preference(regime, inflation_tilt=inflation_tilt)
        s = scores[ticker]
        return float(sum(weights.get(f, 0.0) * s.get(f, 0.0) for f in FACTOR_NAMES))

    def get_top_factor_tickers(
        self, factor_name: str, date: str, n: int = 3
    ) -> List[str]:
        """Return top-n tickers ranked descending by the given factor z-score."""
        scores = self.compute_factor_scores(date)
        ranked = sorted(scores.items(), key=lambda kv: kv[1].get(factor_name, 0.0), reverse=True)
        return [t for t, _ in ranked[:n]]

    def get_dominant_factor(self, regime: str) -> str:
        """Factor with highest weight in current regime."""
        weights = self.get_factor_preference(regime)
        return max(weights, key=weights.get) if weights else 'momentum'

    def compute_factor_ic(
        self,
        date:             str,
        regime:           str,
        realized_returns: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Spearman rank IC between each factor z-score and realized returns.
        Requires at least 3 tickers in common. Returns {factor: IC}.
        """
        scores      = self.compute_factor_scores(date)
        common      = [t for t in scores if t in realized_returns]
        default_ic  = {f: 0.0 for f in FACTOR_NAMES}
        if len(common) < 3:
            return default_ic

        rets      = np.array([realized_returns[t] for t in common])
        ret_ranks = rets.argsort().argsort().astype(float)

        ic: Dict[str, float] = {}
        for factor in FACTOR_NAMES:
            fvals   = np.array([scores[t].get(factor, 0.0) for t in common])
            f_ranks = fvals.argsort().argsort().astype(float)
            if f_ranks.std() < 1e-9 or ret_ranks.std() < 1e-9:
                ic[factor] = 0.0
            else:
                ic[factor] = float(np.corrcoef(f_ranks, ret_ranks)[0, 1])

        return ic


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf

    tickers = ["AAPL", "NVDA", "AMZN", "GOOGL", "META", "MSFT", "SPY", "QQQ", "JPM", "AVGO"]
    price_data = {}
    for t in tickers:
        df = yf.download(t, start="2022-01-01", end="2024-12-31",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        price_data[t] = df[["open", "high", "low", "close", "volume"]].dropna()

    engine = FactorEngine(tickers=tickers)
    engine.update_price_data(price_data)

    test_date = "2023-06-15"
    scores = engine.compute_factor_scores(test_date)
    print(f"\nFactor scores on {test_date}:")
    for ticker, s in sorted(scores.items()):
        row = "  ".join(f"{f[:4]}={s[f]:+.2f}" for f in FACTOR_NAMES)
        print(f"  {ticker:<6}  {row}")

    for regime in ('Bull-Trending', 'Bull-Stable', 'Bear-Stable', 'Bear-Stress'):
        print(f"\n{regime}:")
        print(f"  dominant factor : {engine.get_dominant_factor(regime)}")
        for t in tickers[:5]:
            cfs = engine.get_combined_factor_score(t, test_date, regime)
            print(f"  {t:<6}  combined={cfs:+.3f}")
