"""
realized_covariance.py
======================
Realized Covariance Tracker for dynamic position sizing and risk caps.
Replaces hardcoded sector caps with rolling covariance-based risk management.

Marchenko-Pastur filter:
  Removes noise eigenvalues below λ+ = (1 + sqrt(N/T))^2 * σ²
  where N = number of tickers, T = lookback, σ² = median eigenvalue.

Correlation regimes:
  avg_corr < 0.45  → 'normal'
  0.45 ≤ avg_corr < 0.65 → 'elevated'
  avg_corr ≥ 0.65  → 'crisis'
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class RealizedCovarianceTracker:

    CORR_THRESHOLDS = {'normal': 0.45, 'elevated': 0.65}

    def __init__(self, tickers: List[str]):
        # Exclude inverse ETFs — they are not part of the long portfolio universe
        self.tickers = [t for t in tickers if t not in ('SH', 'PSQ')]
        self._price_data: Optional[Dict[str, pd.DataFrame]] = None

        # Single-entry cache keyed by (date, lookback) for same-bar reuse
        self._cache_date:    Optional[str]            = None
        self._cache_lookback: Optional[int]           = None
        self._cache_ret:     Optional[pd.DataFrame]   = None

    # ── data ingestion ────────────────────────────────────────────────────────

    def update_price_data(self, price_data: Dict[str, pd.DataFrame]):
        """Store full price_data dict. Called once after DataLoader.load()."""
        self._price_data = price_data
        self._cache_date = None  # invalidate cache

    # ── internal return computation ───────────────────────────────────────────

    def _get_returns(self, date: str, lookback: int = 30) -> Optional[pd.DataFrame]:
        """Return a (lookback, N) DataFrame of daily returns ending at `date`.
        Causally correct: only uses data up to and including `date`."""
        if date == self._cache_date and lookback == self._cache_lookback:
            return self._cache_ret

        if self._price_data is None:
            self._cache_date, self._cache_lookback, self._cache_ret = date, lookback, None
            return None

        ts = pd.Timestamp(date)
        returns_dict = {}
        for t in self.tickers:
            if t not in self._price_data:
                continue
            close = self._price_data[t]['close']
            close_up = close[close.index <= ts]
            if len(close_up) < lookback + 1:
                continue
            ret = close_up.iloc[-(lookback + 1):].pct_change().dropna()
            if len(ret) >= max(lookback // 3, 5):
                returns_dict[t] = ret

        if len(returns_dict) < 2:
            self._cache_date, self._cache_lookback, self._cache_ret = date, lookback, None
            return None

        ret_df = pd.DataFrame(returns_dict).dropna()
        result = ret_df if len(ret_df) >= 5 else None

        self._cache_date    = date
        self._cache_lookback = lookback
        self._cache_ret     = result
        return result

    # ── covariance matrix ─────────────────────────────────────────────────────

    def compute_rcov(self, date: str, lookback: int = 30) -> np.ndarray:
        """NxN realized covariance matrix (N = len(self.tickers) or available)."""
        ret_df = self._get_returns(date, lookback)
        if ret_df is None:
            n = len(self.tickers)
            return np.eye(n) * 4e-4   # default: ~2% daily vol on diagonal
        return ret_df.cov().values

    def _rcov_to_corr(self, rcov: np.ndarray) -> np.ndarray:
        """Convert covariance → correlation matrix."""
        d = np.sqrt(np.diag(rcov))
        d = np.where(d > 1e-12, d, 1e-12)
        D_inv = np.diag(1.0 / d)
        return D_inv @ rcov @ D_inv

    # ── public scalar metrics ─────────────────────────────────────────────────

    def get_avg_correlation(self, date: str, lookback: int = 30) -> float:
        """Average off-diagonal pairwise correlation."""
        rcov = self.compute_rcov(date, lookback)
        corr = self._rcov_to_corr(rcov)
        n    = corr.shape[0]
        upper = corr[np.triu_indices(n, k=1)]
        return float(np.nanmean(upper)) if len(upper) > 0 else 0.30

    def get_max_correlation(self, date: str, lookback: int = 30) -> float:
        """Maximum pairwise correlation."""
        rcov = self.compute_rcov(date, lookback)
        corr = self._rcov_to_corr(rcov)
        n    = corr.shape[0]
        upper = corr[np.triu_indices(n, k=1)]
        return float(np.nanmax(upper)) if len(upper) > 0 else 0.50

    def get_portfolio_variance(self, weights: Dict[str, float], date: str) -> float:
        """Predicted portfolio variance: w'Σw.
        `weights` is a dict {ticker: portfolio_weight_fraction}."""
        ret_df = self._get_returns(date, lookback=30)
        if ret_df is None:
            return 1e-4

        available = [t for t in self.tickers if t in ret_df.columns]
        if not available:
            return 1e-4

        rcov = ret_df[available].cov().values
        w    = np.array([weights.get(t, 0.0) for t in available])
        if w.sum() < 1e-12:
            return 0.0

        var = float(w @ rcov @ w)
        return max(var, 0.0)

    def get_diversification_ratio(self, weights: Dict[str, float], date: str) -> float:
        """DR = sum(w_i * σ_i) / sqrt(w'Σw).  DR ≥ 1; higher = more diversified."""
        ret_df = self._get_returns(date, lookback=30)
        if ret_df is None:
            return 1.0

        available = [t for t in self.tickers if t in ret_df.columns]
        if not available:
            return 1.0

        rcov = ret_df[available].cov().values
        w    = np.array([weights.get(t, 0.0) for t in available])
        if w.sum() < 1e-12:
            return 1.0

        port_var = float(w @ rcov @ w)
        port_vol = np.sqrt(max(port_var, 1e-12))
        weighted_vol = float(w @ np.sqrt(np.maximum(np.diag(rcov), 0.0)))
        return float(weighted_vol / port_vol)

    def get_correlation_regime(self, date: str) -> str:
        """'normal' / 'elevated' / 'crisis' based on avg off-diagonal correlation."""
        avg_corr = self.get_avg_correlation(date)
        if avg_corr < self.CORR_THRESHOLDS['normal']:
            return 'normal'
        elif avg_corr < self.CORR_THRESHOLDS['elevated']:
            return 'elevated'
        return 'crisis'

    # ── Marchenko-Pastur eigenvalue filter ───────────────────────────────────

    def apply_marchenko_pastur_filter(self, rcov: np.ndarray) -> np.ndarray:
        """Replace sub-noise eigenvalues with the median eigenvalue.
        Noise threshold: λ+ = (1 + sqrt(N/T))^2 * σ² where σ² = median eigenvalue."""
        N = rcov.shape[0]
        T = 30  # assumed lookback

        # Work in correlation space for numerical stability
        d     = np.sqrt(np.diag(rcov))
        d_safe = np.where(d > 1e-12, d, 1e-12)
        D_inv  = np.diag(1.0 / d_safe)
        corr   = D_inv @ rcov @ D_inv

        eigenvalues, eigenvectors = np.linalg.eigh(corr)

        sigma2      = float(np.median(eigenvalues))
        lambda_plus = (1.0 + np.sqrt(N / T)) ** 2 * max(sigma2, 1e-12)

        # Replace noise eigenvalues with σ²
        filtered = np.where(eigenvalues > lambda_plus, eigenvalues, sigma2)

        # Reconstruct correlation matrix
        corr_clean = eigenvectors @ np.diag(filtered) @ eigenvectors.T

        # Convert back to covariance space
        D = np.diag(d_safe)
        return D @ corr_clean @ D

    # ── per-ticker variance contribution ─────────────────────────────────────

    def get_ticker_var_contribution(self, ticker: str, date: str,
                                    lookback: int = 30) -> float:
        """Variance of `ticker` normalized by cross-section mean variance.
        Returns 1.0 for the average-volatility ticker;
        >1.0 = higher vol (size ↓), <1.0 = lower vol (size ↑)."""
        ret_df = self._get_returns(date, lookback)
        if ret_df is None or ticker not in ret_df.columns:
            return 1.0

        rcov    = ret_df.cov()
        if ticker not in rcov.columns:
            return 1.0

        var_i   = float(rcov.loc[ticker, ticker])
        avg_var = float(np.mean(np.diag(rcov.values)))
        if avg_var <= 1e-12:
            return 1.0

        return var_i / avg_var


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

    tracker = RealizedCovarianceTracker(tickers=tickers)
    tracker.update_price_data(price_data)

    test_date = "2023-06-15"
    rcov = tracker.compute_rcov(test_date, lookback=30)
    print(f"rcov shape: {rcov.shape}")
    print(f"avg correlation : {tracker.get_avg_correlation(test_date):.3f}")
    print(f"max correlation : {tracker.get_max_correlation(test_date):.3f}")
    print(f"correlation regime: {tracker.get_correlation_regime(test_date)}")

    equal_weights = {t: 1.0 / len(tickers) for t in tickers}
    print(f"portfolio variance  : {tracker.get_portfolio_variance(equal_weights, test_date):.6f}")
    print(f"diversification ratio: {tracker.get_diversification_ratio(equal_weights, test_date):.3f}")

    rcov_mp = tracker.apply_marchenko_pastur_filter(rcov)
    eigs_raw = np.linalg.eigvalsh(rcov)
    eigs_mp  = np.linalg.eigvalsh(rcov_mp)
    print(f"eigenvalues raw : {np.round(eigs_raw, 6)}")
    print(f"eigenvalues MP  : {np.round(eigs_mp, 6)}")

    for t in ["AAPL", "SPY", "JPM"]:
        vc = tracker.get_ticker_var_contribution(t, test_date)
        print(f"  {t:<6} var_contrib={vc:.3f}  iv_scale={max(0.5, min(1.5, 1/np.sqrt(max(vc, 1e-6)))):.3f}")
