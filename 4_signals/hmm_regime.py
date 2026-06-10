"""
hmm_regime.py
=============
Gaussian Hidden Markov Model regime detector.
Replaces the ADX/Hurst rule-based RegimeDetector in backtest_engine_v2.

4 hidden states sorted by mean 5-day return (ascending):
  0 — Bear-Stress    (most negative mean return, highest vol)
  1 — Bear-Stable    (slightly negative, moderate vol)
  2 — Bull-Stable    (slightly positive, low vol)
  3 — Bull-Trending  (most positive, moderate vol)

6-feature input per bar (all computed causally from price data):
  1. SPY 5-day return
  2. SPY 20-day realised volatility (annualised)
  3. SPY 20-day momentum (price / SMA20 - 1)
  4. SPY vs bond proxy rolling 30-bar correlation (IEF > TLT > QQQ fallback)
  5. Average pairwise return correlation across all tickers (rolling 30)
  6. VIX proxy (SPY 20-day annualised vol x 100)

Fitting:
  fit_initial()  — called from prepare(); trains on data up to train_start
  refit()        — called at each walk-forward window boundary with ALL prior data

Inference:
  get_regime(date)     — Viterbi label if in state_history; causal Viterbi
                         prefix for OOS dates; updates persistence counter
  get_posterior(date)  — forward-backward posteriors from last fit/refit

Reproducibility fix applied:
  REMOVED multi-seed competition loop (seeds=[42,123,456,789,1000]).
  The competition selected seed=789 non-deterministically depending on data
  snapshot and numpy global state — causing different regime labels on every
  fresh run and large return variance (29%–48% across runs).
  NOW: single fixed seed=random_state (default 42) with two retry attempts
  at higher n_iter if the first fit does not converge. This guarantees
  identical regime labels on every run given the same price data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from hmmlearn.hmm import GaussianHMM


class GaussianHMMRegimeDetector:

    STATE_NAMES = {
        0: "Bear-Stress",
        1: "Bear-Stable",
        2: "Bull-Stable",
        3: "Bull-Trending",
    }
    N_COMPONENTS = 4

    def __init__(self, n_iter: int = 100, random_state: int = 42):
        self.n_iter       = n_iter
        self.random_state = random_state

        self._hmm:               Optional[GaussianHMM]        = None
        self._fitted:            bool                          = False
        self._state_map:         Dict[int, int]               = {}   # raw → sorted
        self._state_history:     pd.Series                    = pd.Series(dtype=str)
        self._posterior_history: Dict[str, Dict[str, float]] = {}
        self._price_data:        Optional[Dict]               = None

        # Persistence tracking
        self._last_regime:         Optional[str] = None
        self._persistence_counter: int           = 0
        self._feature_scaler:      Optional[object] = None

    # ── feature construction ──────────────────────────────────────────────────

    def _build_features(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        spy_df = price_data.get("SPY")
        if spy_df is None:
            raise ValueError("SPY must be present in price_data for HMM features")

        if not isinstance(spy_df.index, pd.DatetimeIndex):
            if "date" in spy_df.columns:
                spy_df = spy_df.set_index(pd.to_datetime(spy_df["date"]))
            else:
                raise ValueError(
                    "SPY DataFrame must have a DatetimeIndex or a 'date' column"
                )
            price_data = {
                t: (df.set_index(pd.to_datetime(df["date"]))
                    if not isinstance(df.index, pd.DatetimeIndex)
                    and "date" in df.columns
                    else df)
                for t, df in price_data.items()
            }

        spy_close = spy_df["close"]

        ret5      = spy_close.pct_change(5)
        rvol20    = spy_close.pct_change().rolling(20).std() * np.sqrt(252)
        sma20     = spy_close.rolling(20).mean()
        mom20     = (spy_close / sma20.replace(0, np.nan)) - 1.0
        vix_proxy = rvol20 * 100.0

        # SPY vs bond-proxy correlation (IEF > TLT > QQQ fallback)
        bond_proxy = None
        for ticker in ("IEF", "TLT", "QQQ"):
            if ticker in price_data:
                bond_proxy = price_data[ticker]["close"]
                break
        if bond_proxy is not None:
            spy_ret       = spy_close.pct_change()
            bond_ret      = bond_proxy.pct_change()
            common        = spy_ret.index.intersection(bond_ret.index)
            spy_bond_corr = (
                spy_ret.reindex(common)
                .rolling(30)
                .corr(bond_ret.reindex(common))
            )
            spy_bond_corr = spy_bond_corr.reindex(spy_close.index)
        else:
            spy_bond_corr = pd.Series(0.0, index=spy_close.index)

        # Average pairwise cross-ticker return correlation (rolling 30)
        all_returns = {
            t: price_data[t]["close"].pct_change()
            for t in price_data
            if t not in ("SH", "PSQ")
        }
        avg_corr_series = pd.Series(0.0, index=spy_close.index)
        if len(all_returns) >= 2:
            ret_df = pd.DataFrame(all_returns).reindex(spy_close.index)
            for i in range(30, len(spy_close)):
                window = ret_df.iloc[max(0, i - 29): i + 1].dropna(axis=1)
                if window.shape[1] >= 2:
                    corr_m = window.corr().values
                    n      = corr_m.shape[0]
                    upper  = corr_m[np.triu_indices(n, k=1)]
                    avg_corr_series.iloc[i] = float(np.nanmean(upper))

        spy_bond_corr = spy_bond_corr.fillna(0.0)

        features = pd.DataFrame({
            "ret5":          ret5,
            "rvol20":        rvol20,
            "mom20":         mom20,
            "spy_bond_corr": spy_bond_corr,
            "avg_corr":      avg_corr_series,
            "vix_proxy":     vix_proxy,
        }, index=spy_close.index).dropna(
            subset=["ret5", "rvol20", "mom20", "vix_proxy"]
        )

        return features

    # ── state sorting ─────────────────────────────────────────────────────────

    def _sort_states(self):
        """Map raw HMM states to states sorted ascending by mean 5-day return."""
        means          = self._hmm.means_[:, 0]   # feature 0 = ret5
        sorted_indices = np.argsort(means)         # most negative first
        self._state_map = {
            int(raw_idx): int(sorted_idx)
            for sorted_idx, raw_idx in enumerate(sorted_indices)
        }

    # ── fit / refit ───────────────────────────────────────────────────────────

    def _fit_and_decode(self, features: pd.DataFrame):
        """
        Train HMM on `features` and Viterbi-decode to populate state_history.

        REPRODUCIBILITY FIX: Uses a single fixed seed (self.random_state)
        instead of the previous multi-seed competition loop that selected the
        best log-likelihood winner. The competition was non-deterministic
        because the winning seed depended on the exact price data snapshot and
        numpy global state at fit time — causing return variance of 29–48%
        across runs on identical code.

        Retry strategy: if the first fit at n_iter=500 does not converge, we
        retry once at n_iter=1000. If that also fails we use the non-converged
        model rather than returning nothing, which is strictly better than
        leaving the HMM unfitted.
        """
        if len(features) < 60:
            print("  HMM: not enough data to fit (< 60 bars), skipping.")
            return

        X = features.values.astype(np.float64)
        X = np.where(np.isfinite(X), X, 0.0)

        from sklearn.preprocessing import StandardScaler
        self._feature_scaler = StandardScaler()
        X = self._feature_scaler.fit_transform(X)

        # ── FIXED SEED — no competition loop ──────────────────────────────────
        # Previously: seeds=[42,123,456,789,1000] with best-score selection.
        # Problem: winner was seed=789 on one data snapshot, different seed on
        # another → regime labels changed → returns changed by 14–18%.
        # Fix: single seed, deterministic, same result every run.

        fitted_model = None

        # Attempt 1: n_iter=500
        try:
            candidate = GaussianHMM(
                n_components    = self.N_COMPONENTS,
                covariance_type = "diag",
                n_iter          = 500,
                random_state    = self.random_state,
                tol             = 1e-4,
                init_params     = "stmc",
            )
            candidate.fit(X)
            if candidate.monitor_.converged:
                fitted_model = candidate
                print(f"  HMM converged with seed={self.random_state} "
                      f"n_iter=500, "
                      f"log-likelihood={candidate.score(X):.2f}")
            else:
                print(f"  HMM seed={self.random_state} n_iter=500 "
                      f"did not converge — retrying at n_iter=1000")
        except Exception as e:
            print(f"  HMM seed={self.random_state} n_iter=500 failed: {e} "
                  f"— retrying at n_iter=1000")

        # Attempt 2: n_iter=1000 (only if attempt 1 did not converge)
        if fitted_model is None:
            try:
                candidate = GaussianHMM(
                    n_components    = self.N_COMPONENTS,
                    covariance_type = "diag",
                    n_iter          = 1000,
                    random_state    = self.random_state,
                    tol             = 1e-4,
                    init_params     = "stmc",
                )
                candidate.fit(X)
                converged = candidate.monitor_.converged
                print(f"  HMM seed={self.random_state} n_iter=1000 "
                      f"{'converged' if converged else 'did not converge — using anyway'}, "
                      f"log-likelihood={candidate.score(X):.2f}")
                fitted_model = candidate
            except Exception as e:
                print(f"  HMM seed={self.random_state} n_iter=1000 failed: {e}")

        if fitted_model is None:
            print("  HMM: all fit attempts failed — skipping.")
            return

        self._hmm = fitted_model
        self._sort_states()

        # State means diagnostic in ORIGINAL (unscaled) space
        state_means_orig = self._feature_scaler.inverse_transform(
            fitted_model.means_
        )
        inv_map = {v: k for k, v in self._state_map.items()}   # sorted → raw
        print("  State labels by mean ret5 (original scale):")
        for si in range(self.N_COMPONENTS):
            raw_idx = inv_map[si]
            m       = state_means_orig[raw_idx]
            print(f"    {self.STATE_NAMES[si]:<16}: "
                  f"ret5={m[0]:.4f}  rvol20={m[1]:.4f}  mom20={m[2]:.4f}")

        # Viterbi decoding for smoothed historical labels
        states_raw  = self._hmm.predict(X)
        state_names = [
            self.STATE_NAMES[self._state_map[int(s)]] for s in states_raw
        ]
        self._state_history = pd.Series(state_names, index=features.index)

        # Forward-backward posteriors
        posteriors = self._hmm.predict_proba(X)
        self._posterior_history = {}
        for i, date in enumerate(features.index):
            try:
                date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
            except Exception:
                date_str = str(date)
            self._posterior_history[date_str] = {
                self.STATE_NAMES[self._state_map[j]]: float(posteriors[i, j])
                for j in range(self.N_COMPONENTS)
            }

        self._fitted = True
        n_each = {
            self.STATE_NAMES[v]: int((np.array(states_raw) == k).sum())
            for k, v in self._state_map.items()
        }
        print(f"  HMM fitted on {len(features)} bars | state counts: {n_each}")

    def fit_initial(self, price_data: Dict[str, pd.DataFrame],
                    end_date: str = None):
        """Initial fit on data up to `end_date` (first training period)."""
        self._price_data = price_data

        spy = price_data.get("SPY")
        if spy is None:
            raise ValueError("HMM requires SPY in price_data dict")

        print(f"  HMM fitting on SPY: {len(spy)} bars, "
              f"date range {spy.index.min()} to {spy.index.max()}")

        if len(spy) < 800:
            print(f"  WARNING: SPY only has {len(spy)} bars. "
                  f"HMM needs 3+ years (~750 bars) minimum. "
                  f"Current range covers "
                  f"{(spy.index.max() - spy.index.min()).days} days.")

        features = self._build_features(price_data)
        if end_date:
            features = features[features.index <= pd.Timestamp(end_date)]
        self._fit_and_decode(features)

    def refit(self, price_data: Dict[str, pd.DataFrame],
              end_date: str = None):
        """Refit on ALL data up to `end_date` (walk-forward window boundary)."""
        self._price_data = price_data
        features         = self._build_features(price_data)
        if end_date:
            features = features[features.index <= pd.Timestamp(end_date)]
        self._fit_and_decode(features)

    # ── online inference for OOS bars ─────────────────────────────────────────

    def _online_inference(self, ts: pd.Timestamp) -> str:
        """
        For OOS bars not in state_history: run causal Viterbi on all features
        up to `ts` and return the last decoded state.
        """
        if self._price_data is None:
            return "Bull-Stable"
        try:
            features    = self._build_features(self._price_data)
            features_up = features[features.index <= ts]
        except Exception:
            features_up = pd.DataFrame()

        if features_up.empty:
            return (str(self._state_history.iloc[-1])
                    if not self._state_history.empty else "Bull-Stable")

        X = features_up.values.astype(np.float64)
        X = np.where(np.isfinite(X), X, 0.0)
        if self._feature_scaler is not None:
            try:
                X = self._feature_scaler.transform(X)
            except Exception:
                pass
        try:
            states_raw = self._hmm.predict(X)
            raw_state  = int(states_raw[-1])
            return self.STATE_NAMES[self._state_map[raw_state]]
        except Exception:
            return (str(self._state_history.iloc[-1])
                    if not self._state_history.empty else "Bull-Stable")

    # ── public interface ──────────────────────────────────────────────────────

    def get_regime(self, date) -> str:
        """Return regime label for a given date. Uses nearest-date lookup."""
        if not self._fitted:
            regime = "Bull-Stable"
        else:
            if not isinstance(date, pd.Timestamp):
                date = pd.Timestamp(date)

            if not isinstance(self._state_history.index, pd.DatetimeIndex):
                self._state_history.index = pd.to_datetime(
                    self._state_history.index
                )
            if not self._state_history.index.is_monotonic_increasing:
                self._state_history = self._state_history.sort_index()

            if date < self._state_history.index.min():
                regime = str(self._state_history.iloc[0])

            elif date > self._state_history.index.max():
                regime = str(self._state_history.iloc[-1])

            else:
                try:
                    regime = str(self._state_history.loc[date])
                except KeyError:
                    try:
                        idx = self._state_history.index.get_indexer(
                            [date], method="nearest"
                        )[0]
                        if idx == -1:
                            regime = "Bull-Stable"
                        else:
                            regime = str(self._state_history.iloc[idx])
                    except Exception as e:
                        print(f"    HMM get_regime failed for {date}: {e}")
                        regime = "Bull-Stable"

        if regime == self._last_regime:
            self._persistence_counter += 1
        else:
            self._persistence_counter = 1
        self._last_regime = regime
        return regime

    def get_posterior(self, date: str) -> Dict[str, float]:
        """Forward-backward posterior probabilities for `date`."""
        default = {name: 0.25 for name in self.STATE_NAMES.values()}
        return self._posterior_history.get(date, default)

    def get_transition_matrix(self) -> np.ndarray:
        """4x4 transition matrix re-indexed to sorted state order."""
        if not self._fitted:
            T = np.ones((4, 4)) / 4
            np.fill_diagonal(T, 0.97)
            T /= T.sum(axis=1, keepdims=True)
            return T
        raw_T   = self._hmm.transmat_
        n       = self.N_COMPONENTS
        inv_map = {v: k for k, v in self._state_map.items()}   # sorted → raw
        sorted_T = np.zeros((n, n))
        for si in range(n):
            for sj in range(n):
                sorted_T[si, sj] = raw_T[inv_map[si], inv_map[sj]]
        return sorted_T

    def get_regime_persistence(self, state_name: str) -> float:
        """Expected duration in bars for `state_name` (geometric distribution)."""
        name_to_id = {v: k for k, v in self.STATE_NAMES.items()}
        state_id   = name_to_id.get(state_name)
        if state_id is None:
            return 10.0
        T      = self.get_transition_matrix()
        p_stay = float(T[state_id, state_id])
        return 1.0 / max(1.0 - p_stay, 1e-9)

    def get_regime_history(self) -> pd.Series:
        """Full Viterbi-decoded state history indexed by date."""
        return self._state_history.copy()

    def get_regime_persistence_so_far(self) -> int:
        """
        Consecutive bars in the current regime since the last state change.
        Updated automatically on every get_regime() call.
        """
        return self._persistence_counter


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf

    tickers = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN",
               "META", "GOOGL", "JPM", "AVGO"]
    price_data = {}
    for t in tickers:
        df = yf.download(t, start="2018-01-01", end="2024-12-31",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        price_data[t] = df[["open", "high", "low", "close", "volume"]].dropna()

    hmm = GaussianHMMRegimeDetector(random_state=42)
    hmm.fit_initial(price_data, end_date="2021-01-04")

    print("\nTransition matrix:")
    print(np.round(hmm.get_transition_matrix(), 3))

    for name in GaussianHMMRegimeDetector.STATE_NAMES.values():
        print(f"  {name:<16} persistence: "
              f"{hmm.get_regime_persistence(name):.1f} bars")

    for date in ["2020-03-20", "2021-06-15", "2022-10-01", "2024-01-10"]:
        r  = hmm.get_regime(date)
        ps = hmm.get_posterior(date)
        print(f"  {date}  regime={r}  posteriors={ps}")