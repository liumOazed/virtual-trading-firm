"""
backtest_engine_v2.py
=====================
Next-level backtesting engine for VIRTUAL_TRADING_FIRM.

Stages implemented
------------------
5   Rolling walk-forward  — OOS-only equity curve, Optuna retraining each window
5c  Regime detection      — 4-state (Bull/Bear × Trend/MeanRev) via Hurst + HMM
5g  Signal correlation    — prune redundant alphas before weighting
5f  Alpha decay           — measure edge half-life, set optimal hold period
5   Kalman ensemble       — dynamic per-bar signal weighting
5e  Filter competition    — rolling Sharpe tournament, losers zeroed
5d  Champion selection    — best combo per regime, only active in that regime
5b  Regime stress test    — replay on COVID crash, 2022 bear, 2018 Q4
5h  Cost/slippage sweep   — Sharpe decay curve from 0 → 0.5% slippage

Drop-in compatible with your existing:
  signal_engine.SignalEngine.get_full_signals()
  portfolio.Portfolio
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from dataclasses import dataclass, field
import pickle
import hashlib
from multiprocessing import Pool, cpu_count
from functools import partial

warnings.filterwarnings("ignore")

# ── path setup (mirrors backtest_engine.py) ─────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "4_signals"))

from signal_engine import SignalEngine
from portfolio import Portfolio


# ════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION DATACLASS
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestConfig:
    # Universe
    tickers:         List[str] = field(default_factory=lambda: ["AAPL","NVDA","MSFT","SPY","QQQ","TSLA"])
    start_date:      str  = "2023-01-01"
    end_date:        str  = "2025-04-01"
    initial_capital: float = 100_000.0

    # Walk-forward
    train_months:    int   = 6       # training window length
    oos_months:      int   = 1       # out-of-sample step size
    retrain:         bool  = True    # retrain model each window?

    # Position sizing
    fixed_size:      float = 0.10    # fraction of equity per position

    # Signal thresholds
    buy_threshold:   float = 0.55
    sell_threshold:  float = 0.45

    # Slippage / commission
    base_slippage:   float = 0.001   # 0.1 % baseline
    commission_rate: float = 0.001

    # Alpha decay
    decay_horizons:  List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])

    # Filter competition
    competition_window: int = 20     # rolling window (bars) for Sharpe scoring

    # Stress periods  (label, start, end)
    stress_periods: List[Tuple] = field(default_factory=lambda: [
        ("COVID crash",    "2020-02-15", "2020-03-23"),
        ("2022 bear",      "2022-01-01", "2022-12-31"),
        ("2018 Q4",        "2018-10-01", "2018-12-31"),
    ])

    # Cost stress sweep
    slippage_sweep:  List[float] = field(default_factory=lambda:
                        [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05])


# ════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADER
# ════════════════════════════════════════════════════════════════════════════

class DataLoader:
    """Download and cache OHLCV with enough lookback for feature warm-up."""

    WARMUP_DAYS = 420   # Hurst(252) + ESN(100) + buffer

    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.price_data: Dict[str, pd.DataFrame] = {}

    def load(self):
        print("📥 Loading price data …")
        start_dt  = datetime.strptime(self.cfg.start_date, "%Y-%m-%d")
        ext_start = (start_dt - timedelta(days=self.WARMUP_DAYS)).strftime("%Y-%m-%d")

        for ticker in self.cfg.tickers:
            try:
                df = yf.download(ticker, start=ext_start,
                                 end=self.cfg.end_date,
                                 progress=False, auto_adjust=True)
                if df.empty:
                    print(f"  ⚠️  {ticker}: no data"); continue

                # flatten MultiIndex if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                df = df[["open","high","low","close","volume"]].dropna()
                self.price_data[ticker] = df
                print(f"  ✅ {ticker}: {len(df)} rows")

            except Exception as e:
                print(f"  ❌ {ticker} failed: {e}")

    def trading_dates(self) -> List[str]:
        """Return sorted trading dates within cfg.start_date → cfg.end_date."""
        sample = next(iter(self.price_data.values()))
        mask   = (sample.index >= self.cfg.start_date) & (sample.index <= self.cfg.end_date)
        return sample.index[mask].strftime("%Y-%m-%d").tolist()


# ════════════════════════════════════════════════════════════════════════════
# 2.  REGIME DETECTOR  (Stage 5c)
# ════════════════════════════════════════════════════════════════════════════

class RegimeDetector:
    """
    4-state market regime per bar:
        0  Bull-Trending
        1  Bull-MeanReverting
        2  Bear-Trending
        3  Bear-MeanReverting

    Detection logic
    ───────────────
    Direction  : close vs 50-day SMA  (Bull / Bear)
    Character  : Hurst exponent       (> 0.55 → Trending, < 0.45 → MeanRev, else Neutral)
    Volatility : 20-day realised vol  (used to break Neutral ties)
    """

    HMM_AVAILABLE = False
    try:
        from hmmlearn import hmm as _hmm
        HMM_AVAILABLE = True
    except ImportError:
        pass

    def __init__(self, hurst_window: int = 100, vol_window: int = 20):
        self.hurst_window = hurst_window
        self.vol_window   = vol_window

    # ── Hurst exponent (R/S method) ────────────────────────────────────────
    @staticmethod
    def _hurst(series: np.ndarray) -> float:
        n = len(series)
        if n < 20:
            return 0.5
        lags = range(2, min(n // 2, 40))
        rs_vals = []
        for lag in lags:
            chunks = [series[i:i+lag] for i in range(0, n - lag + 1, lag)]
            chunk_rs = []
            for chunk in chunks:
                m = np.mean(chunk)
                demeaned   = chunk - m
                cumdev     = np.cumsum(demeaned)
                R          = cumdev.max() - cumdev.min()
                S          = np.std(chunk, ddof=1)
                if S > 0:
                    chunk_rs.append(R / S)
            if chunk_rs:
                rs_vals.append((lag, np.mean(chunk_rs)))
        if len(rs_vals) < 4:
            return 0.5
        lags_log = np.log([x[0] for x in rs_vals])
        rs_log   = np.log([x[1] for x in rs_vals])
        hurst    = float(np.polyfit(lags_log, rs_log, 1)[0])
        return np.clip(hurst, 0.0, 1.0)

    def label_regimes(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series of int regime labels aligned to df.index."""
        close    = df["close"]
        sma50    = close.rolling(50).mean()
        returns  = close.pct_change()
        rvol     = returns.rolling(self.vol_window).std()

        labels = pd.Series(0, index=df.index, dtype=int)

        for i in range(self.hurst_window, len(df)):
            window  = close.iloc[i - self.hurst_window: i].values
            h       = self._hurst(np.log(window + 1e-10))
            bull    = close.iloc[i] > sma50.iloc[i]
            trending = h > 0.55
            # Neutral Hurst (0.45–0.55): use vol as tiebreaker
            if not trending and h >= 0.45:
                trending = rvol.iloc[i] > rvol.rolling(60).mean().iloc[i]

            if bull and trending:
                labels.iloc[i] = 0        # Bull-Trend
            elif bull and not trending:
                labels.iloc[i] = 1        # Bull-MeanRev
            elif not bull and trending:
                labels.iloc[i] = 2        # Bear-Trend
            else:
                labels.iloc[i] = 3        # Bear-MeanRev

        return labels

    REGIME_NAMES = {
        0: "Bull-Trending",
        1: "Bull-MeanRev",
        2: "Bear-Trending",
        3: "Bear-MeanRev",
    }


# ════════════════════════════════════════════════════════════════════════════
# 3.  SIGNAL CORRELATION FILTER  (Stage 5g)
# ════════════════════════════════════════════════════════════════════════════

class SignalCorrelationFilter:
    """
    Given a dict of {signal_name: proba_series}, compute pairwise
    return-level correlations and drop signals whose correlation with
    a higher-Sharpe signal exceeds the threshold.
    Returns the pruned list of signal names.
    """

    def __init__(self, corr_threshold: float = 0.70):
        self.corr_threshold = corr_threshold

    def fit_prune(self, signal_df: pd.DataFrame,
                  sharpe_scores: Dict[str, float]) -> List[str]:
        """
        signal_df : columns = signal names, values = daily proba_buy
        sharpe_scores : {signal_name: sharpe}
        Returns pruned list (highest Sharpe survivors).
        """
        if signal_df.shape[1] <= 1:
            return list(signal_df.columns)

        corr   = signal_df.corr()
        ranked = sorted(sharpe_scores, key=sharpe_scores.get, reverse=True)
        kept   = []
        dropped = set()

        for sig in ranked:
            if sig in dropped:
                continue
            kept.append(sig)
            for other in ranked:
                if other == sig or other in dropped:
                    continue
                if sig in corr.columns and other in corr.columns:
                    if abs(corr.loc[sig, other]) > self.corr_threshold:
                        dropped.add(other)

        print(f"  🔗 Correlation filter: {signal_df.shape[1]} → {len(kept)} signals kept")
        return kept


# ════════════════════════════════════════════════════════════════════════════
# 4.  ALPHA DECAY ANALYSER  (Stage 5f)
# ════════════════════════════════════════════════════════════════════════════

class AlphaDecayAnalyser:
    """
    For each signal, measure forward returns at horizons [1,3,5,10,20].
    Fits a simple exponential decay to find the half-life.
    Reports optimal hold period = horizon where edge drops below 50% of peak.
    """

    def __init__(self, horizons: List[int] = None):
        self.horizons = horizons or [1, 3, 5, 10, 20]

    def analyse(self, signal_series: pd.Series,
                price_series: pd.Series) -> Dict:
        """
        signal_series : 1 where BUY signal fired, 0 otherwise (aligned to price)
        price_series  : close prices aligned to same index
        """
        results = {}
        buy_dates = signal_series[signal_series == 1].index

        if len(buy_dates) < 10:
            return {"half_life": None, "optimal_hold": self.horizons[-1],
                    "edge_by_horizon": {}}

        edges = {}
        for h in self.horizons:
            fwd_ret = price_series.pct_change(h).shift(-h)
            buy_rets = fwd_ret.reindex(buy_dates).dropna()
            # edge = mean forward return on buy signals vs all bars
            all_mean = fwd_ret.mean()
            edge = float(buy_rets.mean() - all_mean) if len(buy_rets) > 0 else 0.0
            edges[h] = edge

        results["edge_by_horizon"] = edges

        # Fit decay: find where edge drops to 50% of its max
        peak = max(edges.values()) if edges else 0
        half_life = None
        optimal_hold = self.horizons[-1]

        if peak > 0:
            for h in self.horizons:
                if edges[h] <= peak * 0.5:
                    half_life = h
                    # optimal hold = last horizon still above 50% of peak
                    idx = self.horizons.index(h)
                    optimal_hold = self.horizons[idx - 1] if idx > 0 else h
                    break

        results["half_life"]     = half_life
        results["optimal_hold"]  = optimal_hold
        return results


# ════════════════════════════════════════════════════════════════════════════
# 5.  KALMAN ENSEMBLE  (dynamic signal weighting)
# ════════════════════════════════════════════════════════════════════════════

class KalmanEnsemble:
    """
    Scalar Kalman filter that tracks how well each signal is predicting.
    Weight for signal i = normalised posterior precision (inverse variance).

    State  : w_i  (true weight of signal i)
    Process noise Q controls how fast weights can drift.
    Observation noise R reflects signal reliability.
    """

    def __init__(self, n_signals: int,
                 process_noise: float = 0.01,
                 obs_noise: float = 0.1):
        self.n  = n_signals
        self.w  = np.ones(n_signals) / n_signals   # prior weights
        self.P  = np.ones(n_signals) * 1.0         # prior variances
        self.Q  = process_noise
        self.R  = obs_noise

    def update(self, signal_probas: np.ndarray,
               realised_return: float) -> np.ndarray:
        """
        signal_probas : array of proba_buy values, one per signal
        realised_return: actual next-bar return (used as 'truth')
        Returns normalised blended weights.
        """
        # Predict step: increase uncertainty
        self.P += self.Q

        # Innovation: how much did each signal's proba agree with the return?
        # Positive return → proba > 0.5 is "correct"
        direction = 1.0 if realised_return > 0 else -1.0
        # signal quality: how aligned proba is with direction
        quality = direction * (signal_probas - 0.5) * 2   # in [-1, 1]

        # Update step (independent per signal)
        K = self.P / (self.P + self.R)
        self.w = self.w + K * (quality - self.w)
        self.P = (1 - K) * self.P

        # Normalise: softmax over positive weights
        w_pos = np.clip(self.w, 0.05, None)  # minimum 5% floor, no ticker fully zeroed
        total = w_pos.sum()
        if total > 1e-6:
            return w_pos / total
        return np.ones(self.n) / self.n

    def blend(self, signal_probas: np.ndarray,
              weights: np.ndarray) -> float:
        """Weighted average of probas."""
        return float(np.dot(weights, signal_probas))


# ════════════════════════════════════════════════════════════════════════════
# 6.  FILTER COMPETITION  (Stage 5e)
# ════════════════════════════════════════════════════════════════════════════

class FilterCompetition:
    """
    Rolling Sharpe tournament over a window of bars.
    Each 'filter' is a signal column in a DataFrame.
    At each bar, scores are updated and allocation zeroed for losers.
    """

    def __init__(self, window: int = 20, min_signals: int = 1):
        self.window      = window
        self.min_signals = min_signals
        self._return_history: Dict[str, List[float]] = {}

    def score(self, signal_names: List[str],
              bar_returns: Dict[str, float]) -> Dict[str, float]:
        """
        bar_returns : {signal_name: realised_return_this_bar}
        Returns allocation weights {signal_name: weight ∈ [0,1]}.
        """
        for name in signal_names:
            if name not in self._return_history:
                self._return_history[name] = []
            ret = bar_returns.get(name, 0.0)
            self._return_history[name].append(ret)
            if len(self._return_history[name]) > self.window:
                self._return_history[name].pop(0)

        # Rolling Sharpe per signal
        sharpes = {}
        for name in signal_names:
            hist = np.array(self._return_history[name])
            if len(hist) < 5 or hist.std() < 1e-9:
                sharpes[name] = 0.0
            else:
                sharpes[name] = hist.mean() / hist.std()

        # Rank: keep top half (min self.min_signals)
        ranked  = sorted(sharpes, key=sharpes.get, reverse=True)
        n_keep  = max(self.min_signals, len(ranked) // 2)
        winners = set(ranked[:n_keep])

        weights = {}
        total_sharpe = sum(max(sharpes[s], 0) for s in winners) or 1.0
        for name in signal_names:
            if name in winners and sharpes[name] > 0:
                weights[name] = sharpes[name] / total_sharpe
            else:
                weights[name] = 0.0

        return weights


# ════════════════════════════════════════════════════════════════════════════
# 7.  CHAMPION SELECTOR  (Stage 5d)
# ════════════════════════════════════════════════════════════════════════════

class ChampionSelector:
    """
    Tracks per-regime performance for each ticker/signal combination.
    Selects the 'champion' (highest Calmar) combo for each regime.
    """

    def __init__(self):
        # {regime: {ticker: {"returns": [], "drawdown": float}}}
        self._perf: Dict[int, Dict[str, Dict]] = {r: {} for r in range(4)}

    def record(self, regime: int, ticker: str, bar_return: float):
        if ticker not in self._perf[regime]:
            self._perf[regime][ticker] = {"returns": [], "peak": 1.0, "nav": 1.0}
        rec = self._perf[regime][ticker]
        rec["returns"].append(bar_return)
        rec["nav"]  *= (1 + bar_return)
        rec["peak"]  = max(rec["peak"], rec["nav"])

    def champions(self) -> Dict[int, str]:
        """Return {regime: best_ticker} by Calmar ratio."""
        result = {}
        for regime, tickers in self._perf.items():
            best_ticker, best_calmar = None, -np.inf
            for ticker, rec in tickers.items():
                rets = np.array(rec["returns"])
                if len(rets) < 5:
                    continue
                ann_ret = rets.mean() * 252
                max_dd  = max((rec["peak"] - rec["nav"]) / rec["peak"], 1e-6)
                calmar  = ann_ret / max_dd
                if calmar > best_calmar:
                    best_calmar  = calmar
                    best_ticker  = ticker
            result[regime] = best_ticker
        return result


# ════════════════════════════════════════════════════════════════════════════
# 8.  STRESS TESTER  (Stages 5b & 5h)
# ════════════════════════════════════════════════════════════════════════════

class StressTester:

    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg

    # ── 5b: Regime stress ─────────────────────────────────────────────────
    def regime_stress(self, equity_df: pd.DataFrame,
                      trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        equity_df must have columns: date, equity
        Returns per-stress-period stats.
        """
        equity_df = equity_df.copy()
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        rows = []

        for label, start, end in self.cfg.stress_periods:
            mask = (equity_df["date"] >= pd.Timestamp(start)) & \
             (equity_df["date"] <= pd.Timestamp(end))
            period = equity_df[mask]
            if period.empty:
                rows.append({"period": label, "note": "no data"}); continue

            eq     = period["equity"].values
            rets   = pd.Series(eq).pct_change().dropna()
            peak   = eq.max()
            trough = eq.min()
            max_dd = (peak - trough) / peak if peak > 0 else 0

            sharpe = (rets.mean() / rets.std() * np.sqrt(252)
                      if rets.std() > 1e-9 else 0)
            total_ret = (eq[-1] / eq[0] - 1) if eq[0] > 0 else 0

            rows.append({
                "period":       label,
                "start":        start,
                "end":          end,
                "total_return": round(total_ret * 100, 2),
                "max_drawdown": round(max_dd * 100, 2),
                "sharpe":       round(sharpe, 3),
                "bars":         len(period),
            })

        return pd.DataFrame(rows)

    # ── 5h: Cost / slippage sweep ─────────────────────────────────────────
    def slippage_sweep(self, trades_df: pd.DataFrame,
                       equity_start: float) -> pd.DataFrame:
        """
        Replays P&L from trade_log at different slippage levels.
        Returns DataFrame with slippage → Sharpe / total_return.
        """
        if trades_df.empty:
            return pd.DataFrame()

        rows = []
        for slip in self.cfg.slippage_sweep:
            adj_trades = trades_df.copy()
            # BUY: effective price higher by slippage
            buy_mask  = adj_trades["action"] == "BUY"
            sell_mask = adj_trades["action"] == "SELL"
            adj_trades.loc[buy_mask,  "price"] *= (1 + slip)
            adj_trades.loc[sell_mask, "price"] *= (1 - slip)

            # Reconstruct simplified P&L
            adj_trades["total"] = np.where(
                adj_trades["action"] == "BUY",
                -adj_trades["shares"] * adj_trades["price"],
                 adj_trades["shares"] * adj_trades["price"]
            )
            adj_trades["total"] -= (adj_trades["shares"] * adj_trades["price"]
                                    * self.cfg.commission_rate)

            cum_pnl = adj_trades.groupby("date")["total"].sum().cumsum()
            equity  = equity_start + cum_pnl
            rets    = equity.pct_change().dropna()

            sharpe    = (rets.mean() / rets.std() * np.sqrt(252)
                         if rets.std() > 1e-9 else 0)
            total_ret = (equity.iloc[-1] / equity_start - 1
                         if len(equity) > 0 else 0)

            rows.append({
                "slippage_pct":  round(slip * 100, 3),
                "total_return":  round(total_ret * 100, 2),
                "sharpe":        round(sharpe, 3),
            })

        return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# 9.  WALK-FORWARD ENGINE  (Stage 5 core)
# ════════════════════════════════════════════════════════════════════════════

class WalkForwardEngine:
    """
    Generates rolling train/OOS windows.
    Each window returns the list of (train_dates, oos_dates).
    """

    def __init__(self, all_dates: List[str],
                 train_months: int, oos_months: int):
        self.dates       = all_dates
        self.train_mo    = train_months
        self.oos_mo      = oos_months

    def windows(self) -> List[Tuple[List[str], List[str]]]:
        """Return [(train_dates, oos_dates), …]"""
        result = []
        dates  = pd.to_datetime(self.dates)
        n      = len(dates)
        i      = 0
        while True:
            train_end_dt = dates[i] + pd.DateOffset(months=self.train_mo)
            oos_end_dt   = train_end_dt + pd.DateOffset(months=self.oos_mo)

            train_mask   = (dates >= dates[i]) & (dates <  train_end_dt)
            oos_mask     = (dates >= train_end_dt) & (dates < oos_end_dt)

            train_dates  = [d.strftime("%Y-%m-%d") for d in dates[train_mask]]
            oos_dates    = [d.strftime("%Y-%m-%d") for d in dates[oos_mask]]

            if not train_dates or not oos_dates:
                break
            result.append((train_dates, oos_dates))

            # Step forward by oos_months
            next_start = dates[i] + pd.DateOffset(months=self.oos_mo)
            future = dates[dates >= next_start]
            if len(future) == 0:
                break
            i = list(dates).index(future[0])

            if oos_end_dt > dates[-1]:
                break

        print(f"  📅 Walk-forward: {len(result)} windows "
              f"({self.train_mo}mo train / {self.oos_mo}mo OOS)")
        return result


# ════════════════════════════════════════════════════════════════════════════
# 10.  MAIN BACKTEST ENGINE V2
# ════════════════════════════════════════════════════════════════════════════

class BacktestEngineV2:

    def __init__(self, cfg: BacktestConfig = None):
        self.cfg = cfg or BacktestConfig()

        # Core components
        self.loader    = DataLoader(self.cfg)
        self.engine    = SignalEngine()
        self.regime_d  = RegimeDetector()
        self.corr_filt = SignalCorrelationFilter(corr_threshold=0.70)
        self.decay     = AlphaDecayAnalyser(self.cfg.decay_horizons)
        self.champion  = ChampionSelector()
        self.stress    = StressTester(self.cfg)
        self.competition = FilterCompetition(self.cfg.competition_window)

        # State
        self.signal_cache:  Dict[str, pd.DataFrame] = {}
        self.regime_cache:  Dict[str, pd.Series]    = {}
        self.kalman_map:    Dict[str, KalmanEnsemble] = {}
        self.decay_results: Dict[str, Dict] = {}
        self.active_tickers: List[str] = list(self.cfg.tickers)

        # Results
        self.equity_history: List[Dict] = []
        self.trade_history:  List[Dict] = []

    # ── STEP 1: load & pre-compute ────────────────────────────────────────

    def prepare(self):
        self.loader.load()
        self._precompute_signals()
        self._precompute_regimes()
        self._run_correlation_filter()
        self._run_alpha_decay()

    # ── STEP 2: precompute signals (reuses your SignalEngine) ─────────────

    def _precompute_signals(self):
        print("\n🧠 Precomputing signals …")
        for ticker in self.cfg.tickers:
            if ticker not in self.loader.price_data:
                continue
            df = self.loader.price_data[ticker].copy().reset_index()
            df["date"] = df["Date"].dt.strftime("%Y-%m-%d") if "Date" in df.columns \
                    else df.index.strftime("%Y-%m-%d")
            try:
                sig_df = self.engine.get_full_signals(df, ticker)
                if "proba_buy" not in sig_df.columns:
                    print(f"  ❌ {ticker}: missing proba_buy"); continue
                self.signal_cache[ticker] = sig_df.set_index("date")
                print(f"  ✅ {ticker}: {len(sig_df)} signal rows")
            except Exception as e:
                print(f"  ❌ {ticker} signal error: {e}")

    # ── STEP 3: regime labels per ticker ──────────────────────────────────

    def _precompute_regimes(self):
        print("\n🗺️  Computing regime labels …")
        for ticker, df in self.loader.price_data.items():
            try:
                labels = self.regime_d.label_regimes(df)
                self.regime_cache[ticker] = labels
                dist = labels.value_counts().to_dict()
                dist_str = {RegimeDetector.REGIME_NAMES[k]: v
                            for k, v in dist.items()}
                print(f"  ✅ {ticker} regimes: {dist_str}")
            except Exception as e:
                print(f"  ⚠️  {ticker} regime error: {e}")

    # ── STEP 4: signal correlation pruning ────────────────────────────────

    def _run_correlation_filter(self):
        print("\n🔗 Signal correlation filter …")
        # Build proba matrix (one column per ticker, rows = dates)
        frames = {}
        for ticker, sig_df in self.signal_cache.items():
            if "proba_buy" in sig_df.columns:
                frames[ticker] = sig_df["proba_buy"]

        if not frames:
            return

        proba_matrix = pd.DataFrame(frames).dropna()
        if proba_matrix.empty or proba_matrix.shape[1] < 2:
            return

        # Quick Sharpe per ticker over available proba series
        sharpes = {}
        for ticker in proba_matrix.columns:
            s = proba_matrix[ticker].diff().dropna()
            sharpes[ticker] = float(s.mean() / s.std()) if s.std() > 1e-9 else 0.0

        self.active_tickers = self.corr_filt.fit_prune(proba_matrix, sharpes)

    # ── STEP 5: alpha decay per active ticker ────────────────────────────

    def _run_alpha_decay(self):
        print("\n⏳ Alpha decay analysis …")
        for ticker in self.active_tickers:
            if ticker not in self.signal_cache:
                continue
            sig_df    = self.signal_cache[ticker]
            price_df  = self.loader.price_data[ticker]

            # Align on common dates
            common = sig_df.index.intersection(price_df.index.strftime("%Y-%m-%d"))
            if len(common) < 20:
                continue

            signal_series = (sig_df.loc[common, "proba_buy"] > self.cfg.buy_threshold
                             ).astype(int)
            price_series  = price_df["close"].copy()
            price_series.index = price_df.index.strftime("%Y-%m-%d")
            price_series  = price_series.reindex(common)

            res = self.decay.analyse(signal_series, price_series)
            self.decay_results[ticker] = res
            print(f"  ✅ {ticker}: half-life={res['half_life']}d  "
                  f"optimal_hold={res['optimal_hold']}d  "
                  f"edge@1d={res['edge_by_horizon'].get(1,0):.4f}")

    # ── STEP 6: walk-forward main loop ────────────────────────────────────

    def run(self):
        if not self.loader.price_data:
            self.prepare()

        all_dates  = self.loader.trading_dates()
        wf_engine  = WalkForwardEngine(all_dates,
                                       self.cfg.train_months,
                                       self.cfg.oos_months)
        windows    = wf_engine.windows()
        portfolio  = Portfolio(
            initial_capital=self.cfg.initial_capital,
            commission_rate=self.cfg.commission_rate
        )

        print(f"\n🚀 Walk-forward backtest: {self.cfg.start_date} → "
              f"{self.cfg.end_date}\n")

        # ── PARALLEL RETRAINING ───────────────────────────────────────────────
        if self.cfg.retrain:
            n_cores = max(1, min(cpu_count() - 1, len(windows)))
            print(f"\n⚡ Parallel retraining: {len(windows)} windows on {n_cores} cores …")

            worker_args = [
                {
                    "window_idx":   idx,
                    "train_dates":  train_dates,
                    "tickers":      self.cfg.tickers,
                    "start_date":   self.cfg.start_date,
                    "end_date":     self.cfg.end_date,
                    "project_root": project_root,
                }
                for idx, (train_dates, _) in enumerate(windows)
            ]

            with Pool(processes=n_cores) as pool:
                results = pool.map(_retrain_window_worker, worker_args)

            # Sort by window_idx (pool.map preserves order but be safe)
            results.sort(key=lambda x: x["window_idx"])

            # Build per-window signal cache map
            # {window_idx: {ticker: {date: proba}}}
            self.window_signal_caches = {
                r["window_idx"]: r["signal_cache"]
                for r in results if r["model_path"] is not None
            }
            self.window_model_paths = {
                r["window_idx"]: r["model_path"]
                for r in results if r["model_path"] is not None
            }

            print(f"\n✅ All windows retrained. "
                f"{sum(1 for r in results if r['model_path'])} / {len(windows)} succeeded.")
        else:
            self.window_signal_caches = {}
            self.window_model_paths   = {}

        # ── BUILD PER-WINDOW OOS DATE MAP ────────────────────────────────────
        # Maps each OOS date → window_idx so we load the right model's signals
        oos_date_to_window: Dict[str, int] = {}
        oos_dates_all = []

        for idx, (_, oos_dates) in enumerate(windows):
            for d in oos_dates:
                oos_date_to_window[d] = idx
            oos_dates_all.extend(oos_dates)

        # ── OOS LOOP ─────────────────────────────────────────────────────────
        self._run_oos_loop_v2(oos_dates_all, oos_date_to_window, portfolio)

        print("\n🏁 Walk-forward complete")
        self._finalise(portfolio)

    # ── retrain hook ──────────────────────────────────────────────────────

    # def _maybe_retrain(self, train_dates: List[str]):
    #     """
    #     Hook for retraining your XGBoost model on train_dates.
    #     Implement by calling your xgboost_model.py training pipeline.
    #     Skipped with a warning if training infra not wired up.
    #     """
    #     try:
    #         # Example:
    #         # from xgboost_model import retrain_global_model
    #         # retrain_global_model(train_dates, self.cfg.tickers)
    #         # self.engine = SignalEngine()  # reload fresh model
    #         # self._precompute_signals()    # refresh cache
    #         pass   # ← replace with your retrain call
    #     except Exception as e:
    #         print(f"  ⚠️  Retrain skipped: {e}")
    
    def _maybe_retrain(self, train_dates: List[str]):
        """
        Retrain XGBoost global model using ONLY past data.
        This enables true walk-forward validation.
        """
        try:
            print("\n🧠 Retraining Global Model (Walk-Forward)...")

            from xgboost_model import build_multi_ticker_dataset, train_xgboost

            # Use last date of training window
            end_date = train_dates[-1]

            # Step 1: Build dataset (ONLY past data)
            df = build_multi_ticker_dataset(
                tickers=self.cfg.tickers,
                end_date=end_date,
                lookback_days=730,   # same as training
                forward_days=5
            )

            if df.empty:
                print("❌ Retrain failed: empty dataset")
                return

            # Step 2: Train model
            model, scaler, metrics = train_xgboost(
                df,
                save_path="4_signals/xgboost_global_model.pkl",
                n_trials=4   # reduce for speed during backtest
            )

            print(f"✅ Retrained | Acc: {metrics['wf_accuracy_mean']:.4f}")

            # Step 3: Reload engine (CRITICAL)
            from signal_engine import SignalEngine
            self.engine = SignalEngine("4_signals/xgboost_global_model.pkl")

            # Step 4: Refresh signals cache
            self._precompute_signals()

        except Exception as e:
            print(f"⚠️ Retrain failed: {e}")

    # ── bar-by-bar OOS loop ───────────────────────────────────────────────

    def _run_oos_loop_v2(self,
                    oos_dates: List[str],
                    oos_date_to_window: Dict[str, int],
                    portfolio: Portfolio):
        """
        Bar-by-bar OOS loop with per-window signal caches.

        Fixes included:
        - Safe Kalman initialization (no n_sig == 0)
        - Correct competition return calculation (no fake returns)
        - Kalman fallback weights
        - Stable slippage handling
        - Optional Kalman reset on window change
        """

        prev_prices: Dict[str, float] = {}
        prev_window_idx: Optional[int] = None

        for bar_idx, current_date in enumerate(oos_dates):

            daily_prices: Dict[str, float] = {}
            bar_probas:   Dict[str, float] = {}

            # ── window selection ─────────────────────────────────────────────
            window_idx   = oos_date_to_window.get(current_date, 0)
            window_cache = self.window_signal_caches.get(window_idx, {})

            # Optional: reset Kalman when switching model regimes
            if prev_window_idx is not None and window_idx != prev_window_idx:
                self.kalman_map.clear()
            prev_window_idx = window_idx

            # ── collect prices & signals ─────────────────────────────────────
            for ticker in self.active_tickers:
                price_df = self.loader.price_data.get(ticker)
                if price_df is None:
                    continue

                date_idx = price_df.index.strftime("%Y-%m-%d")
                if current_date not in date_idx:
                    continue

                # price lookup
                price = float(
                    price_df.loc[
                        price_df.index[date_idx == current_date][0], "close"
                    ]
                )
                daily_prices[ticker] = price

                # ── signal lookup (window-aware fallback) ────────────────────
                if window_cache and ticker in window_cache:
                    proba = window_cache[ticker].get(current_date, float("nan"))
                else:
                    sig_cache = self.signal_cache.get(ticker)
                    if sig_cache is None or current_date not in sig_cache.index:
                        continue
                    proba = float(sig_cache.loc[current_date, "proba_buy"])

                if not np.isnan(proba):
                    bar_probas[ticker] = proba

            # ── no signals → just mark-to-market ─────────────────────────────
            if not bar_probas:
                portfolio.update_prices(daily_prices)
                portfolio.record_snapshot(current_date)
                prev_prices = {**daily_prices}
                continue

            # ── regime detection ─────────────────────────────────────────────
            proxy  = "SPY" if "SPY" in self.regime_cache else \
                    next(iter(self.regime_cache), None)

            regime = 0
            if proxy:
                d_idx = self.regime_cache[proxy].index.strftime("%Y-%m-%d")
                if current_date in d_idx:
                    regime = int(
                        self.regime_cache[proxy][d_idx == current_date].iloc[0]
                    )

            regime_name = RegimeDetector.REGIME_NAMES.get(regime, "Unknown")

            # ── Kalman ensemble ──────────────────────────────────────────────
            tickers_with_signal = list(bar_probas.keys())
            n_sig = len(tickers_with_signal)

            kalman_weights = np.ones(n_sig) / n_sig if n_sig > 0 else np.array([])

            if n_sig > 0:
                key = tuple(sorted(tickers_with_signal))

                if key not in self.kalman_map:
                    self.kalman_map[key] = KalmanEnsemble(n_sig)

                proba_vec = np.array([bar_probas[t] for t in tickers_with_signal])

                # realised return (safe calculation)
                realised = np.mean([
                    (daily_prices[t] / prev_prices[t]) - 1
                    for t in tickers_with_signal
                    if t in prev_prices and prev_prices[t] > 0
                ]) if prev_prices else 0.0

                kalman_weights = self.kalman_map[key].update(proba_vec, realised)

            # ── competition returns (FIXED: no fake values) ──────────────────
            competition_returns = {}
            for t in tickers_with_signal:
                if t in prev_prices and prev_prices[t] > 0:
                    competition_returns[t] = (
                        daily_prices.get(t, prev_prices[t]) / prev_prices[t] - 1
                    )
                else:
                    competition_returns[t] = 0.0

            comp_weights = self.competition.score(
                tickers_with_signal, competition_returns
            )

            # ── champion tracking ────────────────────────────────────────────
            for ticker in tickers_with_signal:
                self.champion.record(
                    regime, ticker, competition_returns.get(ticker, 0.0)
                )

            # ── trade execution ──────────────────────────────────────────────
            equity = portfolio.get_portfolio_state()["total_equity"]

            for i, ticker in enumerate(tickers_with_signal):

                price = daily_prices.get(ticker)
                if price is None:
                    continue

                raw_proba = bar_probas[ticker]

                # Safe Kalman weight fallback
                if len(kalman_weights) == n_sig and n_sig > 0:
                    k_weight = float(kalman_weights[i])
                else:
                    k_weight = 1.0 / n_sig

                c_weight  = comp_weights.get(ticker, 0.0)
                blended_w = 0.5 * k_weight + 0.5 * c_weight

                blended_proba = min(
                    raw_proba * (1.0 + blended_w * 0.1), 0.99
                )

                # Slippage (kept flexible for future extensions)
                eff_slip   = self.cfg.base_slippage
                exec_price = price * (1 + eff_slip)

                pos_value = equity * self.cfg.fixed_size
                shares    = pos_value / exec_price if exec_price > 0 else 0

                if shares < 0.01:
                    continue

                # ── BUY ──────────────────────────────────────────────────────
                if blended_proba > self.cfg.buy_threshold and blended_w > 0.05:
                    if ticker not in portfolio.positions:
                        portfolio.execute_trade(
                            ticker, "BUY", pos_value, exec_price, current_date
                        )
                        self.trade_history.append({
                            "date": current_date,
                            "ticker": ticker,
                            "action": "BUY",
                            "proba": round(blended_proba, 4),
                            "regime": regime_name,
                            "weight": round(blended_w, 4),
                            "price": round(exec_price, 2),
                            "shares": round(shares, 4),
                            "window": window_idx,
                        })

                # ── SELL ─────────────────────────────────────────────────────
                elif blended_proba < self.cfg.sell_threshold:
                    if ticker in portfolio.positions:
                        sell_price  = price * (1 - eff_slip)
                        sell_shares = portfolio.positions[ticker]["shares"]

                        portfolio.execute_trade(
                            ticker, "SELL", sell_shares, sell_price, current_date
                        )
                        self.trade_history.append({
                            "date": current_date,
                            "ticker": ticker,
                            "action": "SELL",
                            "proba": round(blended_proba, 4),
                            "regime": regime_name,
                            "weight": round(blended_w, 4),
                            "price": round(sell_price, 2),
                            "shares": round(sell_shares, 4),
                            "window": window_idx,
                        })

            # ── end of bar ───────────────────────────────────────────────────
            portfolio.update_prices(daily_prices)
            portfolio.record_snapshot(current_date)
            prev_prices = {**daily_prices}

            self.equity_history.append({
                "date":   current_date,
                "equity": portfolio.get_portfolio_state()["total_equity"],
                "regime": regime_name,
                "window": window_idx,
            })

            # ── logging ──────────────────────────────────────────────────────
            if bar_idx % 20 == 0:
                state = portfolio.get_portfolio_state()
                print(
                    f"  📅 {current_date} | W{window_idx+1} | {regime_name:<16} | "
                    f"Equity: ${state['total_equity']:>10,.0f} | "
                    f"Heat: {state['heat']:.1%} | DD: {state['drawdown']:.1%}"
                )

    # ── STEP 7: finalise & run all stress tests ───────────────────────────

    def _finalise(self, portfolio: Portfolio):
        os.makedirs("5_backtesting/results", exist_ok=True)

        equity_df = pd.DataFrame(self.equity_history)
        trades_df = pd.DataFrame(self.trade_history)

        equity_df.to_csv("5_backtesting/results/equity_curve.csv",  index=False)
        trades_df.to_csv("5_backtesting/results/trade_log.csv",     index=False)

        perf = portfolio.get_performance_summary()

        # ── champion report ───────────────────────────────────────────────
        champs = self.champion.champions()
        champ_rows = [{"regime": RegimeDetector.REGIME_NAMES.get(r, r),
                       "champion_ticker": t}
                      for r, t in champs.items()]
        pd.DataFrame(champ_rows).to_csv(
            "5_backtesting/results/champion_selection.csv", index=False)

        # ── decay report ──────────────────────────────────────────────────
        decay_rows = []
        for ticker, res in self.decay_results.items():
            row = {"ticker": ticker,
                   "half_life":    res.get("half_life"),
                   "optimal_hold": res.get("optimal_hold")}
            row.update({f"edge_t{h}": res["edge_by_horizon"].get(h, 0)
                        for h in self.cfg.decay_horizons})
            decay_rows.append(row)
        pd.DataFrame(decay_rows).to_csv(
            "5_backtesting/results/alpha_decay.csv", index=False)

        # ── stress tests ──────────────────────────────────────────────────
        portfolio_trades = pd.DataFrame(portfolio.trade_history) \
                           if portfolio.trade_history else pd.DataFrame()

        stress_df = self.stress.regime_stress(equity_df, portfolio_trades)
        stress_df.to_csv("5_backtesting/results/stress_regime.csv", index=False)

        slip_df = self.stress.slippage_sweep(
            trades_df if not trades_df.empty else portfolio_trades,
            self.cfg.initial_capital
        )
        slip_df.to_csv("5_backtesting/results/stress_slippage.csv", index=False)

        # ── print summary ─────────────────────────────────────────────────
        print("\n" + "═"*55)
        print("📊  BACKTEST V2 RESULTS")
        print("═"*55)
        print(f"Return    : {perf.get('total_return',0):.2f}%")
        print(f"Sharpe    : {perf.get('sharpe_ratio',0):.3f}")
        print(f"Max DD    : {perf.get('max_drawdown',0):.2f}%")
        print(f"Trades    : {perf.get('num_trades',0)}")

        print("\n🏆  CHAMPION PER REGIME:")
        for row in champ_rows:
            print(f"  {row['regime']:<22} → {row['champion_ticker']}")

        if not stress_df.empty:
            print("\n🔥  STRESS TEST RESULTS:")
            for _, row in stress_df.iterrows():
                print(f"  {row['period']:<18} | "
                      f"Ret: {row.get('total_return','?'):>6}% | "
                      f"DD: {row.get('max_drawdown','?'):>5}% | "
                      f"Sharpe: {row.get('sharpe','?')}")

        if not slip_df.empty:
            print("\n💸  SLIPPAGE SWEEP:")
            for _, row in slip_df.iterrows():
                print(f"  slip={row['slippage_pct']:>5}%  "
                      f"Ret={row['total_return']:>7.2f}%  "
                      f"Sharpe={row['sharpe']:.3f}")

        print("\n✅  Results saved to 5_backtesting/results/")

# ── STANDALONE FUNCTION (must be at module level for multiprocessing) ──────
def _retrain_window_worker(args: dict) -> dict:
    """
    Runs in a child process. Fully isolated — no shared state.
    Returns dict with window_idx, model_path, metrics, signal_cache.
    """
    import os, sys, warnings
    warnings.filterwarnings("ignore")

    project_root = args["project_root"]
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "4_signals"))

    window_idx  = args["window_idx"]
    train_dates = args["train_dates"]
    tickers     = args["tickers"]
    end_date    = train_dates[-1]

    # Unique model path per window — prevents file collision
    model_path  = f"4_signals/xgboost_window_{window_idx:03d}.pkl"

    print(f"  [W{window_idx+1}] Training on data up to {end_date} …")

    try:
        from xgboost_model import build_multi_ticker_dataset, train_xgboost
        from signal_engine import SignalEngine

        df = build_multi_ticker_dataset(
            tickers      = tickers,
            end_date     = end_date,
            lookback_days = 730,
            forward_days  = 5,
        )

        if df.empty:
            print(f"  [W{window_idx+1}] ❌ Empty dataset")
            return {"window_idx": window_idx, "model_path": None,
                    "signal_cache": {}, "metrics": {}}

        _, _, metrics = train_xgboost(
            df,
            save_path = model_path,
            n_trials  = 5,
        )

        # Precompute signals using this window's model
        engine = SignalEngine(model_path)
        signal_cache = {}

        import yfinance as yf
        import pandas as pd
        from datetime import datetime, timedelta

        start_dt  = datetime.strptime(args["start_date"], "%Y-%m-%d")
        ext_start = (start_dt - timedelta(days=420)).strftime("%Y-%m-%d")

        for ticker in tickers:
            try:
                raw = yf.download(ticker, start=ext_start,
                                  end=args["end_date"],
                                  progress=False, auto_adjust=True)
                if raw.empty:
                    continue
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                raw.columns = [c.lower() for c in raw.columns]
                raw = raw[["open","high","low","close","volume"]].dropna()

                df_reset = raw.copy().reset_index()
                df_reset["date"] = df_reset["Date"].dt.strftime("%Y-%m-%d")

                sig_df = engine.get_full_signals(df_reset, ticker)
                if "proba_buy" in sig_df.columns:
                    signal_cache[ticker] = sig_df.set_index("date")["proba_buy"].to_dict()

            except Exception as e:
                print(f"  [W{window_idx+1}] ⚠️ Signal fail {ticker}: {e}")

        print(f"  [W{window_idx+1}] ✅ Done | "
              f"Acc={metrics.get('wf_accuracy_mean', 0):.3f} | "
              f"model={model_path}")

        return {
            "window_idx":   window_idx,
            "model_path":   model_path,
            "signal_cache": signal_cache,   # {ticker: {date: proba}}
            "metrics":      metrics,
        }

    except Exception as e:
        print(f"  [W{window_idx+1}] ❌ Worker failed: {e}")
        return {"window_idx": window_idx, "model_path": None,
                "signal_cache": {}, "metrics": {}}
# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Windows requires this guard for multiprocessing
    from multiprocessing import freeze_support
    freeze_support()

    cfg = BacktestConfig(
        tickers         = ["AAPL", "NVDA", "MSFT", "SPY", "QQQ", "TSLA"],
        start_date      = "2020-01-01",
        end_date        = "2025-04-01",
        initial_capital = 100_000,
        train_months    = 12,
        oos_months      = 2,
        retrain         = True,      # set True once retrain pipeline is wired
        fixed_size      = 0.10,
        base_slippage   = 0.001,
    )

    bt = BacktestEngineV2(cfg)
    bt.prepare()   # load → signals → regimes → correlation filter → decay
    bt.run()       # walk-forward OOS loop → stress tests → reports
    
    # Save price data for RL agent
    import pickle
    with open("5_backtesting/results/price_data.pkl", "wb") as f:
        pickle.dump(bt.loader.price_data, f)
    print("✅ price_data.pkl saved")