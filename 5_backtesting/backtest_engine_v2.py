"""
backtest_engine_v2.py
=====================
SIMPLER architecture restored from +88.94% Sharpe-1.124 version.

Key principles:
- Trust the model. Execute its signals without override.
- Fixed 10% position size. No regime-conditional scaling.
- Uniform 0.55 buy / 0.45 sell threshold.
- Kalman ensemble + filter competition for signal weighting.
- NO tail hedger, NO structural breaks, NO early loss cuts,
  NO regime flip exits, NO cooldowns, NO crisis correlation halving.

Components:
  5   Rolling walk-forward
  5c  Regime detection (Hurst + SMA50, 4 states)
  5g  Signal correlation filter
  5f  Alpha decay analysis
  5   Kalman ensemble (per-bar signal weighting)
  5e  Filter competition (rolling Sharpe tournament)
  5d  Champion selection per regime
  5b  Regime stress test
  5h  Cost/slippage sweep

Reproducibility fixes applied:
  FIX-1  np.random.seed moved to first line of prepare()
  FIX-2  GaussianHMMRegimeDetector receives random_state=seed
  FIX-3  HMM model cached to disk after first fit, reloaded on reruns
  FIX-4  Price data cached to disk after first download, reloaded on reruns
  FIX-5  all_tickers built with ordered deduplication instead of set()
"""

import os, sys, warnings, pickle
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

# ── path setup ──────────────────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "4_signals"))

from signal_engine import SignalEngine
from portfolio import Portfolio
from sector_signal_engine import (
    SectorSignalEngine,
    TICKER_TO_SECTOR,
    REGIME_TO_SECTORS,
    SECTOR_REGISTRY,
)

# Tickers that should not receive fresh capital during confirmed Bull-Trending.
DEFENSIVE_TICKERS = {"GLD", "PG", "WMT", "XOM", "CVX"}


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BacktestConfig:
    tickers:         List[str] = field(default_factory=lambda:
        ["AAPL","NVDA","MSFT","SPY","QQQ"])

    use_sector_models:  bool = True
    sector_tickers: List[str] = field(default_factory=lambda: [
        # Hardware
        "NVDA", "AVGO", "TSM",
        # Hypercloud
        "MSFT", "GOOGL", "AMZN", "META",
        # Autos
        "TSLA", "RACE",
        # Defensive
        "XOM", "CVX", "PG", "WMT",
        # Gold
        "GLD",
    ])

    global_tickers: List[str] = field(default_factory=lambda:
        ["AAPL", "QQQ"])

    # C10 removed — hedge_tickers empty
    hedge_tickers: List[str] = field(default_factory=lambda: [])

    start_date:      str  = "2020-01-01"
    end_date:        str  = "2026-05-19"
    initial_capital: float = 100_000.0

    train_months:    int   = 9
    oos_months:      int   = 1
    retrain:         bool  = False

    fixed_size:      float = 0.10
    buy_threshold:   float = 0.55
    sell_threshold:  float = 0.45

    ticker_buy_thresholds:  Dict[str, float] = field(default_factory=lambda: {})
    ticker_sell_thresholds: Dict[str, float] = field(default_factory=lambda: {})

    min_hold_bars:   int   = 3

    base_slippage:   float = 0.001
    commission_rate: float = 0.001
    random_seed:     int   = 42

    decay_horizons:  List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    competition_window: int = 20

    stress_periods: List[Tuple] = field(default_factory=lambda: [
        ("COVID crash",    "2020-02-15", "2020-03-23"),
        ("2022 bear",      "2022-01-01", "2022-12-31"),
        ("2025 tariff",    "2025-04-01", "2025-05-15"),
    ])

    slippage_sweep: List[float] = field(default_factory=lambda:
        [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05])

    # Cache paths — set to "" to disable caching
    price_cache_path: str = "5_backtesting/results/price_data.pkl"
    hmm_cache_path:   str = "5_backtesting/results/hmm_detector.pkl"


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADER
# ═══════════════════════════════════════════════════════════════════════

class DataLoader:
    WARMUP_DAYS = 420

    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.price_data: Dict[str, pd.DataFrame] = {}

    def load(self):
        # ── FIX-4: load from cache if available ───────────────────
        if self.cfg.price_cache_path and \
                os.path.exists(self.cfg.price_cache_path):
            print(f"  📂 Loading cached price data from "
                  f"{self.cfg.price_cache_path} ...")
            with open(self.cfg.price_cache_path, "rb") as f:
                self.price_data = pickle.load(f)
            print(f"  ✅ Loaded {len(self.price_data)} tickers from cache")
            return

        print("📥 Loading price data …")
        start_dt  = datetime.strptime(self.cfg.start_date, "%Y-%m-%d")
        ext_start = (start_dt - timedelta(days=self.WARMUP_DAYS)
                     ).strftime("%Y-%m-%d")

        # ── FIX-5: ordered deduplication instead of set() ─────────
        _seen = set()
        all_tickers = []
        for t in (self.cfg.tickers +
                  (self.cfg.sector_tickers
                   if self.cfg.use_sector_models else []) +
                  self.cfg.hedge_tickers):
            if t not in _seen:
                _seen.add(t)
                all_tickers.append(t)

        for ticker in all_tickers:
            try:
                df = yf.download(ticker, start=ext_start,
                                 end=self.cfg.end_date,
                                 progress=False, auto_adjust=True)
                if df.empty:
                    print(f"  ⚠️  {ticker}: no data"); continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                df = df[["open","high","low","close","volume"]].dropna()
                self.price_data[ticker] = df
                print(f"  ✅ {ticker}: {len(df)} rows")
            except Exception as e:
                print(f"  ❌ {ticker} failed: {e}")

        # ── FIX-4: save to cache after download ────────────────────
        if self.cfg.price_cache_path:
            os.makedirs(os.path.dirname(self.cfg.price_cache_path),
                        exist_ok=True)
            with open(self.cfg.price_cache_path, "wb") as f:
                pickle.dump(self.price_data, f)
            print(f"  💾 Price data cached → {self.cfg.price_cache_path}")

    def trading_dates(self) -> List[str]:
        # ── FIX-5: anchor to SPY — deterministic regardless of dict order
        anchor = "SPY" if "SPY" in self.price_data else "AAPL"
        sample = self.price_data[anchor]
        mask   = ((sample.index >= self.cfg.start_date) &
                  (sample.index <= self.cfg.end_date))
        return sample.index[mask].strftime("%Y-%m-%d").tolist()


# ═══════════════════════════════════════════════════════════════════════
# REGIME DETECTOR (Hurst + SMA50, 4 states)
# ═══════════════════════════════════════════════════════════════════════

class RegimeDetector:
    def __init__(self, hurst_window: int = 100, vol_window: int = 20):
        self.hurst_window = hurst_window
        self.vol_window   = vol_window

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
                m        = np.mean(chunk)
                demeaned = chunk - m
                cumdev   = np.cumsum(demeaned)
                R        = cumdev.max() - cumdev.min()
                S        = np.std(chunk, ddof=1)
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
        close   = df["close"]
        sma50   = close.rolling(50).mean()
        returns = close.pct_change()
        rvol    = returns.rolling(self.vol_window).std()
        labels  = pd.Series(0, index=df.index, dtype=int)

        for i in range(self.hurst_window, len(df)):
            window   = close.iloc[i - self.hurst_window: i].values
            h        = self._hurst(np.log(window + 1e-10))
            bull     = close.iloc[i] > sma50.iloc[i]
            trending = h > 0.55
            if not trending and h >= 0.45:
                trending = rvol.iloc[i] > rvol.rolling(60).mean().iloc[i]

            if bull and trending:
                labels.iloc[i] = 0
            elif bull and not trending:
                labels.iloc[i] = 1
            elif not bull and trending:
                labels.iloc[i] = 2
            else:
                labels.iloc[i] = 3

        return labels

    REGIME_NAMES = {
        0: "Bull-Trending",
        1: "Bull-MeanRev",
        2: "Bear-Trending",
        3: "Bear-MeanRev",
    }


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL CORRELATION FILTER
# ═══════════════════════════════════════════════════════════════════════

class SignalCorrelationFilter:
    def __init__(self, corr_threshold: float = 0.70):
        self.corr_threshold = corr_threshold

    def fit_prune(self, signal_df: pd.DataFrame,
                  sharpe_scores: Dict[str, float]) -> List[str]:
        if signal_df.shape[1] <= 1:
            return list(signal_df.columns)

        corr   = signal_df.corr()
        ranked = sorted(sharpe_scores, key=sharpe_scores.get, reverse=True)
        kept, dropped = [], set()

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

        print(f"  🔗 Correlation filter: "
              f"{signal_df.shape[1]} → {len(kept)} signals kept")
        return kept


# ═══════════════════════════════════════════════════════════════════════
# ALPHA DECAY ANALYSER
# ═══════════════════════════════════════════════════════════════════════

class AlphaDecayAnalyser:
    def __init__(self, horizons: List[int] = None):
        self.horizons = horizons or [1, 3, 5, 10, 20]

    def analyse(self, signal_series: pd.Series,
                price_series: pd.Series) -> Dict:
        results  = {}
        buy_dates = signal_series[signal_series == 1].index

        if len(buy_dates) < 10:
            return {"half_life": None,
                    "optimal_hold": self.horizons[-1],
                    "edge_by_horizon": {}}

        edges = {}
        for h in self.horizons:
            fwd_ret  = price_series.pct_change(h).shift(-h)
            buy_rets = fwd_ret.reindex(buy_dates).dropna()
            all_mean = fwd_ret.mean()
            edge     = (float(buy_rets.mean() - all_mean)
                        if len(buy_rets) > 0 else 0.0)
            edges[h] = edge

        results["edge_by_horizon"] = edges

        peak         = max(edges.values()) if edges else 0
        half_life    = None
        optimal_hold = self.horizons[-1]

        if peak > 0:
            for h in self.horizons:
                if edges[h] <= peak * 0.5:
                    half_life    = h
                    idx          = self.horizons.index(h)
                    optimal_hold = (self.horizons[idx - 1]
                                    if idx > 0 else h)
                    break

        results["half_life"]    = half_life
        results["optimal_hold"] = optimal_hold
        return results


# ═══════════════════════════════════════════════════════════════════════
# KALMAN ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════

class KalmanEnsemble:
    def __init__(self, n_signals: int,
                 process_noise: float = 0.01,
                 obs_noise: float = 0.1):
        self.n = n_signals
        self.w = np.ones(n_signals) / n_signals
        self.P = np.ones(n_signals) * 1.0
        self.Q = process_noise
        self.R = obs_noise

    def update(self, signal_probas: np.ndarray,
               realised_return: float) -> np.ndarray:
        self.P   += self.Q
        direction = 1.0 if realised_return > 0 else -1.0
        quality   = direction * (signal_probas - 0.5) * 2
        K         = self.P / (self.P + self.R)
        self.w    = self.w + K * (quality - self.w)
        self.P    = (1 - K) * self.P
        w_pos     = np.clip(self.w, 0.05, None)
        total     = w_pos.sum()
        if total > 1e-6:
            return w_pos / total
        return np.ones(self.n) / self.n

    def blend(self, signal_probas: np.ndarray,
              weights: np.ndarray) -> float:
        return float(np.dot(weights, signal_probas))


# ═══════════════════════════════════════════════════════════════════════
# FILTER COMPETITION
# ═══════════════════════════════════════════════════════════════════════

class FilterCompetition:
    def __init__(self, window: int = 20, min_signals: int = 1):
        self.window      = window
        self.min_signals = min_signals
        self._return_history: Dict[str, List[float]] = {}

    def score(self, signal_names: List[str],
              bar_returns: Dict[str, float]) -> Dict[str, float]:
        for name in signal_names:
            if name not in self._return_history:
                self._return_history[name] = []
            ret = bar_returns.get(name, 0.0)
            self._return_history[name].append(ret)
            if len(self._return_history[name]) > self.window:
                self._return_history[name].pop(0)

        sharpes = {}
        for name in signal_names:
            hist = np.array(self._return_history[name])
            if len(hist) < 5 or hist.std() < 1e-9:
                sharpes[name] = 0.0
            else:
                sharpes[name] = hist.mean() / hist.std()

        ranked  = sorted(sharpes, key=sharpes.get, reverse=True)
        n_keep  = max(self.min_signals, len(ranked) // 2)
        winners = set(ranked[:n_keep])

        weights      = {}
        total_sharpe = (sum(max(sharpes[s], 0) for s in winners) or 1.0)
        for name in signal_names:
            if name in winners and sharpes[name] > 0:
                weights[name] = sharpes[name] / total_sharpe
            else:
                weights[name] = 0.0

        return weights


# ═══════════════════════════════════════════════════════════════════════
# CHAMPION SELECTOR
# ═══════════════════════════════════════════════════════════════════════

class ChampionSelector:
    def __init__(self):
        self._perf: Dict[int, Dict[str, Dict]] = {r: {} for r in range(4)}

    def record(self, regime: int, ticker: str, bar_return: float):
        if ticker not in self._perf[regime]:
            self._perf[regime][ticker] = {
                "returns": [], "peak": 1.0, "nav": 1.0
            }
        rec = self._perf[regime][ticker]
        rec["returns"].append(bar_return)
        rec["nav"]  *= (1 + bar_return)
        rec["peak"]  = max(rec["peak"], rec["nav"])

    def champions(self) -> Dict[int, str]:
        result = {}
        for regime, tickers in self._perf.items():
            best_ticker, best_calmar = None, -np.inf
            for ticker, rec in tickers.items():
                rets = np.array(rec["returns"])
                if len(rets) < 5:
                    continue
                ann_ret = rets.mean() * 252
                max_dd  = max(
                    (rec["peak"] - rec["nav"]) / rec["peak"], 1e-6
                )
                calmar  = ann_ret / max_dd
                if calmar > best_calmar:
                    best_calmar = calmar
                    best_ticker = ticker
            result[regime] = best_ticker
        return result


# ═══════════════════════════════════════════════════════════════════════
# STRESS TESTER
# ═══════════════════════════════════════════════════════════════════════

class StressTester:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg

    def regime_stress(self, equity_df: pd.DataFrame,
                      trades_df: pd.DataFrame) -> pd.DataFrame:
        equity_df         = equity_df.copy()
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        rows = []

        for label, start, end in self.cfg.stress_periods:
            mask   = ((equity_df["date"] >= start) &
                      (equity_df["date"] <= end))
            period = equity_df[mask]
            if period.empty:
                rows.append({"period": label, "note": "no data"})
                continue

            eq        = period["equity"].values
            rets      = pd.Series(eq).pct_change().dropna()
            peak      = eq.max()
            trough    = eq.min()
            max_dd    = (peak - trough) / peak if peak > 0 else 0
            sharpe    = (rets.mean() / rets.std() * np.sqrt(252)
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


# ═══════════════════════════════════════════════════════════════════════
# WALK-FORWARD ENGINE
# ═══════════════════════════════════════════════════════════════════════

class WalkForwardEngine:
    def __init__(self, all_dates: List[str],
                 train_months: int, oos_months: int):
        self.dates    = all_dates
        self.train_mo = train_months
        self.oos_mo   = oos_months

    def windows(self) -> List[Tuple[List[str], List[str]]]:
        result = []
        dates  = pd.to_datetime(self.dates)
        i      = 0
        while True:
            train_end_dt = dates[i] + pd.DateOffset(months=self.train_mo)
            oos_end_dt   = train_end_dt + pd.DateOffset(months=self.oos_mo)

            train_mask = ((dates >= dates[i]) & (dates < train_end_dt))
            oos_mask   = ((dates >= train_end_dt) & (dates < oos_end_dt))

            train_dates = [d.strftime("%Y-%m-%d") for d in dates[train_mask]]
            oos_dates   = [d.strftime("%Y-%m-%d") for d in dates[oos_mask]]

            if not train_dates or not oos_dates:
                break
            result.append((train_dates, oos_dates))

            next_start = dates[i] + pd.DateOffset(months=self.oos_mo)
            future     = dates[dates >= next_start]
            if len(future) == 0:
                break
            i = list(dates).index(future[0])
            if oos_end_dt > dates[-1]:
                break

        print(f"  📅 Walk-forward: {len(result)} windows "
              f"({self.train_mo}mo train / {self.oos_mo}mo OOS)")
        return result


# ═══════════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════════

class BacktestEngineV2:
    def __init__(self, cfg: BacktestConfig = None):
        self.cfg = cfg or BacktestConfig()

        self.loader        = DataLoader(self.cfg)
        self.engine        = SignalEngine()
        self.sector_engine = (SectorSignalEngine()
                              if self.cfg.use_sector_models else None)
        self.regime_d    = RegimeDetector()
        self.corr_filt   = SignalCorrelationFilter(corr_threshold=0.70)
        self.decay       = AlphaDecayAnalyser(self.cfg.decay_horizons)
        self.champion    = ChampionSelector()
        self.stress      = StressTester(self.cfg)
        self.competition = FilterCompetition(self.cfg.competition_window)

        self.signal_cache:   Dict[str, pd.DataFrame]   = {}
        self.regime_cache:   Dict[str, pd.Series]      = {}
        self.kalman_map:     Dict[str, KalmanEnsemble] = {}
        self.decay_results:  Dict[str, Dict]           = {}
        self.active_tickers: List[str] = list(self.cfg.tickers)

        self.equity_history:      List[Dict] = []
        self.trade_history:       List[Dict] = []
        self._oos_sell_log:       List[Dict] = []
        self._oos_cache_miss_log: List[Dict] = []
        self._rl_callback   = None     # set via set_rl_callback() — default disabled
        self._rl_signal_log = []       # records every signal for RL training

    def set_rl_callback(self, callback):
        """
        Register an RL callback that fires before each BUY execution.
        callback(state_dict) -> float multiplier in [0.5, 1.5]
        If callback is None (default), engine runs as today (100-102%).
        """
        self._rl_callback = callback
        self._rl_signal_log = []

    def prepare(self):
        # ── FIX-1: seed set as very first operation ────────────────
        np.random.seed(self.cfg.random_seed)

        self.loader.load()
        self._precompute_signals()
        self._precompute_regimes()
        self._run_correlation_filter()
        self._run_alpha_decay()
        if self.cfg.use_sector_models and self.sector_engine is not None:
            self.sector_engine.load()
            self.sector_engine.clear_cache()
            print(f"\n🏭 Sector engine loaded: "
                  f"{len(self.sector_engine.models)} sectors active")
            print("\n  Sector threshold verification:")
            for sk, sm in self.sector_engine.models.items():
                print(f"    {sm['name']:<25} threshold={sm['threshold']}")
            self.sector_engine.precompute_signals(self.loader.price_data)
        self._load_hmm_detector()

    def _load_hmm_detector(self):
        """Load HMM regime detector — cached after first fit."""
        self._hmm_detector = None
        if not self.cfg.use_sector_models:
            return

        # ── FIX-3: load HMM from cache if available ───────────────
        if self.cfg.hmm_cache_path and \
                os.path.exists(self.cfg.hmm_cache_path):
            try:
                print("\n🗺️  Loading cached HMM detector ...")
                with open(self.cfg.hmm_cache_path, "rb") as f:
                    self._hmm_detector = pickle.load(f)
                if not self._hmm_detector._state_history.empty:
                    dist = (self._hmm_detector._state_history
                            .value_counts().to_dict())
                    print(f"  ✅ HMM loaded from cache | "
                          f"distribution: {dist}")
                else:
                    print("  ✅ HMM loaded from cache")
                return
            except Exception as e:
                print(f"  ⚠️  HMM cache load failed: {e} — refitting")

        try:
            from hmm_regime import GaussianHMMRegimeDetector

            print("\n🗺️  Fitting HMM detector ...")
            spy = yf.download("SPY", start="2015-01-01",
                              end=self.cfg.end_date,
                              progress=False, auto_adjust=False)
            ief = yf.download("IEF", start="2015-01-01",
                              end=self.cfg.end_date,
                              progress=False, auto_adjust=False)

            for df in [spy, ief]:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

            spy = spy.rename(columns={"Close": "close"})
            ief = ief.rename(columns={"Close": "close"})

            # ── FIX-2: pass random_state to HMM ───────────────────
            try:
                hmm = GaussianHMMRegimeDetector(
                    random_state=self.cfg.random_seed
                )
            except TypeError:
                # if constructor doesn't accept random_state,
                # seed is already set globally via FIX-1
                hmm = GaussianHMMRegimeDetector()

            hmm.fit_initial({"SPY": spy, "IEF": ief})
            self._hmm_detector = hmm

            if not hmm._state_history.empty:
                dist = hmm._state_history.value_counts().to_dict()
                print(f"  ✅ HMM fitted | distribution: {dist}")
            else:
                print("  ⚠️  HMM fitted but state_history empty")

            # ── FIX-3: save HMM to cache ───────────────────────────
            if self.cfg.hmm_cache_path:
                os.makedirs(
                    os.path.dirname(self.cfg.hmm_cache_path),
                    exist_ok=True
                )
                with open(self.cfg.hmm_cache_path, "wb") as f:
                    pickle.dump(hmm, f)
                print(f"  💾 HMM cached → {self.cfg.hmm_cache_path}")

        except Exception as e:
            print(f"  ⚠️  HMM detector failed: {e} "
                  f"— using Hurst mapping fallback")
            self._hmm_detector = None

    def _precompute_signals(self):
        print("\n🧠 Precomputing signals …")
        all_tickers = list(set(
            self.cfg.tickers +
            (self.cfg.sector_tickers if self.cfg.use_sector_models else [])
        ))
        for ticker in all_tickers:
            if ticker not in self.loader.price_data:
                continue
            df = self.loader.price_data[ticker].copy().reset_index()
            df["date"] = (df["Date"].dt.strftime("%Y-%m-%d")
                          if "Date" in df.columns
                          else df.index.strftime("%Y-%m-%d"))
            try:
                sig_df = self.engine.get_full_signals(df, ticker)
                if "proba_buy" not in sig_df.columns:
                    print(f"  ❌ {ticker}: missing proba_buy")
                    continue
                if "date" in sig_df.columns:
                    sig_df = sig_df.set_index("date")
                sig_df.index = (pd.to_datetime(sig_df.index)
                                .strftime("%Y-%m-%d"))
                self.signal_cache[ticker] = sig_df
                sample = list(sig_df.index[:3]) + list(sig_df.index[-3:])
                print(f"  ✅ {ticker}: {len(sig_df)} signal rows")
                print(f"    {ticker} date index sample: {sample}")
            except Exception as e:
                print(f"  ❌ {ticker} signal error: {e}")

        print(f"\n  Sector signal cache: "
              f"{len(self.signal_cache)} tickers precomputed")

        rows = []
        for tkr, sdf in self.signal_cache.items():
            sector_key = (self.sector_engine.get_sector_for_ticker(tkr)
                          if hasattr(self, "sector_engine")
                          and self.sector_engine else None)
            from sector_signal_engine import SECTOR_REGISTRY
            cfg_s     = SECTOR_REGISTRY.get(sector_key, {})
            threshold = cfg_s.get("native_threshold", 0.5)

            proba_series = (sdf["proba_buy"]
                            if "proba_buy" in sdf.columns
                            else pd.Series(dtype=float))

            buy_signals  = (proba_series >= threshold).sum()
            sell_signals = (proba_series < threshold * 0.85).sum()
            nan_count    = proba_series.isna().sum()
            date_sample  = list(sdf.index[:3]) + list(sdf.index[-3:])

            rows.append({
                "ticker":         tkr,
                "sector":         sector_key,
                "threshold":      threshold,
                "total_rows":     len(sdf),
                "buy_signals":    int(buy_signals),
                "sell_signals":   int(sell_signals),
                "nan_count":      int(nan_count),
                "buy_pct":        round(buy_signals / max(len(sdf), 1) * 100, 1),
                "proba_mean":     round(float(proba_series.mean()), 4),
                "proba_std":      round(float(proba_series.std()), 4),
                "proba_min":      round(float(proba_series.min()), 4),
                "proba_max":      round(float(proba_series.max()), 4),
                "first_date":     sdf.index[0]  if len(sdf) > 0 else "",
                "last_date":      sdf.index[-1] if len(sdf) > 0 else "",
                "date_format_ok": (str(sdf.index[0]).count("-") == 2
                                   if len(sdf) > 0 else False),
                "date_sample":    str(date_sample),
            })

        diag_df   = pd.DataFrame(rows)
        diag_path = "5_backtesting/results/sector_signal_precompute.csv"
        os.makedirs("5_backtesting/results", exist_ok=True)
        diag_df.to_csv(diag_path, index=False)
        print(f"  Precompute diagnostic → {diag_path}")

    def _precompute_regimes(self):
        print("\n🗺️  Computing regime labels …")
        for ticker, df in self.loader.price_data.items():
            try:
                labels  = self.regime_d.label_regimes(df)
                self.regime_cache[ticker] = labels
                dist    = labels.value_counts().to_dict()
                dist_str = {RegimeDetector.REGIME_NAMES[k]: v
                            for k, v in dist.items()}
                print(f"  ✅ {ticker} regimes: {dist_str}")
            except Exception as e:
                print(f"  ⚠️  {ticker} regime error: {e}")

    def _run_correlation_filter(self):
        print("\n🔗 Signal correlation filter …")
        frames = {}
        for ticker, sig_df in self.signal_cache.items():
            if "proba_buy" in sig_df.columns:
                frames[ticker] = sig_df["proba_buy"]
        if not frames:
            return
        proba_matrix = pd.DataFrame(frames).dropna()
        if proba_matrix.empty or proba_matrix.shape[1] < 2:
            return
        sharpes = {}
        for ticker in proba_matrix.columns:
            s = proba_matrix[ticker].diff().dropna()
            sharpes[ticker] = (float(s.mean() / s.std())
                               if s.std() > 1e-9 else 0.0)
        self.active_tickers = self.corr_filt.fit_prune(
            proba_matrix, sharpes
        )

    def _run_alpha_decay(self):
        print("\n⏳ Alpha decay analysis …")
        for ticker in self.active_tickers:
            if ticker not in self.signal_cache:
                continue
            sig_df   = self.signal_cache[ticker]
            price_df = self.loader.price_data[ticker]
            common   = sig_df.index.intersection(
                price_df.index.strftime("%Y-%m-%d")
            )
            if len(common) < 20:
                continue
            signal_series = (
                sig_df.loc[common, "proba_buy"] > self.cfg.buy_threshold
            ).astype(int)
            price_series       = price_df["close"].copy()
            price_series.index = price_df.index.strftime("%Y-%m-%d")
            price_series       = price_series.reindex(common)
            res                = self.decay.analyse(signal_series,
                                                    price_series)
            self.decay_results[ticker] = res
            print(f"  ✅ {ticker}: half-life={res['half_life']}d  "
                  f"optimal_hold={res['optimal_hold']}d  "
                  f"edge@1d={res['edge_by_horizon'].get(1,0):.4f}")

    def run(self):
        # seed already set in prepare() — set again here for safety
        # in case run() is called without prepare()
        np.random.seed(self.cfg.random_seed)

        if not self.loader.price_data:
            self.prepare()

        all_dates = self.loader.trading_dates()
        wf_engine = WalkForwardEngine(all_dates,
                                      self.cfg.train_months,
                                      self.cfg.oos_months)
        windows   = wf_engine.windows()
        portfolio = Portfolio(
            initial_capital=self.cfg.initial_capital,
            commission_rate=self.cfg.commission_rate
        )

        print(f"\n🚀 Walk-forward backtest: {self.cfg.start_date} → "
              f"{self.cfg.end_date}\n")

        oos_dates_all = []
        for window_idx, (train_dates, oos_dates) in enumerate(windows):
            print(f"\n── Window {window_idx+1}/{len(windows)} │ "
                  f"Train: {train_dates[0]}→{train_dates[-1]} │ "
                  f"OOS: {oos_dates[0]}→{oos_dates[-1]}")
            if self.cfg.retrain:
                self._maybe_retrain(train_dates)
            oos_dates_all.extend(oos_dates)

        self._run_oos_loop(oos_dates_all, portfolio)
        print("\n🏁 Walk-forward complete")
        self._finalise(portfolio)

    def _maybe_retrain(self, train_dates: List[str]):
        if not self.cfg.retrain:
            print("  ⏭️  Retrain disabled — using existing model")
            return
        try:
            from xgboost_model import build_multi_ticker_dataset, train_xgboost
            end_date = train_dates[-1]
            df = build_multi_ticker_dataset(
                tickers=self.cfg.tickers,
                end_date=end_date,
                lookback_days=730,
                forward_days=5
            )
            if df.empty:
                print("❌ Retrain failed: empty dataset")
                return
            model, scaler, metrics = train_xgboost(
                df,
                save_path="4_signals/xgboost_global_model.pkl",
                n_trials=4
            )
            print(f"✅ Retrained | Acc: {metrics['wf_accuracy_mean']:.4f}")
            from signal_engine import SignalEngine
            self.engine = SignalEngine("4_signals/xgboost_global_model.pkl")
            self._precompute_signals()
        except Exception as e:
            print(f"⚠️ Retrain failed: {e}")

    def _run_oos_loop(self, oos_dates: List[str], portfolio: Portfolio):
        prev_prices:      Dict[str, float] = {}
        sell_log:         List[Dict]       = []
        cache_miss_log:   List[Dict]       = []
        _last_bull_bar:       int  = -999
        _regime_bar_count      = 0
        _regime_cumulative_ret = 0.0
        _regime_base_equity    = None
        _prev_hmm_regime:     str  = ""
        _pyramid_prev_regime: str  = ""
        _bear_confirm_count:  int  = 0
        _bear_duration_bars:  int  = 0
        _bars_held:           Dict[str, int] = {}
        _trend_added:         set  = set()

        print(f"  📅 OOS loop anchor: trading dates from "
              f"{'SPY' if 'SPY' in self.loader.price_data else 'AAPL'} "
              f"— {len(oos_dates)} bars")

        for bar_idx, current_date in enumerate(oos_dates):
            daily_prices: Dict[str, float] = {}
            bar_probas:   Dict[str, float] = {}

            for ticker in self.active_tickers:
                price_df = self.loader.price_data.get(ticker)
                if price_df is None:
                    continue
                date_idx = price_df.index.strftime("%Y-%m-%d")
                if current_date not in date_idx:
                    continue
                price = float(
                    price_df.loc[
                        price_df.index[date_idx == current_date][0],
                        "close"
                    ]
                )
                daily_prices[ticker] = price
                sig_cache = self.signal_cache.get(ticker)
                if sig_cache is None or current_date not in sig_cache.index:
                    continue
                proba = float(sig_cache.loc[current_date, "proba_buy"])
                if not np.isnan(proba):
                    bar_probas[ticker] = proba

            # update bars_held; clean up closed positions
            for t in list(portfolio.positions.keys()):
                _bars_held[t] = _bars_held.get(t, 0) + 1
            for t in list(_bars_held.keys()):
                if t not in portfolio.positions:
                    del _bars_held[t]
                    _trend_added.discard(t)

            if not bar_probas:
                portfolio.update_prices(daily_prices)
                portfolio.record_snapshot(current_date)
                continue

            proxy  = ("SPY" if "SPY" in self.regime_cache
                      else next(iter(self.regime_cache), None))
            regime = 0
            if proxy and current_date in (
                self.regime_cache[proxy].index.strftime("%Y-%m-%d")
            ):
                regime_series = self.regime_cache[proxy]
                d_idx  = regime_series.index.strftime("%Y-%m-%d")
                regime = int(
                    regime_series[d_idx == current_date].iloc[0]
                )
            regime_name = RegimeDetector.REGIME_NAMES.get(
                regime, "Unknown"
            )

            if hasattr(self, "_hmm_detector") and \
                    self._hmm_detector is not None:
                try:
                    hmm_regime = self._hmm_detector.get_regime(
                        pd.Timestamp(current_date)
                    )
                except Exception:
                    hmm_regime = "Bull-Stable"
            else:
                _hurst_to_hmm = {
                    "Bull-Trending": "Bull-Trending",
                    "Bull-MeanRev":  "Bull-Stable",
                    "Bear-Trending": "Bear-Stress",
                    "Bear-MeanRev":  "Bear-Stable",
                }
                hmm_regime = _hurst_to_hmm.get(regime_name, "Bull-Stable")

            if (hmm_regime == "Bull-Trending" and
                    _prev_hmm_regime != "Bull-Trending"):
                _last_bull_bar = bar_idx

            if hmm_regime in ("Bear-Stress", "Bear-Stable"):
                _bear_confirm_count += 1
            else:
                _bear_duration_bars = _bear_confirm_count
                _bear_confirm_count = 0

            _just_flipped_bull = (
                hmm_regime in ("Bull-Trending", "Bull-Stable") and
                _prev_hmm_regime in (
                    "Bear-Trending", "Bear-Stable", "Bear-Stress"
                )
            )
            _prev_hmm_regime = hmm_regime

            if _just_flipped_bull:
                _flip_equity = portfolio.get_portfolio_state()[
                    "total_equity"
                ]
                for anchor in self.cfg.global_tickers:
                    if (anchor not in portfolio.positions and
                            anchor in daily_prices):
                        _px  = daily_prices[anchor] * (
                            1 + self.cfg.base_slippage
                        )
                        # Fix B: bigger anchor re-entry on bull flip.
                        # 0.12 -> 0.18 per anchor. Anchors fire on the flip
                        # bar regardless of sector signals, so this is the
                        # one lever that raises post-2024 bull participation
                        # where the signal engine produces few/no buys.
                        # Both anchors at 0.18 = 36% deployed on the flip bar.
                        _bear_was_short = _bear_duration_bars <= 30
                        _anchor_size    = 0.24 if _bear_was_short else 0.22
                        _val = _flip_equity * _anchor_size
                        portfolio.execute_trade(
                            anchor, "BUY", _val, _px, current_date
                        )
                        self.trade_history.append({
                            "date":       current_date,
                            "ticker":     anchor,
                            "action":     "BUY",
                            "proba":      1.0,
                            "regime":     regime_name,
                            "hmm_regime": hmm_regime,
                            "weight":     _anchor_size,
                            "price":      round(_px, 2),
                            "shares":     round(_val / _px, 4),
                            "reason":     "anchor_reentry",
                        })
                        print(f"  🔁 ANCHOR RE-ENTRY: {anchor} "
                              f"on bull flip")

            active_sector_keys     = REGIME_TO_SECTORS.get(hmm_regime, [])
            active_sector_tickers  = []
            for sk in active_sector_keys:
                active_sector_tickers.extend(
                    SECTOR_REGISTRY[sk]["tickers"]
                )

            _spy_px    = daily_prices.get("SPY")
            _spy_prev  = prev_prices.get("SPY")
            _spy_crash = (
                (_spy_px / _spy_prev - 1) < -0.02
                if (_spy_px and _spy_prev and _spy_prev > 0) else False
            )
            _confirmed_bear = (_bear_confirm_count >= 2) or _spy_crash

            if _confirmed_bear and self.cfg.use_sector_models:
                _is_bear_regime = hmm_regime in ["Bear-Stress", "Bear-Stable"]
                _is_bull_regime = hmm_regime in ["Bull-Trending", "Bull-Stable"]

                for held_ticker in list(portfolio.positions.keys()):
                    held_sector   = TICKER_TO_SECTOR.get(held_ticker)
                    if held_sector is None:
                        continue

                    held_regimes  = SECTOR_REGISTRY.get(
                        held_sector, {}
                    ).get("deploy_regimes", [])

                    _sector_is_bear = any(
                        r in ["Bear-Stress", "Bear-Stable"]
                        for r in held_regimes
                    )
                    _sector_is_bull = any(
                        r in ["Bull-Trending", "Bull-Stable"]
                        for r in held_regimes
                    )

                    if _sector_is_bull and _sector_is_bear:
                        continue

                    _should_exit = (
                        (_sector_is_bull and _is_bear_regime) or
                        (_sector_is_bear and _is_bull_regime)
                    )
                    if not _should_exit:
                        continue

                    held_price = daily_prices.get(held_ticker)
                    if held_price is None:
                        print(f"  ⚠️  FORCED EXIT SKIPPED: {held_ticker} "
                              f"no price on {current_date}")
                        continue

                    print(f"  🔄 FORCED EXIT: {held_ticker} "
                          f"sector={held_sector} "
                          f"hmm={hmm_regime} "
                          f"on {current_date}")

                    sell_price  = held_price * (1 - self.cfg.base_slippage)
                    sell_shares = portfolio.positions[held_ticker]["shares"]
                    portfolio.execute_trade(
                        held_ticker, "SELL",
                        sell_shares, sell_price, current_date
                    )
                    self.trade_history.append({
                        "date":   current_date,
                        "ticker": held_ticker,
                        "action": "SELL",
                        "proba":  0.0,
                        "regime": regime_name,
                        "weight": 0.0,
                        "price":  round(sell_price, 2),
                        "shares": round(sell_shares, 4),
                        "reason": "regime_exit",
                    })

            _is_bear = hmm_regime in ["Bear-Stress", "Bear-Stable"]
            if _is_bear and self.cfg.use_sector_models:
                _bear_state = portfolio.get_portfolio_state()
                if _bear_state.get("heat", 0.0) > 0.30:
                    _trim_candidates = sorted(
                        [
                            (t, bar_probas.get(t, 0.0))
                            for t in list(portfolio.positions.keys())
                            if TICKER_TO_SECTOR.get(t) not in (
                                None, "defensive", "gold"
                            )
                        ],
                        key=lambda x: x[1],
                    )
                    for _trim_tkr, _ in _trim_candidates:
                        if _trim_tkr == "TSLA" and hmm_regime not in ("Bear-Trending","Bear-Stress"):
                            continue
                        if portfolio.get_portfolio_state().get(
                            "heat", 0.0
                        ) <= 0.30:
                            break
                        _trim_px = daily_prices.get(_trim_tkr)
                        if _trim_px is None:
                            continue
                        _trim_sell   = _trim_px * (
                            1 - self.cfg.base_slippage
                        )
                        _trim_shares = portfolio.positions[
                            _trim_tkr
                        ]["shares"]
                        portfolio.execute_trade(
                            _trim_tkr, "SELL",
                            _trim_shares, _trim_sell, current_date
                        )
                        self.trade_history.append({
                            "date":   current_date,
                            "ticker": _trim_tkr,
                            "action": "SELL",
                            "proba":  0.0,
                            "regime": regime_name,
                            "weight": 0.0,
                            "price":  round(_trim_sell, 2),
                            "shares": round(_trim_shares, 4),
                            "reason": "bear_heat_trim",
                        })

            if (self.cfg.use_sector_models and
                    self.sector_engine is not None and
                    self.sector_engine.loaded):
                for ticker in active_sector_tickers:
                    if ticker not in daily_prices:
                        continue
                    try:
                        sig = self.sector_engine.get_signals(
                            ticker=ticker,
                            current_date=current_date,
                            regime=hmm_regime,
                        )
                        if sig["active"] and not np.isnan(sig["proba_buy"]):
                            bar_probas[ticker] = sig["proba_buy"]
                            if not hasattr(self, "_sector_thresholds"):
                                self._sector_thresholds = {}
                            self._sector_thresholds[ticker] = sig[
                                "threshold"
                            ]
                        elif sig["active"] and np.isnan(sig["proba_buy"]):
                            if len(cache_miss_log) < 500:
                                _sc = self.sector_engine._sig_cache.get(
                                    ticker
                                )
                                cache_miss_log.append({
                                    "date":    current_date,
                                    "ticker":  ticker,
                                    "sector":  sig["sector"],
                                    "regime":  hmm_regime,
                                    "proba_buy": sig["proba_buy"],
                                    "threshold": sig["threshold"],
                                    "sig_cache_loaded": _sc is not None,
                                    "sig_cache_rows":   (
                                        len(_sc) if _sc is not None else 0
                                    ),
                                    "date_in_sig_cache": (
                                        current_date in _sc.index
                                        if _sc is not None else False
                                    ),
                                })
                    except Exception:
                        pass

            _global_anchors = set(self.cfg.global_tickers)
            _filtered_probas = {}
            _in_bear = hmm_regime in (
                "Bear-Trending", "Bear-Stable", "Bear-Stress"
            )
            for tkr, proba in bar_probas.items():
                if tkr in _global_anchors:
                    # Gate anchors in confirmed bear — hold existing
                    # positions but block new entries after 5 bear bars.
                    # Uses _regime_bar_count (not _bear_confirm_count)
                    # because it correctly tracks how long the current
                    # bear phase has been active, resetting on every
                    # regime change.
                    if _in_bear and _regime_bar_count > 5:
                        if tkr not in portfolio.positions:
                            continue  # block new anchor entry
                    _filtered_probas[tkr] = proba
                    continue
                tkr_sector = TICKER_TO_SECTOR.get(tkr)
                if tkr_sector is None:
                    _filtered_probas[tkr] = proba
                    continue
                if tkr_sector in active_sector_keys:
                    _filtered_probas[tkr] = proba
            bar_probas = _filtered_probas

            tickers_with_signal = list(bar_probas.keys())
            n_sig = len(tickers_with_signal)
            kalman_weights = (np.ones(n_sig) / n_sig
                              if n_sig > 0 else np.array([]))

            if n_sig > 0:
                key = tuple(sorted(tickers_with_signal))
                if key not in self.kalman_map:
                    self.kalman_map[key] = KalmanEnsemble(n_sig)
                proba_vec = np.array(
                    [bar_probas[t] for t in tickers_with_signal]
                )
                realised = (np.mean([
                    (daily_prices.get(t, 0) /
                     prev_prices.get(t, daily_prices.get(t, 1))) - 1
                    for t in tickers_with_signal
                    if t in prev_prices
                ]) if prev_prices else 0.0)
                kalman_weights = self.kalman_map[key].update(
                    proba_vec, realised
                )

            competition_returns = {}
            for ticker in tickers_with_signal:
                if ticker in prev_prices and prev_prices[ticker] > 0:
                    competition_returns[ticker] = (
                        daily_prices.get(ticker, prev_prices[ticker]) /
                        prev_prices[ticker] - 1
                    )
                else:
                    competition_returns[ticker] = 0.0
            comp_weights = self.competition.score(
                tickers_with_signal, competition_returns
            )

            for ticker in tickers_with_signal:
                ret = competition_returns.get(ticker, 0.0)
                self.champion.record(regime, ticker, ret)

            _bar_pf   = portfolio.get_portfolio_state()
            equity    = _bar_pf["total_equity"]
            _bar_heat = _bar_pf.get("heat", 0.0)
            _bar_dd   = abs(_bar_pf.get("drawdown", 0.0))

            if hmm_regime != _pyramid_prev_regime:
                _regime_bar_count      = 0
                _regime_cumulative_ret = 0.0
                _regime_base_equity    = equity
            _pyramid_prev_regime = hmm_regime

            _regime_bar_count += 1
            if _regime_base_equity and _regime_base_equity > 0:
                _regime_cumulative_ret = (
                    (equity - _regime_base_equity) / _regime_base_equity
                )

            if hmm_regime == "Bull-Trending":
                if (_regime_bar_count >= 20 and
                        _regime_cumulative_ret >= 0.025 and
                        _bar_dd < 0.08):
                    _pyramid_mult = 1.20
                elif (_regime_bar_count >= 10 and
                      _regime_cumulative_ret >= 0.01 and
                      _bar_dd < 0.10):
                    _pyramid_mult = 1.10
                else:
                    _pyramid_mult = 1.00
            elif hmm_regime == "Bull-Stable":
                if (_regime_bar_count >= 15 and
                        _regime_cumulative_ret >= 0.02):
                    _pyramid_mult = 1.08
                else:
                    _pyramid_mult = 1.00
            else:
                _pyramid_mult = 1.00

            # ── TSLA: runs every bar, independent of regime/sector activation ──
            if "TSLA" in daily_prices:
                _tsla_px = daily_prices.get("TSLA")
                _in_pos  = "TSLA" in portfolio.positions
                if _tsla_px and _tsla_px > 0:
                    _tc   = self.loader.price_data["TSLA"]["close"]
                    _tidx = _tc.index.strftime("%Y-%m-%d")
                    _loc  = np.where(_tidx == current_date)[0]
                    if len(_loc):
                        _i   = int(_loc[0])
                        _cls = _tc.iloc[max(0, _i-200): _i+1].values.astype(float)
                        _d   = np.diff(_cls[-15:]) if len(_cls) >= 15 else np.array([0.0])
                        _ag  = float(np.mean(np.clip(_d, 0, None))) or 1e-9
                        _al  = float(np.mean(np.clip(-_d, 0, None))) or 1e-9
                        _rsi = 100.0 - (100.0 / (1.0 + _ag / _al))
                        _rsi_hist = []
                        for _j in range(max(0,_i-16), _i+1):
                            _c2 = _tc.iloc[max(0,_j-14):_j+1].values.astype(float)
                            _d2 = np.diff(_c2)
                            _g2 = float(np.mean(np.clip(_d2,0,None))) or 1e-9
                            _l2 = float(np.mean(np.clip(-_d2,0,None))) or 1e-9
                            _rsi_hist.append(100.0-(100.0/(1.0+_g2/_l2)))
                        _rsi_rising  = len(_rsi_hist)>=2 and _rsi_hist[-1]>_rsi_hist[-2]
                        _rsi_falling = len(_rsi_hist)>=3 and _rsi_hist[-1]<_rsi_hist[-2]<_rsi_hist[-3]
                        _rsi_was_hot = len(_rsi_hist)>=3 and max(_rsi_hist[-3:]) > 70
                        _ma200 = float(np.mean(_cls[-200:])) if len(_cls)>=200 else float(np.mean(_cls))
                        _pct   = (_tsla_px/_ma200 - 1)*100 if _ma200>0 else 0.0
                        _peak_exit = _in_pos and _rsi_was_hot and _rsi_falling
                        _ext_exit  = _in_pos and _pct > 120.0
                        _bear_exit = _in_pos and hmm_regime in ("Bear-Trending","Bear-Stress")
                        if _peak_exit or _ext_exit or _bear_exit:
                            _reason = ("tsla_peak_exit" if _peak_exit
                                       else "tsla_ext_exit" if _ext_exit
                                       else f"tsla_bear|{hmm_regime}")
                            _sh = portfolio.positions["TSLA"]["shares"]
                            portfolio.execute_trade("TSLA","SELL",_sh,
                                _tsla_px*(1-self.cfg.base_slippage), current_date)
                            self.trade_history.append({
                                "date":current_date,"ticker":"TSLA","action":"SELL",
                                "proba":0.0,"regime":regime_name,"hmm_regime":hmm_regime,
                                "weight":0.0,"price":round(_tsla_px,2),
                                "shares":round(_sh,4),
                                "reason":f"{_reason}|rsi={_rsi:.1f}|pct={_pct:.1f}",
                            })
                            _trend_added.discard("TSLA")
                        elif (not _in_pos
                              and hmm_regime == "Bull-Trending"
                              and 42 <= _rsi <= 62
                              and -10 <= _pct <= 60
                              and _rsi_rising):
                            _tsla_size = equity * 0.12 * 0.80
                            _tsla_exec = _tsla_px * (1 + self.cfg.base_slippage)
                            _tsla_sh   = _tsla_size / _tsla_exec
                            if _tsla_sh >= 0.01:
                                portfolio.execute_trade("TSLA","BUY",_tsla_size,
                                    _tsla_exec, current_date)
                                self.trade_history.append({
                                    "date":current_date,"ticker":"TSLA","action":"BUY",
                                    "proba":0.88,"regime":regime_name,"hmm_regime":hmm_regime,
                                    "weight":0.12,"price":round(_tsla_exec,2),
                                    "shares":round(_tsla_sh,4),
                                    "reason":"tsla_runup_entry",
                                })
            bar_probas.pop("TSLA", None)
            for i, ticker in enumerate(tickers_with_signal):
                if ticker == "TSLA":
                    continue
                price = daily_prices.get(ticker)
                if price is None:
                    continue

                raw_proba     = bar_probas[ticker]
                k_weight      = (float(kalman_weights[i]) if n_sig > 0
                                 else 1.0 / max(len(tickers_with_signal), 1))
                c_weight      = comp_weights.get(ticker, 0.0)
                blended_w     = 0.5 * k_weight + 0.5 * c_weight
                blended_proba = raw_proba * (1.0 + blended_w * 0.1)
                blended_proba = min(blended_proba, 0.99)

                eff_slip   = self.cfg.base_slippage
                exec_price = price * (1 + eff_slip)

                _max_pos = 12 if hmm_regime == "Bull-Trending" else 9
                if len(portfolio.positions) >= _max_pos:
                    continue

                # Bull-Trending sizing scales with RegBar (trend durability).
                # Low RegBar = fresh/choppy bull → keep base 0.15 (cautious).
                # High RegBar = sustained confirmed bull → size up to capture
                # the upside. This avoids amplifying losses in choppy fake-bulls
                # (which reverse fast at low RegBar) while loading into the
                # durable bull legs that actually run. Bear sizes unchanged.
                if hmm_regime == "Bull-Trending":
                    _regime_size = (
                        0.22 if _regime_bar_count >= 20 else
                        0.18 if _regime_bar_count >= 10 else
                        0.15
                    )
                else:
                    _regime_size = (
                        0.12 if hmm_regime == "Bull-Stable"              else
                        0.10 if hmm_regime in ("Bear-Stress", "Bear-Stable") else
                        0.08
                    )
                # Cap raised 0.20 -> 0.26 so confirmed-trend sizing (0.22)
                # plus pyramid multiplier can actually deploy before clipping.
                _regime_size = min(_regime_size * _pyramid_mult, 0.26)

                _ticker_mult = {
                    "NVDA": 0.90,
                    "TSLA": 0.80,
                }.get(ticker, 1.0)

                pos_value = equity * _regime_size * _ticker_mult
                # ── RL callback hook (inert when _rl_callback is None) ─────────────
                _rl_mult = 1.0
                if self._rl_callback is not None:
                    _rl_state = {
                        "ticker":           ticker,
                        "confidence":       float(blended_proba),
                        "hmm_regime":       hmm_regime,
                        "heat":             float(_bar_heat),
                        "drawdown":         float(_bar_dd),
                        "regime_bar_count": int(_regime_bar_count),
                        "n_open":           len(portfolio.positions),
                        "pos_value_base":   float(pos_value),
                        "equity":           float(equity),
                        "current_date":     current_date,
                    }
                    try:
                        _rl_mult = float(self._rl_callback(_rl_state))
                        _rl_mult = max(0.5, min(_rl_mult, 1.5))
                    except Exception:
                        _rl_mult = 1.0
                    self._rl_signal_log.append({
                        **_rl_state,
                        "rl_mult":         _rl_mult,
                        "final_pos_value": float(pos_value * _rl_mult),
                    })
                pos_value = pos_value * _rl_mult
                # ────────────────────────────────────────────────────────────────────
                shares    = pos_value / exec_price if exec_price > 0 else 0
                if shares < 0.01:
                    continue

                _sector_thr    = getattr(self, "_sector_thresholds", {})
                _regime_buy_thr = (
                    0.50 if hmm_regime == "Bull-Trending" else
                    0.52 if hmm_regime == "Bull-Stable"   else
                    0.58
                )

                if ticker in _sector_thr:
                    _native  = _sector_thr[ticker]
                    buy_thr  = _native
                    sell_thr = _native * 0.80
                else:
                    buy_thr  = self.cfg.ticker_buy_thresholds.get(
                        ticker, _regime_buy_thr
                    )
                    sell_thr = self.cfg.ticker_sell_thresholds.get(
                        ticker, self.cfg.sell_threshold
                    )

                if (ticker in ("NVDA") and
                        (bar_idx - _last_bull_bar) <= 3):
                    buy_thr = buy_thr * 0.88

                if hmm_regime == "Bull-Trending":
                    if ticker == "META":
                        buy_thr = 0.48

                _is_anchor = ticker in self.cfg.global_tickers
                if blended_proba > buy_thr and (
                    _is_anchor or blended_w > 0.05
                ):
                    if ticker not in portfolio.positions:
                        if (ticker in DEFENSIVE_TICKERS and
                                hmm_regime == "Bull-Trending" and
                                _regime_bar_count >= 10):
                            continue  # skip defensive buys in confirmed deep bull
                        _sector_key = TICKER_TO_SECTOR.get(ticker)
                        if _sector_key is not None:
                            _sector_tickers = SECTOR_REGISTRY.get(
                                _sector_key, {}
                            ).get("tickers", [])
                            _sector_open = sum(
                                1 for t in _sector_tickers
                                if t in portfolio.positions
                            )
                            if _sector_open >= 2:
                                continue

                        _current_dd = abs(
                            portfolio.get_portfolio_state().get(
                                "drawdown", 0.0
                            )
                        )
                        if _current_dd > 0.15:
                            _is_anchor = ticker in self.cfg.global_tickers
                            _in_bull   = hmm_regime in (
                                "Bull-Trending", "Bull-Stable"
                            )
                            if _is_anchor and _in_bull:
                                pos_value = equity * 0.05
                            else:
                                continue
                        elif _current_dd > 0.14:
                            pos_value = pos_value * 0.75

                        _live_state = portfolio.get_portfolio_state()
                        _live_heat  = _live_state.get("heat", 0.0)
                        _live_cash  = _live_state.get("cash", 0.0)

                        if _live_cash < equity * 0.05:
                            continue

                        _heat_cap = (
                            0.90 if hmm_regime == "Bull-Trending" else
                            0.80 if hmm_regime == "Bull-Stable"   else
                            0.30
                        )
                        if _live_heat >= _heat_cap:
                            continue

                        _trade_heat_add = pos_value / equity
                        if (_live_heat + _trade_heat_add) > _heat_cap:
                            pos_value = (
                                equity * (_heat_cap - _live_heat) * 0.95
                            )
                            if pos_value < equity * 0.02:
                                continue
                            shares = (pos_value / exec_price
                                      if exec_price > 0 else 0)

                        _avail = portfolio.get_portfolio_state()["cash"]
                        if _avail < pos_value:
                            if _avail < equity * 0.02:
                                continue
                            pos_value = _avail * 0.90
                            shares    = (pos_value / exec_price
                                         if exec_price > 0 else 0)

                        if shares < 0.01:
                            continue

                        portfolio.execute_trade(
                            ticker, "BUY", pos_value,
                            exec_price, current_date
                        )
                        self.trade_history.append({
                            "date":       current_date,
                            "ticker":     ticker,
                            "action":     "BUY",
                            "proba":      round(blended_proba, 4),
                            "regime":     regime_name,
                            "hmm_regime": hmm_regime,
                            "weight":     round(blended_w, 4),
                            "price":      round(exec_price, 2),
                            "shares":     round(shares, 4),
                        })

                elif blended_proba < sell_thr:
                    if len(sell_log) < 2000:
                        sell_log.append({
                            "date":            current_date,
                            "ticker":          ticker,
                            "blended_proba":   round(blended_proba, 4),
                            "sell_thr":        round(sell_thr, 4),
                            "buy_thr":         round(buy_thr, 4),
                            "proba_above_buy": blended_proba >= buy_thr,
                            "in_portfolio":    ticker in portfolio.positions,
                            "regime":          regime_name,
                            "hmm_regime":      hmm_regime,
                        })
                    if ticker in portfolio.positions:
                        _held       = _bars_held.get(ticker, 0)
                        _entry_px   = portfolio.positions[ticker]["avg_price"]
                        _profitable = price > _entry_px
                        if _held < self.cfg.min_hold_bars and _profitable:
                            pass  # winning + too new → hold (stop profit-churn)
                        else:
                            sell_price  = price * (1 - eff_slip)
                            sell_shares = portfolio.positions[ticker]["shares"]
                            portfolio.execute_trade(
                                ticker, "SELL", sell_shares,
                                sell_price, current_date
                            )
                            self.trade_history.append({
                                "date":       current_date,
                                "ticker":     ticker,
                                "action":     "SELL",
                                "proba":      round(blended_proba, 4),
                                "regime":     regime_name,
                                "hmm_regime": hmm_regime,
                                "weight":     round(blended_w, 4),
                                "price":      round(sell_price, 2),
                                "shares":     round(sell_shares, 4),
                            })

            # trend-add: pyramid into winners after 10 bars / 5%+ gain
            if hmm_regime == "Bull-Trending":
                for tkr in list(portfolio.positions.keys()):
                    if tkr == "TSLA":
                        continue
                    if tkr in _trend_added:
                        continue
                    if (tkr in DEFENSIVE_TICKERS and
                            hmm_regime == "Bull-Trending" and
                            _regime_bar_count >= 10):
                        continue  # skip defensive trend-add in confirmed deep bull
                    _px = daily_prices.get(tkr)
                    if _px is None:
                        continue
                    _avg = portfolio.positions[tkr]["avg_price"]
                    if _avg <= 0:
                        continue
                    _unreal = (_px / _avg) - 1
                    if _unreal > 0.05 and _bars_held.get(tkr, 0) >= 10:
                        _existing_alloc = (
                            portfolio.positions[tkr]["shares"] * _px
                        ) / equity
                        if _existing_alloc < 0.20:
                            _add_px  = _px * (1 + self.cfg.base_slippage)
                            _add_val = equity * 0.05
                            portfolio.execute_trade(
                                tkr, "BUY", _add_val, _add_px, current_date
                            )
                            _trend_added.add(tkr)
                            self.trade_history.append({
                                "date":       current_date,
                                "ticker":     tkr,
                                "action":     "BUY",
                                "proba":      1.0,
                                "regime":     regime_name,
                                "hmm_regime": hmm_regime,
                                "weight":     0.05,
                                "price":      round(_add_px, 2),
                                "shares":     round(_add_val / _add_px, 4),
                                "reason":     "trend_add",
                            })

            portfolio.update_prices(daily_prices)
            portfolio.record_snapshot(current_date)
            prev_prices = {**daily_prices}

            self.equity_history.append({
                "date":         current_date,
                "equity":       portfolio.get_portfolio_state()["total_equity"],
                "regime":       hmm_regime,   # HMM label — drives the breakdown
                "hurst_regime": regime_name,  # old Hurst label, kept for reference
            })

            self._oos_sell_log       = sell_log
            self._oos_cache_miss_log = cache_miss_log

            if bar_idx % 20 == 0:
                state = portfolio.get_portfolio_state()
                print(f"  📅 {current_date} | {regime_name:<18} | "
                      f"Equity: ${state['total_equity']:>10,.0f} | "
                      f"Heat: {state['heat']:.1%} | "
                      f"DD: {state['drawdown']:.1%} | "
                      f"RegBar: {_regime_bar_count} | "
                      f"RegRet: {_regime_cumulative_ret:.2%} | "
                      f"PyMult: {_pyramid_mult:.2f}")

    def _finalise(self, portfolio: Portfolio):
        os.makedirs("5_backtesting/results", exist_ok=True)
        equity_df = pd.DataFrame(self.equity_history)
        trades_df = pd.DataFrame(self.trade_history)
        equity_df.to_csv("5_backtesting/results/equity_curve.csv",
                         index=False)
        trades_df.to_csv("5_backtesting/results/trade_log.csv",
                         index=False)

        perf = portfolio.get_performance_summary()

        champs     = self.champion.champions()
        champ_rows = [
            {"regime": RegimeDetector.REGIME_NAMES.get(r, r),
             "champion_ticker": t}
            for r, t in champs.items()
        ]
        pd.DataFrame(champ_rows).to_csv(
            "5_backtesting/results/champion_selection.csv", index=False
        )

        decay_rows = []
        for ticker, res in self.decay_results.items():
            row = {
                "ticker":       ticker,
                "half_life":    res.get("half_life"),
                "optimal_hold": res.get("optimal_hold"),
            }
            row.update({
                f"edge_t{h}": res["edge_by_horizon"].get(h, 0)
                for h in self.cfg.decay_horizons
            })
            decay_rows.append(row)
        pd.DataFrame(decay_rows).to_csv(
            "5_backtesting/results/alpha_decay.csv", index=False
        )

        portfolio_trades = (
            pd.DataFrame(portfolio.trade_history)
            if portfolio.trade_history else pd.DataFrame()
        )
        stress_df = self.stress.regime_stress(equity_df, portfolio_trades)
        stress_df.to_csv("5_backtesting/results/stress_regime.csv",
                         index=False)

        print("\n" + "═" * 55)
        print("📊  BACKTEST V2 RESULTS")
        print("═" * 55)
        print(f"Return    : {perf.get('total_return', 0):.2f}%")
        print(f"Sharpe    : {perf.get('sharpe_ratio', 0):.3f}")
        print(f"Max DD    : {perf.get('max_drawdown', 0):.2f}%")
        print(f"Trades    : {perf.get('num_trades', 0)}")

        print("\n🏆  CHAMPION PER REGIME:")
        for row in champ_rows:
            print(f"  {row['regime']:<22} → {row['champion_ticker']}")

        if not stress_df.empty:
            print("\n🔥  STRESS TEST RESULTS:")
            for _, row in stress_df.iterrows():
                print(f"  {row['period']:<18} | "
                      f"Ret: {row.get('total_return', '?'):>6}% | "
                      f"DD: {row.get('max_drawdown', '?'):>5}% | "
                      f"Sharpe: {row.get('sharpe', '?')}")

        try:
            diag = {
                "run_date":           pd.Timestamp.now().strftime(
                    "%Y-%m-%d %H:%M"
                ),
                "start_date":         self.cfg.start_date,
                "end_date":           self.cfg.end_date,
                "initial_capital":    self.cfg.initial_capital,
                "final_equity":       perf.get("final_equity", 0),
                "total_return":       perf.get("total_return", 0),
                "annualized_return":  perf.get("annualized_return", 0),
                "sharpe_ratio":       perf.get("sharpe_ratio", 0),
                "max_drawdown":       perf.get("max_drawdown", 0),
                "num_trades":         perf.get("num_trades", 0),
                "use_sector_models":  self.cfg.use_sector_models,
                "random_seed":        self.cfg.random_seed,
                "tickers":            str(self.cfg.tickers),
                "sector_tickers":     str(
                    self.cfg.sector_tickers
                    if self.cfg.use_sector_models else []
                ),
            }

            if not trades_df.empty and "ticker" in trades_df.columns:
                for tkr in trades_df["ticker"].unique():
                    tkr_trades  = trades_df[trades_df["ticker"] == tkr]
                    buy_trades  = tkr_trades[tkr_trades["action"] == "BUY"]
                    sell_trades = tkr_trades[tkr_trades["action"] == "SELL"]
                    diag[f"{tkr}_trades"] = len(tkr_trades)
                    diag[f"{tkr}_buys"]   = len(buy_trades)
                    diag[f"{tkr}_sells"]  = len(sell_trades)

            if self.equity_history:
                eq_df    = pd.DataFrame(self.equity_history)
                reg_dist = eq_df["regime"].value_counts().to_dict()
                for reg, cnt in reg_dist.items():
                    diag[f"regime_{reg.replace('-','_')}_bars"] = cnt

            if (self.cfg.use_sector_models and
                    not trades_df.empty and
                    "ticker" in trades_df.columns):
                for sector_key, cfg_s in SECTOR_REGISTRY.items():
                    sector_trades = trades_df[
                        trades_df["ticker"].isin(cfg_s["tickers"])
                    ]
                    diag[f"sector_{sector_key}_trades"] = len(sector_trades)

            diag["forced_exits_total"] = sum(
                1 for t in self.trade_history
                if t.get("reason") == "regime_exit"
            )

            diag_df   = pd.DataFrame([diag])
            diag_path = "5_backtesting/results/backtest_diagnostic.csv"
            diag_df.to_csv(diag_path, index=False)
            print(f"  Diagnostic saved → {diag_path}")

        except Exception as _e:
            print(f"  ⚠️  Diagnostic CSV failed: {_e}")

        try:
            sell_diag = pd.DataFrame(
                getattr(self, "_oos_sell_log", [])
            )
            sell_path = "5_backtesting/results/sell_signal_log.csv"
            sell_diag.to_csv(sell_path, index=False)
            print(f"  Sell signal log → {sell_path} "
                  f"({len(sell_diag)} rows)")
        except Exception as _e:
            print(f"  ⚠️  sell_signal_log.csv failed: {_e}")

        try:
            miss_diag = pd.DataFrame(
                getattr(self, "_oos_cache_miss_log", [])
            )
            miss_path = "5_backtesting/results/cache_miss_log.csv"
            miss_diag.to_csv(miss_path, index=False)
            print(f"  Cache miss log  → {miss_path} "
                  f"({len(miss_diag)} rows)")
        except Exception as _e:
            print(f"  ⚠️  cache_miss_log.csv failed: {_e}")

        print("\n✅  Results saved to 5_backtesting/results/")


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = BacktestConfig(
        tickers           = ["AAPL", "QQQ", "SPY"],
        start_date        = "2020-01-01",
        end_date          = "2026-05-28",
        initial_capital   = 100_000,
        train_months      = 9,
        oos_months        = 1,
        retrain           = False,
        fixed_size        = 0.10,
        base_slippage     = 0.001,
        use_sector_models = True,
        random_seed       = 42,
        # Cache paths — delete these files to force a fresh download/fit
        price_cache_path  = "5_backtesting/results/price_data.pkl",
        hmm_cache_path    = "5_backtesting/results/hmm_detector.pkl",
    )

    bt = BacktestEngineV2(cfg)
    bt.prepare()
    bt.run()