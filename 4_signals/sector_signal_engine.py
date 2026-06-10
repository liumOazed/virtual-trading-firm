"""
sector_signal_engine.py
=======================
VIRTUAL TRADING FIRM | Stage 4 — Sector Signal Engine

Routes each ticker to its correct sector PKL at inference time.
Returns proba_buy with sector-native threshold per ticker.

Sector → PKL → Tickers → Native threshold:
  Hardware   → hardware_model.pkl   → NVDA, AVGO, ASML, AMD, TSM  → 0.38
  Hypercloud → hypercloud_model.pkl → MSFT, GOOGL, AMZN, META     → 0.53
  Autos      → autos_model.pkl      → GM, F, TSLA, TM, RACE       → 0.30
  Defensive  → defensive_model.pkl  → XOM, CVX, PG, WMT, GLD      → 0.34

HMM regime → sector deployment (Option A gating):
  Bull-Trending → Hardware + Autos
  Bull-Stable   → Hypercloud
  Bear-Stress   → Defensive
  Bear-Stable   → Defensive

Usage:
  engine = SectorSignalEngine()
  engine.load()
  signals = engine.get_signals(ticker, price_df, current_date)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Optional, List

warnings.filterwarnings("ignore")

# ── path setup ──────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "4_signals"))

# ── TSLA Strategy A override ─────────────────────────────────────────────────
try:
    from tsla_strategy_a import TSLASignalOverride as _TSLAOverride
    _tsla_override_instance: _TSLAOverride = None
except ImportError:
    _TSLAOverride = None
    _tsla_override_instance = None


# ═══════════════════════════════════════════════════════════════════════
# SECTOR REGISTRY
# ═══════════════════════════════════════════════════════════════════════

SECTOR_REGISTRY = {
    "hardware": {
        "name":            "Hardware / Semiconductor",
        "pkl":             "4_signals/models/hardware_model.pkl",
        "tickers":         ["NVDA", "AVGO", "ASML", "AMD", "TSM"],
        "native_threshold": 0.52,
        "deploy_regimes":  ["Bull-Trending"],
    },
    "hypercloud": {
        "name":            "Hypercloud",
        "pkl":             "4_signals/models/hypercloud_model.pkl",
        "tickers":         ["MSFT", "GOOGL", "AMZN", "META"],
        "native_threshold": 0.53,
        "deploy_regimes":  ["Bull-Trending", "Bull-Stable"],
    },
    "autos": {
        "name":            "Autos / EV",
        "pkl":             "4_signals/models/autos_model.pkl",
        "tickers":         ["TSLA", "RACE"],
        "native_threshold": 0.55,
        "deploy_regimes":  ["Bull-Trending"],
    },
    "defensive": {
        "name":            "Defensive",
        "pkl":             "4_signals/models/defensive_model.pkl",
        "tickers":         ["XOM", "CVX", "PG", "WMT"],
        "native_threshold": 0.55,
        "deploy_regimes":  ["Bear-Stress", "Bear-Stable"],
    },
    "gold": {
        "name":            "Gold / Macro Hedge",
        "pkl":             "4_signals/models/defensive_model.pkl",
        "tickers":         ["GLD"],
        "native_threshold": 0.55,
        "deploy_regimes":  ["Bull-Trending", "Bull-Stable", "Bear-Stress", "Bear-Stable"],
    },
}

# HMM regime → active sectors mapping (Option A gating)
REGIME_TO_SECTORS = {
    "Bull-Trending": ["hardware", "autos", "gold"],
    "Bull-Stable":   ["hypercloud", "gold"],
    "Bear-Stress":   ["defensive", "gold"],
    "Bear-Stable":   ["defensive", "gold"],
}

# ticker → sector lookup (built from registry)
TICKER_TO_SECTOR = {
    ticker: sector_key
    for sector_key, cfg in SECTOR_REGISTRY.items()
    for ticker in cfg["tickers"]
}


# ═══════════════════════════════════════════════════════════════════════
# SECTOR SIGNAL ENGINE
# ═══════════════════════════════════════════════════════════════════════

class SectorSignalEngine:
    """
    Routes tickers to sector PKLs for inference.
    Loads all 4 sector models at startup.
    Returns proba_buy and buy/sell signal per ticker per bar.
    """

    def __init__(self, model_dir: str = "4_signals/models"):
        self.model_dir   = model_dir
        self.models:     Dict[str, dict] = {}
        self.loaded:     bool            = False
        self._cache:     Dict[str, dict] = {}            # date+ticker → result dict
        self._sig_cache: Dict[str, pd.DataFrame] = {}   # ticker → full sig_df (date-indexed)

    # ── loading ────────────────────────────────────────────────────────

    def load(self):
        """Load all 4 sector PKLs into memory."""
        print("\n📦 Loading sector models...")
        failed = []

        for sector_key, cfg in SECTOR_REGISTRY.items():
            pkl_path = cfg["pkl"]
            if not os.path.exists(pkl_path):
                # try relative to ROOT
                pkl_path = os.path.join(ROOT, cfg["pkl"])

            if not os.path.exists(pkl_path):
                print(f"  ❌ {cfg['name']}: PKL not found at {cfg['pkl']}")
                failed.append(sector_key)
                continue

            try:
                data = joblib.load(pkl_path)
                from signal_engine import SignalEngine
                se = SignalEngine(pkl_path)
                self.models[sector_key] = {
                    "data":      data,
                    "name":      cfg["name"],
                    "tickers":   cfg["tickers"],
                    "threshold": cfg["native_threshold"],
                    "pkl_path":  pkl_path,
                    "engine":    se,
                }
                size_mb = round(os.path.getsize(pkl_path) / 1e6, 1)
                print(f"  ✅ {cfg['name']:<25} | "
                      f"threshold={cfg['native_threshold']} | "
                      f"{size_mb}MB | "
                      f"tickers={cfg['tickers']}")
            except Exception as e:
                print(f"  ❌ {cfg['name']}: load failed — {e}")
                failed.append(sector_key)

        if failed:
            print(f"\n  ⚠️  Failed to load: {failed}")
            print(f"      Run sector_model_trainer.py to retrain missing sectors")

        self.loaded = len(self.models) > 0
        print(f"\n  Loaded {len(self.models)}/5 sector models")
        return self

    # ── ticker routing ─────────────────────────────────────────────────

    def get_sector_for_ticker(self, ticker: str) -> Optional[str]:
        """Return sector_key for a ticker, or None if not in any sector."""
        return TICKER_TO_SECTOR.get(ticker)

    def get_active_sectors_for_regime(self, regime: str) -> List[str]:
        """Return list of active sector keys for a given HMM regime."""
        return REGIME_TO_SECTORS.get(regime, [])

    def is_ticker_active(self, ticker: str, regime: str) -> bool:
        """
        Return True if ticker's sector is active in the given regime.
        Used by backtest engine to gate position sizing.
        """
        sector_key = self.get_sector_for_ticker(ticker)
        if sector_key is None:
            return False
        active = self.get_active_sectors_for_regime(regime)
        return sector_key in active

    # ── inference ──────────────────────────────────────────────────────

    def get_signals(self,
                    ticker:       str,
                    current_date: str,
                    regime:       str = "Bull-Stable",
                    price_df:     pd.DataFrame = None) -> dict:
        """
        Run inference for a single ticker on a given date.

        Args:
            ticker:       Ticker symbol e.g. 'NVDA'
            price_df:     OHLCV dataframe with date index
            current_date: Date string 'YYYY-MM-DD'
            regime:       Current HMM regime label

        Returns dict with:
            proba_buy:   float — buy probability from sector model
            signal:      int   — 1=BUY, 0=SELL/HOLD
            threshold:   float — sector-native threshold used
            sector:      str   — sector name
            active:      bool  — whether sector is active in current regime
            regime:      str   — regime passed in
        """
        # default response — NaN so callers can detect misses (date mismatch, empty sig_df, etc.)
        default = {
            "proba_buy": float("nan"),
            "signal":    0,
            "threshold": 0.5,
            "sector":    "unknown",
            "active":    False,
            "regime":    regime,
        }

        if not self.loaded:
            return default

        # route ticker to sector
        sector_key = self.get_sector_for_ticker(ticker)
        if sector_key is None or sector_key not in self.models:
            return default

        sector_model = self.models[sector_key]
        threshold    = sector_model["threshold"]
        is_active    = self.is_ticker_active(ticker, regime)

        # check cache
        cache_key = f"{current_date}_{ticker}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            cached["active"] = is_active
            cached["regime"] = regime
            return cached

        try:
            # reuse the pre-loaded engine; build sig_df once per ticker
            engine = sector_model["engine"]

            if ticker not in self._sig_cache:
                if price_df is None:
                    return {**default,
                            "sector": sector_model["name"],
                            "active": is_active}
                df_reset = price_df.copy().reset_index()
                if "Date" in df_reset.columns:
                    df_reset["date"] = df_reset["Date"].dt.strftime("%Y-%m-%d")
                elif "date" not in df_reset.columns:
                    df_reset["date"] = df_reset.index.strftime("%Y-%m-%d")

                sig_df = engine.get_full_signals(df_reset, ticker)

                if sig_df is None or sig_df.empty:
                    return {**default,
                            "sector": sector_model["name"],
                            "active": is_active}

                sig_df = sig_df.set_index("date") \
                         if "date" in sig_df.columns else sig_df
                self._sig_cache[ticker] = sig_df

            sig_df = self._sig_cache[ticker]

            # Exact match preferred; else fall back to most recent prior signal date.
            # Feature-builder dropna() can drop the latest bar, so the newest
            # available signal may predate current_date. Use it rather than missing.
            if current_date in sig_df.index:
                lookup_date = current_date
            else:
                _prior = [d for d in sig_df.index if d <= current_date]
                if not _prior:
                    return {**default,
                            "sector": sector_model["name"],
                            "active": is_active}
                lookup_date = max(_prior)

            proba_buy = float(sig_df.loc[lookup_date, "proba_buy"])

            if np.isnan(proba_buy):
                return {**default,
                        "sector": sector_model["name"],
                        "active": is_active}

            signal = 1 if proba_buy >= threshold else 0

            result = {
                "proba_buy": round(proba_buy, 4),
                "signal":    signal,
                "threshold": threshold,
                "sector":    sector_model["name"],
                "active":    is_active,
                "regime":    regime,
            }

            self._cache[cache_key] = result
            return result

        except Exception as e:
            print(f"  ⚠️  {ticker} signal failed: {e}")
            return {**default,
                    "sector": sector_model["name"],
                    "active": is_active}

    def get_signals_batch(self,
                          tickers:      List[str],
                          price_data:   Dict[str, pd.DataFrame],
                          current_date: str,
                          regime:       str = "Bull-Stable") -> Dict[str, dict]:
        """
        Run inference for multiple tickers on a given date.
        Returns dict of ticker → signal dict.
        """
        results = {}
        for ticker in tickers:
            if ticker not in price_data:
                continue
            results[ticker] = self.get_signals(
                ticker       = ticker,
                price_df     = price_data[ticker],
                current_date = current_date,
                regime       = regime,
            )
        return results

    def precompute_signals(self, price_data: Dict[str, "pd.DataFrame"]):
        """
        Pre-warm _sig_cache for all sector tickers using price_data dict.
        Call once after load() so get_signals() never needs price_df at bar time.
        """
        print("\n🔄 Precomputing sector signals...")
        for sector_key, sector_model in self.models.items():
            engine = sector_model["engine"]
            for ticker in sector_model["tickers"]:
                if ticker in self._sig_cache:
                    continue
                price_df = price_data.get(ticker)
                if price_df is None:
                    print(f"  ⚠️  {ticker}: no price data — skipping")
                    continue
                try:
                    df_reset = price_df.copy().reset_index()
                    if "Date" in df_reset.columns:
                        df_reset["date"] = df_reset["Date"].dt.strftime("%Y-%m-%d")
                    elif "date" not in df_reset.columns:
                        df_reset["date"] = df_reset.index.strftime("%Y-%m-%d")

                    sig_df = engine.get_full_signals(df_reset, ticker)
                    if sig_df is None or sig_df.empty:
                        print(f"  ⚠️  {ticker}: empty sig_df")
                        continue

                    sig_df = sig_df.set_index("date") \
                             if "date" in sig_df.columns else sig_df
                    self._sig_cache[ticker] = sig_df
                    print(f"  ✅ {ticker}: {len(sig_df)} rows cached")
                except Exception as e:
                    print(f"  ❌ {ticker}: precompute failed — {e}")

        print(f"  Sector sig_cache: {len(self._sig_cache)} tickers ready")

    def clear_cache(self):
        """Clear inference cache. Call at start of each backtest run."""
        self._cache.clear()
        self._sig_cache.clear()

    # ── diagnostics ────────────────────────────────────────────────────

    def print_registry(self):
        """Print sector registry and regime routing table."""
        print("\n" + "═" * 55)
        print("  SECTOR SIGNAL ENGINE — REGISTRY")
        print("═" * 55)
        for sector_key, cfg in SECTOR_REGISTRY.items():
            loaded = "✅" if sector_key in self.models else "❌"
            print(f"\n  {loaded} {cfg['name']}")
            print(f"     Tickers  : {cfg['tickers']}")
            print(f"     Threshold: {cfg['native_threshold']}")
            print(f"     Regimes  : {cfg['deploy_regimes']}")
            print(f"     PKL      : {cfg['pkl']}")

        print(f"\n  Regime → Active sectors:")
        for regime, sectors in REGIME_TO_SECTORS.items():
            names = [SECTOR_REGISTRY[s]["name"] for s in sectors]
            print(f"    {regime:<20} → {names}")
        print("═" * 55)

    def get_all_tickers(self) -> List[str]:
        """Return all tickers across all loaded sector models."""
        tickers = []
        for cfg in SECTOR_REGISTRY.values():
            tickers.extend(cfg["tickers"])
        return list(set(tickers))


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT — smoke test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  SECTOR SIGNAL ENGINE — SMOKE TEST")
    print("=" * 55)

    engine = SectorSignalEngine()
    engine.load()
    engine.print_registry()

    # test ticker routing
    print("\n  Ticker routing test:")
    test_cases = [
        ("NVDA",  "Bull-Trending"),
        ("MSFT",  "Bull-Stable"),
        ("TSLA",  "Bull-Trending"),
        ("XOM",   "Bear-Stress"),
        ("GLD",   "Bear-Stable"),
        ("SPY",   "Bull-Trending"),  # not in any sector
    ]
    for ticker, regime in test_cases:
        sector  = TICKER_TO_SECTOR.get(ticker, "none")
        active  = engine.is_ticker_active(ticker, regime)
        flag    = "✅" if active else "⚠️ "
        print(f"  {flag} {ticker:<6} | regime={regime:<15} | "
              f"sector={sector:<12} | active={active}")

    print("\n✅ Smoke test complete")
