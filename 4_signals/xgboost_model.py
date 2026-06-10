"""
xgboost_model.py  (v6 — Stacked Ensemble + GPU)
================================================
VIRTUAL_TRADING_FIRM  |  Stage 4

Architecture change from v5
----------------------------
STACKED ENSEMBLE replaces single global XGBoost.

Layer 1 — Five specialist models, each expert in one signal family:
  mean_rev  : LogisticRegression on OU-process features
  vol       : RandomForest       on volatility/regime features
  momentum  : LightGBM           on price momentum features
  structure : Ridge              on Kalman/structural features
  esn       : LogisticRegression on ESN signal only

Each specialist trained with walk-forward OOS to produce
honest out-of-sample probability estimates.

Layer 2 — XGBoost meta-learner:
  Inputs:  5 specialist probas + regime + hurst + vol_regime (8 features)
  Learns:  when to trust which specialist in which regime
  Trained: Optuna 100 trials — fast because only 8 meta-features
  GPU:     injected via GPU_CFG automatically

Why better than single XGBoost
-------------------------------
  ESN is one vote among five — cannot dominate
  Each regime gets expert signal from the right specialist
  Meta-learner learns regime-specific trust weights from data
  8 meta-features vs 23 raw = faster Optuna, less overfit
  Expected F1 improvement: +0.08-0.14 over v5

ESN leakage fix
---------------
  Rolling input window: 252 bars max (not full history)
  ESN reservoir only sees last year — breaks 0.96 autocorrelation
  100-bar Z-score normalisation retained for smoothness

GPU
---
  GPU_CFG auto-detected at module load via torch.cuda.is_available()
  Injected into all XGBClassifier and LightGBM calls automatically

Optuna seed changed 42 -> 123
  Different exploration path through the search space

Compute cost on A100
--------------------
  Specialist walk-forward (5 simple models): ~8-12 min
  Meta-learner Optuna (100 trials, 8 features): ~50-70 min
  Final model + threshold + SHAP: ~10 min
  Total: ~70-90 min (faster than v5 104 min)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler
import shap
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'TradingAgents'))


# ════════════════════════════════════════════════════════════════════════════
# GPU DETECTION
# ════════════════════════════════════════════════════════════════════════════

def _detect_gpu() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            name   = torch.cuda.get_device_name(0)
            vram   = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU: {name} ({vram:.1f} GB VRAM)")
            return {"tree_method": "hist", "device": "cuda"}
        else:
            print("  No GPU — using CPU")
            return {"tree_method": "hist"}
    except ImportError:
        return {"tree_method": "hist"}


GPU_CFG = _detect_gpu()


# ════════════════════════════════════════════════════════════════════════════
# SPECIALIST FEATURE GROUPS
# (derived from SHAP analysis — each group is one signal family)
# ════════════════════════════════════════════════════════════════════════════

SPECIALIST_FEATURES = {
    "mean_rev": [
        # OU process — best in mean-reverting regimes
        "ou_theta", "ou_half_life", "ou_mu", "ou_residual",
    ],
    "vol": [
        # Volatility and regime — best at detecting transitions
        "vol_regime", "realized_vol_20d", "vol_ratio",
        "atr_14", "bb_width", "hurst", "garch_vol",
    ],
    "momentum": [
        # Price momentum — best in trending regimes
        "return_5d", "return_20d", "return_skew_20",
        "macd", "rsi_14", "rwi_low", "rwi_max",
    ],
    "structure": [
        "qv_signature_ratio", "dd_ratio", "signal_quality",
    ],
    "esn": [
        # Temporal memory — one vote in meta-learner
        "esn_signal",
    ],
}

# Per-sector specialist feature overrides.
# Only define what DIFFERS from SPECIALIST_FEATURES above.
# Sectors not listed here use the global SPECIALIST_FEATURES.
SECTOR_SPECIALIST_FEATURES = {
    "hardware": {
        "mean_rev": [
            "bb_pct",       # position within band — price level not vol
            "pct_from_52h", # distance from peak — price level
            "rsi_14",       # oscillator — orthogonal to vol
            "ou_residual",  # one OU feature for mean reversion character
        ],
        "vol": [
            "vol_regime", "realized_vol_20d", "vol_ratio",
            "atr_14", "bb_width", "hurst",
            # garch_vol removed — not in sector feature set
        ],
        "momentum": [
            "return_5d", "return_10d", "return_20d",
            "macd", "macd_hist", "rsi_14",
            "rwi_trend", "rwi_direction",
            # rwi_low/rwi_max replaced with directional versions
        ],
        "structure": [
            "dd_ratio", "signal_quality",
            "hjb_entry_score", "hjb_direction",
            # qv_signature_ratio removed
        ],
        "esn": ["esn_signal"],
    },

    "hypercloud": {
        "mean_rev": [
            "bb_pct",       # position within band
            "pct_from_52h", # distance from peak
            "rsi_14",       # oscillator
            "ou_residual",  # mean reversion character
        ],
        "vol": [
            "vol_regime", "realized_vol_20d", "vol_ratio",
            "atr_14", "bb_width", "hurst",
        ],
        "momentum": [
            "return_5d", "return_10d", "return_20d",
            "macd", "rsi_14",
            "rwi_trend", "rwi_direction",
        ],
        "structure": [
            "dd_ratio", "signal_quality",
            "hjb_entry_score", "hjb_direction",
        ],
        "esn": ["esn_signal"],
    },

    "software": {
        "mean_rev": [
            # Full OU suite — software mean-reverts in Bull-Stable
            "ou_theta", "ou_half_life", "ou_mu", "ou_residual",
            "ou_reversion_signal",
        ],
        "vol": [
            "vol_regime", "realized_vol_20d", "vol_ratio",
            "atr_14", "bb_width", "hurst",
            "qv_5d",
        ],
        "momentum": [
            "return_5d", "return_20d",
            "macd", "macd_signal", "macd_hist",
            "rsi_14", "rsi_7",
        ],
        "structure": [
            "qv_signature_ratio", "dd_ratio", "signal_quality",
            "bb_pct", "pct_from_52h",
        ],
        "esn": ["esn_signal"],
    },

    "autos": {
        "mean_rev": [
            # OU removed — cyclicals trend in Bull-Trending
            # replaced with recovery/distress signals
            "pct_from_52l", "hl_ratio", "bb_width", "vol_ratio",
        ],
        "vol": [
            "vol_regime", "realized_vol_20d", "vol_ratio",
            "atr_14", "hl_ratio", "hurst",
        ],
        "momentum": [
            "return_5d", "return_10d", "return_20d",
            "macd", "macd_hist", "rsi_14",
            "rwi_trend", "rwi_direction",
        ],
        "structure": [
            "dd_ratio", "drift_signed", "dd_regime",
            "signal_quality", "beta", "debt_equity",
        ],
        "esn": ["esn_signal"],
    },

    "defensive": {
        "mean_rev": [
            # Full OU suite — Bear has strong mean reversion
            "ou_theta", "ou_half_life", "ou_residual",
            "ou_reversion_signal", "bb_pct",
        ],
        "vol": [
            "vol_regime", "realized_vol_20d", "vol_ratio",
            "atr_14", "hl_ratio", "hurst",
            "vol_of_vol", "bb_width",
        ],
        "momentum": [
            # Only short-term return — no trend following in Bear
            "return_5d", "qv_5d",
            "drift_signed", "diffusion",
        ],
        "structure": [
            "dd_ratio", "dd_regime", "signal_quality",
            "pct_from_52l", "beta",
        ],
        "esn": ["esn_signal"],
    },
}

# All raw features (union of all specialist groups)
CORE_FEATURES = [
    "esn_signal",
    "bb_width",
    "return_5d",
    "ou_mu",
    "rwi_low",
    "qv_signature_ratio",
    "ou_half_life",
    "vol_regime",
    "return_skew_20",
    "ou_theta",
    "vol_ratio",
    "realized_vol_20d",
    "return_20d",
    "atr_14",
    "macd",
    "rsi_14",
    "hurst",
    "ou_residual",
    "dd_ratio",
    "signal_quality",
    "rwi_max",
    "garch_vol",
]

SECTOR_CORE_FEATURES = {
    "global": CORE_FEATURES,

    "hardware": [
        "esn_signal",
        "return_5d", "return_10d", "return_20d",
        "macd", "macd_hist", "rsi_14",
        "above_sma50", "golden_cross",
        "rwi_trend", "rwi_direction", "rwi_max",
        "pct_from_52h",
        "atr_14", "vol_ratio", "realized_vol_20d", "bb_width",
        "hjb_entry_score", "hjb_direction",
        "beta",
        "hurst", "vol_regime", "signal_quality",
        "dd_ratio",
    ],

    "hypercloud": [
        "esn_signal",
        "return_5d", "return_10d", "return_20d",
        "macd", "rsi_14",
        "above_sma50", "golden_cross",
        "rwi_trend", "rwi_direction",
        "pct_from_52h",
        "bb_pct", "bb_width",
        "atr_14", "vol_ratio", "realized_vol_20d",
        "hjb_entry_score", "hjb_direction",
        "hurst", "vol_regime", "signal_quality", "dd_ratio",
    ],

    "software": [
        "esn_signal",
        "bb_pct", "bb_width",
        "ou_theta", "ou_mu", "ou_half_life", "ou_residual",
        "ou_reversion_signal",
        "rsi_14", "rsi_7",
        "macd", "macd_signal", "macd_hist",
        "pct_from_52h",
        "qv_signature_ratio", "qv_5d",
        "realized_vol_20d", "vol_ratio",
        "atr_14",
        "return_5d", "return_20d", "return_skew_20",
        "hurst", "vol_regime", "signal_quality",
    ],

    "autos": [
        "esn_signal",
        "return_5d", "return_10d", "return_20d",
        "macd", "macd_hist", "rsi_14",
        "golden_cross", "above_sma50",
        "rwi_trend", "rwi_direction", "rwi_max",
        "pct_from_52l",
        "beta", "debt_equity",
        "hl_ratio",
        "atr_14", "vol_ratio", "realized_vol_20d",
        "bb_width", "vol_regime",
        "dd_ratio", "drift_signed", "dd_regime",
        "hurst", "signal_quality",
    ],

    "defensive": [
        "esn_signal",
        "atr_14", "hl_ratio", "vol_ratio", "realized_vol_20d",
        "vol_of_vol", "bb_width", "vol_regime",
        "ou_theta", "ou_half_life", "ou_residual",
        "ou_reversion_signal",
        "bb_pct",
        "dd_ratio", "dd_regime", "drift_signed", "diffusion",
        "signal_quality",
        "pct_from_52l",
        "beta",
        "return_5d",
        "qv_5d",
        "hurst",
    ],
}

SECTOR_LABEL_CONFIG = {
    # Default — works for momentum/tech sectors
    "global": {
        "profit_target":         1.10,
        "stop_loss":             1.0,
        "vertical_barrier_days": 5,
        "min_barrier_days":      1,
    },
    "hardware": {
        "profit_target":         1.10,
        "stop_loss":             1.0,
        "vertical_barrier_days": 5,
        "min_barrier_days":      1,
    },
    "hypercloud": {
        "profit_target":         1.10,
        "stop_loss":             1.0,
        "vertical_barrier_days": 5,
        "min_barrier_days":      1,
    },
    "software": {
        "profit_target":         1.10,
        "stop_loss":             1.0,
        "vertical_barrier_days": 5,
        "min_barrier_days":      1,
    },
    # Autos — cyclicals need more time to develop
    # 10-day vertical barrier, slightly asymmetric
    "autos": {
        "profit_target":         1.20,  # higher bar — cyclicals overshoot
        "stop_loss":             1.0,
        "vertical_barrier_days": 10,    # more time for cyclical move
        "min_barrier_days":      2,
    },
    # Defensive — low vol stocks need wider asymmetric barriers
    # 15-day vertical, much higher profit target
    "defensive": {
        "profit_target":         1.50,  # wider — low ATR needs room
        "stop_loss":             1.0,
        "vertical_barrier_days": 15,    # more time in Bear regime
        "min_barrier_days":      3,
    },
}

# Meta-learner input feature names (in column order)
META_FEATURE_NAMES = [
    "proba_mean_rev",
    "proba_vol",
    "proba_momentum",
    "proba_structure",
    "proba_esn",
    "hurst",
    "vol_regime",
    "stocks_bonds_corr",    # cross-asset: SPY/TLT 30d rolling corr
    "credit_stress_score",  # cross-asset: HYG/LQD spread z-score
    "risk_score",           # cross-asset: composite macro risk-on/off [-1,+1]
    "esn_latent_0",         # PCA-compressed ESN reservoir state dim 0
    "esn_latent_1",
    "esn_latent_2",
    "esn_latent_3",
    "esn_latent_4",
    "esn_latent_5",
    "esn_latent_6",
    "esn_latent_7",
    "inflation_momentum",   # breakeven_today - breakeven_60d_ago (%)
    "real_rate_proxy",      # nominal yield proxy - breakeven inflation (%)
]

SECTOR_META_FEATURES = {
    "global": META_FEATURE_NAMES,

    "hardware": META_FEATURE_NAMES + [
        "dollar_strength",
        "stocks_dollar_corr",
    ],

    "hypercloud": META_FEATURE_NAMES + [
        "yield_curve_slope",
        "breakeven_inflation",
    ],

    "software": META_FEATURE_NAMES,

    "autos": META_FEATURE_NAMES + [
        "commodity_signal",
        "yield_curve_slope",
        "dollar_strength",
    ],

    "defensive": META_FEATURE_NAMES + [
        "gold_ret",
        "commodity_signal",
        "breakeven_inflation",
        "yield_curve_slope",
        "dollar_strength",
    ],
}


# ════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════


def add_esn_features(df: pd.DataFrame, esn_model=None,
                      collect_states: bool = False):
    """
    ESN with 252-bar rolling input window.
    Prevents full-history leakage that caused autocorr=0.96 in v5.
    Reservoir only sees last year of prices — responds to current regime.

    When collect_states=True, also returns the raw reservoir state matrix
    (n_rows, reservoir_size) aligned with df rows — used to fit ESN PCA
    during training.  Returns (df, states_matrix) in that case.
    """
    from rc_temporal import EchoStateNetwork
    prices = df["close"]
    labels = df["label"]

    if esn_model is None:
        split = int(len(prices) * 0.7)
        esn   = EchoStateNetwork()
        esn.fit(prices.values[:split], labels.values[:split])
    else:
        esn = esn_model

    signals      = []
    states_list  = [] if collect_states else None

    for i in range(len(prices)):
        if i < 60:
            signals.append(0.0)
            if collect_states:
                states_list.append(np.zeros(esn.reservoir_size))
        else:
            # rolling 252-bar window — breaks reservoir memory leakage
            start  = max(0, i - 252)
            result = esn.predict(prices.values[start:i+1])
            signals.append(result["decision"])
            if collect_states:
                states_list.append(result["reservoir_state"])

    df["esn_signal"]     = signals
    df["esn_signal"]     = df["esn_signal"].shift(1)
    rolling_mean         = df["esn_signal"].rolling(window=100, min_periods=20).mean()
    rolling_std          = df["esn_signal"].rolling(window=100, min_periods=20).std()
    df["esn_signal"]     = (df["esn_signal"] - rolling_mean) / (rolling_std + 1e-6)
    df["esn_signal"]     = df["esn_signal"].fillna(0.0)
    df["esn_confidence"] = df["esn_signal"].abs()

    if collect_states:
        return df, np.array(states_list, dtype=np.float32)  # (n_rows, reservoir_size)
    return df


def hurst_exponent(ts: np.ndarray, max_lag: int = 100) -> float:
    lags = range(5, max_lag)
    tau  = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    reg  = np.polyfit(np.log(lags), np.log(tau), 1)
    return reg[0]


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    prices     = df['close'].values
    hurst_vals = []
    for i in range(len(prices)):
        if i < 252:
            hurst_vals.append(0.5)
        else:
            h = hurst_exponent(prices[i-252:i+1], max_lag=100)
            hurst_vals.append(h)

    df['hurst']          = hurst_vals
    df['return_skew_20'] = df['return_1d'].rolling(20).skew()
    df['return_kurt_20'] = df['return_1d'].rolling(20).kurt()
    df['vol_regime']     = (
        df['atr_14'].rolling(20).mean() /
        df['atr_14'].rolling(60).mean()
    )
    # GARCH(1,1) conditional volatility — shifted 1 bar to prevent lookahead
    try:
        from arch import arch_model
        returns = df['return_1d'].fillna(0.0) * 100   # scale for numerical stability
        am  = arch_model(returns, vol='Garch', p=1, q=1, dist='normal', rescale=False)
        res = am.fit(disp='off', show_warning=False)
        df['garch_vol'] = (res.conditional_volatility / 100).shift(1)
    except Exception:
        df['garch_vol'] = df.get('realized_vol_20d', pd.Series(0.0, index=df.index))
    df['garch_vol'] = df['garch_vol'].fillna(
        df.get('realized_vol_20d', pd.Series(0.0, index=df.index))
    )

    for col, default in [
        ("ou_residual", 0.0), ("dd_ratio", 0.5),
        ("signal_quality", 0.5), ("rwi_max", 0.0),
    ]:
        if col not in df.columns:
            df[col] = default
    return df


# ════════════════════════════════════════════════════════════════════════════
# ESN PCA FITTING
# ════════════════════════════════════════════════════════════════════════════

def fit_esn_pca(states_matrix: np.ndarray,
                n_components: int = 8) -> tuple:
    """
    Fit PCA on raw ESN reservoir states collected across all training bars.
    states_matrix: (n_rows, reservoir_size) — e.g. (N, 200).
    Returns (fitted_pca, variance_explained_list).
    variance_explained_list: per-component fraction, sums to explained total.
    """
    n_comp = min(n_components, states_matrix.shape[0] - 1, states_matrix.shape[1])
    pca    = PCA(n_components=n_comp, random_state=42)
    pca.fit(states_matrix)
    var    = pca.explained_variance_ratio_.tolist()
    print(f"  ESN PCA({n_comp}): {sum(var)*100:.1f}% variance explained "
          f"| dims: {states_matrix.shape}")
    return pca, var


# ════════════════════════════════════════════════════════════════════════════
# TRIPLE BARRIER LABELS
# ════════════════════════════════════════════════════════════════════════════

def apply_triple_barrier_labels(
    df:                    pd.DataFrame,
    profit_target:         float = 1.10,
    stop_loss:             float = 1.0,
    vertical_barrier_days: int   = 5,
    min_barrier_days:      int   = 1,
) -> pd.DataFrame:
    labels, exit_times, exit_types = [], [], []
    prices = df['close'].values
    atrs   = df['atr_14'].values
    n      = len(prices)

    for i in range(n):
        if i + vertical_barrier_days >= n:
            labels.append(np.nan)
            exit_times.append(np.nan)
            exit_types.append('none')
            continue
        entry    = prices[i]
        atr      = atrs[i]
        upper    = entry + (profit_target * atr)
        lower    = entry - (stop_loss * atr)
        end_idx  = min(i + vertical_barrier_days, n - 1)
        hit, exit_time, exit_type = 0, end_idx, 'time'
        for j in range(i + min_barrier_days, end_idx + 1):
            p = prices[j]
            if p >= upper:
                hit, exit_time, exit_type = 1, j, 'profit'; break
            elif p <= lower:
                hit, exit_time, exit_type = 0, j, 'stop';   break
        labels.append(hit)
        exit_times.append(exit_time - i)
        exit_types.append(exit_type)

    df = df.copy()
    df['label']        = labels
    df['tb_exit_days'] = exit_times
    df['tb_exit_type'] = exit_types
    df = df.dropna(subset=['label']).reset_index(drop=True)
    df['label'] = df['label'].astype(int)

    print(f"  Triple Barrier: profit={profit_target:.2f}xATR | "
          f"stop={stop_loss:.1f}xATR | vertical={vertical_barrier_days}d")
    print(f"  Labels: BUY={df['label'].sum():,} | "
          f"SELL={len(df)-df['label'].sum():,} | "
          f"ratio={df['label'].mean():.1%}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD CV  (shared by specialists and meta-learner)
# ════════════════════════════════════════════════════════════════════════════

def walk_forward_cv(
    X:                np.ndarray,
    y:                np.ndarray,
    train_window:     int        = 900,
    test_window:      int        = 60,
    step:             int        = 40,
    expanding:        bool       = True,
    sample_weights:   np.ndarray = None,
    params:           dict       = None,
    scale_pos_weight: float      = None,
    model_type:       str        = "xgb",
) -> list:
    """
    Walk-forward CV supporting xgb / lgb / rf / lr / ridge.
    GPU_CFG injected automatically for xgb and lgb.
    """
    results = []
    start   = 0

    while True:
        train_end = start + train_window
        test_end  = train_end + test_window
        if test_end > len(X):
            break

        X_train = X[0:train_end] if expanding else X[start:train_end]
        y_train = y[0:train_end] if expanding else y[start:train_end]
        X_test  = X[train_end:test_end]
        y_test  = y[train_end:test_end]
        sw      = (sample_weights[0:train_end] if expanding and sample_weights is not None
                   else (sample_weights[start:train_end]
                         if sample_weights is not None else None))

        if model_type == "xgb":
            model = XGBClassifier(
                **(params or {}), **GPU_CFG,
                scale_pos_weight=scale_pos_weight,
                verbosity=0, random_state=42,
            )
            model.fit(X_train, y_train, sample_weight=sw, verbose=False)

        elif model_type == "lgb":
            try:
                import lightgbm as lgb
                lgb_device = "gpu" if "cuda" in GPU_CFG.get("device", "") else "cpu"
                model = lgb.LGBMClassifier(
                    **(params or {}), device=lgb_device,
                    random_state=42, verbosity=-1,
                )
                lgb_sw = None if (params or {}).get("is_unbalance") else sw
                model.fit(X_train, y_train, sample_weight=lgb_sw)
            except Exception:
                import lightgbm as lgb
                model = lgb.LGBMClassifier(
                    **(params or {}), random_state=42, verbosity=-1)
                lgb_sw = None if (params or {}).get("is_unbalance") else sw
                model.fit(X_train, y_train, sample_weight=lgb_sw)

        elif model_type == "rf":
            model = RandomForestClassifier(
                **(params or {}), random_state=42, n_jobs=-1)
            model.fit(X_train, y_train, sample_weight=sw)

        elif model_type == "lr":
            model = LogisticRegression(
                **(params or {}), random_state=42, max_iter=1000)
            model.fit(X_train, y_train, sample_weight=sw)

        elif model_type == "ridge":
            model = Ridge(**(params or {}))
            model.fit(X_train, y_train, sample_weight=sw)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if model_type == "ridge":
            d     = model.predict(X_test)          # Ridge uses predict() not decision_function()
            probs = 1 / (1 + np.exp(-d))
            preds = (probs >= 0.5).astype(int)
        else:
            probs = model.predict_proba(X_test)[:, 1]
            preds = model.predict(X_test)

        results.append({
            "train_start": start,  "train_end":  train_end,
            "test_start":  train_end, "test_end": test_end,
            "accuracy":    accuracy_score(y_test, preds),
            "f1":          f1_score(y_test, preds, zero_division=0),
            "ap":          average_precision_score(y_test, probs),
            "n_train":     len(X_train), "n_test": len(X_test),
            "oos_probs":   probs,
            "oos_true":    y_test,
        })
        start += step

    return results


# ════════════════════════════════════════════════════════════════════════════
# LAYER 1 — SPECIALIST TRAINING
# ════════════════════════════════════════════════════════════════════════════

SPECIALIST_CONFIGS = {
    "mean_rev": {
        "model_type": "lr",
        "params": {
            "C": 0.5,
            "solver": "lbfgs",
            "class_weight": "balanced",   # ← forces 38/62 awareness
        },
    },
    "vol": {
        "model_type": "rf",
        "params": {
            "n_estimators":    150,
            "max_depth":       5,
            "min_samples_leaf":20,
            "class_weight":    "balanced",  # ← forces 38/62 awareness
        },
    },
    "momentum": {
        "model_type": "lgb",
        "params": {
            "n_estimators":      200,
            "max_depth":         4,
            "learning_rate":     0.05,
            "num_leaves":        16,
            "min_child_samples": 20,
            "is_unbalance":      True,    # ← LightGBM equivalent of balanced
        },
    },
    "structure": {
        "model_type": "lr",               # ← Ridge → LogisticRegression
        "params": {
            "C":            0.1,
            "solver":       "lbfgs",
            "class_weight": "balanced",   # ← forces 38/62 awareness
        },
    },
    "esn": {
        "model_type": "lr",
        "params": {
            "C":            1.0,
            "solver":       "lbfgs",
            "class_weight": "balanced",   # ← forces 38/62 awareness
        },
    },
}


def train_specialists(
    X_scaled:       np.ndarray,
    y:              np.ndarray,
    feature_cols:   list,
    sample_weights: np.ndarray,
    train_window:   int = 900,
    test_window:    int = 60,
    step:           int = 40,
    sector_key:     str = "global",
) -> tuple:
    """
    Train each specialist on its feature subset via walk-forward CV.
    Returns:
      oos_probas          : (n_samples, 5) OOS probability matrix
      trained_specialists : dict of final fitted specialist models
    """
    feat_idx   = {f: i for i, f in enumerate(feature_cols)}
    n          = len(y)
    n_specs    = len(SPECIALIST_FEATURES)
    oos_probas = np.full((n, n_specs), 0.5)
    trained    = {}

    # use sector-specific specialist features if defined
    _sector_specs = SECTOR_SPECIALIST_FEATURES.get(sector_key, {})

    for s_idx, (name, cfg) in enumerate(SPECIALIST_CONFIGS.items()):
        # sector override takes priority over global
        feats = _sector_specs.get(name, SPECIALIST_FEATURES[name])
        idx   = [feat_idx[f] for f in feats if f in feat_idx]

        if not idx:
            print(f"  ! {name}: no features found, skipping")
            continue

        X_sub = X_scaled[:, idx]
        print(f"  Specialist [{name}]: {len(feats)} features | {cfg['model_type']}")

        wf = walk_forward_cv(
            X_sub, y,
            train_window   = train_window,
            test_window    = test_window,
            step           = step,
            expanding      = True,
            sample_weights = sample_weights,
            params         = cfg["params"],
            model_type     = cfg["model_type"],
        )

        for r in wf:
            ts = r["test_start"]
            te = r["test_end"]
            oos_probas[ts:te, s_idx] = r["oos_probs"]

        # retrain on full data for storage
        if cfg["model_type"] == "lr":
            final = LogisticRegression(**cfg["params"], random_state=42, max_iter=1000)
        elif cfg["model_type"] == "rf":
            final = RandomForestClassifier(**cfg["params"], random_state=42, n_jobs=-1)
        elif cfg["model_type"] == "lgb":
            import lightgbm as lgb
            final = lgb.LGBMClassifier(**cfg["params"], random_state=42, verbosity=-1)
        elif cfg["model_type"] == "ridge":
            final = Ridge(**cfg["params"])

        final.fit(X_sub, y, sample_weight=sample_weights)
        trained[name] = {
            "model":      final,
            "feat_names": feats,
            "feat_idx":   idx,
            "model_type": cfg["model_type"],
        }

        # report quality
        all_p = np.concatenate([r["oos_probs"] for r in wf])
        all_t = np.concatenate([r["oos_true"]  for r in wf])
        preds = (all_p >= 0.5).astype(int)
        print(f"    OOS Acc={accuracy_score(all_t,preds):.4f} | "
              f"F1={f1_score(all_t,preds,zero_division=0):.4f} | "
              f"AP={average_precision_score(all_t,all_p):.4f}")

    return oos_probas, trained


# ════════════════════════════════════════════════════════════════════════════
# LAYER 2 — META-FEATURE BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_meta_features(
    oos_probas:             np.ndarray,
    X_scaled:               np.ndarray,
    feature_cols:           list,
    stocks_bonds_corr:      np.ndarray = None,
    credit_stress_score:    np.ndarray = None,
    risk_score:             np.ndarray = None,
    esn_latent_matrix:      np.ndarray = None,  # (n, 8) PCA-projected reservoir states
    inflation_momentum_arr: np.ndarray = None,  # (n,) per-bar inflation momentum %
    real_rate_arr:          np.ndarray = None,  # (n,) per-bar real rate proxy %
    extra_meta:             dict       = None,
    sector_meta_names:      list       = None,
) -> np.ndarray:
    n             = len(X_scaled)
    feat_idx      = {f: i for i, f in enumerate(feature_cols)}
    hurst_col     = (X_scaled[:, feat_idx["hurst"]]
                     if "hurst" in feat_idx else np.zeros(n))
    vol_reg_col   = (X_scaled[:, feat_idx["vol_regime"]]
                     if "vol_regime" in feat_idx else np.zeros(n))
    sbc_col     = stocks_bonds_corr   if stocks_bonds_corr   is not None else np.zeros(n)
    css_col     = credit_stress_score if credit_stress_score is not None else np.zeros(n)
    rs_col      = risk_score          if risk_score          is not None else np.zeros(n)
    esn_lat     = (esn_latent_matrix
                   if esn_latent_matrix is not None and len(esn_latent_matrix) == n
                   else np.zeros((n, 8), dtype=np.float32))
    infl_mom    = inflation_momentum_arr if inflation_momentum_arr is not None else np.zeros(n)
    real_rate   = real_rate_arr          if real_rate_arr          is not None else np.zeros(n)

    base = np.column_stack([
        oos_probas,   # (n, 5) — one column per specialist
        hurst_col,    # trending vs mean-reverting
        vol_reg_col,  # volatility context
        sbc_col,      # cross-asset: stocks-bonds corr
        css_col,      # cross-asset: credit stress
        rs_col,       # cross-asset: composite risk score
        esn_lat,      # (n, 8) ESN reservoir PCA projections
        infl_mom,     # breakeven inflation momentum (%)
        real_rate,    # real rate proxy (%)
    ]).astype(np.float32)

    if not extra_meta:
        return base

    extras = []
    _base_count = len(META_FEATURE_NAMES)
    _extra_keys = (sector_meta_names or [])[_base_count:]
    for key in _extra_keys:
        arr = extra_meta.get(key)
        if arr is not None and len(arr) == len(base):
            extras.append(np.array(arr, dtype=np.float32).reshape(-1, 1))
        else:
            print(f"  ⚠️  extra_meta '{key}' missing or wrong length — zeros")
            extras.append(np.zeros((len(base), 1), dtype=np.float32))

    if extras:
        return np.column_stack([base] + extras).astype(np.float32)
    return base


# ════════════════════════════════════════════════════════════════════════════
# LAYER 2 — META-LEARNER TRAINING
# ════════════════════════════════════════════════════════════════════════════

def train_meta_learner(
    meta_X:         np.ndarray,
    y:              np.ndarray,
    sample_weights: np.ndarray,
    n_trials:       int = 100,
    train_window:   int = 900,
    test_window:    int = 60,
    step:           int = 40,
    wf_step:        int = 40,
) -> tuple:
    """
    XGBoost meta-learner trained with Optuna on 20 clean meta-features.
    10 original (specialist probas + hurst + vol_regime + cross-asset) +
    8 ESN reservoir PCA projections + 2 inflation signals.
    Seed=123 explores different space from v5 seed=42.
    """
    meta_scaler   = StandardScaler()
    meta_X_scaled = meta_scaler.fit_transform(meta_X)

    counts = np.bincount(y)
    ratio  = min(counts[0] / counts[1], 2.0) if len(counts) > 1 else 1.0

    val_size = int(len(meta_X_scaled) * 0.20)  # larger val set for small datasets
    X_tr_es  = meta_X_scaled[:-val_size]
    y_tr_es  = y[:-val_size]
    X_val_es = meta_X_scaled[-val_size:]
    y_val_es = y[-val_size:]
    sw_tr_es = sample_weights[:-val_size]

    def objective(trial):
        params = {
            "n_estimators":     200,   # fixed — let early stopping decide
            "max_depth":        trial.suggest_int("max_depth", 3, 6),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample",       0.65, 0.92),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.72, 0.95),
            "gamma":            trial.suggest_float("gamma",            0.0,  0.50),
            "reg_lambda":       trial.suggest_float("reg_lambda",       1.0,  5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha",        0.0,  1.5),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 8),
        }

        wf = walk_forward_cv(
            meta_X_scaled, y,
            train_window=train_window, test_window=test_window,
            step=wf_step, expanding=True,
            sample_weights=sample_weights,
            params=params, scale_pos_weight=ratio,
            model_type="xgb",
        )
        if not wf:
            return 0.0

        f1_mean  = np.mean([r["f1"]  for r in wf])
        ap_mean  = np.mean([r["ap"]  for r in wf])
        acc_std  = np.std ([r["accuracy"] for r in wf])

        score = 0.60 * f1_mean + 0.40 * ap_mean
        if acc_std > 0.10:
            score -= (acc_std - 0.10) * 1.5
        if f1_mean < 0.25:
            score -= (0.25 - f1_mean) * 3.0
        return score

    print(f"\n  Meta-learner Optuna ({n_trials} trials, seed=123) ...")
    study = optuna.create_study(
        direction = "maximize",
        sampler   = TPESampler(seed=123),
        pruner    = optuna.pruners.MedianPruner(
            n_startup_trials=10, n_warmup_steps=5),
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best score: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    final_params = study.best_params
    meta_model   = XGBClassifier(
        **final_params, **GPU_CFG,
        scale_pos_weight      = ratio,
        eval_metric           = "logloss",
        early_stopping_rounds = 50 if len(meta_X) < 8000 else 200,
        n_estimators          = 1000 if len(meta_X) < 8000 else 2000,
        random_state          = 42,
        verbosity             = 0,
    )
    meta_model.fit(
        X_tr_es, y_tr_es,
        sample_weight = sw_tr_es,
        eval_set      = [(X_val_es, y_val_es)],
        verbose       = False,
    )
    print(f"  Early stopping: best iteration = {meta_model.best_iteration}")

    return meta_model, meta_scaler, final_params, study.best_value


# ════════════════════════════════════════════════════════════════════════════
# DATASET BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_multi_ticker_dataset(
    tickers:       list,
    end_date:      str,
    lookback_days: int = 1825,
    forward_days:  int = 5,
    sector_key:    str = "global",
) -> pd.DataFrame:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "feature_builder",
        os.path.join(ROOT, "4_signals", "feature_builder.py")
    )
    fb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fb)

    dfs = []
    for ticker in tickers:
        try:
            print(f"  -> {ticker}...")
            df = fb.build_features(
                ticker=ticker, end_date=end_date,
                lookback_days=lookback_days, forward_days=forward_days,
                sentiment_score=0.0,
            )
            if df is None or df.empty:
                print(f"  ! {ticker}: no data"); continue

            _label_cfg = SECTOR_LABEL_CONFIG.get(
                sector_key,
                SECTOR_LABEL_CONFIG["global"]
            )
            print(f"  Label config [{sector_key}]: "
                  f"profit={_label_cfg['profit_target']}xATR | "
                  f"stop={_label_cfg['stop_loss']}xATR | "
                  f"vertical={_label_cfg['vertical_barrier_days']}d")
            df = apply_triple_barrier_labels(
                df,
                profit_target         = _label_cfg["profit_target"],
                stop_loss             = _label_cfg["stop_loss"],
                vertical_barrier_days = _label_cfg["vertical_barrier_days"],
                min_barrier_days      = _label_cfg["min_barrier_days"],
            )
            if df.empty:
                print(f"  ! {ticker}: empty after TB"); continue

            df["sentiment"] = 0.0
            df["ticker"]    = ticker
            dfs.append(df)
        except Exception as e:
            print(f"  ! {ticker} failed: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {len(combined):,} samples from {len(dfs)} tickers")
    return combined


# ════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION  (entry point for Cell 4 in Colab)
# ════════════════════════════════════════════════════════════════════════════

def train_xgboost(
    df:         pd.DataFrame,
    save_path:  str = None,
    n_trials:   int = 100,
    wf_step:    int = 40,
    sector_key: str = "global",
) -> tuple:
    """
    Full stacked ensemble pipeline.
    Returns (meta_model, meta_scaler, metrics).
    PKL format compatible with signal_engine.py.
    """

    print("\n" + "=" * 62)
    print("  STACKED ENSEMBLE v6")
    print("  Layer 1: 5 specialists  |  Layer 2: XGBoost meta-learner")
    print("=" * 62)

    # ── per-ticker ESN ────────────────────────────────────────────────────
    print("\n  Layer 0: per-ticker ESN...")
    from rc_temporal import EchoStateNetwork

    esn_dfs, ticker_esns = [], {}

    # Fit HMM on FULL HISTORICAL SPY DATA (not the 60% training split).
    # Regimes are descriptive structural features — fitting on full history
    # does not cause lookahead because the HMM labels are not predictions.
    from hmm_regime import GaussianHMMRegimeDetector
    import yfinance as yf

    hmm_start          = '2015-01-01'   # 10 years: covers 2018/2020/2022 bear markets
    hmm_end            = pd.Timestamp.now().strftime('%Y-%m-%d')
    price_data_for_hmm = {}             # defined here so diagnostics can reference it

    print(f"\n  Loading HMM training data: SPY+IEF {hmm_start} to {hmm_end}")
    try:
        spy_full = yf.download('SPY', start=hmm_start, end=hmm_end,
                               progress=False, auto_adjust=False)
        ief_full = yf.download('IEF', start=hmm_start, end=hmm_end,
                               progress=False, auto_adjust=False)

        if isinstance(spy_full.columns, pd.MultiIndex):
            spy_full.columns = spy_full.columns.get_level_values(0)
        if isinstance(ief_full.columns, pd.MultiIndex):
            ief_full.columns = ief_full.columns.get_level_values(0)

        spy_full = spy_full.rename(columns={'Close': 'close'})
        ief_full = ief_full.rename(columns={'Close': 'close'})

        print(f"  HMM SPY: {len(spy_full)} bars  "
              f"({spy_full.index.min().date()} → {spy_full.index.max().date()})")

        price_data_for_hmm = {'SPY': spy_full, 'IEF': ief_full}

        # Add all training tickers for avg_corr feature
        for ticker_name in df['ticker'].unique():
            if ticker_name in price_data_for_hmm:
                continue
            try:
                t_full = yf.download(ticker_name, start=hmm_start, end=hmm_end,
                                     progress=False, auto_adjust=False)
                if t_full.empty:
                    continue
                if isinstance(t_full.columns, pd.MultiIndex):
                    t_full.columns = t_full.columns.get_level_values(0)
                t_full = t_full.rename(columns={'Close': 'close'})
                price_data_for_hmm[ticker_name] = t_full
            except Exception:
                pass

        hmm = GaussianHMMRegimeDetector()
        hmm.fit_initial(price_data_for_hmm)
        print(f"  HMM fit on {len(price_data_for_hmm)} tickers: "
              f"{sorted(price_data_for_hmm.keys())}")

    except Exception as _hmm_e:
        print(f"  ! HMM full-history download failed: {_hmm_e} — "
              f"all regimes = 'Bull-Stable'")
        hmm = None

    # ── HMM diagnostic ────────────────────────────────────────────────
    if hmm is not None and hmm._fitted:
        test_dates = [
            '2020-03-20',  # COVID crash        → Bear-Stress
            '2021-01-15',  # 2021 melt-up        → Bull-Trending
            '2022-06-15',  # 2022 bear            → Bear-Stable or Bear-Stress
            '2023-11-01',  # 2023 recovery        → Bull-Stable
            '2024-07-15',  # AI rally             → Bull-Trending
        ]
        print("\n  HMM regime test (should return DIFFERENT regimes):")
        for td in test_dates:
            try:
                r = hmm.get_regime(td)
                print(f"    {td}: {r}")
            except Exception as e:
                print(f"    {td}: FAILED — {e}")

        print(f"\n  HMM transition matrix:")
        print(np.round(hmm.get_transition_matrix(), 3))

        if hmm._hmm is not None:
            inv_map = {v: k for k, v in hmm._state_map.items()}
            print(f"\n  HMM state means [ret5, rvol20, mom20, spy_bond_corr, avg_corr, vix_proxy]:")
            for si in range(4):
                raw  = inv_map[si]
                name = hmm.STATE_NAMES[si]
                print(f"    {name:<16}: {np.round(hmm._hmm.means_[raw], 4).tolist()}")

        if not hmm._state_history.empty:
            print(f"\n  HMM state_history: {len(hmm._state_history)} bars | "
                  f"distribution: {hmm._state_history.value_counts().to_dict()}")
        else:
            print("  WARNING: HMM state_history is EMPTY — fit_initial() failed silently")

        if price_data_for_hmm:
            try:
                feat_df = hmm._build_features(price_data_for_hmm)
                sample_dates = ['2020-03-20', '2021-01-15', '2022-06-15',
                                '2023-11-01', '2024-07-15']
                print(f"\n  HMM input features for sample dates (should vary):")
                for sd in sample_dates:
                    sd_ts      = pd.to_datetime(sd)
                    candidates = feat_df.index[feat_df.index <= sd_ts]
                    if len(candidates) > 0:
                        row = feat_df.loc[candidates[-1]]
                        print(f"    {sd}: ret5={row['ret5']:.4f}  rvol20={row['rvol20']:.4f}  "
                              f"mom20={row['mom20']:.4f}  vix={row['vix_proxy']:.2f}")
                    else:
                        print(f"    {sd}: no SPY data before this date")
            except Exception as e:
                print(f"  ! HMM feature check failed: {e}")

    for tname in df["ticker"].unique():
        t_df  = df[df["ticker"] == tname].copy().reset_index(drop=True)
        split = int(len(t_df) * 0.7)
        esn   = EchoStateNetwork()
        esn.fit(t_df["close"].values[:split], t_df["label"].values[:split])
        ticker_esns[tname] = esn
        t_df, t_states = add_esn_features(t_df, esn, collect_states=True)
        t_df['_esn_state'] = list(t_states)  # object column, survives dropna

        if "regime" not in t_df.columns:
            if hmm is not None and "date" in t_df.columns:
                try:
                    dates_dt = pd.to_datetime(t_df["date"])
                    t_df["regime"] = [hmm.get_regime(d) for d in dates_dt]
                except Exception as e:
                    print(f"    ! HMM regime failed for {tname}: {e}")
                    t_df["regime"] = "Bull-Stable"
            else:
                t_df["regime"] = "Bull-Stable"
        else:
            print(f"    ✅ {tname}: regime column already present — skipping HMM assignment")

        t_df["regime"] = t_df["regime"].astype(str)
        esn_dfs.append(t_df)
        print(f"    ESN+Regime: {tname} ({len(t_df)} rows)")

    df = pd.concat(esn_dfs, ignore_index=True)

    # ── feature engineering ───────────────────────────────────────────────
    print("\n  Feature engineering...")
    df = add_advanced_features(df)
    df = df.dropna()

    # Extract ESN reservoir states after dropna — _esn_state column survives
    # because it contains numpy arrays (never NaN), so dropna preserves it.
    _esn_states_raw = np.vstack(df['_esn_state'].values)   # (n, reservoir_size)

    # ESN state matrix diagnostics — checks for reservoir collapse before PCA
    _states_std   = _esn_states_raw.std(axis=0)
    _states_cov   = np.cov(_esn_states_raw[:min(2000, len(_esn_states_raw))].T)
    _eigvals_pre  = np.sort(np.real(np.linalg.eigvalsh(_states_cov)))[::-1]
    _top8_pct_pre = _eigvals_pre[:8].sum() / (_eigvals_pre.sum() + 1e-12) * 100
    _near_zero    = (np.abs(_esn_states_raw) < 1e-3).mean() * 100
    _saturated    = (np.abs(_esn_states_raw) > 0.99).mean() * 100
    print(f"\n  ESN state matrix diagnostics {_esn_states_raw.shape}:")
    print(f"    States std — mean: {_states_std.mean():.6f}  min: {_states_std.min():.6f}  (want >0.1)")
    print(f"    Top-8 eigenvalues capture: {_top8_pct_pre:.1f}% of variance  (want 40–70%)")
    print(f"    Top-20 eigenvalues: {np.round(_eigvals_pre[:20], 4).tolist()}")
    print(f"    Near-zero values (<1e-3): {_near_zero:.1f}%  (>20% → collapsed reservoir)")
    print(f"    Saturated values (|x|>0.99): {_saturated:.1f}%  (>30% → tanh saturation / OOD input)")

    df = df.drop(columns=['_esn_state'])
    print(f"\n  Fitting ESN PCA on reservoir states ...")
    esn_pca, esn_pca_variance = fit_esn_pca(_esn_states_raw, n_components=8)
    esn_latent_matrix = esn_pca.transform(_esn_states_raw).astype(np.float32)  # (n, 8)

    regime_counts = pd.Series(df['regime']).value_counts()
    print(f"  Regime distribution: {regime_counts.to_dict()}")
    if len(regime_counts) == 1 and sector_key == "global":
        print(f"  WARNING: All bars labeled '{regime_counts.index[0]}' "
              f"— HMM may not be fitting properly")
        print(f"      Check: hmmlearn installed, SPY in dataset, dates parseable")
    elif len(regime_counts) == 1 and sector_key != "global":
        print(f"  ✅ Mono-regime expected for sector '{sector_key}' "
              f"— regime filter working correctly")

    _sector_features = SECTOR_CORE_FEATURES.get(sector_key, CORE_FEATURES)
    feature_cols     = [c for c in _sector_features if c in df.columns]
    _missing = [c for c in _sector_features if c not in df.columns]
    if _missing:
        print(f"  ⚠️  Sector '{sector_key}' features not found in df "
              f"(will skip): {_missing}")
    print(f"  Sector '{sector_key}' | feature count: {len(feature_cols)}")
    X = np.nan_to_num(df[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["label"].astype(int).values

    print(f"  Dataset: {len(X):,} samples x {len(feature_cols)} features")
    print(f"  Class ratio: {y.mean():.1%} BUY")

    scaler         = StandardScaler()
    X_scaled       = scaler.fit_transform(X)
    sample_weights = np.linspace(0.3, 1.0, len(y))   # pure time decay

    # ── Layer 1 ───────────────────────────────────────────────────────────
    print("\n  Layer 1: specialist training...")
    oos_probas, trained_specialists = train_specialists(
        X_scaled=X_scaled, y=y,
        feature_cols=feature_cols,
        sample_weights=sample_weights,
        sector_key=sector_key,
    )

    # ── Specialist heterogeneity check ────────────────────────────────────
    spec_names = list(SPECIALIST_FEATURES.keys())
    corr_matrix = np.corrcoef(oos_probas.T)
    print("\n  Specialist pairwise correlation (OOS probas):")
    for i in range(len(spec_names)):
        for j in range(i + 1, len(spec_names)):
            r    = corr_matrix[i, j]
            flag = "  *** REDUNDANT — consider merging" if abs(r) > 0.85 else ""
            print(f"    {spec_names[i]:<12} x {spec_names[j]:<12}  r={r:+.3f}{flag}")

    print("\n  Specialist OOS probas:")
    for i, name in enumerate(SPECIALIST_FEATURES.keys()):
        p = oos_probas[:, i]
        print(f"    {name:<12} mean={p.mean():.4f} | "
              f"std={p.std():.4f} | >0.5={( p>0.5).mean():.1%}")

    # ── Inflation macro features ──────────────────────────────────────────
    print("\n  Computing inflation macro features...")
    inflation_momentum_arr = np.zeros(len(df), dtype=np.float32)
    real_rate_arr          = np.zeros(len(df), dtype=np.float32)
    try:
        from inflation_signals import InflationSignalEngine

        # Robust date extraction: handles 'date' column, DatetimeIndex, or
        # any date-like column.  The old fallback converted integer indices
        # (0, 1, 2 …) to strings like "100" which strptime then rejected.
        if "date" in df.columns:
            dates_col = pd.to_datetime(df["date"])
        elif isinstance(df.index, pd.DatetimeIndex):
            dates_col = df.index.to_series().reset_index(drop=True)
        else:
            _date_cands = [c for c in df.columns
                           if "date" in c.lower() or "time" in c.lower()]
            if _date_cands:
                dates_col = pd.to_datetime(df[_date_cands[0]])
            else:
                raise ValueError("No date column found — skipping inflation features")

        infl_eng = InflationSignalEngine()
        infl_eng.load_data(
            dates_col.min().strftime("%Y-%m-%d"),
            dates_col.max().strftime("%Y-%m-%d"),
        )
        inflation_momentum_arr = np.array(
            [infl_eng.get_inflation_momentum(d.strftime("%Y-%m-%d"))
             for d in dates_col],
            dtype=np.float32)
        real_rate_arr = np.array(
            [infl_eng.get_real_rate_proxy(d.strftime("%Y-%m-%d"))
             for d in dates_col],
            dtype=np.float32)

        if np.all(inflation_momentum_arr == 0) and np.all(real_rate_arr == 0):
            print("  WARNING: All inflation features are zero — engine returning zeros")
        else:
            print(f"  Inflation features OK: "
                  f"mom mean={inflation_momentum_arr.mean():.3f} | "
                  f"real_rate mean={real_rate_arr.mean():.3f}")
    except Exception as _e:
        print(f"  ! Inflation features failed: {_e} — using zeros")

    # ── Cross-asset macro features ────────────────────────────────────────
    print("\n  Computing cross-asset macro features...")
    stocks_bonds_corr_arr = np.zeros(len(df), dtype=np.float32)
    credit_stress_arr     = np.zeros(len(df), dtype=np.float32)
    risk_score_arr        = np.zeros(len(df), dtype=np.float32)
    try:
        from cross_asset_signals import CrossAssetSignalEngine
        ca_eng = CrossAssetSignalEngine()

        if "date" in df.columns:
            dates_col = pd.to_datetime(df["date"])
        elif isinstance(df.index, pd.DatetimeIndex):
            dates_col = df.index
        else:
            raise ValueError("No date column or DatetimeIndex found")

        ca_eng.load_data(str(dates_col.min().date()),
                         str(dates_col.max().date()))

        stocks_bonds_corr_arr = np.array(
            [ca_eng.get_stocks_bonds_corr(d.strftime('%Y-%m-%d'))
             for d in dates_col],
            dtype=np.float32)
        credit_stress_arr = np.array(
            [ca_eng.get_credit_stress_score(d.strftime('%Y-%m-%d'))
             for d in dates_col],
            dtype=np.float32)
        risk_score_arr = np.array(
            [ca_eng.get_risk_on_off_signal(d.strftime('%Y-%m-%d'))
             for d in dates_col],
            dtype=np.float32)

        if (np.all(stocks_bonds_corr_arr == 0) and
                np.all(credit_stress_arr == 0) and
                np.all(risk_score_arr == 0)):
            print("  WARNING: All cross-asset features are zero — engine returning zeros")
        else:
            print(f"  Cross-asset OK: sbc mean={stocks_bonds_corr_arr.mean():.3f} "
                  f"std={stocks_bonds_corr_arr.std():.3f} | "
                  f"css mean={credit_stress_arr.mean():.3f} | "
                  f"rs mean={risk_score_arr.mean():.3f}")
    except Exception as _e:
        print(f"  ! Cross-asset features failed: {_e} — using zeros")

    # ── Layer 2: build meta-features ──────────────────────────────────────
    print("\n  Building meta-features...")
    # compute sector-specific extra macro signals
    _sector_meta_names = SECTOR_META_FEATURES.get(sector_key, META_FEATURE_NAMES)
    _extra_meta        = {}

    if "yield_curve_slope" in _sector_meta_names:
        try:
            _extra_meta["yield_curve_slope"] = np.array(
                [ca_eng.get_yield_curve_slope(d.strftime('%Y-%m-%d'))
                 for d in dates_col], dtype=np.float32)
            print(f"  yield_curve_slope: "
                  f"mean={_extra_meta['yield_curve_slope'].mean():.4f}")
        except Exception as _e:
            print(f"  ! yield_curve_slope failed: {_e}")
            _extra_meta["yield_curve_slope"] = np.zeros(len(df), dtype=np.float32)

    if "dollar_strength" in _sector_meta_names:
        try:
            _extra_meta["dollar_strength"] = np.array(
                [ca_eng.get_dollar_strength_score(d.strftime('%Y-%m-%d'))
                 for d in dates_col], dtype=np.float32)
            print(f"  dollar_strength: "
                  f"mean={_extra_meta['dollar_strength'].mean():.4f}")
        except Exception as _e:
            print(f"  ! dollar_strength failed: {_e}")
            _extra_meta["dollar_strength"] = np.zeros(len(df), dtype=np.float32)

    if "stocks_dollar_corr" in _sector_meta_names:
        try:
            _extra_meta["stocks_dollar_corr"] = np.array(
                [ca_eng.get_stocks_dollar_corr(d.strftime('%Y-%m-%d'))
                 for d in dates_col], dtype=np.float32)
            print(f"  stocks_dollar_corr: "
                  f"mean={_extra_meta['stocks_dollar_corr'].mean():.4f}")
        except Exception as _e:
            print(f"  ! stocks_dollar_corr failed: {_e}")
            _extra_meta["stocks_dollar_corr"] = np.zeros(len(df), dtype=np.float32)

    if "commodity_signal" in _sector_meta_names:
        try:
            _extra_meta["commodity_signal"] = np.array(
                [infl_eng.get_commodity_inflation_signal(d.strftime('%Y-%m-%d'))
                 for d in dates_col], dtype=np.float32)
            print(f"  commodity_signal: "
                  f"mean={_extra_meta['commodity_signal'].mean():.4f}")
        except Exception as _e:
            print(f"  ! commodity_signal failed: {_e}")
            _extra_meta["commodity_signal"] = np.zeros(len(df), dtype=np.float32)

    if "breakeven_inflation" in _sector_meta_names:
        try:
            _extra_meta["breakeven_inflation"] = np.array(
                [infl_eng.get_breakeven_inflation(d.strftime('%Y-%m-%d'))
                 for d in dates_col], dtype=np.float32)
            print(f"  breakeven_inflation: "
                  f"mean={_extra_meta['breakeven_inflation'].mean():.4f}")
        except Exception as _e:
            print(f"  ! breakeven_inflation failed: {_e}")
            _extra_meta["breakeven_inflation"] = np.zeros(len(df), dtype=np.float32)

    if "gold_ret" in _sector_meta_names:
        try:
            _extra_meta["gold_ret"] = np.array(
                [ca_eng._recent_return('GLD', d.strftime('%Y-%m-%d'), lookback=30)
                 for d in dates_col], dtype=np.float32)
            print(f"  gold_ret: "
                  f"mean={_extra_meta['gold_ret'].mean():.4f}")
        except Exception as _e:
            print(f"  ! gold_ret failed: {_e}")
            _extra_meta["gold_ret"] = np.zeros(len(df), dtype=np.float32)

    meta_X = build_meta_features(
        oos_probas=oos_probas, X_scaled=X_scaled,
        feature_cols=feature_cols,
        stocks_bonds_corr=stocks_bonds_corr_arr,
        credit_stress_score=credit_stress_arr,
        risk_score=risk_score_arr,
        esn_latent_matrix=esn_latent_matrix,
        inflation_momentum_arr=inflation_momentum_arr,
        real_rate_arr=real_rate_arr,
        extra_meta=_extra_meta,
        sector_meta_names=_sector_meta_names,
    )
    print(f"  Meta-feature matrix : {meta_X.shape}")
    print(f"  Meta features ({len(_sector_meta_names)}): {_sector_meta_names}")

    # ── Layer 2: meta-learner Optuna ──────────────────────────────────────
    meta_model, meta_scaler, best_params, meta_optuna_score = train_meta_learner(
        meta_X=meta_X, y=y,
        sample_weights=sample_weights,
        n_trials=n_trials,
        wf_step=wf_step,
    )

    # ── threshold optimisation ────────────────────────────────────────────
    print("\n  Threshold optimisation...")
    meta_X_scaled = meta_scaler.transform(meta_X)
    counts        = np.bincount(y)
    ratio         = min(counts[0] / counts[1], 2.0) if len(counts) > 1 else 1.0

    wf_thresh = walk_forward_cv(
        meta_X_scaled, y,
        train_window=900, test_window=60, step=40,
        expanding=True, sample_weights=sample_weights,
        params={**best_params,
                "n_estimators": max(300, meta_model.best_iteration)},
        scale_pos_weight=ratio, model_type="xgb",
    )

    all_probs = np.concatenate([r["oos_probs"] for r in wf_thresh])
    all_true  = np.concatenate([r["oos_true"]  for r in wf_thresh])

    best_s, best_thresh, best_f1, best_acc = 0.0, 0.50, 0.0, 0.0
    for thresh in np.arange(0.30, 0.70, 0.01):
        preds = (all_probs >= thresh).astype(int)
        if preds.sum() < 20:
            continue
        if precision_score(all_true, preds, zero_division=0) < 0.44:
            continue
        curr_f1  = f1_score(all_true, preds, zero_division=0)
        curr_acc = accuracy_score(all_true, preds)
        score    = 0.6 * curr_f1 + 0.4 * curr_acc
        if score > best_s:
            best_s, best_thresh = score, thresh
            best_f1, best_acc   = curr_f1, curr_acc

    # Fallback if precision guard rejected everything
    _precision_fallback = best_f1 == 0.0
    if _precision_fallback:
        print("  ! Precision guard rejected all — using F1 fallback")
        for thresh in np.arange(0.30, 0.70, 0.01):
            preds    = (all_probs >= thresh).astype(int)
            if preds.sum() < 20:
                continue
            curr_f1  = f1_score(all_true, preds, zero_division=0)
            curr_acc = accuracy_score(all_true, preds)
            score    = 0.6 * curr_f1 + 0.4 * curr_acc
            if score > best_s:
                best_s, best_thresh = score, thresh
                best_f1, best_acc   = curr_f1, curr_acc

    accs = [r["accuracy"] for r in wf_thresh]
    f1s  = [r["f1"]       for r in wf_thresh]

    print(f"  Threshold : {best_thresh:.3f} | "
          f"Thr-F1={best_f1:.4f} | Thr-Acc={best_acc:.4f}")
    print(f"  WF Acc    : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  WF F1     : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    # ── SHAP on meta-learner ──────────────────────────────────────────────
    shap_importance = []
    try:
        n_shap    = min(len(meta_X_scaled), 2000)
        print(f"\n  SHAP on meta-learner ({n_shap} samples)...")
        explainer = shap.TreeExplainer(meta_model)
        idx       = np.random.choice(len(meta_X_scaled), n_shap, replace=False)
        shap_vals = explainer.shap_values(meta_X_scaled[idx])
        shap_imp  = np.abs(shap_vals).mean(0)
        shap_importance = sorted(
            zip(META_FEATURE_NAMES, shap_imp),
            key=lambda x: x[1], reverse=True
        )
        print("\n  Specialist trust (SHAP on meta-learner):")
        for feat, score in shap_importance:
            bar = "#" * int(score * 200)
            print(f"    {feat:<22} {bar:<30} {score:.4f}")
    except Exception as e:
        print(f"  ! SHAP failed: {e}")

    # ESN latent SHAP importance (subset of full shap_importance)
    esn_latent_shap = [(f, s) for f, s in shap_importance
                       if f.startswith("esn_latent")]

    # ── metrics ───────────────────────────────────────────────────────────
    metrics = {
        "wf_accuracy_mean":   round(np.mean(accs), 4),
        "wf_accuracy_std":    round(np.std(accs),  4),
        "wf_f1_mean":         round(np.mean(f1s),  4),
        "wf_f1_std":          round(np.std(f1s),   4),
        "threshold_acc":      round(best_acc,       4),
        "threshold_f1":       round(best_f1,        4),
        "optimal_threshold":  round(best_thresh,    3),
        "n_windows":          len(wf_thresh),
        "n_features":         len(feature_cols),
        "n_meta_features":    meta_X.shape[1],
        "n_samples":          len(X),
        "feature_cols":       feature_cols,
        "meta_feature_names": META_FEATURE_NAMES,
        "best_params":        best_params,
        "meta_optuna_score":  round(meta_optuna_score, 4),
        "gpu_used":           "cuda" in GPU_CFG.get("device", "cpu"),
        "best_iteration":     meta_model.best_iteration,
        "shap_importance":    shap_importance,
        "architecture":       "stacked_ensemble_v6",
        "precision_fallback": _precision_fallback,
        "esn_pca_variance":   esn_pca_variance,
        "esn_latent_shap":    esn_latent_shap,
    }

    # ── save — backward compatible with signal_engine.py ─────────────────
    if save_path:
        joblib.dump({
            "model":               meta_model,
            "global_model":        meta_model,
            "scaler":              scaler,
            "meta_scaler":         meta_scaler,
            "feature_cols":        feature_cols,
            "meta_feature_names":  META_FEATURE_NAMES,
            "sector_key":          sector_key,
            "sector_meta_names":   SECTOR_META_FEATURES.get(
                                       sector_key, META_FEATURE_NAMES),
            "best_params":         best_params,
            "optimal_threshold":   best_thresh,
            "metrics":             metrics,
            "global_esn":          ticker_esns,
            "X_train_sample":      X_scaled[-200:],
            "specialists":         trained_specialists,
            "specialist_features": SPECIALIST_FEATURES,
            "architecture":        "stacked_ensemble_v6",
            "esn_pca":             esn_pca,
            "esn_pca_variance":    esn_pca_variance,
        }, save_path)
        size = os.path.getsize(save_path) / 1e6
        print(f"\n  Saved: {save_path} ({size:.1f} MB)")
        print(f"  GPU  : {'Yes (CUDA)' if metrics['gpu_used'] else 'No (CPU)'}")

    return meta_model, meta_scaler, metrics


# ════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _specialist_proba_single_row(
    X_raw:               np.ndarray,
    specialists:         dict,
    scaler:              StandardScaler,
    feature_cols:        list,
    esn_latent_row:      np.ndarray = None,
    stocks_bonds_corr:   float = 0.0,
    credit_stress_score: float = 0.0,
    risk_score:          float = 0.0,
    inflation_momentum:  float = 0.0,
    real_rate:           float = 0.0,
) -> np.ndarray:
    """Run one row through all specialists, return meta-feature row (1, 20)."""
    feat_idx  = {f: i for i, f in enumerate(feature_cols)}
    X_scaled  = scaler.transform(X_raw)
    probas    = []

    for name in SPECIALIST_FEATURES.keys():
        spec_data  = specialists.get(name)
        if spec_data is None:
            probas.append(0.5); continue
        idx        = spec_data["feat_idx"]
        model      = spec_data["model"]
        mtype      = spec_data["model_type"]
        X_sub      = X_scaled[:, idx]
        try:
            if mtype == "ridge":
                d    = model.predict(X_sub)
                prob = float(1 / (1 + np.exp(-d[0])))
            else:
                prob = float(model.predict_proba(X_sub)[0, 1])
        except Exception:
            prob = 0.5
        probas.append(prob)

    while len(probas) < 5:
        probas.append(0.5)

    hurst_s   = (float(X_scaled[0, feat_idx["hurst"]])
                 if "hurst" in feat_idx else 0.5)
    vol_reg_s = (float(X_scaled[0, feat_idx["vol_regime"]])
                 if "vol_regime" in feat_idx else 1.0)
    esn_lat   = (esn_latent_row.flatten().tolist()
                 if esn_latent_row is not None else [0.0] * 8)

    # order matches META_FEATURE_NAMES (20 total):
    # 5 probas + hurst + vol_regime + 3 cross-asset + 8 esn_latent + 2 inflation
    return np.array(
        probas + [hurst_s, vol_reg_s,
                  stocks_bonds_corr, credit_stress_score, risk_score]
        + esn_lat + [inflation_momentum, real_rate],
        dtype=np.float32,
    ).reshape(1, -1)


def _specialist_probas_batch(
    X_raw:                  np.ndarray,
    specialists:            dict,
    scaler:                 StandardScaler,
    feature_cols:           list,
    hurst_series:           np.ndarray = None,
    vol_regime_series:      np.ndarray = None,
    esn_latent_matrix:      np.ndarray = None,  # (n, 8) PCA-projected reservoir states
    inflation_momentum_arr: np.ndarray = None,  # (n,) inflation momentum %
    real_rate_arr:          np.ndarray = None,  # (n,) real rate proxy %
) -> np.ndarray:
    """
    Batch specialist inference for get_full_signals().
    Single call per ticker — not row-by-row.
    Returns meta_X of shape (n_rows, 20).
    """
    feat_idx    = {f: i for i, f in enumerate(feature_cols)}
    X_scaled    = scaler.transform(X_raw)
    n           = len(X_raw)
    spec_probas = np.full((n, len(SPECIALIST_FEATURES)), 0.5)

    for s_idx, name in enumerate(SPECIALIST_FEATURES.keys()):
        spec_data = specialists.get(name)
        if spec_data is None:
            continue
        idx   = spec_data["feat_idx"]
        model = spec_data["model"]
        mtype = spec_data["model_type"]
        X_sub = X_scaled[:, idx]

        try:
            if mtype == "ridge":
                d = model.predict(X_sub)
                spec_probas[:, s_idx] = 1 / (1 + np.exp(-d))
            else:
                spec_probas[:, s_idx] = model.predict_proba(X_sub)[:, 1]
        except Exception as e:
            print(f"  ! Specialist {name} batch failed: {e}")

    hurst_col   = (hurst_series if hurst_series is not None
                   else X_scaled[:, feat_idx["hurst"]]
                   if "hurst" in feat_idx else np.full(n, 0.5))
    vol_reg_col = (vol_regime_series if vol_regime_series is not None
                   else X_scaled[:, feat_idx["vol_regime"]]
                   if "vol_regime" in feat_idx else np.ones(n))
    esn_lat     = (esn_latent_matrix
                   if esn_latent_matrix is not None and len(esn_latent_matrix) == n
                   else np.zeros((n, 8), dtype=np.float32))
    infl_mom    = (inflation_momentum_arr
                   if inflation_momentum_arr is not None and len(inflation_momentum_arr) == n
                   else np.zeros(n, dtype=np.float32))
    real_rate   = (real_rate_arr
                   if real_rate_arr is not None and len(real_rate_arr) == n
                   else np.zeros(n, dtype=np.float32))

    return np.column_stack([
        spec_probas, hurst_col, vol_reg_col,
        np.zeros(n), np.zeros(n), np.zeros(n),  # stocks_bonds_corr, credit_stress, risk_score
        esn_lat,                                 # (n, 8) ESN reservoir PCA projections
        infl_mom,                                # inflation momentum %
        real_rate,                               # real rate proxy %
    ]).astype(np.float32)



# ════════════════════════════════════════════════════════════════════════════
# PREDICT SIGNAL  (stacked ensemble inference, backward compatible)
# ════════════════════════════════════════════════════════════════════════════

def predict_signal(df: pd.DataFrame, model_path: str, ticker: str) -> dict:
    data         = joblib.load(model_path)
    meta_model   = data.get("model") or data.get("global_model")
    scaler       = data["scaler"]
    meta_scaler  = data.get("meta_scaler")
    feature_cols = data["feature_cols"]
    threshold    = data.get("optimal_threshold", 0.50)
    specialists  = data.get("specialists", {})
    architecture = data.get("architecture", "unknown")
    esn_pca      = data.get("esn_pca")

    esn_data   = data.get("global_esn")
    ticker_esn = (esn_data.get(ticker) if isinstance(esn_data, dict)
                  else esn_data)

    esn_states = None
    if esn_pca is not None:
        df, esn_states = add_esn_features(df, ticker_esn, collect_states=True)
        df["_esn_state"] = list(esn_states)
    else:
        df = add_esn_features(df, ticker_esn)
    df = add_advanced_features(df)
    df = df.dropna()

    if esn_pca is not None and "_esn_state" in df.columns:
        esn_states = np.vstack(df["_esn_state"].values)
        df = df.drop(columns=["_esn_state"])
        esn_latent_for_row = esn_pca.transform(esn_states[[-1]]).flatten()
    else:
        esn_latent_for_row = None

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    target_date = (df.index[-1].strftime("%Y-%m-%d")
                   if hasattr(df.index[-1], 'strftime')
                   else str(df.index[-1]))

    X_raw    = df[feature_cols].iloc[[-1]].values
    X_raw    = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = scaler.transform(X_raw)

    if architecture == "stacked_ensemble_v6" and specialists and meta_scaler:
        from cross_asset_signals import CrossAssetSignalEngine
        from inflation_signals import InflationSignalEngine
        ca   = CrossAssetSignalEngine()
        infl = InflationSignalEngine()
        start_date = (pd.Timestamp(target_date) - pd.DateOffset(days=450)).strftime("%Y-%m-%d")
        try:
            ca.load_data(start_date, target_date)
            infl.load_data(start_date, target_date)
            sbc = ca.get_stocks_bonds_corr(target_date)
            css = ca.get_credit_stress_score(target_date)
            rs  = ca.get_risk_on_off_signal(target_date)
            im  = infl.get_inflation_momentum(target_date)
            rr  = infl.get_real_rate_proxy(target_date)
        except Exception as _e:
            print(f"  ! Cross-asset/inflation features failed at inference: {_e}")
            sbc, css, rs, im, rr = 0.0, 0.0, 0.0, 0.0, 0.0
        meta_row        = _specialist_proba_single_row(
            X_raw=X_raw, specialists=specialists,
            scaler=scaler, feature_cols=feature_cols,
            esn_latent_row=esn_latent_for_row,
            stocks_bonds_corr=sbc, credit_stress_score=css,
            risk_score=rs, inflation_momentum=im, real_rate=rr,
        )
        meta_row_scaled = meta_scaler.transform(meta_row)
        proba           = meta_model.predict_proba(meta_row_scaled)[0]
    else:
        proba = meta_model.predict_proba(X_scaled)[0]

    proba_buy  = float(proba[1])
    confidence = float(max(proba))
    signal     = 1 if proba_buy >= threshold else 0

    return {
        "signal":         signal,
        "label":          "BUY" if signal == 1 else "WAIT/SELL",
        "confidence":     round(confidence, 4),
        "proba_buy":      round(proba_buy, 4),
        "proba_sell":     round(float(proba[0]), 4),
        "threshold_used": round(threshold, 3),
        "architecture":   architecture,
    }


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from datetime import date

    TODAY = date.today().strftime("%Y-%m-%d")
    TICKERS = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","JPM","SPY","QQQ","AVGO"]

    print("=" * 62)
    print("  VIRTUAL TRADING FIRM — XGBoost v6 Stacked Ensemble")
    print("=" * 62)

    df_global = build_multi_ticker_dataset(
        tickers=TICKERS, end_date=TODAY,
        lookback_days=1825, forward_days=5,
    )
    if df_global.empty:
        print("ERROR: dataset empty"); sys.exit()

    df_train_list = []
    for ticker in df_global['ticker'].unique():
        df_t  = df_global[df_global['ticker'] == ticker].copy()
        split = int(len(df_t) * 0.6)
        df_train_list.append(df_t.iloc[:split])
    df_train = pd.concat(df_train_list, ignore_index=True)

    print(f"\n  Training: {len(df_train):,} samples "
          f"({len(df_train)/len(df_global)*100:.0f}% of {len(df_global):,})\n")

    model, scaler, metrics = train_xgboost(
        df_train,
        save_path = "4_signals/xgboost_global_model.pkl",
        n_trials  = 100,
    )

    print("\n" + "=" * 62)
    print("  RESULTS")
    print("=" * 62)
    print(f"  Architecture : {metrics['architecture']}")
    print(f"  WF Accuracy  : {metrics['wf_accuracy_mean']:.4f} +/- {metrics['wf_accuracy_std']:.4f}")
    print(f"  WF F1        : {metrics['wf_f1_mean']:.4f} +/- {metrics['wf_f1_std']:.4f}")
    print(f"  Threshold    : {metrics['optimal_threshold']:.3f}")
    print(f"  Thr-Acc      : {metrics['threshold_acc']:.4f}")
    print(f"  Thr-F1       : {metrics['threshold_f1']:.4f}")
    print(f"  Best iter    : {metrics['best_iteration']}")
    print(f"  Meta Optuna  : {metrics['meta_optuna_score']:.4f}")
    print(f"  GPU used     : {'Yes' if metrics['gpu_used'] else 'No'}")
    print(f"  Samples      : {metrics['n_samples']:,}")
    print("=" * 62)