"""
xgboost_model.py  (v3 — regime-conditional + 5yr + cross-asset)
================================================================
Changes from previous version
------------------------------
1. lookback_days 730 → 1825  (5 years, ~12k samples vs 2.8k)
2. add_cross_asset_features() — VIX, TLT, DXY added as features
3. train_regime_models() — 4 separate XGBoost models, one per regime
4. predict_signal() routes to correct regime model automatically
5. Threshold search now requires precision >= 0.52 (stops always-BUY collapse)
6. metrics dict uses np.mean(accs/f1s) not broken threshold values
7. profit_target confirmed at 1.1 (not 1.5, not 1.12)
8. learning_rate Optuna range widened: (0.005, 0.15)
9. hybrid_score weights: 0.40 AP + 0.60 F1
10. sample_weights linspace: 0.3 → 1.0 (stronger recency)
11. Threshold search starts at 0.35
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler
import shap
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

from finbert_sentiment import get_sentiment


# ════════════════════════════════════════════════════════════════════════════
# REGIME CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

REGIME_NAMES = {
    0: "Bull-Trending",
    1: "Bull-MeanRev",
    2: "Bear-Trending",
    3: "Bear-MeanRev",
}

MODEL_SAVE_KEY = "regime_models"   # key inside the pkl dict


# ════════════════════════════════════════════════════════════════════════════
# REGIME DETECTOR  (minimal — Hurst + SMA50)
# ════════════════════════════════════════════════════════════════════════════

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
            demeaned = chunk - m
            cumdev   = np.cumsum(demeaned)
            R = cumdev.max() - cumdev.min()
            S = np.std(chunk, ddof=1)
            if S > 0:
                chunk_rs.append(R / S)
        if chunk_rs:
            rs_vals.append((lag, np.mean(chunk_rs)))
    if len(rs_vals) < 4:
        return 0.5
    lags_log = np.log([x[0] for x in rs_vals])
    rs_log   = np.log([x[1] for x in rs_vals])
    h = float(np.polyfit(lags_log, rs_log, 1)[0])
    return float(np.clip(h, 0.0, 1.0))


def get_regime(df: pd.DataFrame, hurst_window: int = 100) -> pd.Series:
    """
    Returns a Series of int regime labels aligned to df.index.
    0 = Bull-Trending, 1 = Bull-MeanRev, 2 = Bear-Trending, 3 = Bear-MeanRev
    """
    close  = df["close"]
    sma50  = close.rolling(50).mean()
    rvol   = close.pct_change().rolling(20).std()
    labels = pd.Series(0, index=df.index, dtype=int)

    for i in range(hurst_window, len(df)):
        window   = close.iloc[i - hurst_window: i].values
        h        = _hurst(np.log(window + 1e-10))
        bull     = close.iloc[i] > sma50.iloc[i]
        trending = h > 0.55
        if not trending and h >= 0.45:
            trending = rvol.iloc[i] > rvol.rolling(60).mean().iloc[i]

        if   bull and     trending: labels.iloc[i] = 0
        elif bull and not trending: labels.iloc[i] = 1
        elif not bull and trending: labels.iloc[i] = 2
        else:                       labels.iloc[i] = 3

    return labels


# ════════════════════════════════════════════════════════════════════════════
# FEATURE ADDITIONS
# ════════════════════════════════════════════════════════════════════════════

def add_kalman_features(df: pd.DataFrame) -> pd.DataFrame:
    from kalman_risk import run_kalman, compute_risk_signals
    prices = df["close"]
    if len(prices) < 2:
        return df
    kf_df = run_kalman(prices)
    kf_df = compute_risk_signals(kf_df)
    df["kalman_deviation"]    = kf_df["kalman_deviation"].values
    df["kalman_innovation_z"] = kf_df["innovation_zscore"].values
    df["above_kalman_upper"]  = kf_df["above_upper"].values
    df["below_kalman_lower"]  = kf_df["below_lower"].values
    df["kalman_in_band"]      = kf_df["in_band"].values
    return df


def add_esn_features(df: pd.DataFrame, esn_model=None) -> pd.DataFrame:
    from rc_temporal import EchoStateNetwork
    prices = df["close"]
    labels = df["label"]

    if esn_model is None:
        split = int(len(prices) * 0.7)
        esn   = EchoStateNetwork()
        esn.fit(prices.values[:split], labels.values[:split])
    else:
        esn = esn_model

    signals = []
    for i in range(len(prices)):
        if i < 60:
            signals.append(0.0)
        else:
            result = esn.predict(prices.values[:i+1])
            signals.append(result["decision"])

    df["esn_signal"]     = signals
    df["esn_signal"]     = df["esn_signal"].shift(1)
    rolling_mean         = df["esn_signal"].rolling(window=100, min_periods=20).mean()
    rolling_std          = df["esn_signal"].rolling(window=100, min_periods=20).std()
    df["esn_signal"]     = (df["esn_signal"] - rolling_mean) / (rolling_std + 1e-6)
    df["esn_signal"]     = df["esn_signal"].fillna(0.0)
    df["esn_confidence"] = df["esn_signal"].abs()
    return df


def hurst_exponent(ts: np.ndarray, max_lag: int = 100) -> float:
    lags = range(5, max_lag)
    tau  = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    reg  = np.polyfit(np.log(lags), np.log(tau), 1)
    return reg[0]


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    prices = df['close'].values

    hurst_vals = []
    for i in range(len(prices)):
        if i < 252:
            hurst_vals.append(0.5)
        else:
            window = prices[i-252:i+1]
            h = hurst_exponent(window, max_lag=100)
            hurst_vals.append(h)
    df['hurst'] = hurst_vals

    df['return_skew_20'] = df['return_1d'].rolling(20).skew()
    df['return_kurt_20'] = df['return_1d'].rolling(20).kurt()
    df['vol_regime']     = (
        df['atr_14'].rolling(20).mean() /
        df['atr_14'].rolling(60).mean()
    )

    for col, default in [
        ("ou_residual",    0.0),
        ("dd_ratio",       0.5),
        ("signal_quality", 0.5),
        ("rwi_max",        0.0),
    ]:
        if col not in df.columns:
            df[col] = default

    return df


def add_cross_asset_features(df: pd.DataFrame,
                              start_date: str,
                              end_date:   str) -> pd.DataFrame:
    """
    FIX 2: Add VIX, TLT, DXY as cross-asset features.
    These are proven macro edges — fear gauge, bond direction, dollar strength.
    Falls back to 0.0 silently if download fails.
    """
    proxies = {
        "vix":  "^VIX",
        "tlt":  "TLT",
        "dxy":  "DX-Y.NYB",
    }

    # Use df index as the date index for alignment
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else None

    for name, ticker in proxies.items():
        try:
            raw = yf.download(
                ticker, start=start_date, end=end_date,
                progress=False, auto_adjust=True
            )
            if raw.empty:
                df[f"{name}_ret5d"]   = 0.0
                df[f"{name}_level"]   = 0.0
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            close = raw["Close"].squeeze()
            close.index = pd.to_datetime(close.index)

            ret5d = close.pct_change(5)
            level = (close / close.rolling(20).mean()) - 1   # deviation from 20d MA

            if idx is not None:
                ret5d = ret5d.reindex(idx, method="ffill").fillna(0)
                level = level.reindex(idx, method="ffill").fillna(0)
            else:
                ret5d = ret5d.reindex(df.index, method="ffill").fillna(0)
                level = level.reindex(df.index, method="ffill").fillna(0)

            df[f"{name}_ret5d"] = ret5d.values
            df[f"{name}_level"] = level.values

        except Exception as e:
            print(f"  ! Cross-asset {ticker} failed: {e}")
            df[f"{name}_ret5d"] = 0.0
            df[f"{name}_level"] = 0.0

    return df


# ════════════════════════════════════════════════════════════════════════════
# FEATURE SET
# ════════════════════════════════════════════════════════════════════════════

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
    "kalman_deviation",
    "kalman_innovation_z",
    "rsi_14",
    "hurst",
    # cross-asset
    "vix_ret5d",
    "vix_level",
    "tlt_ret5d",
    "tlt_level",
    "dxy_ret5d",
    "dxy_level",
]

MULTI_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "JPM",  "SPY",  "QQQ",  "TSLA",
]


# ════════════════════════════════════════════════════════════════════════════
# TRIPLE BARRIER
# ════════════════════════════════════════════════════════════════════════════

def apply_triple_barrier_labels(
    df: pd.DataFrame,
    profit_target:         float = 1.1,   # FIX 7: confirmed 1.1, not 1.5 or 1.12
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

        entry  = prices[i]
        atr    = atrs[i]
        upper  = entry + (profit_target * atr)
        lower  = entry - (stop_loss * atr)
        end_idx= min(i + vertical_barrier_days, n - 1)

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

    print(f"  Triple Barrier: profit={profit_target:.1f}×ATR | "
          f"stop={stop_loss:.1f}×ATR | vertical={vertical_barrier_days}d")
    print(f"  Labels → BUY: {df['label'].sum():,} | "
          f"SELL: {len(df)-df['label'].sum():,} | "
          f"ratio: {df['label'].mean():.1%}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD CV
# ════════════════════════════════════════════════════════════════════════════

def walk_forward_cv(
    X:                np.ndarray,
    y:                np.ndarray,
    train_window:     int   = 600,
    test_window:      int   = 40,
    step:             int   = 30,
    expanding:        bool  = True,
    sample_weights:   np.ndarray = None,
    params:           dict  = None,
    scale_pos_weight: float = None,
) -> list:
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
        sw      = (sample_weights[0:train_end] if expanding
                   else sample_weights[start:train_end])

        model = XGBClassifier(
            **(params or {}),
            scale_pos_weight = scale_pos_weight,
            verbosity        = 0,
            random_state     = 42,
        )
        model.fit(X_train, y_train, sample_weight=sw, verbose=False)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds, zero_division=0)
        ap    = average_precision_score(y_test, probs)

        results.append({
            "train_start": start, "train_end": train_end,
            "test_start": train_end, "test_end": test_end,
            "accuracy": acc, "f1": f1, "ap": ap,
            "n_train": len(X_train), "n_test": len(X_test),
        })
        start += step

    return results


# ════════════════════════════════════════════════════════════════════════════
# DATASET BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_multi_ticker_dataset(
    tickers:      list,
    end_date:     str,
    lookback_days:int = 1825,   # FIX 1: 730 → 1825 (5 years)
    forward_days: int = 5,
) -> pd.DataFrame:
    import importlib.util
    from datetime import datetime, timedelta

    spec = importlib.util.spec_from_file_location(
        "feature_builder",
        os.path.join(ROOT, "4_signals", "feature_builder.py")
    )
    fb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fb)

    end_dt    = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt  = end_dt - timedelta(days=lookback_days)
    start_str = start_dt.strftime("%Y-%m-%d")

    dfs = []
    for ticker in tickers:
        df = None
        try:
            print(f"  → {ticker}...")
            df = fb.build_features(
                ticker        = ticker,
                end_date      = end_date,
                lookback_days = lookback_days,
                forward_days  = forward_days,
                sentiment_score = 0.0,
            )
            if df is None or df.empty:
                print(f"  ! {ticker}: no data"); continue

            # FIX 2: add cross-asset features
            df = add_cross_asset_features(df, start_str, end_date)

            df = apply_triple_barrier_labels(
                df,
                profit_target        = 1.1,
                stop_loss            = 1.0,
                vertical_barrier_days= forward_days,
            )
            if df.empty:
                print(f"  ! {ticker}: empty after TB"); continue

            # FIX 3: add regime label to each row
            df["regime"] = get_regime(df).values

            df["sentiment"] = 0.0
            df["ticker"]    = ticker
            dfs.append(df)

        except Exception as e:
            print(f"  ! {ticker} failed: {e}")
            continue

    if not dfs:
        print("  ERROR: all tickers failed.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {len(combined):,} samples from {len(dfs)} tickers")

    regime_dist = combined["regime"].value_counts().sort_index()
    for r_id, count in regime_dist.items():
        print(f"    {REGIME_NAMES.get(r_id, r_id)}: {count:,} rows")

    return combined


# ════════════════════════════════════════════════════════════════════════════
# SINGLE XGB TRAINER  (used per regime and for global model)
# ════════════════════════════════════════════════════════════════════════════

def _train_single_xgb(
    X_scaled:       np.ndarray,
    y:              np.ndarray,
    sample_weights: np.ndarray,
    ratio:          float,
    n_trials:       int,
    feature_cols:   list,
    label:          str = "Global",
) -> tuple:
    """
    Runs Optuna + threshold optimisation on a given (X, y) subset.
    Returns (model, scaler already applied externally, metrics, best_thresh).
    """

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int  ('n_estimators',    200, 1000),
            'max_depth':        trial.suggest_int  ('max_depth',        3,    7),
            'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.15, log=True),  # FIX 8
            'subsample':        trial.suggest_float('subsample',      0.6,  0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree',0.6, 0.95),
            'gamma':            trial.suggest_float('gamma',           0.0,  1.0),
            'reg_lambda':       trial.suggest_float('reg_lambda',      0.5,  6.0),
            'reg_alpha':        trial.suggest_float('reg_alpha',       0.0,  3.0),
            'min_child_weight': trial.suggest_int  ('min_child_weight', 1,    6),
        }
        wf = walk_forward_cv(
            X_scaled, y,
            train_window    = 600,
            test_window     = 40,
            step            = 30,
            expanding       = True,
            sample_weights  = sample_weights,
            params          = params,
            scale_pos_weight= ratio,
        )
        if not wf:
            return 0.0
        ap_mean = np.mean([r["ap"] for r in wf])
        f1_mean = np.mean([r["f1"] for r in wf])
        return 0.40 * ap_mean + 0.60 * f1_mean   # FIX 9

    print(f"\n  Optuna [{label}] — {n_trials} trials...")
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    final_model = XGBClassifier(
        **best_params,
        scale_pos_weight = ratio,
        eval_metric      = "logloss",
        random_state     = 42,
        verbosity        = 0,
    )
    final_model.fit(X_scaled, y, sample_weight=sample_weights)

    # Threshold optimisation with precision guard
    wf_final = walk_forward_cv(
        X_scaled, y,
        train_window    = 700,
        test_window     = 40,
        step            = 30,
        expanding       = True,
        sample_weights  = sample_weights,
        params          = best_params,
        scale_pos_weight= ratio,
    )

    all_probs, all_true = [], []
    for w in wf_final:
        X_tr = X_scaled[0:w["train_end"]]
        y_tr = y      [0:w["train_end"]]
        X_te = X_scaled[w["test_start"]:w["test_end"]]
        y_te = y      [w["test_start"]:w["test_end"]]
        sw   = sample_weights[0:w["train_end"]]
        tm   = XGBClassifier(**best_params, scale_pos_weight=ratio,
                             random_state=42, verbosity=0)
        tm.fit(X_tr, y_tr, sample_weight=sw)
        all_probs.extend(tm.predict_proba(X_te)[:, 1])
        all_true.extend(y_te)

    all_probs = np.array(all_probs)
    all_true  = np.array(all_true)

    # FIX 5: require precision >= 0.52, search from 0.35
    best_score  = 0.0
    best_thresh = 0.50
    best_f1     = 0.0
    best_acc    = 0.0

    for thresh in np.arange(0.35, 0.75, 0.01):   # FIX 11
        preds = (all_probs >= thresh).astype(int)
        if preds.sum() < 10:
            continue
        prec = precision_score(all_true, preds, zero_division=0)
        if prec < 0.52:                            # FIX 5: precision guard
            continue
        curr_f1  = f1_score(all_true, preds, zero_division=0)
        curr_acc = accuracy_score(all_true, preds)
        score    = 0.6 * curr_f1 + 0.4 * curr_acc
        if score > best_score:
            best_score  = score
            best_thresh = thresh
            best_f1     = curr_f1
            best_acc    = curr_acc

    # FIX 6: use window-level averages not threshold values for WF metrics
    accs = [r["accuracy"] for r in wf_final]
    f1s  = [r["f1"]       for r in wf_final]

    metrics = {
        "wf_accuracy_mean":  round(np.mean(accs), 4),   # FIX 6
        "wf_accuracy_std":   round(np.std(accs),  4),
        "wf_f1_mean":        round(np.mean(f1s),  4),   # FIX 6
        "wf_f1_std":         round(np.std(f1s),   4),
        "threshold_f1":      round(best_f1,        4),
        "threshold_acc":     round(best_acc,       4),
        "optimal_threshold": round(best_thresh,    3),
        "n_windows":         len(wf_final),
        "n_features":        len(feature_cols),
        "n_samples":         len(X_scaled),
        "feature_cols":      feature_cols,
        "best_params":       best_params,
    }

    print(f"  [{label}] WF Acc: {metrics['wf_accuracy_mean']:.4f} | "
          f"WF F1: {metrics['wf_f1_mean']:.4f} | "
          f"Threshold: {best_thresh:.3f} | "
          f"Thr-F1: {best_f1:.4f} | Thr-Acc: {best_acc:.4f}")

    return final_model, metrics, best_thresh


# ════════════════════════════════════════════════════════════════════════════
# REGIME-CONDITIONAL TRAINING  (FIX 3 — main new function)
# ════════════════════════════════════════════════════════════════════════════

def train_regime_models(
    df:        pd.DataFrame,
    save_path: str  = None,
    n_trials:  int  = 30,
) -> tuple:
    """
    FIX 3: Train one XGBoost model per regime (4 models total).
    Also trains a global fallback model.

    Returns (regime_models dict, global_model, scaler, all_metrics).
    """
    from rc_temporal import EchoStateNetwork

    print("\n  Training Global ESN...")
    global_esn = EchoStateNetwork()
    split      = int(len(df) * 0.7)
    global_esn.fit(df["close"].values[:split], df["label"].values[:split])

    print("  Adding Kalman features...")
    df = add_kalman_features(df)
    print("  Adding ESN features...")
    df = add_esn_features(df, global_esn)
    print("  Adding advanced features...")
    df = add_advanced_features(df)
    df = df.dropna()

    feature_cols = [c for c in CORE_FEATURES if c in df.columns]
    print(f"  Features available: {len(feature_cols)} / {len(CORE_FEATURES)}")

    # Global scaler fitted on all data
    X_all    = df[feature_cols].values
    X_all    = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    y_all    = df["label"].astype(int).values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # FIX 10: stronger recency weights 0.3 → 1.0
    n_samples      = len(y_all)
    sample_weights = np.linspace(0.3, 1.0, n_samples)
    sample_weights = sample_weights * (1 + (y_all == 1) * 0.5)

    counts = np.bincount(y_all)
    ratio  = counts[0] / counts[1] if len(counts) > 1 else 1.0

    # ── Train 4 regime-specific models ────────────────────────────────────
    regime_models  = {}
    regime_metrics = {}

    print(f"\n  Training regime-conditional models...")

    for regime_id, regime_name in REGIME_NAMES.items():
        mask = (df["regime"] == regime_id).values
        n_regime = mask.sum()

        if n_regime < 200:
            print(f"  ! {regime_name}: only {n_regime} rows — skipping "
                  f"(will use global model for this regime)")
            continue

        print(f"\n  Regime {regime_id}: {regime_name} — {n_regime} samples")

        X_r  = X_scaled[mask]
        y_r  = y_all[mask]
        sw_r = sample_weights[mask]

        counts_r = np.bincount(y_r)
        ratio_r  = counts_r[0] / counts_r[1] if len(counts_r) > 1 else 1.0

        print(f"    BUY ratio: {y_r.mean():.1%} | "
              f"Class ratio: {ratio_r:.2f}")

        # Use fewer trials per regime to keep total time reasonable
        regime_trials = max(10, n_trials // 2)

        model_r, metrics_r, thresh_r = _train_single_xgb(
            X_r, y_r, sw_r, ratio_r, regime_trials,
            feature_cols, label=regime_name
        )

        regime_models[regime_id]  = {
            "model":     model_r,
            "threshold": thresh_r,
            "metrics":   metrics_r,
        }
        regime_metrics[regime_name] = metrics_r

    # ── Train global fallback model ────────────────────────────────────────
    print(f"\n  Training global fallback model ({len(X_scaled)} samples)...")
    global_model, global_metrics, global_thresh = _train_single_xgb(
        X_scaled, y_all, sample_weights, ratio, n_trials,
        feature_cols, label="Global"
    )

    # ── SHAP on global model ───────────────────────────────────────────────
    try:
        n_shap    = min(len(X_scaled), 2000)
        explainer = shap.TreeExplainer(global_model)
        idx       = np.random.choice(len(X_scaled), n_shap, replace=False)
        shap_vals = explainer.shap_values(X_scaled[idx])
        shap_imp  = np.abs(shap_vals).mean(0)
        global_metrics["shap_importance"] = sorted(
            zip(feature_cols, shap_imp),
            key=lambda x: x[1], reverse=True
        )
        print("\n  Top SHAP features (global model):")
        for feat, score in global_metrics["shap_importance"][:12]:
            print(f"    {feat:<24} {score:.4f}")
    except Exception as e:
        print(f"  ! SHAP failed: {e}")

    # ── Combined accuracy report ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  {'REGIME-CONDITIONAL MODEL PERFORMANCE':^63}")
    print("=" * 65)
    print(f"  {'Model':<22} {'WF Acc':>8} {'WF F1':>8} {'Thr-Acc':>8} {'Thr-F1':>8} {'Thresh':>7}")
    print("  " + "-" * 63)

    all_wf_accs, all_wf_f1s = [], []

    for regime_id, regime_name in REGIME_NAMES.items():
        if regime_id in regime_models:
            m = regime_models[regime_id]["metrics"]
        else:
            m = global_metrics
            regime_name += " (global)"

        all_wf_accs.append(m["wf_accuracy_mean"])
        all_wf_f1s.append(m["wf_f1_mean"])

        print(f"  {regime_name:<22} "
              f"{m['wf_accuracy_mean']:>8.4f} "
              f"{m['wf_f1_mean']:>8.4f} "
              f"{m.get('threshold_acc',0):>8.4f} "
              f"{m.get('threshold_f1',0):>8.4f} "
              f"{m['optimal_threshold']:>7.3f}")

    print("  " + "-" * 63)
    combined_acc = np.mean(all_wf_accs)
    combined_f1  = np.mean(all_wf_f1s)
    print(f"  {'Combined (weighted avg)':<22} "
          f"{combined_acc:>8.4f} "
          f"{combined_f1:>8.4f}")
    print("=" * 65)

    all_metrics = {
        "regime_metrics":    regime_metrics,
        "global_metrics":    global_metrics,
        "combined_wf_acc":   round(combined_acc, 4),
        "combined_wf_f1":    round(combined_f1,  4),
        # For backtest compatibility — use global metrics as top-level
        "wf_accuracy_mean":  global_metrics["wf_accuracy_mean"],
        "wf_accuracy_std":   global_metrics["wf_accuracy_std"],
        "wf_f1_mean":        global_metrics["wf_f1_mean"],
        "wf_f1_std":         global_metrics["wf_f1_std"],
        "n_samples":         len(X_scaled),
        "n_features":        len(feature_cols),
        "feature_cols":      feature_cols,
    }

    if save_path:
        joblib.dump({
            MODEL_SAVE_KEY:      regime_models,
            "global_model":      global_model,
            "global_threshold":  global_thresh,
            "scaler":            scaler,
            "feature_cols":      feature_cols,
            "optimal_threshold": global_thresh,
            "metrics":           all_metrics,
            "global_esn":        global_esn,
            "X_train_sample":    X_scaled[-200:] if len(X_scaled) > 200 else X_scaled,
        }, save_path)
        print(f"\n  Model saved: {save_path}")

    return regime_models, global_model, scaler, all_metrics


# ════════════════════════════════════════════════════════════════════════════
# LEGACY train_xgboost (kept for backtest_engine_v2 retrain hook)
# ════════════════════════════════════════════════════════════════════════════

def train_xgboost(
    df:        pd.DataFrame,
    save_path: str  = None,
    n_trials:  int  = 30,
) -> tuple:
    """
    Thin wrapper around train_regime_models for backward compatibility.
    Returns (global_model, scaler, metrics) — same signature as before.
    """
    _, global_model, scaler, metrics = train_regime_models(
        df, save_path=save_path, n_trials=n_trials
    )
    return global_model, scaler, metrics


# ════════════════════════════════════════════════════════════════════════════
# DRIFT DETECTION
# ════════════════════════════════════════════════════════════════════════════

def check_adversarial_drift(X_train: np.ndarray,
                             X_live:  np.ndarray) -> float:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import warnings

    try:
        if X_train.ndim == 1: X_train = X_train.reshape(1, -1)
        if X_live.ndim  == 1: X_live  = X_live .reshape(1, -1)

        y_drift = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_live))])
        X_drift = np.concatenate([X_train, X_live])

        def _mahal():
            try:
                mu  = X_train.mean(axis=0)
                cov = np.cov(X_train.T)
                inv = (np.linalg.pinv(cov) if np.linalg.det(cov) == 0
                       else np.linalg.inv(cov))
                d   = np.sqrt((X_live[0]-mu) @ inv @ (X_live[0]-mu).T)
                auc = min(1.0, d / 5.0)
                return auc if not np.isnan(auc) else 0.5
            except Exception:
                return 0.5

        if len(X_drift) < 20:
            return _mahal()

        n_splits     = min(3, len(X_drift) // 2)
        class_counts = np.bincount(y_drift.astype(int))
        if min(class_counts) < n_splits:
            return _mahal()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf    = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
            scores = cross_val_score(clf, X_drift, y_drift, cv=n_splits, scoring='roc_auc')
            auc    = np.nanmean(scores)

        return 0.5 if np.isnan(auc) else float(auc)

    except Exception as e:
        print(f"  ! Drift detection failed: {e}")
        return 0.5


# ════════════════════════════════════════════════════════════════════════════
# SENTIMENT OVERLAY
# ════════════════════════════════════════════════════════════════════════════

def apply_sentiment_overlay(signal_dict: dict,
                             sentiment_score: float) -> dict:
    confidence = signal_dict["confidence"]
    label      = signal_dict["label"]

    if sentiment_score < -0.3 and label == "BUY":
        confidence *= 0.75
        if confidence < 0.5:
            label = "WAIT/SELL"
    elif sentiment_score > 0.3 and label == "BUY":
        confidence = min(0.99, confidence * 1.15)

    signal_dict["confidence"]        = round(confidence, 4)
    signal_dict["label"]             = label
    signal_dict["sentiment_overlay"] = sentiment_score
    return signal_dict


# ════════════════════════════════════════════════════════════════════════════
# PREDICTION  (routes to correct regime model)
# ════════════════════════════════════════════════════════════════════════════

def predict_signal(
    df:         pd.DataFrame,
    model_path: str,
    ticker:     str,
) -> dict:
    data         = joblib.load(model_path)
    scaler       = data["scaler"]
    feature_cols = data["feature_cols"]
    global_esn   = data["global_esn"]

    df = add_kalman_features(df)
    df = add_esn_features(df, global_esn)
    df = add_advanced_features(df)
    df = df.dropna()

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X        = df[feature_cols].iloc[[-1]].values
    X_scaled = scaler.transform(X)

    # FIX 3: detect regime, route to correct model
    regime_models = data.get(MODEL_SAVE_KEY, {})
    current_regime = int(get_regime(df).iloc[-1]) if len(df) >= 50 else 0

    if current_regime in regime_models:
        model     = regime_models[current_regime]["model"]
        threshold = regime_models[current_regime]["threshold"]
        model_used = REGIME_NAMES[current_regime]
    else:
        model     = data["global_model"]
        threshold = data.get("global_threshold", data.get("optimal_threshold", 0.50))
        model_used = "Global (fallback)"

    train_sample = data.get("X_train_sample")
    drift_score  = (check_adversarial_drift(train_sample[-200:], X_scaled)
                    if train_sample is not None and len(train_sample) >= 6
                    else 0.5)

    proba      = model.predict_proba(X_scaled)[0]
    proba_buy  = float(proba[1])
    confidence = float(max(proba))
    signal     = 1 if proba_buy >= threshold else 0

    result = {
        "signal":          int(signal),
        "label":           "BUY" if (signal == 1 and drift_score < 0.70) else "WAIT/SELL",
        "drift_auc":       round(drift_score, 3),
        "regime_warning":  drift_score > 0.70,
        "confidence":      round(confidence, 4),
        "proba_buy":       round(proba_buy, 4),
        "proba_sell":      round(float(proba[0]), 4),
        "threshold_used":  round(threshold, 3),
        "model_used":      model_used,
        "regime":          current_regime,
    }

    target_date     = (df.index[-1].strftime("%Y-%m-%d")
                       if hasattr(df.index[-1], 'strftime')
                       else str(df.index[-1]))
    sentiment_score = get_sentiment(ticker, target_date, use_local=True)["score"]
    result          = apply_sentiment_overlay(result, sentiment_score)
    return result


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from datetime import date

    TODAY = date.today().strftime("%Y-%m-%d")

    tickers_to_train = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "JPM",  "SPY",  "QQQ",  "TSLA",
    ]

    print("=" * 65)
    print("  TRAINING REGIME-CONDITIONAL GLOBAL BRAIN")
    print("=" * 65)

    df_global = build_multi_ticker_dataset(
        tickers       = tickers_to_train,
        end_date      = TODAY,
        lookback_days = 1825,   # FIX 1: 5 years
        forward_days  = 5,
    )

    if df_global.empty:
        print("ERROR: could not build dataset.")
        sys.exit()

    # 60% per ticker for training
    df_train_list = []
    for ticker in df_global['ticker'].unique():
        df_t = df_global[df_global['ticker'] == ticker].copy()
        split = int(len(df_t) * 0.6)
        df_train_list.append(df_t.iloc[:split])
    df_train = pd.concat(df_train_list, ignore_index=True)

    print(f"\n  Training on {len(df_train):,} samples "
          f"({len(df_train)/len(df_global)*100:.0f}% of {len(df_global):,} total)")

    regime_models, global_model, scaler, metrics = train_regime_models(
        df_train,
        save_path = "4_signals/xgboost_global_model.pkl",
        n_trials  = 50,
    )

    # Final AAPL test
    print(f"\nTesting on AAPL live signal...")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "feature_builder",
        os.path.join(ROOT, "4_signals", "feature_builder.py")
    )
    fb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fb)

    aapl_df = fb.build_features("AAPL", TODAY, 1825, 5, 0.0)
    result  = predict_signal(
        aapl_df, "4_signals/xgboost_global_model.pkl", "AAPL"
    )
    print(f"\n  Signal      : {result['label']}")
    print(f"  Confidence  : {result['confidence']:.4f}")
    print(f"  Model used  : {result['model_used']}")
    print(f"  Regime      : {REGIME_NAMES.get(result['regime'], result['regime'])}")
    print(f"  Drift AUC   : {result['drift_auc']:.3f}")