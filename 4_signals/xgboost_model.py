import os
import sys
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler
import shap
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'TradingAgents'))

from finbert_sentiment import get_sentiment


def add_kalman_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Kalman filter features to feature dataframe."""
    from kalman_risk import run_kalman, compute_risk_signals

    prices = df["close"]
    if len(prices) < 2:
        return df  # not enough data

    kf_df = run_kalman(prices)
    kf_df = compute_risk_signals(kf_df)

    df["kalman_deviation"] = kf_df["kalman_deviation"].values
    df["kalman_innovation_z"] = kf_df["innovation_zscore"].values
    df["above_kalman_upper"] = kf_df["above_upper"].values
    df["below_kalman_lower"] = kf_df["below_lower"].values
    df["kalman_in_band"] = kf_df["in_band"].values

    return df


def add_esn_features(df: pd.DataFrame, esn_model=None) -> pd.DataFrame:
    """Add rolling ESN signal as a feature."""
    from rc_temporal import EchoStateNetwork

    prices = df["close"]
    labels = df["label"]

    if esn_model is None:
        # Train a new ESN if not provided
        split = int(len(prices) * 0.7)
        esn = EchoStateNetwork()
        esn.fit(prices.values[:split], labels.values[:split])
    else:
        # Use the provided pre-trained ESN
        esn = esn_model

    signals = []
    for i in range(len(prices)):
        if i < 60:
            signals.append(0.0)
        else:
            result = esn.predict(prices.values[:i+1])
            signals.append(result["decision"])

    # 1. Store the raw signals first
    df["esn_signal"] = signals
    df["esn_signal"] = df["esn_signal"].shift(1)  # Shift to prevent leakage

    # 2. Apply Rolling Z-Score Normalization (Window of 100 days)
    rolling_mean = df["esn_signal"].rolling(window=100, min_periods=20).mean()
    rolling_std = df["esn_signal"].rolling(window=100, min_periods=20).std()

    # Z-Score formula: (x - mean) / std (with epsilon to prevent division by zero)
    df["esn_signal"] = (df["esn_signal"] - rolling_mean) / (rolling_std + 1e-6)

    # 3. Handle the first few rows (fill NaNs with 0.0)
    df["esn_signal"] = df["esn_signal"].fillna(0.0)

    # 4. Confidence is the absolute value of the normalized signal
    df["esn_confidence"] = df["esn_signal"].abs()

    return df

def hurst_exponent(ts: np.ndarray, max_lag: int = 100) -> float:
    """Calculate Hurst Exponent using R/S method."""
    lags = range(5, max_lag)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    reg = np.polyfit(np.log(lags), np.log(tau), 1)
    return reg[0]  # Hurst exponent


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Hurst, Skewness, and Volatility Regime"""
    prices = df['close'].values
    returns = df['return_1d'].values

    # Hurst Exponent (rolling)
    hurst_vals = []
    for i in range(len(prices)):
        # Increase window to 252 (1 trading year)
        if i < 252:
            hurst_vals.append(0.5)  # Default to random walk
        else:
            window = prices[i-252:i+1]
            h = hurst_exponent(window, max_lag=100)
            hurst_vals.append(h)
    df['hurst'] = hurst_vals

    # Rolling Skewness & Kurtosis
    df['return_skew_20'] = df['return_1d'].rolling(20).skew()
    df['return_kurt_20'] = df['return_1d'].rolling(20).kurt()

    # Volatility Regime
    df['vol_regime'] = df['atr_14'].rolling(20).mean() / df['atr_14'].rolling(60).mean()

    return df


def walk_forward_cv(
    X: np.ndarray,
    y: np.ndarray,
    train_window: int = 400,
    test_window: int = 30,
    step: int = 15,
    expanding: bool = True,
    sample_weights: np.ndarray = None,
    params: dict = None,
    scale_pos_weight: float = None,
) -> list:
    """
    Walk forward cross validation.

    expanding=False → rolling window (fixed size)
    expanding=True → expanding window (grows over time)
    """
    results = []
    start = 0

    while True:
        train_end = start + train_window
        test_end = train_end + test_window

        if test_end > len(X):
            break

        X_train = X[start:train_end] if not expanding else X[0:train_end]
        y_train = y[start:train_end] if not expanding else y[0:train_end]
        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]

        if params is None:
            model = XGBClassifier(verbosity=0, random_state=42)
        else:
            model = XGBClassifier(
                **(params or {}),
                scale_pos_weight=scale_pos_weight,
                verbosity=0,
                random_state=42
            )

        # Apply weights to the training subset during tuning
        subset_weights = (
            sample_weights[start:train_end]
            if not expanding
            else sample_weights[0:train_end]
        )

        model.fit(X_train, y_train, sample_weight=subset_weights, verbose=False)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]  # Get probabilities for AP score

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        ap = average_precision_score(y_test, probs)

        results.append({
            "train_start": start,
            "train_end": train_end,
            "test_start": train_end,
            "test_end": test_end,
            "accuracy": acc,
            "f1": f1,
            "ap": ap,
            "n_train": len(X_train),
            "n_test": len(X_test),
        })

        start += step

    return results


CORE_FEATURES = [
    # Top SHAP features (pruned to top 15-20 for better performance)
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
    "hurst"
]

MULTI_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "JPM",
    "SPY",
    "QQQ",
    "TSLA"
]

def apply_triple_barrier_labels(
    df: pd.DataFrame,
    profit_target: float = 1.5,      # ATR multiplier for profit
    stop_loss: float = 1.0,          # ATR multiplier for stop
    vertical_barrier_days: int = 5,
    min_barrier_days: int = 1
) -> pd.DataFrame:
    """
    Improved Triple Barrier Labeling (Professional Version)

    Rules:
    - Upper barrier (profit): +1.0 * profit_target * ATR
    - Lower barrier (stop): -1.0 * stop_loss * ATR
    - Vertical barrier: after N days
    - Label = 1 if profit barrier hit first
    - Label = 0 if stop loss or time barrier hit first
    """
    labels = []
    exit_times = []      # For future analysis (optional)
    exit_types = []      # 'profit', 'stop', 'time'

    prices = df['close'].values
    atrs = df['atr_14'].values
    n = len(prices)

    for i in range(n):
        if i + vertical_barrier_days >= n:
            labels.append(np.nan)
            exit_times.append(np.nan)
            exit_types.append('none')
            continue

        entry_price = prices[i]
        atr = atrs[i]

        # Dynamic barriers
        upper_barrier = entry_price + (profit_target * atr)
        lower_barrier = entry_price - (stop_loss * atr)
        end_idx = min(i + vertical_barrier_days, n - 1)

        hit = 0
        exit_time = end_idx
        exit_type = 'time'

        # Scan future bars until first barrier is hit
        for j in range(i + min_barrier_days, end_idx + 1):
            current_price = prices[j]
            if current_price >= upper_barrier:
                hit = 1
                exit_time = j
                exit_type = 'profit'
                break
            elif current_price <= lower_barrier:
                hit = 0
                exit_time = j
                exit_type = 'stop'
                break

        labels.append(hit)
        exit_times.append(exit_time - i)  # holding period in days
        exit_types.append(exit_type)

    # Add results to dataframe
    df = df.copy()
    df['label'] = labels
    df['tb_exit_days'] = exit_times
    df['tb_exit_type'] = exit_types

    # Drop rows where label could not be assigned
    df = df.dropna(subset=['label']).reset_index(drop=True)
    df['label'] = df['label'].astype(int)

    print(f"Triple Barrier applied: Profit={profit_target:.1f}×ATR | "
          f"Stop={stop_loss:.1f}×ATR | Vertical={vertical_barrier_days}d")
    print(f"Final labels → BUY: {df['label'].sum():,} | "
          f"SELL: {len(df)-df['label'].sum():,}")

    return df


def build_multi_ticker_dataset(
    tickers: list,
    end_date: str,
    lookback_days: int = 730,
    forward_days: int = 5,
) -> pd.DataFrame:
    """Build combined dataset with robust error handling and Triple Barrier labels."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "feature_builder",
        os.path.join(ROOT, "4_signals", "feature_builder.py")
    )
    fb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fb)

    dfs = []

    for ticker in tickers:
        # Initialize df as None at the start of every loop
        df = None

        try:
            print(f" → {ticker}...")

            # 1. Fetch Features
            df = fb.build_features(
                ticker=ticker,
                end_date=end_date,
                lookback_days=lookback_days,
                forward_days=forward_days,
                sentiment_score=0.0,
            )

            # Check if features were actually returned
            if df is None or df.empty:
                print(f" ⚠️ {ticker}: No data returned from feature_builder.")
                continue

            # 2. Apply Triple Barrier Labels (Step 1 of Variance Reduction)
            df = apply_triple_barrier_labels(
                df,
                profit_target=1.12,      # you can experiment with 1.0, 1.25, 2.0
                stop_loss=1.0,
                vertical_barrier_days=forward_days
            )

            if df.empty:
                print(f" ⚠️ {ticker}: 0 rows remaining after Triple Barrier filtering.")
                continue

            # 3. Add Sentiment
            print(f" → Fetching sentiment for {ticker}...")
            # For training, keep sentiment as constant 0.0 to avoid spike feature
            df["sentiment"] = 0.0
            df["ticker"] = ticker

            dfs.append(df)

        except Exception as e:
            print(f" ⚠️ {ticker} failed: {e}")
            continue

    if not dfs:
        # This prevents the "No objects to concatenate" crash
        print(" ❌ ERROR: All tickers failed. No data to train on.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    print(f" ✅ Combined: {len(combined)} samples from {len(dfs)} tickers")

    return combined

def train_xgboost(
    df: pd.DataFrame,
    save_path: str = None,
    n_trials: int = 30
) -> tuple:
    print(" → Training Global ESN on all tickers...")
    from rc_temporal import EchoStateNetwork

    global_esn = EchoStateNetwork()
    # Train on first 70% of the combined data
    split = int(len(df) * 0.7)
    global_esn.fit(df["close"].values[:split], df["label"].values[:split])

    print(" → Adding Kalman features...")
    df = add_kalman_features(df)

    print(" → Adding ESN features...")
    df = add_esn_features(df, global_esn)

    print(" → Adding Advanced features (Hurst, Skew, Vol Regime)...")
    df = add_advanced_features(df)

    df = df.dropna()

    feature_cols = [c for c in CORE_FEATURES if c in df.columns]
    X = df[feature_cols].values

    # Handle any remaining NaN or infinity values before scaling
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["label"].astype(int).values

    print(f" → Dataset: {len(X)} samples × {len(feature_cols)} features")
    print(f" → Class distribution: {np.bincount(y)} | BUY ratio: {y.mean():.1%}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Recency Weights
    n_samples = len(y)
    sample_weights = np.linspace(0.5, 1.0, n_samples)
    sample_weights = sample_weights * (1 + (y == 1) * 0.5)  # Boost recent BUYs by up to 50%

    counts = np.bincount(y)
    ratio = counts[0] / counts[1] if len(counts) > 1 else 1.0

    # ==================== OPTUNA TUNING ====================
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 800),      # Expanded range
            'max_depth': trial.suggest_int('max_depth', 4, 8),                # Deeper trees
            'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.02, log=True),
            'subsample': trial.suggest_float('subsample', 0.75, 0.92),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.75, 0.92),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),                  # increased
            'reg_lambda': trial.suggest_float('reg_lambda', 0.8, 4.0),        # stronger L2
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 2.0),          # stronger L1
            'min_child_weight': trial.suggest_int('min_child_weight', 2, 4),  # increased
        }

        wf_results = walk_forward_cv(
            X_scaled,
            y,
            train_window=600,      # Larger training window = more stable
            test_window=40,
            step=30,               # Much bigger step = fewer windows = faster
            expanding=True,
            sample_weights=sample_weights,
            params=params,
            scale_pos_weight=ratio
        )

        ap_mean = np.mean([r["ap"] for r in wf_results])
        f1_mean = np.mean([r["f1"] for r in wf_results])

        # === INCREASED WEIGHT ON F1 ===
        hybrid_score = (0.55 * ap_mean) + (0.45 * f1_mean)
        return hybrid_score

    print(f"\n🔍 Running Optuna hyperparameter tuning ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"✅ Best Hybrid Score: {study.best_value:.4f}")
    print(f"✅ Best params: {study.best_params}")

    # ==================== FINAL MODEL ====================
    final_params = study.best_params
    # scale_pos_weight is fixed, not tuned
    final_model = XGBClassifier(
        **final_params,
        scale_pos_weight=ratio,
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    )
    final_model.fit(X_scaled, y, sample_weight=sample_weights)
    
        # ==================== THRESHOLD OPTIMIZATION ====================
    print("\n🔍 Optimizing Probability Threshold (instead of default 0.5)...")

    # Get out-of-sample probabilities from final walk-forward
    # Final Walk-forward evaluation (more accurate reporting)
    wf_results = walk_forward_cv(
        X_scaled,
        y,
        train_window=700,      # Large training window
        test_window=40,
        step=30,               # Smaller step than tuning → more windows for stable variance estimate
        expanding=True,
        sample_weights=sample_weights,
        params=final_params,
        scale_pos_weight=ratio
    )

    # Collect all test predictions and true labels
    all_probs = []
    all_true = []

    for window in wf_results:
        # Re-train on train set and predict on test set to get clean probs
        X_train = X_scaled[window["train_start"]:window["train_end"]]
        y_train = y[window["train_start"]:window["train_end"]]
        X_test = X_scaled[window["test_start"]:window["test_end"]]
        y_test = y[window["test_start"]:window["test_end"]]

        temp_model = XGBClassifier(
            **final_params,
            scale_pos_weight=ratio,
            random_state=42,
            verbosity=0
        )
        temp_model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights[window["train_start"]:window["train_end"]]
        )
        probs = temp_model.predict_proba(X_test)[:, 1]
        all_probs.extend(probs)
        all_true.extend(y_test)

    all_probs = np.array(all_probs)
    all_true = np.array(all_true)

    # Search for best threshold
    thresholds = np.arange(0.45, 0.75, 0.01)
    best_f1 = 0.0
    best_thresh = 0.50
    best_acc = 0.0

    for thresh in thresholds:
        preds = (all_probs >= thresh).astype(int)
        current_f1 = f1_score(all_true, preds, zero_division=0)
        current_acc = accuracy_score(all_true, preds)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_thresh = thresh
            best_acc = current_acc

    print(f"✅ Optimal Threshold: {best_thresh:.3f} | F1: {best_f1:.4f} | Accuracy: {best_acc:.4f}")

    # Compute window-level statistics for reporting
    accs = [r["accuracy"] for r in wf_results]
    f1s = [r["f1"] for r in wf_results]

    # Store best threshold in metrics
    metrics = {
        "wf_accuracy_mean": round(best_acc, 4),
        "wf_accuracy_std": round(np.std(accs), 4),
        "wf_f1_mean": round(best_f1, 4),
        "wf_f1_std": round(np.std(f1s), 4),
        "optimal_threshold": round(best_thresh, 3),
        "n_windows": len(wf_results),
        "n_features": len(feature_cols),
        "n_samples": len(X),
        "feature_cols": feature_cols,
        "best_params": study.best_params,
    }

    # SHAP (keep your existing SHAP code here)
    try:
        shap_sample_size = min(len(X_scaled), 2000)
        print(f"\n🔍 Generating SHAP explanations (Sample size: {shap_sample_size})...")
        explainer = shap.TreeExplainer(final_model)
        indices = np.random.choice(len(X_scaled), shap_sample_size, replace=False)
        shap_values = explainer.shap_values(X_scaled[indices])
        shap_imp = np.abs(shap_values).mean(0)
        metrics["shap_importance"] = sorted(
            zip(feature_cols, shap_imp),
            key=lambda x: x[1],
            reverse=True
        )
        print("\n🏆 GLOBAL SHAP TOP FEATURES:")
        for feat, score in metrics["shap_importance"][:12]:
            print(f"    {feat:<22} {score:.4f}")
    except Exception as e:
        print(f"⚠️ SHAP failed: {e}")

    if save_path:
        joblib.dump({
            "model": final_model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "best_params": final_params,
            "optimal_threshold": best_thresh,
            "metrics": metrics,
            "global_esn": global_esn,
            "X_train_sample": X_scaled[-200:] if len(X_scaled) > 200 else X_scaled,
        }, save_path)
        print(f"✅ Model saved → {save_path} (with optimal threshold)")

    return final_model, scaler, metrics

def check_adversarial_drift(X_train: np.ndarray, X_live: np.ndarray) -> float:
    """
    Step 2: Detects if the current market (X_live) looks different from the training data.

    Returns: AUC score (0.5 = identical, 1.0 = completely different).
    Handles edge cases like small sample sizes and returns 0.5 (no drift) if detection fails.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import warnings

    try:
        # Ensure we have 2D arrays
        if X_train.ndim == 1:
            X_train = X_train.reshape(1, -1)
        if X_live.ndim == 1:
            X_live = X_live.reshape(1, -1)

        # Create labels: 0 for training, 1 for live
        y_drift = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_live))])
        X_drift = np.concatenate([X_train, X_live])

        # If we have too few samples, use simpler detection (Mahalanobis distance)
        if len(X_drift) < 20:  # Increased threshold for reliability
            try:
                # Use Mahalanobis distance instead
                train_mean = X_train.mean(axis=0)
                train_cov = np.cov(X_train.T)

                # Handle singular covariance matrix
                if np.linalg.det(train_cov) == 0:
                    # Use pseudoinverse if singular
                    inv_cov = np.linalg.pinv(train_cov)
                else:
                    inv_cov = np.linalg.inv(train_cov)

                diff = X_live[0] - train_mean
                mahal_dist = np.sqrt(diff @ inv_cov @ diff.T)

                # Normalize to 0-1 range (assume distances > 5 are "drift")
                auc = min(1.0, mahal_dist / 5.0)
                return auc if not np.isnan(auc) else 0.5
            except:
                return 0.5  # Default: no drift detection

        # For larger sample sizes, use RandomForest CV but ensure balanced folds
        n_splits = min(3, len(X_drift) // 2)
        class_counts = np.bincount(y_drift.astype(int))

        if min(class_counts) < n_splits:
            # Not enough samples per class for CV, fall back to Mahalanobis
            try:
                train_mean = X_train.mean(axis=0)
                train_cov = np.cov(X_train.T)

                if np.linalg.det(train_cov) == 0:
                    inv_cov = np.linalg.pinv(train_cov)
                else:
                    inv_cov = np.linalg.inv(train_cov)

                diff = X_live[0] - train_mean
                mahal_dist = np.sqrt(diff @ inv_cov @ diff.T)
                auc = min(1.0, mahal_dist / 5.0)
                return auc if not np.isnan(auc) else 0.5
            except:
                return 0.5

        # Suppress sklearn warnings for this operation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            drift_clf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
            auc_scores = cross_val_score(drift_clf, X_drift, y_drift, cv=n_splits, scoring='roc_auc')
            auc = np.nanmean(auc_scores)  # Use nanmean to handle any nan values

        # Handle NaN result
        if np.isnan(auc):
            return 0.5
        return auc

    except Exception as e:
        print(f"⚠️ Drift detection failed: {e}")
        return 0.5  # Default: assume no drift if detection fails


def apply_sentiment_overlay(signal_dict: dict, sentiment_score: float) -> dict:
    """
    Industry-standard sentiment overlay: adjusts confidence based on sentiment.
    - Strong negative sentiment dampens BUY signals
    - Strong positive sentiment boosts BUY signals
    """
    confidence = signal_dict["confidence"]
    label = signal_dict["label"]

    # Strong negative sentiment dampens BUY signals
    if sentiment_score < -0.3 and label == "BUY":
        confidence *= 0.75  # reduce confidence by 25%
        if confidence < 0.5:
            label = "WAIT/SELL"

    # Strong positive sentiment boosts BUY signals
    elif sentiment_score > 0.3 and label == "BUY":
        confidence = min(0.99, confidence * 1.15)

    signal_dict["confidence"] = round(confidence, 4)
    signal_dict["label"] = label
    signal_dict["sentiment_overlay"] = sentiment_score

    return signal_dict


def predict_signal(
    df: pd.DataFrame,
    model_path: str,
    ticker: str,
) -> dict:
    """Predict BUY/SELL signal for latest row in df."""
    data = joblib.load(model_path)
    model = data["model"]
    scaler = data["scaler"]
    feature_cols = data["feature_cols"]
    global_esn = data["global_esn"]

    # Add Kalman + ESN + Advanced features
    df = add_kalman_features(df)
    df = add_esn_features(df, global_esn)
    df = add_advanced_features(df)
    df = df.dropna()

    # Then proceed to scaling and prediction
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].iloc[[-1]].values

    # Ensure all required features are present; fill missing with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feature_cols].iloc[[-1]].values
    X_scaled = scaler.transform(X)

    # --- Check for Regime Drift ---
    # We compare the live row against a saved historical training sample.
    train_sample = data.get("X_train_sample")
    if train_sample is None or len(train_sample) < 6:
        drift_score = 0.5
    else:
        drift_score = check_adversarial_drift(train_sample[-200:], X_scaled)

    # Load the optimal threshold that was saved during training
    optimal_threshold = data.get("optimal_threshold", 0.50)
    proba = model.predict_proba(X_scaled)[0]
    proba_buy = float(proba[1])
    confidence = float(max(proba))

    # Use optimized threshold instead of default 0.5
    signal = 1 if proba_buy >= optimal_threshold else 0

    result = {
        "signal": int(signal),
        "label": "BUY" if (signal == 1 and drift_score < 0.70) else "WAIT/SELL",
        "drift_auc": round(drift_score, 3),
        "regime_warning": drift_score > 0.70,
        "confidence": round(confidence, 4),
        "proba_buy": round(proba_buy, 4),
        "proba_sell": round(float(proba[0]), 4),
        "threshold_used": round(optimal_threshold, 3)  # for debugging
    }

    # Apply sentiment overlay (industry standard)
    target_date = df.index[-1].strftime("%Y-%m-%d") if hasattr(df.index[-1], 'strftime') else str(df.index[-1])
    sentiment_score = get_sentiment(ticker, target_date, use_local=True)["score"]
    result = apply_sentiment_overlay(result, sentiment_score)

    return result


if __name__ == "__main__":
    from datetime import date

    TODAY = date.today().strftime("%Y-%m-%d")

    # The tickers we use to build the "Global Brain"
    # We use all 10 to ensure the model sees every type of market behavior
    tickers_to_train = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "SPY", "QQQ", "TSLA"
    ]

    print("=" * 60)
    print("🧠 PHASE 4: TRAINING THE GLOBAL MASTER MODEL")
    print("=" * 60)

    # 1. Build one massive dataset containing ALL tickers
    # This creates a "Diversity Pool" for the model to learn from
    df_global = build_multi_ticker_dataset(
        tickers=tickers_to_train,
        end_date=TODAY,
        lookback_days=730,  # 2 years of history per ticker
        forward_days=5,
    )

    if df_global.empty:
        print("❌ CRITICAL ERROR: Could not build global dataset.")
        sys.exit()

    # 2. Train ONE Master Model on the entire pool
    # The model now learns "How a breakout looks" regardless of the ticker name
    print(f"\n🔥 Training Global Model on {len(df_global)} total samples...")

    df_train_list = []
    for ticker in df_global['ticker'].unique():
        df_t = df_global[df_global['ticker'] == ticker].copy()
        split_idx = int(len(df_t) * 0.6)
        df_train_list.append(df_t.iloc[:split_idx])

    df_train = pd.concat(df_train_list, ignore_index=True)

    model, scaler, metrics = train_xgboost(
        df_train,
        save_path="4_signals/xgboost_global_model.pkl",
        n_trials=50  # Increased for better tuning
    )

    # 3. Report the "Universal" performance
    print("\n" + "=" * 60)
    print(f"🌍 GLOBAL MODEL PERFORMANCE (Weighted Avg)")
    print("-" * 60)
    print(f" Accuracy : {metrics['wf_accuracy_mean']:.4f}")
    print(f" Variance : {metrics['wf_accuracy_std']:.4f}")
    print(f" F1 Score : {metrics['wf_f1_mean']:.4f}")
    print(f" Samples : {metrics['n_samples']}")
    print("=" * 60)

    # 4. Final Test: Predict for AAPL using the Global Brain
    print(f"\n🔮 Testing Global Brain on AAPL live signal...")

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "feature_builder",
        os.path.join(ROOT, "4_signals", "feature_builder.py")
    )
    fb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fb)

    aapl_df = fb.build_features("AAPL", TODAY, 730, 5, 0.0)
    result = predict_signal(aapl_df, "4_signals/xgboost_global_model.pkl", "AAPL")

    print(f" Signal : {result['label']}")
    print(f" Confidence : {result['confidence']:.4f}")

    # ==================== NEW: SEGMENTED PERFORMANCE TABLE ====================
    print("\n" + "=" * 100)
    print("📊 SEGMENTED PERFORMANCE: HOW THE GLOBAL BRAIN SEES EACH TICKER")
    print("=" * 100)
    print(f"{'TICKER':<10} | {'RAW ACCURACY':<10} | {'HIGH-CONF ACC':<15} | "
          f"{'F1 SCORE':<10} | {'SIGNAL VOL':<12} | {'SAMPLES':<10} | {'HC COUNT > 65%':<6}")
    print("-" * 100)

    # feature_cols comes from the metrics dictionary returned by train_xgboost
    feature_cols = metrics["feature_cols"]
    model_data = joblib.load("4_signals/xgboost_global_model.pkl")
    global_esn = model_data["global_esn"]

    for ticker in tickers_to_train:
        # 1. Filter the global dataset for only this ticker's rows
        ticker_df = df_global[df_global['ticker'] == ticker].copy()

        # ✅ ADD THIS CHECK FIRST (BEFORE ANY FEATURES)
        if ticker_df.empty or len(ticker_df) < 100:
            print(f" ⚠️ Skipping {ticker}: not enough data")
            continue

        # NOW safe to compute features
        ticker_df = add_kalman_features(ticker_df)
        ticker_df = add_esn_features(ticker_df, global_esn)
        ticker_df = add_advanced_features(ticker_df)
        ticker_df = ticker_df.dropna()

        # ✅ ADD SECOND CHECK AFTER FEATURES
        if ticker_df.empty:
            print(f" ⚠️ Skipping {ticker}: empty after feature engineering")
            continue

        # 2. Prepare the data (using the GLOBAL scaler we just trained)
        X_ticker = ticker_df[feature_cols].values
        X_ticker = np.nan_to_num(X_ticker, nan=0.0, posinf=0.0, neginf=0.0)
        y_ticker = ticker_df['label'].astype(int).values
        X_ticker_scaled = scaler.transform(X_ticker)  # Use the global scaler!

        # 3. Generate Predictions
        y_preds = model.predict(X_ticker_scaled)
        y_probs = model.predict_proba(X_ticker_scaled)

        # Calculate Confidence (Max probability for the predicted class)
        confidences = np.max(y_probs, axis=1)

        # High Confidence Filter (e.g. > 0.65)
        high_conf_mask = confidences > 0.65
        hc_count = high_conf_mask.sum()

        if high_conf_mask.sum() > 0:
            high_conf_acc = accuracy_score(y_ticker[high_conf_mask], y_preds[high_conf_mask])
        else:
            high_conf_acc = 0.0

        # 4. Calculate Local Metrics
        acc = accuracy_score(y_ticker, y_preds)
        f1 = f1_score(y_ticker, y_preds, zero_division=0)

        # 5. Signal Volatility (Standard Deviation of Predictions)
        # Low volatility here means the model is stable for this ticker
        signal_vol = np.std(y_preds)

        print(f"{ticker:<10} | {acc:<10.4f} | {high_conf_acc:<15.4f} | "
              f"{f1:<10.4f} | {signal_vol:<12.4f} | {len(ticker_df):<10} | HC:{hc_count:<4}")

    print("=" * 100)
    print("✅ Full Global Analysis Complete")
# ==========================================================================

    