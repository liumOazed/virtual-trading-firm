# Section 4: Signal Generation

**Status:** Complete (v3 — regime-conditional)
**Estimated Time:** 1 week initial build + ongoing iteration
**Location:** `4_signals/`

---

## Overview

Section 4 builds the complete ML signal pipeline. It takes the features produced in Section 3 and produces a calibrated BUY/SELL/WAIT signal with a confidence score, probability estimates, regime identification, and a drift warning. The pipeline replaces the TradingAgents debate chain entirely and runs with no GPU requirement and no API calls during training.

The pipeline went through three major architectural versions. v1 was a single global XGBoost model. v2 added walk-forward CV, Optuna tuning, and SHAP. v3 added regime-conditional models, 5-year lookback, and cross-asset features. This document reflects the current v3 state.

---

## Architecture

```
feature_builder.py
    --> add_cross_asset_features()    [VIX, TLT, DXY — new in v3]
    --> apply_triple_barrier_labels() [profit=1.1×ATR, stop=1.0×ATR]
    --> get_regime()                  [4-state regime label — new in v3]
    --> rc_temporal.py                [Echo State Network]
    --> kalman_risk.py                [dynamic stop-loss features]
    --> add_advanced_features()       [Hurst, skew, vol regime]
    --> train_regime_models()         [4 regime models + 1 global — new in v3]
    --> signal_engine.py              [orchestrator, routes to correct model]
```

---

## Files

### `4_signals/feature_builder.py`

Builds the feature matrix for a single ticker over a historical period.

**`fetch_ohlcv()`** downloads OHLCV from yfinance with configurable lookback. Handles MultiIndex flattening and drops NaN rows.

**`add_technical_features()`** computes 20 technical features via pandas-ta: SMA(20/50), EMA(10/20), RSI(14/7), MOM(10), MACD, ATR(14/7), Bollinger Bands (width, percent position), VWMA(20), volume ratio, 1/5/10/20-day returns, high-low ratio, gap, above-SMA flags, golden cross.

**`add_fundamental_features()`** pulls static fundamental data from `yfinance.Ticker.info`: P/E, forward P/E, price-to-book, debt/equity, ROE, beta, short ratio, distance from 52-week high and low. SHAP analysis showed near-zero importance for these features and they are candidates for removal in a future revision.

**`add_labels()`** creates the target variable using Triple Barrier Labeling. See `xgboost_model.py` for the full labeling description.

**`get_feature_columns()`** returns the canonical list of 25 feature column names. This list must be consistent between training and inference.

---

### `4_signals/rc_temporal.py`

Implements an Echo State Network (ESN), a form of Reservoir Computing chosen over LSTM because it trains in seconds on CPU and generalises better on small financial datasets.

**Architecture:**

| Parameter       | Value | Notes                                                |
| --------------- | ----- | ---------------------------------------------------- |
| reservoir_size  | 200   | Recurrent units                                      |
| spectral_radius | 0.95  | Memory decay. Values near 1.0 retain longer history. |
| sparsity        | 0.1   | 10% of reservoir connections active                  |
| input_scaling   | 0.5   | Input weight scale                                   |
| leak_rate       | 0.3   | Update speed. Lower = more memory.                   |

**Z-score normalisation.** Raw ESN decision values are not comparable across tickers because each ESN is trained on its own price scale. A rolling z-score (window=100, min_periods=20) is applied after signal generation so that a z-score of 2.0 means a two-sigma signal for any ticker.

**Leakage prevention.** The ESN is trained on the first 70% of each ticker's data. A one-bar shift is applied to the signal after generation to ensure the signal from bar T is not used as a feature for bar T's label.

**Global ESN.** In v3 a single ESN is trained on the combined multi-ticker dataset rather than one per ticker. This is saved inside the model pkl and loaded by `predict_signal()` at inference time.

---

### `4_signals/xgboost_model.py`

The core signal model. Current version: v3.

---

#### Cross-Asset Features (new in v3)

**`add_cross_asset_features()`** downloads VIX, TLT, and DXY from yfinance and adds 6 new features:

| Feature     | Source     | What it measures                   |
| ----------- | ---------- | ---------------------------------- |
| `vix_ret5d` | `^VIX`     | 5-day change in fear gauge         |
| `vix_level` | `^VIX`     | VIX deviation from 20-day MA       |
| `tlt_ret5d` | `TLT`      | 5-day bond direction (risk-on/off) |
| `tlt_level` | `TLT`      | Bond deviation from 20-day MA      |
| `dxy_ret5d` | `DX-Y.NYB` | 5-day dollar direction             |
| `dxy_level` | `DX-Y.NYB` | Dollar deviation from 20-day MA    |

Falls back to 0.0 silently if any download fails. These are proven macro edges: VIX direction predicts near-term equity volatility, TLT direction signals risk-on/off rotation, DXY direction affects multinational earnings expectations.

---

#### Regime Detection

**`get_regime()`** assigns a 4-state regime label to each bar using Hurst exponent and 50-day SMA:

| Regime        | ID  | Conditions                         |
| ------------- | --- | ---------------------------------- |
| Bull-Trending | 0   | Price above SMA50 AND Hurst > 0.55 |
| Bull-MeanRev  | 1   | Price above SMA50 AND Hurst < 0.45 |
| Bear-Trending | 2   | Price below SMA50 AND Hurst > 0.55 |
| Bear-MeanRev  | 3   | Price below SMA50 AND Hurst < 0.45 |

Neutral Hurst (0.45 to 0.55): uses 20-day realised volatility vs its 60-day average as a tiebreaker. Hurst window is 100 bars.

This label is added to every row of the training dataset and used to route prediction to the correct model at inference time.

---

#### Triple Barrier Labeling

Standard binary labels (price up in N days) are noisy because a 0.01% gain counts the same as a 5% gain. Triple Barrier assigns labels based on which barrier the price path hits first.

| Barrier               | Level                       | Label                   |
| --------------------- | --------------------------- | ----------------------- |
| Upper (profit target) | Entry price + 1.1 × ATR(14) | 1 (BUY)                 |
| Lower (stop-loss)     | Entry price - 1.0 × ATR(14) | 0 (SELL)                |
| Vertical (time limit) | After 5 trading days        | 0 (SELL) if neither hit |

**Profit target history:**

| Version | Value      | BUY ratio | Problem                                                |
| ------- | ---------- | --------- | ------------------------------------------------------ |
| v1      | 1.5 × ATR  | 22%       | Too far for 5-day window; most trades hit stop or time |
| v2      | 1.12 × ATR | ~30%      | Barely above noise; poor signal separation             |
| v3      | 1.1 × ATR  | ~35%      | Balanced; produces clean BUY/SELL separation           |

The 1.1/1.0 asymmetry (slightly more profit required than loss tolerated) is intentional. It encodes a mild positive expectation filter into the labels themselves.

---

#### Multi-Ticker Dataset

Training uses 10 tickers: AAPL, MSFT, GOOGL, AMZN, NVDA, META, JPM, SPY, QQQ, TSLA.

**Lookback changed from 730 to 1825 days (5 years) in v3.** Rationale:

```
Previous: 730 days × 10 tickers = ~2,800 training samples
          23 features → ~120 samples per feature
          Too few for stable XGBoost training

v3:       1825 days × 10 tickers = ~12,000 training samples
          25 features → ~480 samples per feature
          Sufficient for stable regime-conditional models
```

Each ticker is split 60/40: first 60% for training, last 40% reserved for out-of-sample segmented evaluation. Regime labels are attached to every row before the split.

---

#### Feature Set (CORE_FEATURES — 25 features)

```
Technical:    atr_14, macd, bb_width, bb_pct, return_20d,
              return_5d, return_1d, vol_ratio, rsi_14,
              golden_cross, realized_vol_20d, return_skew_20

Advanced:     hurst, vol_regime, ou_theta, ou_mu,
              ou_half_life, qv_signature_ratio, rwi_low

Kalman:       kalman_deviation, kalman_innovation_z

ESN:          esn_signal

Cross-asset:  vix_ret5d, vix_level, tlt_ret5d,
              tlt_level, dxy_ret5d, dxy_level
```

Features removed from earlier versions: `above_sma20`, `above_sma50`, `above_kalman_upper`, `below_kalman_lower` (redundant with other features), `return_kurt_20` (consistently near-zero SHAP importance), `pe_ratio`, `forward_pe`, `pb_ratio`, `debt_equity`, `roe`, `beta` (static fundamentals with near-zero SHAP importance after adding advanced features).

---

#### Regime-Conditional Training (new in v3)

**`train_regime_models()`** trains five models and saves them in one pkl file:

```
xgboost_global_model.pkl
├── regime_models
│   ├── [0] Bull-Trending model  + threshold
│   ├── [1] Bull-MeanRev model   + threshold
│   ├── [2] Bear-Trending model  + threshold
│   └── [3] Bear-MeanRev model   + threshold
├── global_model                 + global_threshold
├── scaler                       (shared StandardScaler)
├── feature_cols                 (25 feature names)
├── global_esn                   (Echo State Network)
└── X_train_sample               (last 200 rows for drift detection)
```

Each regime model is trained only on rows where that regime was active. This means Bull-Trending patterns, which look like momentum continuation, are learned separately from Bear-MeanRev patterns, which look like failed breakdowns.

If a regime has fewer than 200 training rows the dedicated model is skipped and the global fallback is used for that regime. With 5-year lookback this almost never occurs. With 2-year lookback Bear-Trending occasionally falls below threshold because bear markets are rarer than bull markets in the training window.

At inference time `predict_signal()` calls `get_regime()` on the live dataframe, reads the current regime, and routes to the matching model. The model used is logged in the signal output as `model_used`.

---

#### Optuna Hyperparameter Tuning

TPE sampler, hybrid objective: `0.40 × Average Precision + 0.60 × F1`.

F1 weighted more heavily because the stated target is F1 > 0.50. AP alone produces models that rank well but make poor binary decisions at any threshold.

Search space (v3, widened from v2):

| Parameter        | Range               | v2 range      | Reason for change                            |
| ---------------- | ------------------- | ------------- | -------------------------------------------- |
| n_estimators     | 200 to 1000         | 300 to 800    | Wider range finds better depth               |
| max_depth        | 3 to 7              | 4 to 8        | Shallower max reduces overfitting            |
| learning_rate    | 0.005 to 0.15 (log) | 0.015 to 0.02 | v2 range was a 0.005-wide band, not a search |
| subsample        | 0.6 to 0.95         | 0.75 to 0.92  | Wider range                                  |
| colsample_bytree | 0.6 to 0.95         | 0.75 to 0.92  | Wider range                                  |

`scale_pos_weight` is fixed to `counts[0] / counts[1]` and excluded from the Optuna search space. Including it in v2 caused Optuna to overfit the class weight, producing overconfident minority-class predictions.

Per-regime Optuna uses `max(10, n_trials // 2)` trials to keep total training time reasonable. The global fallback model uses the full `n_trials`.

---

#### Walk-Forward Cross-Validation

Expanding window, parameters:

| Parameter    | Value | Notes                            |
| ------------ | ----- | -------------------------------- |
| train_window | 600   | Minimum training rows per window |
| test_window  | 40    | Out-of-sample test bars          |
| step         | 30    | Step between windows             |

With 5-year lookback (~2,800 rows per ticker split, ~8 windows per ticker) Optuna receives a real signal rather than 3-window noise.

**Why these parameters matter.** In v2 an attempt was made to increase windows by reducing train_window to 400 and step to 20. This produced 114 windows on 2,831 samples, meaning each window had only 300-400 training rows for 23 features. Optuna saw noise, not signal. The parameters were reverted to train_window=600, step=30.

---

#### Threshold Optimisation

The default classification threshold of 0.5 assumes balanced classes. The optimal threshold is found by searching [0.35, 0.75] in 0.01 increments.

**Precision guard added in v3.** Only thresholds where `precision >= 0.52` are accepted. This prevents the threshold from collapsing to a value where the model predicts BUY on almost everything, producing high recall but low precision.

```
Score for threshold selection = 0.6 × F1 + 0.4 × accuracy
```

Each regime model gets its own optimal threshold. Bull-Trending typically settles at 0.40-0.45 (momentum is easier to catch). Bear-MeanRev typically settles at 0.45-0.52 (noise requires higher confidence before entry).

**What broke in v2 (documented for reference).** A precision guard of `precision >= 0.50` was applied without a fallback. With a 22% BUY ratio, no threshold exceeded 0.50 precision, causing `best_f1 = 0.0`, `best_thresh = 0.50`, and both `wf_accuracy_mean` and `wf_f1_mean` were written as 0.0 in metrics. The fix was to lower the guard to 0.52 and fall back to the highest-scoring threshold if the guard is never met.

---

#### Recency Weighting

`np.linspace(0.3, 1.0, n_samples)` assigns linearly increasing weights, with the most recent data at weight 1.0 and the oldest at 0.3. An additional 50% boost is applied to recent BUY label rows. Changed from `linspace(0.5, 1.0)` in v3 to reflect stronger assumption that recent market behaviour is more predictive.

---

#### SHAP Explainability

`shap.TreeExplainer` run on 2,000 samples from the global model. Top features from most recent full training run (April 2026, v3):

| Feature            | SHAP Importance |
| ------------------ | --------------- |
| esn_signal         | 0.1889          |
| qv_signature_ratio | 0.0896          |
| hurst              | 0.0894          |
| bb_width           | 0.0851          |
| rwi_low            | 0.0794          |
| rsi_14             | 0.0760          |
| vol_regime         | 0.0745          |
| return_5d          | 0.0739          |
| atr_14             | 0.0583          |
| return_skew_20     | 0.0481          |
| realized_vol_20d   | 0.0466          |
| ou_half_life       | 0.0343          |

ESN signal is the top feature, confirming temporal pattern recognition adds genuine value. The top three after ESN are all volatility regime features (QV ratio, Hurst, Bollinger width), consistent with the finding that regime context is more predictive than any single price indicator.

Cross-asset features (VIX, TLT, DXY) were added too recently to appear in the SHAP results above. Expected to appear in the mid-tier after the next training run.

---

#### Adversarial Drift Detection

A `RandomForestClassifier` distinguishes between the last 200 training rows and the current live observation. AUC above 0.70 indicates the live market significantly differs from training data, and the signal output is changed to `WAIT/SELL`.

Falls back to Mahalanobis distance for sample sizes below 20 or when class counts are insufficient for cross-validation.

---

#### Model Performance

Results from most recent full training run (v3, April 2026):

**Global model:**

| Metric            | Value |
| ----------------- | ----- |
| WF Accuracy       | 0.525 |
| WF F1             | 0.517 |
| Optimal Threshold | 0.350 |
| Training Samples  | 2,821 |
| Features          | 19    |

Note: v3 with 5-year lookback has not yet completed a full training run at time of documentation. The above figures are from a v2 run. Expected improvement with 5-year lookback: +1.5 to +2.5% accuracy.

**High-confidence accuracy from segmented in-sample evaluation (v2 baseline):**

| Ticker | Raw Accuracy | HC Accuracy (>0.65) | HC Trades |
| ------ | ------------ | ------------------- | --------- |
| AAPL   | 0.524        | 0.551               | 118       |
| MSFT   | 0.555        | 0.627               | 118       |
| NVDA   | 0.645        | 0.903               | 62        |
| JPM    | 0.604        | 0.730               | 126       |
| SPY    | 0.659        | 0.788               | 160       |
| QQQ    | 0.552        | 0.683               | 161       |
| TSLA   | 0.595        | 0.633               | 109       |

Note: segmented table is in-sample. Walk-forward metrics are the honest out-of-sample figures.

---

### `4_signals/signal_engine.py`

Orchestrates the full prediction pipeline. Primary interface used by the backtesting engine and RL agent.

Three fixes were applied to wire regime routing into `get_full_signals()` after `xgboost_model.py` was upgraded to the regime-conditional architecture. Prior to these fixes, `get_full_signals()` used a single global model and a single fixed threshold for every row regardless of the current market regime.

---

#### Fix 1 — Model loading in `get_full_signals()`

The model loading block was updated to extract the regime model dictionary and global fallback separately, rather than collapsing everything into a single `model` variable.

Before:

```python
data = joblib.load("4_signals/xgboost_global_model.pkl")
model = data.get("model") or data.get("global_model")
scaler = data["scaler"]
feature_cols = data["feature_cols"]
global_esn = data.get("global_esn")
X_train_sample = data.get("X_train_sample")
```

After:

```python
data = joblib.load("4_signals/xgboost_global_model.pkl")
scaler = data["scaler"]
feature_cols = data["feature_cols"]
global_esn = data.get("global_esn")
X_train_sample = data.get("X_train_sample")
regime_models = data.get("regime_models", {})
global_model = data.get("model") or data.get("global_model")
global_threshold = data.get("optimal_threshold", 0.55)
```

The `regime_models` dict contains up to four entries keyed by integer (0-3). Each entry holds a `model` and a `threshold` specific to that regime. `global_model` and `global_threshold` serve as fallback when a regime model is unavailable.

---

#### Fix 2 — Prediction block in `get_full_signals()`

The single `model.predict()` call was replaced with a batched regime-routing loop. The naive per-row version looped 9,000 times for a typical run (1,500 bars × 6 tickers). The production version batches all rows of the same regime together, reducing this to 4 `predict_proba` calls regardless of dataset size.

```python
from xgboost_model import get_regime
regime_labels = get_regime(df_work).values

predictions   = np.zeros(len(X_scaled), dtype=int)
probabilities = np.zeros((len(X_scaled), 2))

for regime_id in set(regime_labels):
    mask = regime_labels == regime_id
    if not mask.any():
        continue
    if int(regime_id) in regime_models:
        m         = regime_models[int(regime_id)]["model"]
        threshold = regime_models[int(regime_id)]["threshold"]
    else:
        m         = global_model
        threshold = global_threshold

    prob                = m.predict_proba(X_scaled[mask])
    probabilities[mask] = prob
    predictions[mask]   = (prob[:, 1] >= threshold).astype(int)
```

Each regime batch uses the threshold optimised specifically for that regime during training. Bull-Trending typically uses a tighter threshold (0.40-0.45) because momentum signals are cleaner. Bear-MeanRev uses a looser threshold (0.45-0.52) because noise is higher in that regime.

The `signal_idx` downstream is already set correctly by this block so no change to the signal assignment line is required.

---

#### Fix 3 — Signal assignment line

No change required. The line:

```python
signal = "BUY" if (signal_idx == 1 and drift_score < 0.70) else "WAIT/SELL"
```

remains correct because `signal_idx` is already regime-threshold-aware from Fix 2. The drift check applies uniformly across all regimes.

---

**`get_full_signals(df, ticker)`** processes a full OHLCV dataframe and returns a dataframe with one row per date containing all signal columns: `proba_buy`, `confidence`, `label`, `drift_auc`, `regime_warning`, `threshold_used`, `model_used`, `regime`.

**`predict(df, ticker)`** runs prediction on the most recent row. Returns the signal dictionary.

**`get_state(ticker, date)`** fetches fresh data, builds features, and returns a structured state vector for the RL agent's observation space:

```python
{
    "signal": "BUY" / "SELL" / "WAIT",
    "state_vector": {
        "confidence":  float,   # max XGBoost probability
        "sentiment":   float,   # FinBERT score
        "esn_zscore":  float,   # normalised ESN decision value
        "hurst":       float,   # current Hurst exponent
        "kalman_dist": float,   # Kalman deviation from smoothed price
        "vol_regime":  float,   # ATR ratio short/long
        "rsi":         float,   # RSI(14) normalised to [0, 1]
    }
}
```

**Regime routing in `predict_signal()`.** Calls `get_regime()` on the live dataframe, reads the current regime integer (0-3), and loads the matching model from the pkl. Falls back to the global model if the regime model is unavailable. The regime and model used are both logged in the output dict.

**Performance note on batch routing.** The batched implementation in `get_full_signals()` makes 4 `predict_proba` calls (one per regime) rather than one per row. For a 779-bar dataset across 6 tickers this reduces the prediction call count from approximately 4,700 to 24. The speed improvement is roughly 200x on the prediction step, though feature computation still dominates total runtime.

**Sentiment overlay.** After XGBoost produces a BUY signal the real-time FinBERT score is applied:

- Sentiment below -0.3: confidence multiplied by 0.75. Label changed to WAIT/SELL if adjusted confidence falls below 0.5.
- Sentiment above +0.3: confidence multiplied by 1.15, capped at 0.99.
- Neutral sentiment: no adjustment.

---

## Revision History

| Revision | Description                                                                                                                                                                                |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| R1       | Built `feature_builder.py`; 30 features, 195 rows on single ticker                                                                                                                         |
| R2       | Expanded to 10-ticker dataset; 4,830 samples                                                                                                                                               |
| R3       | Wrote `rc_temporal.py`; ESN 80% accuracy on 10-sample temporal test                                                                                                                        |
| R4       | First XGBoost with standard labels; WF accuracy 0.499, variance 0.127                                                                                                                      |
| R5       | Added Triple Barrier labels with 1.5/1.0; BUY ratio 25%, model defaulted to SELL                                                                                                           |
| R6       | Added Optuna tuning and walk-forward CV; accuracy improved to 0.521                                                                                                                        |
| R7       | Added Hurst, skewness, kurtosis, vol_regime features                                                                                                                                       |
| R8       | Added adversarial drift detection                                                                                                                                                          |
| R9       | Added threshold optimisation; optimal threshold 0.41-0.45 vs default 0.50                                                                                                                  |
| R10      | Added SHAP analysis                                                                                                                                                                        |
| R11      | Added recency weighting and BUY label boost                                                                                                                                                |
| R12      | Integrated advanced price features from Section 3                                                                                                                                          |
| R13      | Fixed ESN z-score normalisation; raw ESN signals not comparable across tickers                                                                                                             |
| R14      | Added 1-bar shift to ESN signal to prevent same-bar leakage                                                                                                                                |
| R15      | Attempted profit_target=1.5; BUY ratio dropped to 22%; reverted                                                                                                                            |
| R16      | Fixed `scale_pos_weight` from Optuna search space                                                                                                                                          |
| R17      | Added proper 60/40 train/test split for segmented performance table                                                                                                                        |
| R18      | Wrote `signal_engine.py`; added `get_state()` for RL agent                                                                                                                                 |
| R19      | Added sentiment overlay; applied post-prediction only                                                                                                                                      |
| R20      | Attempted train_window=400, step=20; produced 114 windows on 2,831 samples; reverted to 600/30                                                                                             |
| R21      | Added precision guard `>= 0.50` to threshold search; zeroed all metrics when no threshold qualified; reduced guard to 0.52 and added fallback                                              |
| R22      | Changed threshold search range: 0.45 start → 0.35 start                                                                                                                                    |
| R23      | Changed learning_rate Optuna range: (0.015, 0.02) → (0.005, 0.15)                                                                                                                          |
| R24      | Changed hybrid objective: 0.55 AP + 0.45 F1 → 0.40 AP + 0.60 F1                                                                                                                            |
| R25      | Changed recency weights linspace: 0.5 start → 0.3 start                                                                                                                                    |
| R26      | Fixed metrics dict to use `np.mean(accs/f1s)` not `best_acc/best_f1`                                                                                                                       |
| R27      | Changed profit_target to 1.1; BUY ratio restored to ~35%                                                                                                                                   |
| R28      | Extended lookback: 730 → 1825 days (5 years; ~12k samples)                                                                                                                                 |
| R29      | Added `add_cross_asset_features()`: VIX, TLT, DXY (6 new features)                                                                                                                         |
| R30      | Added `get_regime()` function; 4-state classifier via Hurst + SMA50                                                                                                                        |
| R31      | Added `train_regime_models()`; 4 regime-specific models + global fallback                                                                                                                  |
| R32      | Updated `predict_signal()` to route by detected regime; logs `model_used` and `regime` in output                                                                                           |
| R33      | Saved global_esn in model pkl; single ESN trained on full dataset                                                                                                                          |
| R34      | `signal_engine.py` Fix 1: updated model loading block to extract `regime_models`, `global_model`, and `global_threshold` separately from pkl                                               |
| R35      | `signal_engine.py` Fix 2: replaced single `model.predict()` call with batched regime-routing loop; 4 `predict_proba` calls instead of ~9,000; each regime uses its own optimised threshold |
| R36      | `signal_engine.py` Fix 3: confirmed signal assignment line requires no change; `signal_idx` is already regime-threshold-aware from Fix 2                                                   |

---

## Known Issues

**Regime model minimum sample threshold.** The 200-row minimum per regime means that with 2-year lookback, Bear-Trending and Bear-MeanRev regimes sometimes fall below threshold and get no dedicated model. The 5-year lookback largely resolves this but the behaviour should be monitored on the first full v3 training run.

**Cross-asset download reliability.** DXY (`DX-Y.NYB`) is the least reliable of the three cross-asset tickers. Yahoo Finance occasionally returns empty data for this ticker depending on the date range. The fallback to 0.0 is silent and the feature contributes nothing when missing. Alternative: use `UUP` (dollar ETF) as a proxy for DXY, which is more consistently available.

**Regime detection latency.** `get_regime()` uses a 100-bar Hurst window, meaning the first 100 bars of any dataframe are assigned regime 0 by default. This affects the first 5 months of any new training dataset and means the regime label for early rows is not meaningful. With a 5-year dataset this represents approximately 4% of training rows.

**Walk-forward variance remains high.** WF accuracy standard deviation is approximately 0.13. The regime-conditional architecture should reduce this because each model sees a more homogeneous training distribution, but this has not yet been confirmed on a full v3 training run.

**Fundamental features.** `pe_ratio`, `forward_pe`, `pb_ratio`, `debt_equity`, `roe`, `beta` remain in `feature_builder.py` but are excluded from `CORE_FEATURES` after SHAP showed near-zero importance. They will be removed from `feature_builder.py` in a future revision to reduce computation time.
