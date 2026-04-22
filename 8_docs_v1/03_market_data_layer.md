# Section 3: Market Data Layer

**Status:** Complete  
**Estimated Time:** 2 days  
**Location:** `3_market_data/`

---

## Overview

Section 3 builds the local data pipeline that feeds every downstream component. All computation in this section runs locally with no GPU requirement. The design principle is that Groq should receive only pre-processed summaries, never raw data or computation tasks.

The section produces five modules, each with a single well-defined responsibility.

---

## Modules

### `3_market_data/local_indicators.py`

Computes technical indicators locally using pandas-ta and returns a pre-formatted text summary. Designed as a drop-in replacement for the TradingAgents `get_indicators` tool.

**Indicators computed:**

| Indicator | Parameters | Purpose |
|---|---|---|
| SMA | 20, 50 periods | Trend direction |
| SMA | 200 periods | Long-term trend |
| EMA | 10 periods | Short-term momentum |
| RSI | 14 periods | Overbought / oversold |
| ATR | 14 periods | Volatility level |
| MACD | 12/26/9 | Momentum crossover |
| Bollinger Bands | 20 period, 2 std | Volatility bands, price position |
| VWMA | 20 periods | Volume-weighted trend |

The output is a structured text string with pre-interpreted signals such as "RSI 14: 58.41 → NEUTRAL" and "MACD Direction: BULLISH". The LLM receives interpretations, not raw numbers.

**`patch_tradingagents()`** replaces `get_indicators` references in the TradingAgents namespace. Must be called before `TradingAgentsGraph` is initialised.

---

### `3_market_data/finbert_sentiment.py`

Runs sentiment analysis on financial news headlines using the FinBERT model.

**Model:** `ProsusAI/finbert`  
**Size:** 438 MB  
**Inference:** CPU, approximately 0.3 seconds per headline after first load  
**Output:** Score in range [-1.0, 1.0] where -1.0 is strongly bearish, +1.0 is strongly bullish

**News source hierarchy:**

1. NewsAPI (`NEWS_API_KEY` required, 100 requests per day free tier). Returns up to 10 articles per query with title and description.
2. Yahoo Finance RSS feed. Returns only headlines from the last 48 hours. Suitable for current-day inference, not historical backtesting.
3. yfinance news endpoint. Fallback of last resort.

**Quota tracking.** A JSON file (`newsapi_quota.json`) tracks 24-hour and 12-hour request counts. When either limit is approached, the system skips NewsAPI and proceeds directly to RSS. This prevents silent quota exhaustion mid-run.

**Critical design decision: sentiment in training vs inference.**

During model training, `sentiment = 0.0` for every historical row. This is intentional. NewsAPI's free tier only provides articles from the last 30 days, making genuine historical sentiment unavailable. Assigning today's sentiment score to historical training rows would constitute future data leakage.

During live inference, the real FinBERT score for the current date is fetched and applied as a post-prediction overlay in the signal engine. See Section 4 for the overlay implementation.

---

### `3_market_data/kalman_risk.py`

Implements an Ensemble Kalman Filter on price series to produce a noise-filtered price estimate and a dynamic stop-loss level.

**Filter parameters:**

| Parameter | Value | Effect |
|---|---|---|
| observation_noise | 1.0 | Higher = trusts observations less |
| process_noise | 0.1 | Higher = allows faster state change |
| ensemble size | N/A (scalar KF) | Uses standard KalmanFilter from filterpy |

**Outputs per bar:**

- `smoothed`: Kalman-estimated true price, filtered of noise
- `upper_band`: Smoothed price + 2 standard deviations of uncertainty
- `lower_band`: Smoothed price - 2 standard deviations of uncertainty
- `dynamic_stop_loss`: Equal to `lower_band`. Used as the exit trigger.
- `innovation_zscore`: How surprising the current price move is relative to the Kalman model's expectations
- `risk_level`: HIGH / MEDIUM / LOW based on innovation z-score thresholds (>2.5, >1.5, else LOW)

**Why this is better than a fixed stop-loss percentage.**

A fixed 5% stop-loss gets triggered by routine volatility in high-ATR periods and is too loose in low-ATR periods. The Kalman lower band adapts to the current volatility regime. In a quiet market, the band is narrow and the stop is close. In a volatile market, the band widens to avoid whipsaws.

**`get_kalman_features()`** returns the latest Kalman state as a dictionary for direct use as XGBoost features in Section 4.

**`evaluate_trade_risk()`** takes an entry price and current price and returns a structured risk assessment including whether the current price has breached the dynamic stop-loss.

---

### `3_market_data/advanced_price_features.py`

Computes five advanced quantitative features that characterise the statistical properties of price series. These features were added after the initial XGBoost model showed low accuracy, and SHAP analysis confirmed they are among the most important predictors.

**`random_walk_index()`**

Compares actual price range over a window to the expected range of a pure random walk. When RWI exceeds 1.0, price is moving more directionally than randomness would predict.

- `rwi_high`: Upward trend strength
- `rwi_low`: Downward trend strength
- `rwi_max`: Overall trend strength, max of the two
- `rwi_trend`: Binary flag, 1 if `rwi_max > 1.0`
- `rwi_direction`: +1 uptrend, -1 downtrend, 0 choppy

**`rolling_ou_features()`**

Fits an Ornstein-Uhlenbeck process to each rolling window of price data using OLS regression on lagged prices. The OU process models mean-reverting behaviour.

- `ou_theta`: Mean reversion speed. Higher values indicate faster reversion to equilibrium.
- `ou_mu`: Long-run equilibrium price estimated by the model.
- `ou_half_life`: Days for price to revert halfway to equilibrium. Derived from theta.
- `ou_residual`: Current deviation from equilibrium expressed as a z-score.
- `ou_reversion_signal`: +1 if price is below equilibrium (oversold), -1 if above (overbought), 0 if within normal range.

SHAP analysis ranked `ou_theta` as the fifth most important feature globally.

**`quadratic_variation()`**

Computes realised variance at multiple timescales (1, 5, 10, 20 days) to construct a volatility signature.

- `qv_signature_ratio`: Ratio of short-term to long-term realised variance. When this ratio significantly exceeds 1.0, microstructure noise dominates and signals are less reliable.
- `qv_noise_flag`: Binary flag, 1 if signature ratio exceeds 2.0.
- `realized_vol_20d`: Annualised realised volatility over 20 days.
- `vol_of_vol`: Rolling standard deviation of realised volatility. Measures second-order uncertainty.

SHAP analysis ranked `qv_signature_ratio` as the second most important feature globally.

**`drift_diffusion_ratio()`**

Separates the directional component (drift) of price movement from the random component (diffusion). The ratio is a signal-to-noise measure for the price path.

- `dd_ratio`: Net directional move divided by total realised variation. Values above 0.7 indicate trending; below 0.3 indicate noise-dominated.
- `signal_quality`: Clipped `dd_ratio` expressed as a 0-to-1 score.
- `dd_regime`: +1 trending, -1 noisy, 0 neutral.
- `drift_signed`: Signed net drift over the window, indicating direction.

**`hjb_optimal_entry()`**

A practical approximation of the Hamilton-Jacobi-Bellman optimal stopping solution. The full PDE solver is computationally intractable for daily trading; this approximation captures the core insight by combining four components.

- `hjb_entry_score`: Weighted combination of OU reversion speed (0.30), deviation from equilibrium (0.25), signal quality (0.25), and momentum alignment (0.20). Range: 0 to 1.
- `hjb_entry_signal`: Binary flag, 1 if score exceeds 0.60.
- `hjb_direction`: Follows the OU reversion signal direction.

**`add_advanced_price_features()`** is the master function that calls all five and appends results to the input dataframe. This is the function called by `feature_builder.py` in Section 4.

---

### `3_market_data/news_patch.py`

See Section 2 documentation. This file is physically located in `3_market_data/` because it is part of the data pipeline, but its primary purpose is to patch the TradingAgents framework.

---

## Data Flow

```
yfinance.download()
    --> feature_builder.fetch_ohlcv()
    --> feature_builder.add_technical_features()      [pandas-ta]
    --> feature_builder.add_fundamental_features()    [yfinance.Ticker.info]
    --> advanced_price_features.add_advanced_price_features()
    --> kalman_risk.get_kalman_features()
    --> finbert_sentiment.get_sentiment()             [training: 0.0, inference: real score]
    --> feature_builder.add_labels()                  [Triple Barrier]
```

---

## Revision History

| Revision | Description |
|---|---|
| Initial | Wrote `local_indicators.py` with pandas-ta summary; confirmed 80% token reduction |
| R1 | Wrote `finbert_sentiment.py` with HuggingFace API; discovered model was cold-starting and returning 0.0 |
| R2 | Switched to local FinBERT inference; fixed label parsing bug (`result[0]` not `result[0][0]`) |
| R3 | Added NewsAPI quota tracker after hitting 100/day limit during multi-ticker training runs |
| R4 | Added RSS fallback via feedparser for current-day headlines when NewsAPI is exhausted |
| R5 | Established training vs inference sentiment split; removed historical sentiment fetching from `build_multi_ticker_dataset()` |
| R6 | Wrote `kalman_risk.py`; confirmed dynamic stop-loss adapts correctly to volatility regime |
| R7 | Wrote `advanced_price_features.py` with RWI, OU, QV, D/D, HJB; SHAP confirmed `hurst`, `qv_signature_ratio`, `realized_vol_20d`, `ou_theta` as top-ranked features |
| R8 | Increased Hurst exponent window from 126 to 252 bars; shorter windows measure noise rather than true persistence |

---

## Known Issues

**FinBERT cold start.** The first call to `get_sentiment_local()` in each Python session loads the model weights from the HuggingFace cache. This takes approximately 3 to 5 seconds. Subsequent calls are fast. The model is cached at the HuggingFace default cache location and does not need to be re-downloaded after the first run.

**RSS historical limitation.** Yahoo Finance RSS feeds only contain articles from approximately the last 48 hours. For any date more than two days in the past, RSS returns nothing. This is acceptable because the system uses `sentiment = 0.0` for all training data anyway. It only becomes relevant for live inference on current-day data, where RSS works correctly.

**OU process non-stationarity.** The OU fit via OLS assumes the price process is mean-reverting over the rolling window. In strongly trending markets, the estimated `ou_theta` approaches zero and `ou_half_life` approaches infinity, correctly indicating that mean-reversion assumptions do not apply. The model handles this gracefully but the features become less informative during extended trends.
