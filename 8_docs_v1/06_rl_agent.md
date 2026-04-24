# Section 6: Reinforcement Learning Agent

**Status:** Production  
**Location:** `6_rl_agent/`  
**Last updated:** April 2026

---

## Overview

Section 6 implements the portfolio allocation layer. A Soft Actor-Critic (SAC) agent with a GRU backbone learns to allocate capital across six assets by observing the full system state — signals, regime, portfolio risk, and walk-forward window quality — and outputting portfolio weights that sum to 1.

The RL agent does not generate signals. It learns when to trust them, how much to allocate based on confidence, and how to reduce exposure during adverse conditions. All signal generation remains in Section 4.

---

## Files

| File | Purpose |
|------|---------|
| `rl_agent.py` | Full pipeline: state builder, environment, GRU extractor, SAC agent, data loader. |
| `benchmark.py` | Trains SAC, PPO, A2C, TD3 on identical environment. Side-by-side tearsheet. |
| `rl_equity_curve.csv` | OOS equity curve from RL agent evaluation. |
| `best_model/best_model.zip` | Best checkpoint saved by EvalCallback during training. |
| `sac_gru_final.zip` | Final checkpoint at end of training. Use best_model, not this. |

---

## Dependencies

- Section 5: `equity_curve_v2.csv` — portfolio state and regime labels per bar
- Section 5: `trade_log_v2.csv` — per-ticker proba and weight signals
- Section 5: `price_data.pkl` — raw OHLCV for momentum and vol features
- Section 4: `SignalEngine` — live proba_buy for inference
- stable-baselines3, torch, gymnasium, pandas-ta

---

## Quick Start

```bash
# install dependencies
pip install stable-baselines3 torch gymnasium pandas-ta

# train SAC agent
python 6_rl_agent/rl_agent.py

# benchmark all algorithms
python 6_rl_agent/benchmark.py

# evaluate best model only
agent.load("6_rl_agent/best_model/best_model")
rl_df = agent.evaluate(deterministic=True)
```

---

## Architecture

### State space

The observation at each bar is a flattened window of shape `(seq_len=10, n_features=50)`.

**Global features (8):**

| Feature | Source | Description |
|---------|--------|-------------|
| regime | equity_curve_v2.csv | Encoded 0-3: Bull-Trend, Bull-MR, Bear-Trend, Bear-MR |
| drawdown | equity curve | Current drawdown from peak |
| daily_return | equity curve | Today's equity pct change |
| roll_vol_10 | equity curve | 10-bar rolling volatility |
| roll_sharpe_10 | equity curve | 10-bar rolling annualised Sharpe |
| heat | equity curve | Portfolio exposure fraction |
| kill_switch | computed | 1 if Sharpe < -1.25 AND DD > 4% AND slope < 0 |
| window_quality | computed | sigmoid(roll_sharpe_10), clamped [0.3, 1.0] |

**Per-ticker features (7 x 6 tickers = 42):**

| Feature | Source | Description |
|---------|--------|-------------|
| proba | trade_log_v2.csv | XGBoost buy probability, forward-filled |
| ret5d | price_data.pkl | 5-day price momentum |
| ret20d | price_data.pkl | 20-day price momentum |
| rsi | price_data.pkl | RSI / 100, normalised to [0, 1] |
| above_sma50 | price_data.pkl | Binary trend direction flag |
| atr_norm | price_data.pkl | ATR / close, normalised volatility |
| bb_pct | price_data.pkl | Bollinger %B position |

### Action space

Continuous vector of shape `(n_tickers=6,)` in `[0, 1]`. Softmax-normalised internally to produce portfolio weights summing to 1. Each dimension represents the target allocation to one ticker.

### Reward function

```
reward = 0.7 * sortino + 0.3 * sharpe          # risk-adjusted hybrid
       + top_weight * top_bar_return * 10       # position-aware conviction bonus
       - turnover_penalty * abs_weight_change   # overtrade penalty
       - dd_penalty if drawdown > 5%            # asymmetric drawdown penalty
```

Final reward is clipped to `[-1.0, 1.0]` to prevent critic explosion. Metrics are computed over a rolling 10-bar window, not the full episode, to keep the reward differential rather than cumulative.

Sortino is weighted at 70% because it penalises only downside volatility. Sharpe penalises upside volatility as well, which is undesirable for a strategy with asymmetric returns.

---

## GRU Policy

```
Input  : (batch, seq_len * n_features)
Reshape: (batch, seq_len=10, n_features=50)
GRU    : hidden=128, layers=1, batch_first=True
Output : last hidden state (batch, 128)
LayerNorm + Tanh
MLP head: [128, 64] -> action
```

GRU was chosen over LSTM for two reasons: approximately 30% faster training on the same hardware, and equivalent memory capacity for sequences under 50 bars. LSTM adds no benefit at the 10-bar sequence length used here.

---

## SAC Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| learning_rate | 1e-4 | Conservative. Prevents critic explosion on financial time series. |
| batch_size | 256 | Large batch smooths gradient variance. |
| buffer_size | 100,000 | Sufficient replay diversity for 1,000-bar dataset. |
| gamma | 0.95 | Shorter horizon appropriate for weekly-frequency signals. |
| tau | 0.01 | Faster target network update. Improves stability. |
| ent_coef | auto | Automatic entropy tuning. |
| learning_starts | 2,000 | Random exploration before first gradient update. |
| Adam eps | 1e-5 | Gradient clipping via eps. Prevents critic divergence. |

---

## Training

```python
agent.train(
    total_timesteps = 500_000,
    eval_freq       = 10_000,
    n_eval_episodes = 3,
)
```

EvalCallback saves `best_model/best_model.zip` whenever a new best eval reward is achieved. Early stopping fires after 20 consecutive evaluations with no improvement and a minimum of 30 evaluations completed.

Always load `best_model` for inference, not `sac_gru_final`. The final checkpoint is often past peak performance due to early stopping.

---

## Algorithm Benchmark

`benchmark.py` trains SAC, PPO, A2C, and TD3 on identical environments and data splits. Results from the most recent benchmark run:

| Metric | SAC | PPO | A2C |
|--------|-----|-----|-----|
| Total return | -3.56% | -17.23% | -16.90% |
| Sharpe | -0.371 | -2.314 | -1.921 |
| Max DD | -14.74% | -23.78% | -18.75% |
| Train time | 69 min | 11 min | 8 min |

SAC was selected as the production algorithm. TD3 was excluded from final benchmark due to a policy_kwargs compatibility issue with the shared GRU extractor.

---

## Live Inference

```python
from rl_agent import RLTradingAgent, load_backtest_outputs

obs_matrix, price_matrix, dates, n_features, sb = load_backtest_outputs(
    tickers     = TICKERS,
    equity_file = "5_backtesting/results/equity_curve_v2.csv",
    trade_file  = "5_backtesting/results/trade_log_v2.csv",
    price_pkl   = "5_backtesting/results/price_data.pkl",
)

agent = RLTradingAgent(tickers=TICKERS, device="cpu")
agent.load("6_rl_agent/best_model/best_model")

weights = agent.predict_weights(obs_matrix[-agent.seq_len:])
# returns: {"AAPL": 0.2405, "NVDA": 0.1421, ...}
```

---

## Performance Reference

Most recent training run: 1,047 bars, 85/15 train/eval split.

| Metric | Value |
|--------|-------|
| OOS total return | 21.72% |
| OOS annualised return | 39.75% |
| OOS Sharpe | 1.348 |
| OOS Sortino | 2.301 |
| OOS Calmar | 2.562 |
| OOS max drawdown | -15.51% |
| Live allocations | AAPL 24%, QQQ 23%, MSFT 15%, SPY 14%, NVDA 14%, TSLA 10% |

---

## Retraining

RL retraining is manual because the local machine lacks a GPU. After each monthly backtest retrain (triggered by scheduler.py), a reminder file is written to `logs/COLAB_RETRAIN_NEEDED_DATE.txt`.

Colab workflow:
1. Open Colab notebook. Files are already synced via Google Drive.
2. Run `load_backtest_outputs()` with updated CSV paths.
3. Run `agent.train(total_timesteps=500_000)`.
4. `best_model.zip` saves to Drive automatically.
5. Delete the reminder file.

---

## Revision History

| Version | Change |
|---------|--------|
| v1.0 | Initial SAC agent. Feedforward policy. Flat obs vector. Reward = portfolio return only. |
| v1.1 | GRU feature extractor added. Sequence length 10. Replaces feedforward backbone. |
| v1.2 | Reward redesigned. Differential Sortino/Sharpe hybrid. Turnover penalty. Drawdown penalty. |
| v1.3 | Reward clipped to [-1, 1]. Fixes critic explosion (critic_loss was reaching 1e+14). |
| v1.4 | learning_starts=2000 added. Adam eps=1e-5. Prevents early unstable updates. |
| v1.5 | State enriched with price features from price_data.pkl. proba + ret5d + ret20d + RSI + SMA + ATR + BB per ticker. Replaces stale trade log weights. |
| v1.6 | kill_switch and window_quality added to global state. 50 total features. |
| v1.7 | step_every=5 added to environment. Rebalances every 5 bars to match XGBoost 5-day forward return horizon. |
| v1.8 | train_ratio increased to 0.85. eval_freq increased to 10,000. Early stopping patience increased to 20. |
| v2.0 | Full retrain on extended dataset (2020-2026, 1,047 bars). OOS Sharpe turned positive at 1.348. |

---

## Known Issues

| Issue | Status |
|-------|--------|
| GPU required for training | By design. Use Colab T4. CPU training is possible but takes 8-10 hours. |
| OOS slice is 158 bars (~7 months) | Acceptable. Will grow as monthly retrains extend the dataset. |
| TD3 incompatible with shared GRU policy_kwargs | Known. use_sde conflict. Not investigated further since SAC outperforms. |
