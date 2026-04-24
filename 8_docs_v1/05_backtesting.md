# Section 5: Backtesting Engine

**Status:** Production  
**Location:** `5_backtesting/`  
**Last updated:** April 2026

---

## Overview

Section 5 implements the walk-forward backtesting engine. It orchestrates all upstream layers — signal generation, regime detection, Kalman ensemble, filter competition, and champion selection — into a single pipeline that produces an auditable out-of-sample equity curve.

The engine enforces a strict no-look-ahead constraint. At every bar, only information available up to and including that bar is used. Walk-forward retraining ensures XGBoost is never exposed to future data during the period it is being evaluated on.

---

## Files

| File | Purpose |
|------|---------|
| `backtest_engine_v2.py` | Main orchestrator. Walk-forward loop, kill-switch, blending, trade execution. |
| `portfolio.py` | Position tracking, P&L, drawdown, commission handling. |
| `metrics.py` | Tearsheet generation. Per-regime and per-window breakdown. RL feature export. |
| `scheduler.py` | Monthly auto-retrain and daily explainer scheduling. |

---

## Dependencies

- Section 4: `SignalEngine.get_full_signals()` — XGBoost + ESN proba_buy per bar
- Section 4: `RegimeDetector` — 4-state Hurst-based regime classification
- Section 3: `feature_builder.py` — OHLCV + technical + advanced features
- Section 3: `finbert_sentiment.py` — FinBERT sentiment overlay (inference only)

---

## Quick Start

```bash
# run full backtest
python 5_backtesting/backtest_engine_v2.py

# run metrics tearsheet
python 5_backtesting/metrics.py

# force immediate retrain (closes data gap)
python scheduler.py --retrain-now

# start production scheduler
python scheduler.py
```

---

## Architecture

### Pipeline stages

| Stage | Method | Output |
|-------|--------|--------|
| Data loading | `DataLoader.load()` | OHLCV per ticker with 420-day warmup |
| Signal precomputation | `_precompute_signals()` | proba_buy cache per ticker per date |
| Regime labelling | `_precompute_regimes()` | 4-state label per bar |
| Correlation filter | `_run_correlation_filter()` | Pruned active ticker list |
| Alpha decay | `_run_alpha_decay()` | Optimal hold period per ticker |
| Walk-forward loop | `run()` | OOS-only equity curve |
| Stress testing | `StressTester` | Regime stress + slippage sweep |
| Finalisation | `_finalise()` | CSVs, champion report, decay CSV |

### BacktestConfig key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `start_date` | 2020-01-01 | OOS start. 420-day warmup prepended automatically. |
| `end_date` | 2026-04-01 | Updated monthly by scheduler. |
| `train_months` | 12 | Walk-forward training window. |
| `oos_months` | 2 | OOS step size per window. |
| `retrain` | True | Retrain XGBoost per window with Optuna. |
| `fixed_size` | 0.10 | Base position size as fraction of equity. |
| `base_slippage` | 0.001 | Slippage applied at execution. |
| `buy_threshold` | 0.55 | proba_buy threshold for BUY. |
| `sell_threshold` | 0.45 | proba_buy threshold for SELL. |

---

## Signal Blending

The final blended weight per ticker per bar combines three sources:

```
blended_w = 0.35 * k_weight       # Kalman ensemble
          + 0.35 * c_weight       # Filter competition
          + 0.30 * window_quality # Walk-forward window quality
```

Window quality is `sigmoid(window_sharpe) * recency_decay(0.9^n)`, clamped to `[0.3, 1.0]`.

Position size is then scaled by regime confidence:

```
regime_conf = regime_sharpe / max_regime_sharpe   # clamped to [0.5, 1.0]
pos_value   = equity * fixed_size * regime_conf
```

---

## Kill-Switch

Halts new trade execution when performance degrades. Does not freeze the engine — equity records and resume evaluation run every bar regardless.

**Fire condition** (all three must be true):
- 20-bar rolling annualised Sharpe < -1.25
- Drawdown from peak > 4.0%
- Sharpe slope (10-bar mean minus 20-bar mean) is negative

**Resume condition** (all three must be true):
- Rolling Sharpe > -0.10
- Sharpe slope is positive
- At least 5 bars elapsed since halt

---

## Metrics

### Core metrics

| Metric | Formula |
|--------|---------|
| Sharpe | (ann_return - rf) / ann_vol, rf = 5% |
| Sortino | (ann_return - rf) / downside_vol |
| Calmar | ann_return / abs(max_drawdown) |
| Omega | sum(positive returns) / sum(abs(negative returns)) |
| VaR 95% | 5th percentile of daily return distribution |
| CVaR 95% | Mean of returns below VaR threshold |

### Outputs

```
5_backtesting/results/
  equity_curve_v2.csv      # date, equity, regime, window per bar
  trade_log_v2.csv         # date, ticker, action, proba, regime, weight, price, shares, window
  trade_pnl.csv            # FIFO-matched BUY/SELL pairs with pnl_pct
  window_breakdown.csv     # per-window return, Sharpe, DD, trades, win rate
  champion_selection.csv   # best ticker per regime by Calmar
  alpha_decay.csv          # half-life and edge by horizon per ticker
  stress_regime.csv        # COVID crash, 2022 bear, 2018 Q4 results
  stress_slippage.csv      # Sharpe decay from 0% to 5% slippage
  tearsheet.png            # 5-panel chart
  price_data.pkl           # OHLCV dict for RL agent
```

---

## Scheduler

Keeps the stack current without manual intervention.

| Time | Job | Compute |
|------|-----|---------|
| 1st of month, 02:00 | Backtest retrain (XGBoost walk-forward) | Local CPU (~2 hrs) |
| 1st of month, 02:00 | price_data.pkl refresh + MODEL_CUTOFF update | Local CPU |
| 1st of month, manual | RL agent retrain | Colab T4 (~1 hr) |
| Weekdays, 17:30 | Metrics tearsheet refresh | Local CPU |
| Weekdays, 18:00 | Groq daily explainer | Local CPU + Groq API |

RL retraining is manual because the local machine lacks a GPU. After each backtest retrain completes, a reminder file is written to `logs/COLAB_RETRAIN_NEEDED_DATE.txt`.

---

## Performance Reference

Most recent full run: 2021-01-04 to 2025-03-28 (1,544 trading days).

| Metric | Value |
|--------|-------|
| Total return | 78.8% |
| Annualised return | 14.74% |
| Annualised vol | 6.97% |
| Sharpe | 1.342 |
| Sortino | 2.155 |
| Calmar | 1.067 |
| Max drawdown | -13.82% |
| Win rate | 80.2% |
| Profit factor | 4.394 |
| Bull-Trending Sharpe | 2.916 |
| Bear-Trending Sharpe | 2.065 |
| RL agent OOS Sharpe | 1.348 |
| RL agent OOS return | 21.72% |

---

## Revision History

### backtest_engine_v2.py

| Version | Change |
|---------|--------|
| v1.0 | Initial walk-forward engine. Basic OOS loop, fixed position sizing, single signal cache. |
| v1.1 | RegimeDetector added. 4-state Hurst + SMA classification. SPY as market proxy. |
| v1.2 | SignalCorrelationFilter added. Pairwise return correlation prunes redundant alphas. |
| v1.3 | AlphaDecayAnalyser added. Signal edge measured at t+1,3,5,10,20 bars. |
| v1.4 | KalmanEnsemble added. Dynamic per-bar signal weighting from recent accuracy. |
| v1.5 | FilterCompetition added. Rolling Sharpe tournament. Losers zeroed, winners scaled. |
| v1.6 | ChampionSelector added. Calmar-ranked best ticker per regime. |
| v1.7 | StressTester added. Regime stress replay and slippage sweep. |
| v2.0 | Kill-switch added. Fire: Sharpe < -1.25 AND DD > 4% AND slope < 0. Resume: Sharpe > -0.1 AND slope > 0 AND 5+ bars elapsed. |
| v2.1 | Kill-switch structural fix. Equity recording and resume evaluation now unconditional every bar. Removed continue inside halt block which was preventing resume. |
| v2.2 | Window quality weighting added. sigmoid(window_sharpe) * recency_decay(0.9^n), clamped [0.3, 1.0]. |
| v2.3 | Three-component blending: 35% Kalman + 35% competition + 30% window quality. |
| v2.4 | Regime-aware position sizing. Scaled by regime_sharpe / max_regime_sharpe, floored at 50%. |
| v2.5 | Multiprocessing walk-forward. Worker runs in child process. Per-window model files prevent race conditions. |
| v2.6 | Slippage sweep bug fixed. Compounded equity reconstruction replaces linear P&L sum. Sharpe now correctly degrades with increasing slippage. |
| v2.7 | window column added to equity curve and trade log. Enables per-window breakdown in metrics. |
| v2.8 | price_data.pkl saved at end of run for RL agent. |
| v2.9 | Date range extended to 2020-01-01. Provides 1,500+ bars for RL training. |

### portfolio.py

| Version | Change |
|---------|--------|
| v1.0 | Initial implementation. BUY/SELL execution, cash tracking, equity history. |
| v1.1 | Bug fix: avg_price weighted average corrected. Old code incremented shares before computing average, producing wrong cost basis. |
| v1.2 | Commission applied symmetrically on BUY and SELL. Previously only deducted from SELL. |
| v1.3 | Slippage applied directionally. BUY at ask (price * (1+slip)), SELL at bid (price * (1-slip)). |

### metrics.py

| Version | Change |
|---------|--------|
| v1.0 | Core metrics: Sharpe, Sortino, Calmar, Omega, VaR, CVaR, hit rate. |
| v1.1 | Trade P&L builder added. FIFO matching reconstructs pnl_pct without requiring it in the trade log. |
| v1.2 | Per-regime breakdown added. |
| v1.3 | 4-panel tearsheet chart added. |
| v1.4 | RL feature export added. Normalised flat dict for RL state vector. |
| v2.0 | Per-window breakdown added. Requires window column in equity_curve_v2.csv. |
| v2.1 | 5-panel tearsheet. Per-window return bar chart and window boundary shading on equity curve. |
| v2.2 | Window Sharpe features added to RL export: best, worst, std of window Sharpes. |
| v2.3 | RL feature duplication fix. reset_index(drop=True) before iterrows loops. |
| v2.4 | kill_switch and window_quality added to RL state vector. |
| v2.5 | window_breakdown.csv saved as additional output. |

---

## Known Issues

| Issue | Status |
|-------|--------|
| RL agent must be retrained manually on Colab T4 | By design. Local machine lacks GPU. |
| Stress test returns NaN for periods outside date range | Known. COVID crash and 2018 Q4 are outside range if start_date > 2020. |
| FinBERT returns identical sentiment for all tickers in some sessions | Under investigation. Likely NewsAPI returning market-wide headlines. |
| Bull-MeanRev and Bear-MeanRev return None in champion selection | Known. Insufficient trades in those regimes. |
| Drift AUC approaches 1.0 after 6+ months without retraining | Managed by scheduler monthly retrain. |
