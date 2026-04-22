# Virtual Trading Firm — Documentation Index

**Project:** Virtual Trading Firm  
**Repository root:** `D:\__A Google Drive Project\virtual_trading_firm\`  
**Last updated:** April 2026

---

## Contents

| File | Section | Status |
|---|---|---|
| `01_core_infrastructure.md` | 1. Core Infrastructure Setup | Complete |
| `02_agent_architecture.md` | 2. Agent Architecture | Complete (superseded) |
| `03_market_data_layer.md` | 3. Market Data Layer | Complete |
| `04_signal_generation.md` | 4. Signal Generation | Complete |

Sections 5 through 7 documentation pending completion of those sections.

---

## System Architecture Summary

```
Section 1: Infrastructure
    venv (D: drive) + Google Drive sync + Colab GPU bridge
    TradingAgents cloned, Groq patched into llm_clients

Section 2: Agent Architecture (superseded)
    TradingAgents multi-agent framework confirmed functional
    Replaced by ML pipeline due to Groq token limits

Section 3: Market Data Layer
    local_indicators.py     pandas-ta technical indicators (local)
    finbert_sentiment.py    FinBERT sentiment, NewsAPI + RSS fallback
    kalman_risk.py          dynamic stop-loss via Kalman filter
    advanced_price_features RWI, OU mean-reversion, QV, D/D ratio, HJB

Section 4: Signal Generation
    feature_builder.py      OHLCV + technical + fundamental feature matrix
    rc_temporal.py          Echo State Network temporal patterns
    xgboost_model.py        global brain: 10 tickers, Triple Barrier, Optuna, SHAP
    signal_engine.py        orchestrator + sentiment overlay + RL state vector

Section 5: Backtesting (in progress)
    backtest_engine.py      event-driven v1
    backtest_engine_v2.py   walk-forward, regime detection, Kalman ensemble
    rl_agent.py             SAC + GRU position sizer (Colab T4)
    portfolio.py            position tracking, equity, drawdown
    metrics.py              Sharpe, Sortino, Calmar, Omega, tearsheet

Section 6: RL Agent / Colab Training
    Trains SAC + GRU on Colab T4 GPU
    Saves model to Drive, loaded in VS Code for inference

Section 7: Documentation
    This folder
```

---

## Key Design Decisions

**Groq as explainer only.** Groq was intended as the core reasoning engine. During development, the free tier daily limit of 100,000 tokens was consumed in a single multi-ticker training run. The architecture was redesigned so that all signal generation is performed by local ML models. Groq is reserved for plain-English explanation of decisions, consuming approximately 300 tokens per ticker per day.

**Local-first ML.** Every model in the signal pipeline (pandas-ta, FinBERT, ESN, XGBoost) runs locally with no GPU requirement. The only component that requires GPU is the RL agent, which is trained on Colab's free T4 and loaded for inference on CPU.

**Sentiment in training vs inference.** All training data uses `sentiment = 0.0`. Real FinBERT scores are applied as a post-prediction overlay at inference time only. This design eliminates future data leakage while still incorporating sentiment information for live trading.

**Triple Barrier labels.** Standard binary labels (price up in N days) are noisy. Triple Barrier labels use ATR-scaled barriers (1.1x profit / 1.0x stop) to assign labels only when price makes a meaningful move. This improves label quality and produces a more balanced class distribution.

**Walk-forward validation.** All XGBoost model evaluation uses walk-forward cross-validation with expanding windows. Standard k-fold CV is not used because it allows future data to contaminate earlier training windows.
