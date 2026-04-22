# Section 1: Core Infrastructure Setup

**Status:** Complete  
**Estimated Time:** 1 day  
**Location:** `1_infrastructure/`

---

## Overview

Section 1 establishes the development environment, installs all dependencies, and verifies that every external service the system depends on is reachable and functional. Nothing in Sections 2 through 7 will work correctly unless all five connection tests in this section pass cleanly.

---

## Environment Architecture

The project runs across two environments that share state through Google Drive.

**VS Code (local, D: drive)** handles all code editing and light development tasks. The project folder lives at `D:\__A Google Drive Project\virtual_trading_firm\`. This location was chosen deliberately to keep the project off the C: drive, which had only 20 GB of free space.

**Google Colab (cloud, T4 GPU)** handles all computationally expensive tasks: model training, RL agent training, and full backtesting runs. Colab mounts the same Google Drive folder, so any file saved in VS Code is immediately visible in Colab and vice versa.

**Google Drive for Desktop** acts as the bridge. It syncs `D:\__A Google Drive Project\` to the cloud in real time. When Colab mounts `/content/drive/MyDrive/`, it sees the identical folder structure.

A Python virtual environment (`venv/`) lives inside the project folder on the D: drive. It is activated in VS Code with `venv\Scripts\activate`. Colab uses its own environment and reinstalls dependencies at the start of each session via the Colab setup notebook.

---

## Files

### `1_infrastructure/test_connections.py`

Tests all five external dependencies and prints a pass/fail summary. All five must pass before proceeding to Section 2.

| Service | Test | Notes |
|---|---|---|
| Groq | Sends a minimal prompt, checks response | Model must be non-decommissioned |
| NewsAPI | Fetches one business headline | Free tier: 100 requests/day |
| yfinance | Downloads 2 days of AAPL OHLCV | No API key required |
| pandas-ta | Computes RSI(14) on 30 days of AAPL | Verifies indicator pipeline |
| filterpy | Initialises an EnsembleKalmanFilter | Verifies Kalman stack |

### `1_infrastructure/test_tradingagents.py`

Confirms that the TradingAgents framework can initialise and make a successful call through the Groq client. This is a separate test from the raw Groq connection test because it exercises the monkey-patched `llm_clients/factory.py` and `openai_client.py` modifications.

---

## Dependencies

Full list installed via `pip install` into the venv:

```
langchain
langchain-groq
groq
openai
yfinance
pandas
pandas-ta
filterpy
quantstats
python-dotenv
newsapi-python
xgboost
scikit-learn
transformers
torch
stable-baselines3
sb3-contrib
gymnasium
optuna
shap
feedparser
```

---

## Configuration

### `.env`

Stores API keys. Never committed to version control. Add to `.gitignore` immediately on setup.

```
GROQ_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
HF_TOKEN=your_key_here
```

`HF_TOKEN` is optional but increases HuggingFace API rate limits. The system functions without it.

### `TradingAgents/tradingagents/default_config.py`

Modified to route LLM calls through Groq rather than OpenAI. Changes made:

```python
"llm_provider": "groq",
"deep_think_llm": "llama-3.3-70b-versatile",
"quick_think_llm": "llama-3.3-70b-versatile",
"backend_url": "https://api.groq.com/openai/v1",
```

### `TradingAgents/tradingagents/llm_clients/openai_client.py`

Added Groq to `_PROVIDER_CONFIG`:

```python
"groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
```

### `TradingAgents/tradingagents/llm_clients/factory.py`

Added `"groq"` to the routing condition:

```python
if provider_lower in ("openai", "ollama", "openrouter", "groq"):
    return OpenAIClient(model, base_url, provider=provider_lower, **kwargs)
```

Groq exposes an OpenAI-compatible API endpoint, which is why it can use the existing `OpenAIClient` class without any further modification.

---

## Revision History

| Revision | Description |
|---|---|
| Initial | Set up venv, installed deps, configured .env |
| R1 | Discovered `llama3-70b-8192` decommissioned by Groq. Replaced with `llama-3.3-70b-versatile` in all test files and config |
| R2 | Discovered `llama3-8b-8192` also decommissioned. Replaced with `llama-3.1-8b-instant` in fallback chain |
| R3 | Added `HF_TOKEN` to `.env` after FinBERT integration in Section 3 required it |
| R4 | Added `feedparser`, `xgboost`, `scikit-learn`, `transformers`, `torch`, `stable-baselines3`, `gymnasium`, `optuna`, `shap` to requirements as Sections 3-5 were built |

---

## Known Issues

**Pylance import warnings in VS Code.** Both `test_tradingagents.py` and files in `2_agents/` show red squiggles under `tradingagents.*` imports. This is a Pylance static analysis limitation caused by the `sys.path.insert()` pattern used to locate the TradingAgents package. The imports resolve correctly at runtime. The warnings can be suppressed by adding a `pyrightconfig.json` to the project root, but this has no functional impact and has not been done.

**CUDA version on Colab.** If the Colab runtime is recycled, the torch version installed may not match the CUDA version available on the assigned GPU. Running `!nvidia-smi` and checking the CUDA version before installing torch resolves this if it occurs.
