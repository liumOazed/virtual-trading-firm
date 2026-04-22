# Section 2: Agent Architecture

**Status:** Complete (superseded by ML pipeline for production use)  
**Estimated Time:** 2-3 days original build; ongoing maintenance not required  
**Location:** `2_agents/`, `TradingAgents/`

---

## Overview

Section 2 covers the integration and testing of the TradingAgents multi-agent framework as the initial approach to signal generation. The framework was successfully integrated and confirmed working. However, the architecture was superseded by the ML pipeline built in Sections 3 and 4 due to token consumption constraints on Groq's free tier.

The TradingAgents code remains in the repository and can be used directly if a paid Groq tier or alternative LLM provider is available. The agents, debate loop, and memory system are all functional.

---

## TradingAgents Framework

TradingAgents is an open-source multi-agent trading framework from Tauric Research. Repository: `https://github.com/TauricResearch/TradingAgents`

The framework provides a LangGraph-based workflow with the following agents:

| Agent | Role | File |
|---|---|---|
| Fundamentals Analyst | Pulls balance sheet, cashflow, income statement via yfinance | `agents/analysts/fundamentals_analyst.py` |
| Market Analyst | Computes and interprets technical indicators | `agents/analysts/market_analyst.py` |
| News Analyst | Fetches and summarises recent news | `agents/analysts/news_analyst.py` |
| Social Media Analyst | Analyses sentiment from social sources | `agents/analysts/social_media_analyst.py` |
| Bull Researcher | Argues for the bullish case | `agents/researchers/` |
| Bear Researcher | Argues for the bearish case | `agents/researchers/` |
| Research Manager | Judges the bull/bear debate | `agents/managers/` |
| Trader | Proposes final trade with sizing | `agents/trader/trader.py` |
| Aggressive Debator | Argues for high-risk interpretation | `agents/risk_mgmt/aggressive_debator.py` |
| Conservative Debator | Argues for low-risk interpretation | `agents/risk_mgmt/conservative_debator.py` |
| Neutral Debator | Provides balanced risk view | `agents/risk_mgmt/neutral_debator.py` |
| Portfolio Manager | Final decision maker | `agents/managers/portfolio_manager.py` |

The full workflow graph is defined in `graph/trading_graph.py`. It runs analysts sequentially, then passes all reports through the bull/bear debate loop, then through the risk debate, and finally produces a `BUY/HOLD/SELL` signal with detailed reasoning.

---

## Patches Applied

Two monkey-patches were written to reduce token consumption. Both are applied before `TradingAgentsGraph` is initialised and modify the framework's tool references in-place.

### pandas-ta Patch

**File:** `3_market_data/local_indicators.py`  
**Function:** `patch_tradingagents()`

The market analyst originally called `get_indicators` as a Groq tool call for each indicator separately. With 8 to 11 indicators per run, this consumed approximately 50,000 to 70,000 tokens before any reasoning began.

The patch replaces `get_indicators` with `get_indicators_local`, which computes all indicators locally via pandas-ta and returns a pre-formatted text summary. Groq receives a clean string of numbers and performs only the reasoning step.

Targets patched:
- `tradingagents.agents.utils.agent_utils.get_indicators`
- `tradingagents.agents.utils.technical_indicators_tools.get_indicators`

Token reduction: approximately 80 percent per market analyst run.

### NewsAPI Patch

**File:** `3_market_data/news_patch.py`  
**Function:** `patch_news()`

The news analyst originally used yfinance's news feed, which returns large amounts of unstructured text. The patch replaces both `get_news` and `get_global_news` with NewsAPI versions that return a maximum of five clean, structured articles.

The patch must target four locations to be effective, because Python captures the reference at import time:
- `tradingagents.agents.utils.agent_utils.get_news`
- `tradingagents.agents.utils.agent_utils.get_global_news`
- `tradingagents.agents.utils.news_data_tools.get_news`
- `tradingagents.agents.utils.news_data_tools.get_global_news`
- `tradingagents.agents.analysts.news_analyst.get_news`
- `tradingagents.agents.analysts.news_analyst.get_global_news`
- `TradingAgentsGraph._create_tool_nodes` (overridden to inject patched ToolNodes)

---

## Groq Rate Limit Handling

**File:** `2_agents/test_full_pipeline.py`

Groq's free tier enforces two independent limits:

- **TPM (tokens per minute):** Temporary. Resolved by waiting the exact number of seconds specified in the error response.
- **TPD (tokens per day):** Daily reset. Resolved by skipping to the next model in the fallback chain immediately.

The fallback chain in order:

1. `llama-3.3-70b-versatile` — best quality, 12,000 TPM limit
2. `meta-llama/llama-4-scout-17b-16e-instruct` — 30,000 TPM limit
3. `llama-3.1-8b-instant` — highest RPM, lowest quality

The `run_with_fallback()` function parses the error message to distinguish TPD from TPM errors and handles each appropriately.

---

## Files

### `2_agents/test_full_pipeline.py`

The primary test file for the full TradingAgents pipeline. Applies both patches, initialises the graph with a specified analyst subset, runs `propagate()` for a given ticker and date, and prints the results.

Configurable parameters at the top of the file:
- `MODEL_FALLBACK_CHAIN`: list of models to try in order
- `selected_analysts`: subset of analysts to run (reduces token usage during testing)
- `config`: overrides for `DEFAULT_CONFIG`

---

## Why This Section Was Superseded

The TradingAgents architecture requires 8 to 12 sequential LLM calls per trading day, with each call consuming several thousand tokens. Even with both patches applied, a full run across three analysts regularly exceeded Groq's free tier daily limit of 100,000 tokens during development.

The ML pipeline in Sections 3 and 4 replicates the analytical capabilities of each agent using local computation:

| TradingAgents Agent | ML Replacement |
|---|---|
| Market Analyst | pandas-ta (local, free, instant) |
| Fundamentals Analyst | yfinance direct fetch (local) |
| News Analyst | FinBERT + NewsAPI (1 API call per ticker per day) |
| Bull/Bear Debate | XGBoost signal model |
| Risk Debators | Kalman Filter dynamic stop-loss |
| Portfolio Manager | RL Agent (PPO position sizer) |

The ML pipeline makes one Groq call per day for plain-English explanation only, consuming approximately 300 tokens per ticker. This fits comfortably within the free tier.

---

## Revision History

| Revision | Description |
|---|---|
| Initial | Cloned TradingAgents, confirmed baseline run with OpenAI config |
| R1 | Patched `default_config.py` and `llm_clients/` to route through Groq |
| R2 | Wrote pandas-ta patch; token usage reduced from 70k to ~15k per run |
| R3 | Wrote NewsAPI patch; discovered patching must target module-level references, not just agent_utils |
| R4 | Added model fallback chain with TPD vs TPM error differentiation |
| R5 | Switched from `TradingAgentsGraph` running all analysts together to isolated per-analyst runs; prevented context accumulation across agents |
| R6 | Section superseded by ML pipeline after Groq TPD limit hit repeatedly during development |

---

## Known Issues

**Context accumulation.** When all analysts run in a single `TradingAgentsGraph` instance, each subsequent agent receives the full conversation history of all previous agents. By the time the Portfolio Manager runs, the context window contains the outputs of all preceding agents, regularly exceeding 60,000 tokens. The isolation approach (one graph per analyst) resolves this but produces separate signals that must be aggregated manually.

**Tool call formatting on small models.** `llama-3.1-8b-instant` and similar small models sometimes produce malformed tool call syntax when presented with complex multi-tool prompts. This manifests as a 400 `tool_use_failed` error. The fallback chain handles this by skipping to the next model, but it means the 8B model is effectively only useful for simple single-tool calls.
