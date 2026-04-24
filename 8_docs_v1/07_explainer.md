# Section 7: Groq Explainer

**Status:** Production  
**Location:** `7_explainer/`  
**Last updated:** April 2026

---

## Overview

Section 7 is the post-hoc explanation layer. It uses Groq-hosted LLMs to translate system outputs — signals, regime, portfolio state, RL allocations — into plain English briefings for internal review.

The explainer does not make trading decisions. All decisions are made upstream by XGBoost, the Kalman ensemble, and the RL agent. The explainer's role is to explain those decisions and flag degraded conditions to the risk manager.

---

## Files

| File | Purpose |
|------|---------|
| `groq_explainer.py` | Full pipeline: state collector, Groq client, four briefing modes. |
| `daily_report_DATE.txt` | Full daily briefing saved per run. |
| `explainer_log_DATE.jsonl` | Structured log of all Groq calls per day. |
| `full_log.jsonl` | Complete session log across all modes. |

---

## Dependencies

- Section 4: `SignalEngine.get_state()` — live proba_buy, Hurst, RSI, drift AUC, sentiment
- Section 5: `equity_curve_v2.csv` — portfolio state
- Section 5: `trade_log_v2.csv` — recent trades for audit
- Section 5: `metrics.py` — performance summary
- Section 6: `rl_agent.py` — live portfolio weights
- Groq API key in `.env` as `GROQ_API_KEY`
- `pip install groq`

---

## Quick Start

```bash
# install
pip install groq python-dotenv

# add to .env
GROQ_API_KEY=your_key_here

# run
python 7_explainer/groq_explainer.py
```

---

## Architecture

### Data flow

```
SignalEngine.get_state()     -> live signals per ticker (proba, Hurst, RSI, drift AUC, sentiment)
equity_curve_v2.csv          -> portfolio state (equity, drawdown, rolling Sharpe, kill_switch)
trade_log_v2.csv             -> recent trades for audit mode
metrics.run_metrics()        -> performance summary (Sharpe, Sortino, win rate, profit factor)
rl_agent.predict_weights()   -> live portfolio weights from trained SAC model
         |
         v
SystemStateCollector.collect()   -> unified state dict
         |
         v
TradingExplainer                 -> Groq API calls -> plain English output
```

### Model fallback chain

Groq calls attempt models in order. If a model hits a daily or per-minute token limit, the next model is tried automatically.

```
1. llama-3.3-70b-versatile          (preferred)
2. meta-llama/llama-4-scout-17b-16e-instruct
3. llama-3.1-8b-instant             (fallback)
```

---

## Briefing Modes

### 1. Daily briefing

One briefing per ticker. Approximately 300 tokens each. Covers signal layer, portfolio risk context, and RL allocation rationale.

Inputs used: signal, proba_buy, confidence, drift AUC, Hurst, RSI, ESN signal, FinBERT sentiment, vol regime, portfolio equity, drawdown, rolling Sharpe, kill switch status, regime, RL weight.

### 2. Trade audit

One briefing per trade. Approximately 200 tokens. Explains why a specific trade was executed and flags any risk conditions present at execution time.

Inputs used: ticker, action, price, shares, proba, blended weight, regime, walk-forward window index, drift AUC, Hurst, kill switch.

### 3. Regime warning

Fired automatically when max drift AUC across tickers exceeds 0.70 or kill switch is active. Approximately 200 tokens. Explains what the anomaly metrics indicate and what the system is doing in response.

### 4. Weekly summary

One summary per session. Approximately 500 tokens. Covers performance vs baseline, regime and risk environment, RL allocation logic, and outlook for the following week.

---

## Data Freshness

`MODEL_CUTOFF` in `groq_explainer.py` tracks the date the XGBoost model was last trained on. Every prompt includes a data freshness note:

- If `days_stale > 180`: explicit warning included in every prompt. Groq is instructed to flag signal degradation.
- If drift AUC > 0.70: signal quality label changes to DEGRADED or VERY LOW. Groq is instructed to note reduced reliability.

`MODEL_CUTOFF` is updated automatically by `scheduler.py` after each monthly backtest retrain.

---

## Signal Quality Labels

| Drift AUC | Label |
|-----------|-------|
| >= 0.90 | VERY LOW — market conditions far outside training data |
| >= 0.70 | DEGRADED — possible regime shift |
| >= 0.55 | MODERATE — some drift detected |
| < 0.55 | GOOD |

---

## Token Budget

| Mode | Tokens per call | Calls per day |
|------|----------------|---------------|
| Daily briefing | ~300 | 6 (one per ticker) |
| Trade audit | ~200 | N trades |
| Regime warning | ~200 | 0-1 |
| Weekly summary | ~500 | 1 |
| **Total** | **~3,500** | **Well within 100k/day free tier** |

---

## Outputs

```
7_explainer/
  daily_report_YYYY-MM-DD.txt    # full daily report, all tickers
  explainer_log_YYYY-MM-DD.jsonl # structured log per Groq call
  full_log.jsonl                 # complete session log
```

Each `.jsonl` entry contains: date, mode, ticker, and the full Groq response text. This log is the audit trail for all explanations.

---

## Scheduler Integration

When run via `scheduler.py`, the explainer runs silently at 18:00 on weekdays. All output goes to log files, not stdout. The daily report file is the primary deliverable.

```
scheduler.py
  18:00 weekdays -> job_daily_explainer()
                    -> load RL weights from best_model
                    -> load SignalEngine
                    -> collect state
                    -> run_daily() (silent)
                    -> trade_audit() on last trade
                    -> weekly_summary()
                    -> save_log()
```

---

## Revision History

| Version | Change |
|---------|--------|
| v1.0 | Initial explainer. Four modes: daily briefing, trade audit, regime warning, weekly summary. |
| v1.1 | RL weights hardcoded. Only loaded from backtest CSV, not live model. |
| v1.2 | RL weights replaced with live inference from `rl_agent.predict_weights()`. |
| v1.3 | Equity file fixed. Was reading backtest CSV with historical date filter. Changed to always use last row of RL equity curve. |
| v1.4 | Data freshness system added. MODEL_CUTOFF tracked. Staleness warning injected into all prompts when > 180 days. |
| v1.5 | Signal quality labels added. Drift AUC mapped to GOOD/MODERATE/DEGRADED/VERY LOW. Included in all prompts. |
| v1.6 | as_of_date added to portfolio state. Groq now knows the equity data is from a specific past date, not today. |
| v1.7 | Auto trade file fallback. Tries trade_log_v2.csv, then trade_log.csv. No crash on missing file. |
| v1.8 | ESN signal, vol regime, FinBERT sentiment added to daily briefing prompt inputs. |
| v1.9 | Scheduler integration. Silent mode (verbose=False) when called from scheduler. Output to log files only. |
| v2.0 | MODEL_CUTOFF auto-update added to scheduler.py. Patched after each monthly retrain. |

---

## Known Issues

| Issue | Status |
|-------|--------|
| FinBERT returns identical sentiment for all tickers in some sessions | Under investigation. Likely NewsAPI returning market-wide headlines rather than ticker-specific. |
| Drift AUC is 1.0 for all tickers when model is stale | Expected. Resolved by monthly retrain via scheduler. |
| Portfolio equity reflects last RL training date, not today | By design. A live portfolio tracker is required for real-time equity. Not yet built. |
| Trade audit skips if trade_log_v2.csv is not found | Handled gracefully. Audit is skipped, other modes continue. |
