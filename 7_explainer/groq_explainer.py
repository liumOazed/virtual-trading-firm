"""
groq_explainer.py  (v2 — high fidelity)
========================================
Layer 7 — Post-hoc explainer for VIRTUAL_TRADING_FIRM.
Uses Groq LLM to generate human-readable briefings from system outputs.

Modes
-----
1. daily_briefing  — per-ticker signal + RL allocation
2. trade_audit     — why a specific trade fired
3. regime_warning  — drift / kill-switch alert
4. weekly_summary  — full week in plain English

All ML decisions made UPSTREAM. Groq only explains them.
Token budget: ~3,000/day — well within 100k free tier.
"""

import os
import sys
import json
import time
import warnings
from datetime import date
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "4_signals"))
sys.path.insert(0, os.path.join(ROOT, "5_backtesting"))
sys.path.insert(0, os.path.join(ROOT, "6_rl_agent"))

from dotenv import load_dotenv
load_dotenv()

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️  groq not installed. Run: pip install groq")

RESULTS_DIR   = "5_backtesting/results"
RL_DIR        = "6_rl_agent"
EXPLAINER_DIR = "7_explainer"
os.makedirs(EXPLAINER_DIR, exist_ok=True)

MODEL_CHAIN = [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
]

# Cutoff = last backtest end date — used for data freshness warnings
MODEL_CUTOFF = "2025-03-28"

COMPANY_NAMES = {
    "AAPL": "Apple",
    "NVDA": "NVIDIA",
    "MSFT": "Microsoft",
    "SPY":  "S&P 500 ETF",
    "QQQ":  "Nasdaq 100 ETF",
    "TSLA": "Tesla",
}


# ════════════════════════════════════════════════════════════════════════════
# 1.  GROQ CLIENT
# ════════════════════════════════════════════════════════════════════════════

class GroqClient:

    def __init__(self, api_key: Optional[str] = None):
        if not GROQ_AVAILABLE:
            raise RuntimeError("groq package not installed.")
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    def complete(self,
                 system_prompt: str,
                 user_prompt:   str,
                 max_tokens:    int   = 400,
                 temperature:   float = 0.3) -> str:

        for model in MODEL_CHAIN:
            try:
                resp = self.client.chat.completions.create(
                    model       = model,
                    messages    = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    max_tokens  = max_tokens,
                    temperature = temperature,
                )
                return resp.choices[0].message.content.strip()

            except Exception as e:
                err = str(e)
                if "tokens per day" in err or "TPD" in err:
                    print(f"   🔄 Daily limit on {model} → next")
                    continue
                elif "tokens per minute" in err or "TPM" in err:
                    print(f"   ⏳ TPM limit — waiting 15s")
                    time.sleep(15)
                    try:
                        resp = self.client.chat.completions.create(
                            model       = model,
                            messages    = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user",   "content": user_prompt},
                            ],
                            max_tokens  = max_tokens,
                            temperature = temperature,
                        )
                        return resp.choices[0].message.content.strip()
                    except Exception:
                        continue
                else:
                    print(f"   ❌ {model}: {err[:80]} → next")
                    continue

        return "⚠️ All Groq models unavailable."


# ════════════════════════════════════════════════════════════════════════════
# 2.  SYSTEM PROMPT
# ════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a senior quantitative analyst at a systematic trading firm.
Your role is to explain trading system outputs in plain English for internal review.

The system uses:
- XGBoost + ESN (Echo State Network) for price prediction
- Kalman filter ensemble for dynamic signal weighting
- 4-state regime detection (Bull/Bear × Trending/MeanRev) via Hurst exponent
- SAC + GRU Reinforcement Learning agent for portfolio allocation
- Walk-forward validation with per-window model retraining
- Kill-switch: halts trading when 20d rolling Sharpe < -1.25 AND drawdown > 4%

Rules:
- Be concise. Max 3 sentences per section unless instructed otherwise.
- Never recommend buying or selling — only explain what the system decided.
- Always cite the specific numbers provided.
- When drift AUC > 0.70, explicitly flag signal degradation.
- When data freshness warning is present, acknowledge model staleness.
- Write for a quant trader, not a retail investor."""


# ════════════════════════════════════════════════════════════════════════════
# 3.  STATE COLLECTOR
# ════════════════════════════════════════════════════════════════════════════

class SystemStateCollector:

    def __init__(self, tickers: List[str]):
        self.tickers = tickers

    def collect(self,
                target_date:    str,
                signal_engine=  None,
                rl_weights:     Optional[Dict[str, float]] = None,
                equity_file:    str = f"{RL_DIR}/rl_equity_curve.csv",
                trade_file:     str = f"{RESULTS_DIR}/trade_log.csv",
                metrics_result: Optional[Dict] = None) -> Dict:

        import pandas as pd
        import numpy as np

        today     = date.today().strftime("%Y-%m-%d")
        days_stale = (
            pd.Timestamp(today) - pd.Timestamp(MODEL_CUTOFF)
        ).days

        state = {
            "date":          target_date,
            "tickers":       self.tickers,
            "signals":       {},
            "rl_weights":    rl_weights or {},
            "portfolio":     {},
            "regime":        "Unknown",
            "metrics":       {},
            "recent_trades": [],
            "data_freshness": {
                "model_cutoff":  MODEL_CUTOFF,
                "days_stale":    days_stale,
                "warning":       days_stale > 180,   # warn after 6 months
            },
        }

        # ── live signals ──────────────────────────────────────────────────
        if signal_engine is not None:
            for ticker in self.tickers:
                try:
                    packet = signal_engine.get_state(ticker, target_date)
                    if packet.get("status") == "success":
                        sv = packet.get("state_vector", {})
                        state["signals"][ticker] = {
                            "signal":          packet.get("signal", "WAIT"),
                            "proba_buy":       round(sv.get("proba_buy",  0.5), 4),
                            "confidence":      round(packet.get("confidence", 0.5), 4),
                            "drift_auc":       round(packet.get("drift_auc", 0.5), 4),
                            "hurst":           round(sv.get("hurst",       0.5), 3),
                            "rsi":             round(sv.get("rsi",         50),  1),
                            "sentiment":       round(sv.get("sentiment",   0.0), 3),
                            "esn_signal":      round(sv.get("esn_signal",  0.0), 4),
                            "vol_regime":      round(sv.get("vol_regime",  0.0), 4),
                            "regime_warning":  packet.get("regime_warning", False),
                            "signal_reliable": packet.get("drift_auc", 0.5) < 0.70,
                        }
                        if ticker == "SPY":
                            state["regime"] = (
                                "Bear-Trending"
                                if sv.get("hurst", 0.5) > 0.55
                                and sv.get("proba_buy", 0.5) < 0.5
                                else "Bull-Trending"
                            )
                    else:
                        raise ValueError(packet.get("message", "unknown"))

                except Exception as e:
                    print(f"  ⚠️  Signal failed {ticker}: {e}")
                    state["signals"][ticker] = {
                        "signal": "UNKNOWN", "proba_buy": 0.5,
                        "confidence": 0.0,   "drift_auc": 1.0,
                        "signal_reliable": False,
                    }

        # ── portfolio state (from RL equity curve — last known) ───────────
        try:
            eq  = pd.read_csv(equity_file)
            eq["date"] = eq["date"].astype(str)
            row = eq.iloc[-1]                          # always last known row
            equity_arr = eq["equity"].values
            peak = equity_arr.max()
            curr = float(row["equity"])

            rets = pd.Series(equity_arr).pct_change().dropna().values
            roll_sh = 0.0
            if len(rets) >= 20:
                last20  = rets[-20:]
                roll_sh = (last20.mean() / (last20.std() + 1e-9)) * (252**0.5)

            state["portfolio"] = {
                "equity":            round(curr, 2),
                "as_of_date":        row["date"],
                "drawdown":          round((peak - curr) / max(peak, 1) * 100, 2),
                "regime":            row.get("regime", "Unknown"),
                "roll_sharpe_20d":   round(roll_sh, 3),
                "kill_switch":       bool(roll_sh < -1.25 and
                                         (peak - curr) / max(peak, 1) > 0.04),
            }
            if state["regime"] == "Unknown":
                state["regime"] = row.get("regime", "Unknown")

        except Exception as e:
            print(f"  ⚠️  Equity file failed: {e}")
            state["portfolio"] = {
                "equity": 0, "as_of_date": MODEL_CUTOFF,
                "drawdown": 0, "regime": "Unknown",
                "roll_sharpe_20d": 0, "kill_switch": False,
            }

        # ── recent trades ─────────────────────────────────────────────────
        # check both possible filenames
        for tf in [trade_file,
                   trade_file.replace("_v2", ""),
                   f"{RESULTS_DIR}/trade_log_v2.csv",
                   f"{RESULTS_DIR}/trade_log.csv"]:
            if os.path.exists(tf):
                try:
                    trades = pd.read_csv(tf)
                    trades["date"] = trades["date"].astype(str)
                    state["recent_trades"] = (
                        trades.tail(10).to_dict("records")
                    )
                    break
                except Exception:
                    continue

        # ── metrics ───────────────────────────────────────────────────────
        if metrics_result:
            core = metrics_result.get("core", {})
            trade = metrics_result.get("trade", {})
            state["metrics"] = {
                "total_return": core.get("total_return",  0),
                "sharpe":       core.get("sharpe",        0),
                "sortino":      core.get("sortino",       0),
                "max_drawdown": core.get("max_drawdown",  0),
                "win_rate":     trade.get("win_rate",     0),
                "profit_factor":trade.get("profit_factor",0),
            }

        return state


# ════════════════════════════════════════════════════════════════════════════
# 4.  EXPLAINER
# ════════════════════════════════════════════════════════════════════════════

class TradingExplainer:

    def __init__(self, api_key: Optional[str] = None):
        self.groq = GroqClient(api_key)
        self.log:  List[Dict] = []

    def _freshness_note(self, state: Dict) -> str:
        df = state.get("data_freshness", {})
        if df.get("warning"):
            return (f"DATA FRESHNESS WARNING: Model trained to "
                    f"{df['model_cutoff']} ({df['days_stale']} days ago). "
                    f"Signals may be degraded — treat with caution.")
        return (f"Model trained to {df.get('model_cutoff', 'unknown')}. "
                f"Signal reliability depends on drift AUC.")

    def _signal_quality(self, sig: Dict) -> str:
        auc = sig.get("drift_auc", 0.5)
        if auc >= 0.90:
            return "VERY LOW — market conditions far outside training data"
        elif auc >= 0.70:
            return "DEGRADED — possible regime shift"
        elif auc >= 0.55:
            return "MODERATE — some drift detected"
        return "GOOD"

    def _log(self, mode: str, ticker: str, text: str, dt: str):
        entry = {"date": dt, "mode": mode, "ticker": ticker, "text": text}
        self.log.append(entry)
        path = f"{EXPLAINER_DIR}/explainer_log_{dt}.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ── 1. DAILY BRIEFING ────────────────────────────────────────────────

    def daily_briefing(self, ticker: str, state: Dict,
                       verbose: bool = True) -> str:

        sig    = state["signals"].get(ticker, {})
        weight = state["rl_weights"].get(ticker, 0.0)
        port   = state["portfolio"]

        prompt = f"""
Daily briefing for {ticker} ({COMPANY_NAMES.get(ticker, ticker)}) — {state['date']}.

{self._freshness_note(state)}
Signal quality: {self._signal_quality(sig)}

SIGNAL LAYER:
- Decision: {sig.get('signal', 'N/A')}
- Buy probability: {sig.get('proba_buy', 'N/A')}
- Model confidence: {sig.get('confidence', 'N/A')}
- Drift AUC: {sig.get('drift_auc', 'N/A')} (>0.70 = degraded signal)
- Hurst exponent: {sig.get('hurst', 'N/A')} (>0.55 trending, <0.45 mean-reverting)
- RSI: {sig.get('rsi', 'N/A')}
- ESN signal: {sig.get('esn_signal', 'N/A')}
- FinBERT sentiment: {sig.get('sentiment', 'N/A')} (-1 bearish, +1 bullish)
- Vol regime: {sig.get('vol_regime', 'N/A')}

PORTFOLIO CONTEXT (as of {port.get('as_of_date', 'unknown')}):
- Equity: ${port.get('equity', 0):,.2f}
- Drawdown from peak: {port.get('drawdown', 0)}%
- 20d rolling Sharpe: {port.get('roll_sharpe_20d', 0)}
- Kill switch: {port.get('kill_switch', False)}
- Regime: {state.get('regime', 'Unknown')}

RL ALLOCATION:
- Recommended weight: {weight:.1%}

Write exactly 3 sentences:
1. What the signal layer is showing and why (cite Hurst, proba, drift AUC)
2. Portfolio risk context and whether the kill switch is relevant
3. Why the RL agent allocated {weight:.1%} given this context
If signal quality is degraded, explicitly note reduced reliability.
"""
        text = self.groq.complete(SYSTEM_PROMPT, prompt, max_tokens=320)
        if verbose:
            print(f"\n📊 {ticker} [{state['date']}]")
            print("─" * 50)
            print(text)
        self._log("daily_briefing", ticker, text, state["date"])
        return text

    # ── 2. TRADE AUDIT ───────────────────────────────────────────────────

    def trade_audit(self, trade: Dict, state: Dict,
                    verbose: bool = True) -> str:

        ticker = trade.get("ticker", "UNKNOWN")
        action = trade.get("action", "UNKNOWN")
        sig    = state["signals"].get(ticker, {})

        prompt = f"""
Trade audit — {action} {ticker} on {trade.get('date', 'unknown')}.

EXECUTION DETAILS:
- Action: {action}
- Price: ${trade.get('price', 'N/A')}
- Shares: {trade.get('shares', 'N/A')}
- Signal probability at execution: {trade.get('proba', 'N/A')}
- Blended Kalman+competition weight: {trade.get('weight', 'N/A')}
- Regime at execution: {trade.get('regime', 'N/A')}
- Walk-forward window: W{trade.get('window', 'N/A')}

SIGNAL CONTEXT (current):
- Drift AUC: {sig.get('drift_auc', 'N/A')}
- Hurst: {sig.get('hurst', 'N/A')}
- Kill switch at time: {state['portfolio'].get('kill_switch', False)}
- Signal quality: {self._signal_quality(sig)}

Write exactly 2 sentences:
1. Why the system executed this trade (signal threshold, regime, weight logic)
2. Risk flags present at execution time
"""
        text = self.groq.complete(SYSTEM_PROMPT, prompt, max_tokens=220)
        if verbose:
            print(f"\n📝 AUDIT — {action} {ticker} [{trade.get('date','?')}]")
            print("─" * 50)
            print(text)
        self._log("trade_audit", ticker, text, trade.get("date", state["date"]))
        return text

    # ── 3. REGIME WARNING ────────────────────────────────────────────────

    def regime_warning(self, state: Dict, drift_auc: float,
                       verbose: bool = True) -> str:

        port = state["portfolio"]
        df   = state.get("data_freshness", {})

        prompt = f"""
ALERT: Trading system anomaly detected on {state['date']}.

DETECTION:
- Max adversarial drift AUC: {drift_auc:.3f} (threshold 0.70)
- Model trained to: {df.get('model_cutoff', 'unknown')} ({df.get('days_stale', 0)} days ago)
- Current regime: {state.get('regime', 'Unknown')}
- 20d rolling Sharpe: {port.get('roll_sharpe_20d', 0)}
- Current drawdown: {port.get('drawdown', 0)}%
- Kill switch: {port.get('kill_switch', False)}

Write exactly 2 sentences:
1. What the drift AUC and staleness indicate about signal reliability today
2. What the system is doing in response and what the risk manager should watch
"""
        text = self.groq.complete(SYSTEM_PROMPT, prompt, max_tokens=220)
        if verbose:
            print(f"\n⚠️  REGIME WARNING [{state['date']}]")
            print("─" * 50)
            print(text)
        self._log("regime_warning", "MARKET", text, state["date"])
        return text

    # ── 4. WEEKLY SUMMARY ────────────────────────────────────────────────

    def weekly_summary(self, state: Dict,
                       verbose: bool = True) -> str:

        m     = state["metrics"]
        port  = state["portfolio"]
        top3  = sorted(state["rl_weights"].items(),
                       key=lambda x: x[1], reverse=True)[:3]
        top3s = ", ".join([f"{t} ({w:.1%})" for t, w in top3])
        trades= state["recent_trades"]
        nb    = sum(1 for t in trades if t.get("action") == "BUY")
        ns    = sum(1 for t in trades if t.get("action") == "SELL")
        df    = state.get("data_freshness", {})

        # avg signal quality across tickers
        aucs = [v.get("drift_auc", 0.5) for v in state["signals"].values()]
        avg_auc = sum(aucs) / len(aucs) if aucs else 0.5

        prompt = f"""
Weekly performance summary — {state['date']}.

{self._freshness_note(state)}
Average signal drift AUC this week: {avg_auc:.3f}

PERFORMANCE (backtest baseline):
- Total return: {m.get('total_return', 0)}%
- Sharpe: {m.get('sharpe', 0)}
- Sortino: {m.get('sortino', 0)}
- Max drawdown: {m.get('max_drawdown', 0)}%
- Win rate: {m.get('win_rate', 0)}%
- Profit factor: {m.get('profit_factor', 0)}

LIVE PORTFOLIO (as of {port.get('as_of_date', 'unknown')}):
- Equity: ${port.get('equity', 0):,.2f}
- Drawdown: {port.get('drawdown', 0)}%
- 20d Sharpe: {port.get('roll_sharpe_20d', 0)}
- Regime: {state.get('regime', 'Unknown')}
- Kill switch: {port.get('kill_switch', False)}

ACTIVITY:
- Recent trades: {nb} buys, {ns} sells
- Top RL allocations: {top3s}

Write exactly 4 sentences:
1. Performance vs backtest baseline, noting any divergence
2. Regime and risk environment — is the kill switch relevant?
3. RL allocation logic given current signals
4. Key concern for next week, explicitly noting if model retraining is overdue
"""
        text = self.groq.complete(SYSTEM_PROMPT, prompt, max_tokens=520)
        if verbose:
            print(f"\n📅 WEEKLY SUMMARY [{state['date']}]")
            print("─" * 50)
            print(text)
        self._log("weekly_summary", "PORTFOLIO", text, state["date"])
        return text

    # ── 5. FULL DAILY RUN ────────────────────────────────────────────────

    def run_daily(self, state: Dict,
                  verbose: bool = True) -> Dict[str, str]:

        results = {}
        df      = state.get("data_freshness", {})

        print(f"\n{'='*55}")
        print(f"  GROQ EXPLAINER — {state['date']}")
        print(f"  Regime : {state['regime']}")
        print(f"  Model  : trained to {df.get('model_cutoff')} "
              f"({df.get('days_stale')} days ago)")
        print(f"{'='*55}")

        # regime warning if drift high or kill switch active
        max_drift   = max(
            (v.get("drift_auc", 0.5) for v in state["signals"].values()),
            default=0.5
        )
        kill_active = state["portfolio"].get("kill_switch", False)

        if max_drift > 0.70 or kill_active:
            results["regime_warning"] = self.regime_warning(
                state, max_drift, verbose
            )
            time.sleep(2)

        # per-ticker briefings
        for ticker in state["tickers"]:
            if ticker not in state["signals"]:
                continue
            results[ticker] = self.daily_briefing(ticker, state, verbose)
            time.sleep(1)

        # save report
        path = f"{EXPLAINER_DIR}/daily_report_{state['date']}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"VIRTUAL TRADING FIRM — DAILY REPORT\n")
            f.write(f"Date: {state['date']} | Regime: {state['regime']}\n")
            f.write(f"Model cutoff: {df.get('model_cutoff')} "
                    f"({df.get('days_stale')} days ago)\n")
            f.write("=" * 55 + "\n\n")
            for key, text in results.items():
                f.write(f"[{key}]\n{text}\n\n")
        print(f"\n  💾 Report → {path}")
        return results

    def save_log(self, path: Optional[str] = None):
        path = path or f"{EXPLAINER_DIR}/full_log.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for entry in self.log:
                f.write(json.dumps(entry) + "\n")
        print(f"  💾 Log   → {path}")


# ════════════════════════════════════════════════════════════════════════════
# 5.  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pandas as pd
    import traceback

    TODAY   = date.today().strftime("%Y-%m-%d")
    TICKERS = ["AAPL", "NVDA", "MSFT", "SPY", "QQQ", "TSLA"]

    # ── 1. RL weights from trained model ─────────────────────────────────
    RL_WEIGHTS = {}
    try:
        from rl_agent import RLTradingAgent, load_backtest_outputs

        obs_matrix, price_matrix, dates, n_features, sb = load_backtest_outputs(
            tickers     = TICKERS,
            equity_file = f"{RESULTS_DIR}/equity_curve.csv",
            trade_file  = f"{RESULTS_DIR}/trade_log.csv",
            price_pkl   = f"{RESULTS_DIR}/price_data.pkl",
        )
        agent = RLTradingAgent(tickers=TICKERS, device="cpu")
        agent.load(f"{RL_DIR}/best_model/best_model")
        RL_WEIGHTS = agent.predict_weights(obs_matrix[-agent.seq_len:])
        print("✅ RL weights:")
        for t, w in RL_WEIGHTS.items():
            print(f"   {t:<6} → {w:.2%}")

    except Exception:
        traceback.print_exc()
        RL_WEIGHTS = {t: 1/len(TICKERS) for t in TICKERS}
        print("⚠️  RL fallback: equal weights")

    # ── 2. Signal engine ─────────────────────────────────────────────────
    signal_engine = None
    try:
        from signal_engine import SignalEngine
        signal_engine = SignalEngine()
        print("✅ SignalEngine loaded")
    except Exception as e:
        print(f"⚠️  SignalEngine unavailable: {e}")

    # ── 3. Metrics ───────────────────────────────────────────────────────
    metrics_result = None
    try:
        from metrics import run_metrics
        metrics_result = run_metrics(save_chart=False)
        print("✅ Metrics loaded")
    except Exception as e:
        print(f"⚠️  Metrics unavailable: {e}")

    # ── 4. Collect state ─────────────────────────────────────────────────
    collector = SystemStateCollector(TICKERS)
    try:
        state = collector.collect(
            target_date    = TODAY,
            signal_engine  = signal_engine,
            rl_weights     = RL_WEIGHTS,
            equity_file    = f"{RL_DIR}/rl_equity_curve.csv",
            trade_file     = f"{RESULTS_DIR}/trade_log.csv",
            metrics_result = metrics_result,
        )
    except Exception:
        traceback.print_exc()
        raise

    # ── 5. Run explainer ─────────────────────────────────────────────────
    explainer = TradingExplainer()

    explainer.run_daily(state, verbose=True)

    if state["recent_trades"]:
        explainer.trade_audit(state["recent_trades"][-1], state, verbose=True)

    explainer.weekly_summary(state, verbose=True)

    explainer.save_log()

    print(f"\n✅ Done → {EXPLAINER_DIR}/")