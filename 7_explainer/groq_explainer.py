"""
groq_explainer.py  v3
======================
Layer 7 — Post-hoc audit and briefing for VIRTUAL_TRADING_FIRM.

Uses Groq LLM + local FinBERT to explain every trade decision,
generate daily briefings, and produce weekly summaries.

Architecture
------------
Post-hoc only — reads finished outputs from the engine:
  - 5_backtesting/results/trade_log.csv
  - 5_backtesting/results/equity_curve.csv
  - 6_rl_agent/results/signal_events_enriched.csv  (optional)

No live engine calls. No fake signal methods.

3 Modes
-------
1. trade_audit_batch  — audit all 338 closed round trips (run once)
2. daily_briefing     — what happened on a specific date
3. weekly_summary     — full week recap

FinBERT
-------
Runs locally via HuggingFace transformers.
Zero extra Groq tokens. ~0.5s per ticker on CPU.
News sourced from yfinance (free, no API key).

Model Routing
-------------
Batch (historical): llama-4-scout-17b   (500k TPD)
Daily/Weekly:       llama-3.3-70b       (100k TPD, best quality)
Fallback:           qwen/qwen3-32b      (500k TPD)
Emergency:          llama-3.1-8b-instant(500k TPD)

Token Budget
------------
Trade audits (338):   ~304k tokens  → 1 day on llama-4-scout
Daily briefs (1400):  ~840k tokens  → 2 days on llama-4-scout
Weekly sums  (280):   ~336k tokens  → 4 days on llama-3.3-70b
Daily live op:        ~5k tokens/day → trivial on llama-3.3-70b

Caching
-------
Every output is saved to 7_explainer/cache/ as a JSONL.
Batch runs skip already-completed entries — safe to interrupt/resume.
"""

import os
import sys
import json
import time
import warnings
import traceback
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv()

# ── directories ───────────────────────────────────────────────────────────
RESULTS_DIR  = "5_backtesting/results"
RL_DIR       = "6_rl_agent/results"
OUT_DIR      = "7_explainer"
CACHE_DIR    = os.path.join(OUT_DIR, "cache")
REPORT_DIR   = os.path.join(OUT_DIR, "reports")
for d in [OUT_DIR, CACHE_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Groq ─────────────────────────────────────────────────────────────────
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠  groq not installed — run: pip install groq")

# ── FinBERT ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(ROOT, "4_signals"))
try:
    from finbert_sentiment import get_sentiment as _finbert_get
    FINBERT_AVAILABLE = True
    print("  FinBERT sentiment ready ✓")
except ImportError:
    FINBERT_AVAILABLE = False
    _finbert_get = None
    print("  ⚠ finbert_sentiment.py not found — sentiment disabled")


# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════

# Model chain: batch uses scout, daily uses 70b, fallbacks follow
MODEL_BATCH  = "meta-llama/llama-4-scout-17b-16e-instruct"  # 500k TPD
MODEL_DAILY  = "llama-3.3-70b-versatile"                    # 100k TPD best quality
MODEL_FB1    = "qwen/qwen3-32b"                              # 500k TPD fallback
MODEL_FB2    = "llama-3.1-8b-instant"                        # 500k TPD emergency

COMPANY = {
    "AAPL": "Apple",      "NVDA": "NVIDIA",    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",  "AMZN": "Amazon",    "META": "Meta",
    "TSLA": "Tesla",      "AVGO": "Broadcom",  "TSM": "TSMC",
    "RACE": "Ferrari",    "QQQ":  "Nasdaq ETF","SPY": "S&P500 ETF",
    "XOM":  "ExxonMobil","CVX":  "Chevron",   "GLD": "Gold ETF",
    "PG":   "P&G",        "WMT":  "Walmart",
}

EXIT_REASONS = {
    "bear_heat_trim":  "HMM confirmed bear regime — trimmed highest-heat positions to bring heat below 30%",
    "tsla_peak_exit":  "TSLA-specific RSI momentum-loss exit — RSI was above 70, now falling 2 consecutive bars",
    "tsla_ext_exit":   "TSLA extension exit — price exceeded 120% above 200-day MA",
    "tsla_bear":       "TSLA bear regime exit — HMM flipped to Bear-Trending or Bear-Stress",
    "regime_exit":     "Sector not active in current HMM regime — position force-exited",
    "anchor_reentry":  "Anchor re-entry on bull flip — AAPL/QQQ deployed on regime transition",
    "trend_add":       "Pyramid add — position up 5%+ after 10+ bars, added 5% more",
    "mr_exit":         "Mean-reversion exit — Z-score recovered to target",
    "mr_stop":         "Mean-reversion stop — Z-score breakdown (relationship broken)",
}


# ══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT  (accurate to what we actually built)
# ══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a senior quantitative analyst auditing a systematic trading firm's decisions.

The system you are auditing uses:
- XGBoost (global model + sector-specific models) for BUY/SELL signals with confidence scores
- Gaussian HMM on SPY+IEF for regime detection: Bull-Trending, Bull-Stable, Bear-Stable, Bear-Trending, Bear-Stress
- Kalman ensemble for dynamic signal weighting across models each bar
- Filter competition (rolling Sharpe tournament) selecting winning signals per regime
- TSLA Strategy A: universal RSI momentum-loss exit (RSI was >70, now falling 2 bars)
- Regime-conditional sizing: 0.15x (early bull) → 0.22x (deep confirmed bull)
- Walk-forward: 9-month train / 1-month OOS, 67 windows, Oct 2020 → May 2026

Key exit mechanisms:
- bear_heat_trim: HMM confirmed bear, sells highest-heat positions until heat < 30%
- tsla_peak_exit: RSI momentum-loss from overbought on TSLA specifically
- regime_exit: position's sector not active in current regime
- anchor_reentry: AAPL+QQQ deployed immediately on bull regime flip

Backtest result: +102% total return / Sharpe 1.35 / Max DD -10.1% vs SPY +140% / DD -24.5%

Rules:
- Be concise and precise. Cite specific numbers provided.
- Use markdown tables for quantitative comparisons (max 1 table per response).
- Never recommend buying or selling — only explain system decisions.
- Write for a quantitative trader, not a retail investor.
- When a trade outcome was poor, say so clearly with the reason."""


# ══════════════════════════════════════════════════════════════════════════
# GROQ CLIENT  (with exponential backoff + model chain)
# ══════════════════════════════════════════════════════════════════════════

class GroqClient:

    def __init__(self, api_key: Optional[str] = None):
        if not GROQ_AVAILABLE:
            raise RuntimeError("groq package not installed.")
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self._call_count = 0
        self._token_count = 0

    def complete(
        self,
        prompt:      str,
        model:       str  = MODEL_DAILY,
        max_tokens:  int  = 400,
        temperature: float = 0.25,
    ) -> Tuple[str, int]:
        """
        Returns (text, tokens_used).
        Tries model chain on failures with exponential backoff.
        """
        chain = [model, MODEL_FB1, MODEL_FB2]
        if model not in chain:
            chain.insert(0, model)

        for attempt_model in chain:
            for retry in range(3):
                try:
                    resp = self.client.chat.completions.create(
                        model       = attempt_model,
                        messages    = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt},
                        ],
                        max_tokens  = max_tokens,
                        temperature = temperature,
                    )
                    text   = resp.choices[0].message.content.strip()
                    tokens = resp.usage.total_tokens
                    self._call_count  += 1
                    self._token_count += tokens
                    return text, tokens

                except Exception as e:
                    err = str(e)
                    if "tokens per day" in err.lower() or "tpd" in err.lower():
                        print(f"   🔄 Daily limit on {attempt_model} → next model")
                        break   # try next model in chain
                    elif "tokens per minute" in err.lower() or "rate" in err.lower() or "429" in err:
                        wait = (2 ** retry) * 10   # 10s, 20s, 40s
                        print(f"   ⏳ Rate limit — waiting {wait}s (retry {retry+1})")
                        time.sleep(wait)
                    else:
                        print(f"   ❌ {attempt_model}: {err[:80]}")
                        break

        return "⚠ All Groq models unavailable — try again later.", 0

    @property
    def usage(self) -> Dict:
        return {"calls": self._call_count, "tokens": self._token_count}



# ══════════════════════════════════════════════════════════════════════════
# DATA LOADER  (reads engine outputs, builds round trips)
# ══════════════════════════════════════════════════════════════════════════

class DataLoader:

    def __init__(
        self,
        trade_file:  str = f"{RESULTS_DIR}/trade_log.csv",
        equity_file: str = f"{RESULTS_DIR}/equity_curve.csv",
        events_file: str = f"{RL_DIR}/signal_events_enriched.csv",
    ):
        self.tl = pd.read_csv(trade_file,  parse_dates=["date"])
        self.eq = pd.read_csv(equity_file, parse_dates=["date"])
        self.eq = self.eq.sort_values("date").reset_index(drop=True)
        self.eq["date_str"] = self.eq["date"].dt.strftime("%Y-%m-%d")

        # Drawdown
        peak       = self.eq["equity"].cummax()
        self.eq["dd"] = ((peak - self.eq["equity"]) / peak.clip(lower=1)).fillna(0)

        # Equity index
        self._eq_idx = self.eq.set_index("date_str")
        if self._eq_idx.index.duplicated().any():
            self._eq_idx = self._eq_idx[~self._eq_idx.index.duplicated(keep="first")]

        # Enriched signal events (optional)
        self.events = None
        if os.path.exists(events_file):
            self.events = pd.read_csv(events_file)

        # Build round trips via FIFO
        self._trips: List[Dict] = []
        self._build_trips()

        print(f"  DataLoader: {len(self.tl)} trade rows | "
              f"{len(self._trips)} round trips | "
              f"{len(self.eq)} equity bars")

    def _build_trips(self):
        buy_q = {}
        for _, row in self.tl.sort_values("date").iterrows():
            tk = row["ticker"]
            d  = row["date"].strftime("%Y-%m-%d")
            if row["action"] == "BUY":
                buy_q.setdefault(tk, []).append({
                    "ticker":      tk,
                    "entry_date":  d,
                    "entry_price": round(float(row["price"]), 2),
                    "confidence":  round(float(row["proba"]) if not pd.isna(row["proba"]) else 0.5, 4),
                    "hmm_regime":  str(row.get("hmm_regime", "Unknown")),
                    "weight":      round(float(row["weight"]) if not pd.isna(row["weight"]) else 0.10, 4),
                    "entry_reason": (str(row.get("reason", ""))
                                     if str(row.get("reason", "")) not in ("nan", "", "None")
                                     else "xgboost_buy_signal"),
                    "shares":      round(float(row["shares"]), 4),
                })
            elif row["action"] == "SELL" and tk in buy_q and buy_q[tk]:
                b    = buy_q[tk].pop(0)
                ret  = (row["price"] / b["entry_price"] - 1)
                hold = max((row["date"] - pd.Timestamp(b["entry_date"])).days, 1)
                b.update({
                    "exit_date":    d,
                    "exit_price":   round(float(row["price"]), 2),
                    "exit_reason": (str(row.get("reason", ""))
                                    if str(row.get("reason", "")) not in ("nan", "", "None")
                                    else "xgboost_sell_signal"),
                    "return_pct":   round(ret * 100, 2),
                    "hold_days":    hold,
                    "ann_return":   round(ret * 252 / hold * 100, 2),
                })
                # Merge enriched features if available
                if self.events is not None:
                    match = self.events[
                        (self.events["ticker"] == tk) &
                        (self.events["entry_date"] == b["entry_date"])
                    ]
                    if not match.empty:
                        r = match.iloc[0]
                        b["rsi_at_entry"]   = float(r.get("rsi_at_entry", 50))
                        b["pct_above_200d"] = float(r.get("pct_above_200d", 0))
                        b["momentum_20d"]   = float(r.get("momentum_20d", 0))
                self._trips.append(b)

    @property
    def trips(self) -> List[Dict]:
        return self._trips

    def equity_on(self, date_str: str) -> Dict:
        if date_str in self._eq_idx.index:
            r = self._eq_idx.loc[date_str]
            return {
                "equity":  round(float(r["equity"]), 2),
                "dd":      round(float(r["dd"]) * 100, 2),
                "regime":  str(r.get("regime", "Unknown")),
            }
        return {"equity": 0, "dd": 0, "regime": "Unknown"}

    def trades_on(self, date_str: str) -> List[Dict]:
        mask = self.tl["date"].dt.strftime("%Y-%m-%d") == date_str
        rows = self.tl[mask].to_dict("records")
        return rows

    def trips_in_week(self, week_start: str) -> List[Dict]:
        ws = pd.Timestamp(week_start)
        we = ws + timedelta(days=4)
        return [
            t for t in self._trips
            if ws <= pd.Timestamp(t["entry_date"]) <= we
            or ws <= pd.Timestamp(t["exit_date"])  <= we
        ]

    def sector_avg_return(self, ticker: str) -> float:
        """Average ann_return for all trips with same ticker."""
        same = [t["ann_return"] for t in self._trips if t["ticker"] == ticker]
        return round(float(np.mean(same)), 2) if same else 0.0

    def regime_transitions(self, date_start: str, date_end: str) -> List[str]:
        mask = (
            (self.eq["date_str"] >= date_start) &
            (self.eq["date_str"] <= date_end)
        )
        sub  = self.eq[mask]
        prev = None
        transitions = []
        for _, r in sub.iterrows():
            if r["regime"] != prev and prev is not None:
                transitions.append(f"{r['date_str']}: {prev} → {r['regime']}")
            prev = r["regime"]
        return transitions


# ══════════════════════════════════════════════════════════════════════════
# CACHE MANAGER  (skip already-done entries, resume-safe)
# ══════════════════════════════════════════════════════════════════════════

class CacheManager:

    def __init__(self, name: str):
        self._path = os.path.join(CACHE_DIR, f"{name}.jsonl")
        self._done: set = set()
        self._load()

    def _load(self):
        if os.path.exists(self._path):
            with open(self._path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        self._done.add(entry.get("key", ""))
                    except Exception:
                        pass

    def already_done(self, key: str) -> bool:
        return key in self._done

    def save(self, key: str, data: Dict):
        data["key"] = key
        data["saved_at"] = datetime.now().isoformat()
        with open(self._path, "a") as f:
            f.write(json.dumps(data) + "\n")
        self._done.add(key)

    def load_all(self) -> List[Dict]:
        entries = []
        if os.path.exists(self._path):
            with open(self._path) as f:
                for line in f:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
        return entries


# ══════════════════════════════════════════════════════════════════════════
# GROQ EXPLAINER  (the 3 modes)
# ══════════════════════════════════════════════════════════════════════════

class GroqExplainer:

    def __init__(self, api_key: Optional[str] = None):
        self.groq    = GroqClient(api_key)
        self._tokens_used = 0

    # ── shared helper ─────────────────────────────────────────────────────

    def _sentiment_block(self, ticker: str, date_str: str) -> str:
        if not FINBERT_AVAILABLE or _finbert_get is None:
            return ""
        try:
            result    = _finbert_get(ticker, date_str, use_local=True)
            score     = result.get("score", 0.0)
            label     = result.get("label", "neutral")
            headlines = result.get("headlines", [])[:3]
            if not headlines and score == 0.0:
                return ""
            lines = [f"FinBERT sentiment: **{score:+.2f}** ({label})"]
            if headlines:
                lines.append("Recent headlines:")
                for h in headlines:
                    lines.append(f"  - {h[:90]}")
            return "\n".join(lines)
        except Exception as e:
            return f"FinBERT unavailable: {str(e)[:50]}"

    # ─────────────────────────────────────────────────────────────────────
    # MODE 1: TRADE AUDIT BATCH
    # ─────────────────────────────────────────────────────────────────────

    def audit_trade(
        self,
        trip:    Dict,
        loader:  DataLoader,
        verbose: bool = False,
    ) -> Dict:
        """
        Audit one closed round trip.
        Returns dict with explanation text and metadata.
        """
        tk         = trip["ticker"]
        entry_d    = trip["entry_date"]
        exit_d     = trip["exit_date"]
        ret        = trip["return_pct"]
        hold       = trip["hold_days"]
        confidence = trip.get("confidence", 0.5)
        regime     = trip.get("hmm_regime", "Unknown")
        exit_reason= trip.get("exit_reason", "")
        rsi        = trip.get("rsi_at_entry", None)
        pct_200    = trip.get("pct_above_200d", None)
        mom20      = trip.get("momentum_20d", None)
        sector_avg = loader.sector_avg_return(tk)
        eq_entry   = loader.equity_on(entry_d)
        eq_exit    = loader.equity_on(exit_d)

        # Exit reason human description
        exit_desc  = EXIT_REASONS.get(
            exit_reason.split("|")[0] if "|" in exit_reason else exit_reason,
            exit_reason
        )

        # Technical context
        tech_lines = []
        if rsi is not None:
            rsi_note = ("overbought" if rsi > 70 else
                        "oversold" if rsi < 30 else "neutral")
            tech_lines.append(f"RSI at entry: {rsi:.1f} ({rsi_note})")
        if pct_200 is not None:
            tech_lines.append(f"Price vs 200d MA: {pct_200:+.1f}%")
        if mom20 is not None:
            tech_lines.append(f"20d momentum: {mom20:+.1f}%")
        tech_str = " | ".join(tech_lines) if tech_lines else "Not available"

        prompt = f"""
Audit this trade for the internal risk committee.

## Trade Summary
| Field | Value |
|---|---|
| Ticker | {tk} ({COMPANY.get(tk, tk)}) |
| Entry | {entry_d} @ ${trip['entry_price']:.2f} |
| Exit | {exit_d} @ ${trip['exit_price']:.2f} |
| Return | {ret:+.2f}% over {hold} days ({trip['ann_return']:+.1f}% ann) |
| vs Ticker avg | {sector_avg:+.1f}% ann (all {tk} trades) |

## Entry Context
- Regime: {regime} | Confidence: {confidence:.2f} | Weight: {trip.get('weight', 0):.2f}
- Technical: {tech_str}
- Entry reason: {trip.get('entry_reason', 'signal')}
- Portfolio equity at entry: ${eq_entry['equity']:,.0f} (DD: {eq_entry['dd']:.1f}%)

## Exit Context
- Exit reason: {exit_reason}
- Explanation: {exit_desc}
- Portfolio equity at exit: ${eq_exit['equity']:,.0f}

Write exactly 3 short paragraphs:
1. **Entry quality**: Was this a good entry? Cite regime, confidence, RSI, momentum, sentiment.
2. **Exit quality**: Was the exit well-timed or premature? Was {exit_reason} correct here?
3. **Verdict**: Win/loss verdict with one specific improvement suggestion if applicable.
Keep each paragraph to 2-3 sentences maximum.
"""
        text, tokens = self.groq.complete(prompt, MODEL_BATCH, max_tokens=350)
        self._tokens_used += tokens

        result = {
            "ticker":      tk,
            "entry_date":  entry_d,
            "exit_date":   exit_d,
            "return_pct":  ret,
            "hold_days":   hold,
            "exit_reason": exit_reason,
            "explanation": text,
            "tokens":      tokens,
        }
        if verbose:
            outcome = "✓" if ret > 0 else "✗"
            print(f"\n  {outcome} {tk} {entry_d}→{exit_d} "
                  f"{ret:+.1f}% [{hold}d] | {exit_reason}")
            print(f"  {'─'*60}")
            print(f"  {text}")
        return result

    def audit_batch(
        self,
        loader:    DataLoader,
        max_trades: Optional[int] = None,
        verbose:    bool = True,
    ) -> List[Dict]:
        """
        Audit all closed round trips. Caches results — safe to interrupt.
        """
        cache   = CacheManager("trade_audits")
        trips   = loader.trips
        if max_trades:
            trips = trips[:max_trades]

        total   = len(trips)
        done    = 0
        skipped = 0
        results = []

        print(f"\n  {'═'*55}")
        print(f"  TRADE AUDIT BATCH — {total} round trips")
        print(f"  Model: {MODEL_BATCH} (500k TPD)")
        print(f"  Cache: {CACHE_DIR}/trade_audits.jsonl")
        print(f"  {'═'*55}\n")

        for i, trip in enumerate(trips):
            key = f"{trip['ticker']}_{trip['entry_date']}_{trip['exit_date']}"
            if cache.already_done(key):
                skipped += 1
                if verbose:
                    print(f"  ↩ [{i+1}/{total}] {trip['ticker']} "
                          f"{trip['entry_date']} — cached, skip")
                continue

            if verbose:
                print(f"  [{i+1}/{total}] {trip['ticker']} "
                      f"{trip['entry_date']}→{trip['exit_date']} "
                      f"{trip['return_pct']:+.1f}%", end=" ... ")

            result = self.audit_trade(trip, loader, verbose=False)
            cache.save(key, result)
            results.append(result)
            done += 1

            if verbose:
                outcome = "✓" if result["return_pct"] > 0 else "✗"
                print(f"{outcome} ({result['tokens']} tok)")

            # Brief pause to respect rate limits
            time.sleep(0.3)

        print(f"\n  Done: {done} new | {skipped} cached | "
              f"{self._tokens_used:,} tokens used")
        return results

    # ─────────────────────────────────────────────────────────────────────
    # MODE 2: DAILY BRIEFING
    # ─────────────────────────────────────────────────────────────────────

    def daily_briefing(
        self,
        date_str:       str,
        loader:         DataLoader,
        verbose:        bool = True,
        tickers_filter: Optional[list] = None,
    ) -> str:
        """
        Generate a daily briefing for a specific trading date.
        """
        cache  = CacheManager("daily_briefings")
        key    = f"daily_{date_str}"

        if cache.already_done(key):
            print(f"  ↩ Daily briefing {date_str} — cached")
            entries = cache.load_all()
            for e in entries:
                if e.get("key") == key:
                    return e.get("text", "")

        trades   = loader.trades_on(date_str)
        if tickers_filter:
            trades = [t for t in trades if t.get("ticker") in tickers_filter]
        eq_state = loader.equity_on(date_str)
        tickers  = list({r["ticker"] for r in trades})

        if not trades:
            text = f"No trades on {date_str}. Portfolio held existing positions."
            cache.save(key, {"date": date_str, "text": text, "tokens": 0})
            if verbose:
                print(f"\n  📅 {date_str}: No trades")
            return text

        # Build trade table
        trade_rows = []
        for r in trades:
            reason = str(r.get("reason", "signal"))
            proba  = r.get("proba", None)
            proba_str = f"{proba:.2f}" if proba and not pd.isna(proba) else "N/A"
            trade_rows.append(
                f"| {r['action']:4} | {r['ticker']:5} | "
                f"${r['price']:.2f} | {proba_str} | "
                f"{r.get('hmm_regime','?')} | {reason} |"
            )
        trade_table = (
            "| Action | Ticker | Price | Confidence | Regime | Reason |\n"
            "|--------|--------|-------|------------|--------|--------|\n"
            + "\n".join(trade_rows)
        )

        # FinBERT for traded tickers
        sentiment_lines = []
        for tk in tickers[:4]:   # max 4 tickers to limit compute
            s = _finbert_get(tk, date_str, use_local=True) if _finbert_get else {"score": 0.0, "label": "neutral"}
            if s["score"] != 0.0:
                sentiment_lines.append(f"  {tk}: {s['score']:+.2f} ({s['label']})")
        sentiment_str = "\n".join(sentiment_lines) if sentiment_lines else "  Not available"

        prompt = f"""
Daily trading briefing for {date_str}.

## Portfolio State
- Equity: ${eq_state['equity']:,.0f} | Drawdown: {eq_state['dd']:.1f}% | Regime: {eq_state['regime']}

## Trades Executed
{trade_table}

## FinBERT Sentiment (at time of trades)
{sentiment_str}

Write a concise daily briefing in exactly 3 paragraphs:
1. **Market regime and portfolio state**: What was the HMM regime signalling and what does the current drawdown/equity level indicate?
2. **Trade decisions**: Explain the logic behind today's trades — why did each signal fire, and do the exit reasons make sense?
3. **Risk assessment**: Any flags — high heat, rising drawdown, conflicting signals, sentiment misalignment?
Keep clinical and quantitative. Max 2-3 sentences per paragraph.
"""
        text, tokens = self.groq.complete(prompt, MODEL_DAILY, max_tokens=420)
        self._tokens_used += tokens
        cache.save(key, {"date": date_str, "text": text, "tokens": tokens})

        if verbose:
            print(f"\n  📅 DAILY BRIEFING — {date_str}")
            print(f"  {'─'*55}")
            print(f"  Equity: ${eq_state['equity']:,.0f} | "
                  f"DD: {eq_state['dd']:.1f}% | Regime: {eq_state['regime']}")
            print(f"  Trades: {len(trades)} | Tickers: {', '.join(tickers)}")
            print(f"  {'─'*55}")
            print(f"\n{text}\n")

        return text

    # ─────────────────────────────────────────────────────────────────────
    # MODE 3: WEEKLY SUMMARY
    # ─────────────────────────────────────────────────────────────────────

    def weekly_summary(
        self,
        week_start: str,
        loader:     DataLoader,
        verbose:    bool = True,
    ) -> str:
        """
        Generate a weekly summary starting from week_start (Monday).
        """
        cache    = CacheManager("weekly_summaries")
        week_end = (pd.Timestamp(week_start) + timedelta(days=4)).strftime("%Y-%m-%d")
        key      = f"weekly_{week_start}"

        if cache.already_done(key):
            print(f"  ↩ Weekly summary {week_start} — cached")
            entries = cache.load_all()
            for e in entries:
                if e.get("key") == key:
                    return e.get("text", "")

        # Collect week data
        trips       = loader.trips_in_week(week_start)
        transitions = loader.regime_transitions(week_start, week_end)
        eq_start    = loader.equity_on(week_start)
        eq_end      = loader.equity_on(week_end)
        all_trades  = []
        for d in pd.date_range(week_start, week_end, freq="B"):
            all_trades.extend(loader.trades_on(d.strftime("%Y-%m-%d")))

        # P&L
        if eq_start["equity"] > 0 and eq_end["equity"] > 0:
            week_ret = (eq_end["equity"] / eq_start["equity"] - 1) * 100
        else:
            week_ret = 0.0

        # Best and worst trip this week
        week_closed = [t for t in trips if week_start <= t["exit_date"] <= week_end]
        best  = max(week_closed, key=lambda t: t["return_pct"]) if week_closed else None
        worst = min(week_closed, key=lambda t: t["return_pct"]) if week_closed else None

        # Trip summary table
        if week_closed:
            trip_rows = [
                f"| {t['ticker']:5} | {t['entry_date']} | {t['exit_date']} | "
                f"{t['return_pct']:+.1f}% | {t['hold_days']}d | "
                f"{t['exit_reason'].split('|')[0]} |"
                for t in sorted(week_closed, key=lambda x: x["return_pct"], reverse=True)
            ]
            trip_table = (
                "| Ticker | Entry | Exit | Return | Hold | Exit reason |\n"
                "|--------|-------|------|--------|------|-------------|\n"
                + "\n".join(trip_rows)
            )
        else:
            trip_table = "No closed trades this week."

        prompt = f"""
Weekly trading summary: {week_start} to {week_end}.

## Performance
- Week return: {week_ret:+.2f}%
- Equity: ${eq_start['equity']:,.0f} → ${eq_end['equity']:,.0f}
- Closing drawdown: {eq_end['dd']:.1f}%
- Regime at close: {eq_end['regime']}
- Total trade actions: {len(all_trades)} ({sum(1 for t in all_trades if t.get('action')=='BUY')} buys, {sum(1 for t in all_trades if t.get('action')=='SELL')} sells)

## Closed Positions
{trip_table}

## Regime Transitions
{chr(10).join(transitions) if transitions else 'No regime transitions this week.'}

## Notable
- Best trade: {f"{best['ticker']} {best['return_pct']:+.1f}% ({best['exit_reason'].split('|')[0]})" if best else 'None'}
- Worst trade: {f"{worst['ticker']} {worst['return_pct']:+.1f}% ({worst['exit_reason'].split('|')[0]})" if worst else 'None'}

Write exactly 4 paragraphs:
1. **Performance recap**: Week return vs expectations, drawdown trend, equity trajectory.
2. **Regime and signals**: What HMM was signalling, any transitions, signal quality.
3. **Trade decisions**: Were the key buys and sells well-timed? Any notable exits?
4. **Outlook**: Key risk factors for next week based on current regime/drawdown/sentiment.
Clinical and quantitative. Max 2-3 sentences per paragraph.
"""
        text, tokens = self.groq.complete(prompt, MODEL_DAILY, max_tokens=520)
        self._tokens_used += tokens
        cache.save(key, {
            "week_start": week_start, "week_end": week_end,
            "week_return": week_ret,  "text": text, "tokens": tokens
        })

        csv_path = os.path.join(REPORT_DIR, f"weekly_{week_start}.csv")
        csv_data = {
            "week_start":          week_start,
            "week_end":            week_end,
            "week_return_pct":     round(week_ret, 2),
            "equity_start":        eq_start["equity"],
            "equity_end":          eq_end["equity"],
            "drawdown_pct":        eq_end["dd"],
            "regime":              eq_end["regime"],
            "n_trade_actions":     len(all_trades),
            "n_closed_positions":  len(week_closed),
            "win_rate_pct":        round(sum(1 for t in week_closed if t["return_pct"] > 0)
                                         / max(len(week_closed), 1) * 100, 1),
            "best_ticker":         best["ticker"] if best else "",
            "best_return_pct":     best["return_pct"] if best else 0,
            "worst_ticker":        worst["ticker"] if worst else "",
            "worst_return_pct":    worst["return_pct"] if worst else 0,
            "regime_transitions":  len(transitions),
            "groq_summary":        text,
        }
        pd.DataFrame([csv_data]).to_csv(csv_path, index=False)
        if verbose:
            print(f"  💾 CSV    → {csv_path}")

        if verbose:
            print(f"\n  📅 WEEKLY SUMMARY — {week_start} to {week_end}")
            print(f"  {'─'*55}")
            print(f"  Week return: {week_ret:+.2f}% | "
                  f"Trades: {len(all_trades)} | "
                  f"Closed: {len(week_closed)}")
            if transitions:
                print(f"  Regime transitions: {len(transitions)}")
            print(f"  {'─'*55}")
            print(f"\n{text}\n")

        return text

    # ── convenience: save full report ─────────────────────────────────────

    def save_report(self, content: str, filename: str):
        path = os.path.join(REPORT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  💾 Report → {path}")

    @property
    def usage(self) -> Dict:
        return {**self.groq.usage, "total_tokens_session": self._tokens_used}


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Groq Explainer v3")
    parser.add_argument("-a", "--audit",   action="store_true", help="Batch audit all trades")
    parser.add_argument("-d", "--daily",   action="store_true", help="Daily briefing for a date")
    parser.add_argument("-w", "--weekly",  action="store_true", help="Weekly summary")
    parser.add_argument("--demo",          action="store_true", help="Run 3 demo examples")
    parser.add_argument("--date",  default=None, help="Date for daily mode (YYYY-MM-DD)")
    parser.add_argument("--week",  default=None, help="Week start for weekly mode (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=None, help="Max trades for audit mode")
    parser.add_argument("--ticker", nargs="+", default=None,
                        help="Filter by ticker(s) e.g. --ticker NVDA TSLA")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  GROQ EXPLAINER v3 — Virtual Trading Firm")
    print("═"*60 + "\n")

    # Load data
    print("  Loading engine outputs...")
    loader = DataLoader()

    # Init explainer
    explainer = GroqExplainer()

    if args.demo:
        print("\n  Running DEMO — 3 examples (1 audit, 1 daily, 1 weekly)\n")

        # Pick the best and worst trade for demo
        all_trips = loader.trips
        if all_trips:
            best_trip  = max(all_trips, key=lambda t: t["return_pct"])
            worst_trip = min(all_trips, key=lambda t: t["return_pct"])

            print("  ── BEST TRADE AUDIT ──")
            explainer.audit_trade(best_trip, loader, verbose=True)

            time.sleep(2)

            print("\n  ── WORST TRADE AUDIT ──")
            explainer.audit_trade(worst_trip, loader, verbose=True)

            time.sleep(2)

        # Daily briefing for a date with trades
        dates_with_trades = loader.tl["date"].dt.strftime("%Y-%m-%d").unique()
        if len(dates_with_trades) >= 10:
            demo_date = dates_with_trades[len(dates_with_trades)//2]
            explainer.daily_briefing(demo_date, loader, verbose=True)
            time.sleep(2)

        # Weekly summary for the first full week
        first_date = loader.eq["date_str"].iloc[10]
        week_start = pd.Timestamp(first_date) - timedelta(days=pd.Timestamp(first_date).weekday())
        explainer.weekly_summary(week_start.strftime("%Y-%m-%d"), loader, verbose=True)

    elif args.audit:
        if args.ticker:
            loader.trips = [t for t in loader.trips if t["ticker"] in args.ticker]
        explainer.audit_batch(loader, max_trades=args.limit, verbose=args.verbose)
        print(f"\n  All audits saved to {CACHE_DIR}/trade_audits.jsonl")

    elif args.daily:
        date_str = args.date or date.today().strftime("%Y-%m-%d")
        text = explainer.daily_briefing(date_str, loader, verbose=args.verbose,
                                        tickers_filter=args.ticker)
        explainer.save_report(text, f"daily_{date_str}.txt")

    elif args.weekly:
        week_str = args.week
        if not week_str:
            # Default: most recent Monday
            today = date.today()
            week_str = (today - timedelta(days=today.weekday())).strftime("%Y-%m-%d")
        text = explainer.weekly_summary(week_str, loader, verbose=args.verbose)
        explainer.save_report(text, f"weekly_{week_str}.txt")

    print(f"\n  {'═'*55}")
    print(f"  Session usage: {explainer.usage}")
    print(f"  Outputs: {REPORT_DIR}/  |  Cache: {CACHE_DIR}/")
    print(f"  {'═'*55}")