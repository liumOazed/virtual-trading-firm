"""
regime_selector.py
==================
Regime-conditional asset selection layer.
Sits between signal_engine.py and backtest_engine_v2.py.

Calibrated from xgboost_model.py run — May 2026
------------------------------------------------
  BUY ratio      : 41.5%
  Global F1      : 0.563
  Global Accuracy: 0.473 (WF)
  Threshold      : 0.550
  Early stop     : 295 / 2000

HC Accuracy by ticker (confidence > 0.65):
  JPM   0.837  Tier 1
  META  0.789  Tier 1
  SPY   0.809  Tier 1
  NVDA  0.786  Tier 1
  AMZN  0.707  Tier 2
  GOOGL 0.736  Tier 2
  QQQ   0.674  Tier 2
  MSFT  0.635  Tier 2
  AAPL  0.662  Tier 2
  AVGO  0.527  Tier 3 (below threshold for live trading)

Pipeline position:
  xgboost_model.py  →  signal_engine.py  →  [THIS FILE]  →  backtest_engine_v2.py  →  rl_agent.py

For each trading day this module:
  1. Detects current market regime (0-3) from SPY
  2. Gates tickers by historical HC accuracy in that regime
  3. Gates tickers by today's model confidence
  4. Gates tickers by adversarial drift score
  5. Returns only tradeable tickers ranked by HC accuracy
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, '4_signals'))


# ════════════════════════════════════════════════════════════════════════════
# REGIME CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

REGIME_NAMES = {
    0: "Bear-Stress",
    1: "Bear-Stable",
    2: "Bull-Stable",
    3: "Bull-Trending",
}

# Minimum historical HC trades in a regime before trusting the stats
MIN_REGIME_SAMPLES = 15


# ════════════════════════════════════════════════════════════════════════════
# TICKER TIERS  (calibrated from May 2026 training run)
# ════════════════════════════════════════════════════════════════════════════

# Tier 1 — HC accuracy > 0.78. Trade freely with standard confidence gate.
TIER_1 = {"JPM", "META", "SPY", "NVDA"}

# Tier 2 — HC accuracy 0.60-0.78. Trade with slightly tighter confidence gate.
TIER_2 = {"AMZN", "GOOGL", "QQQ", "MSFT", "AAPL"}

# Tier 3 — HC accuracy < 0.60. Do not trade live until model improves.
TIER_3 = {"AVGO"}

# Hard exclusion — model is actively wrong when confident on these.
# TSLA was excluded from this run; add any ticker with HC < 0.50 here.
BLACKLISTED = set()

# Confidence gate per tier
TIER_CONFIDENCE = {
    1: 0.50,   # Tier 1: lower bar — model is reliable
    2: 0.52,   # Tier 2: standard bar
    3: 1.01,   # Tier 3: effectively excluded (no model output reaches 1.01)
}

# Bear regime overrides — tighter confidence gates in difficult regimes
BEAR_CONFIDENCE_BOOST = 0.03   # added to tier gate in Bear-Trending and Bear-MeanRev

# Maximum concurrent open positions
MAX_POSITIONS = 5

# Minimum proba_buy to consider entry at all
MIN_PROBA    = 0.42

# Maximum drift AUC before skipping
DRIFT_MAX    = 0.70


# ════════════════════════════════════════════════════════════════════════════
# HC ACCURACY BOOTSTRAP  (from May 2026 segmented performance report)
# ════════════════════════════════════════════════════════════════════════════

# These are the in-sample segmented HC accuracy values from the last training run.
# They are used to initialise the performance table before the first backtest
# produces regime-specific OOS data.
#
# Structure: {ticker: {"hc_accuracy": float, "raw_accuracy": float, "hc_trades": int}}

BOOTSTRAP_STATS = {
    "AAPL":  {"hc_accuracy": 0.7147, "raw_accuracy": 0.6324, "hc_trades": 375},
    "MSFT":  {"hc_accuracy": 0.7167, "raw_accuracy": 0.6422, "hc_trades": 413},
    "GOOGL": {"hc_accuracy": 0.7613, "raw_accuracy": 0.6556, "hc_trades": 398},
    "AMZN":  {"hc_accuracy": 0.7576, "raw_accuracy": 0.7022, "hc_trades": 429},
    "NVDA":  {"hc_accuracy": 0.8088, "raw_accuracy": 0.6900, "hc_trades": 387},
    "META":  {"hc_accuracy": 0.8145, "raw_accuracy": 0.7390, "hc_trades": 496},
    "JPM":   {"hc_accuracy": 0.8695, "raw_accuracy": 0.7390, "hc_trades": 429},
    "SPY":   {"hc_accuracy": 0.8230, "raw_accuracy": 0.6654, "hc_trades": 305},
    "QQQ":   {"hc_accuracy": 0.5953, "raw_accuracy": 0.5392, "hc_trades": 299},
}

# HC accuracy gate — ticker must exceed this in current regime to be tradeable
HC_ACC_THRESHOLD = 0.55


# ════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TickerSignal:
    ticker:          str
    proba_buy:       float
    confidence:      float
    regime:          int
    regime_name:     str
    tier:            int
    hc_accuracy:     float
    hc_trades:       int
    drift_auc:       float
    label:           str
    threshold_used:  float
    conf_gate:       float
    selected:        bool
    skip_reason:     Optional[str] = None


@dataclass
class DailySelection:
    date:            str
    regime:          int
    regime_name:     str
    selected:        List[TickerSignal] = field(default_factory=list)
    skipped:         List[TickerSignal] = field(default_factory=list)

    @property
    def all_signals(self) -> List[TickerSignal]:
        return self.selected + self.skipped

    def summary(self) -> str:
        lines = [
            f"Date: {self.date} | Regime: {self.regime_name}",
            f"Selected: {len(self.selected)} / {len(self.all_signals)} tickers",
        ]
        for s in self.selected:
            lines.append(
                f"  TRADE  {s.ticker:<6} | proba={s.proba_buy:.3f} | "
                f"conf={s.confidence:.3f} | hc_acc={s.hc_accuracy:.3f} "
                f"({s.hc_trades} trades) | tier={s.tier}"
            )
        for s in self.skipped[:4]:
            lines.append(f"  SKIP   {s.ticker:<6} | {s.skip_reason}")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# REGIME PERFORMANCE TABLE
# ════════════════════════════════════════════════════════════════════════════

class RegimePerformanceTable:
    """
    Stores per-ticker per-regime historical performance.

    Two data sources (applied in order):
      1. Bootstrap from xgboost segmented report (available immediately)
      2. OOS backtest results (available after first backtest run)

    OOS data overwrites bootstrap data for any regime with enough samples.

    Structure:
      {ticker: {regime_id: {"hc_accuracy": float, "raw_accuracy": float,
                             "hc_trades": int, "win_rate": float,
                             "sharpe": float, "source": str}}}
    """

    def __init__(self):
        self._table: Dict[str, Dict[int, Dict]] = {}

    def update(self, ticker: str, regime: int, stats: dict):
        if ticker not in self._table:
            self._table[ticker] = {}
        self._table[ticker][regime] = stats

    def get(self, ticker: str, regime: int) -> Optional[dict]:
        return self._table.get(ticker, {}).get(regime)

    def get_hc_accuracy(self, ticker: str, regime: int,
                         default: float = 0.50) -> float:
        entry = self.get(ticker, regime)
        if entry is None:
            return default
        if entry.get("hc_trades", 0) < MIN_REGIME_SAMPLES:
            return default
        return float(entry.get("hc_accuracy", default))

    def get_hc_trades(self, ticker: str, regime: int) -> int:
        entry = self.get(ticker, regime)
        return int(entry.get("hc_trades", 0)) if entry else 0

    def rank_for_regime(self, tickers: List[str],
                         regime: int) -> List[Tuple[str, float]]:
        """
        Returns tickers sorted descending by HC accuracy in the given regime.
        Skips tickers with insufficient historical samples.
        """
        ranked = []
        for ticker in tickers:
            if ticker in BLACKLISTED:
                continue
            hc = self.get_hc_accuracy(ticker, regime, default=None)
            if hc is None:
                continue
            ranked.append((ticker, hc))
        return sorted(ranked, key=lambda x: x[1], reverse=True)

    @classmethod
    def from_bootstrap(cls) -> "RegimePerformanceTable":
        """
        Initialise from the hardcoded BOOTSTRAP_STATS.
        Applied equally across all regimes as a prior.
        Regime-specific data from backtest will overwrite these.
        """
        table = cls()
        for ticker, stats in BOOTSTRAP_STATS.items():
            for regime_id in range(4):   # 0=Bear-Stress, 1=Bear-Stable, 2=Bull-Stable, 3=Bull-Trending
                table.update(ticker, regime_id, {
                    "hc_accuracy":  stats["hc_accuracy"],
                    "raw_accuracy": stats["raw_accuracy"],
                    "hc_trades":    stats["hc_trades"],
                    "win_rate":     stats["hc_accuracy"],
                    "sharpe":       0.0,
                    "source":       "bootstrap",
                })
        print("  Performance table: loaded from bootstrap (pre-backtest)")
        return table

    @classmethod
    def from_backtest_results(
        cls,
        trade_file:  str,
        equity_file: str,
    ) -> "RegimePerformanceTable":
        """
        Build regime-specific performance table from completed backtest output.
        Falls back to bootstrap for any regime with insufficient data.
        """
        table = cls.from_bootstrap()

        try:
            trades_df = pd.read_csv(trade_file)
            equity_df = pd.read_csv(equity_file)
        except FileNotFoundError:
            print("  Performance table: backtest files not found, using bootstrap")
            return table

        if "regime" not in trades_df.columns or "action" not in trades_df.columns:
            print("  Performance table: trade log missing columns, using bootstrap")
            return table

        sells = trades_df[trades_df["action"] == "SELL"].copy()
        if sells.empty:
            return table

        regime_name_to_id = {v: k for k, v in REGIME_NAMES.items()}

        for ticker in trades_df["ticker"].unique():
            for regime_name, regime_id in regime_name_to_id.items():
                mask = (
                    (sells["ticker"] == ticker) &
                    (sells["regime"] == regime_name)
                )
                regime_trades = sells[mask]
                n = len(regime_trades)

                if n < MIN_REGIME_SAMPLES:
                    continue

                win_rate = 0.50
                sharpe   = 0.0
                if "pnl_pct" in regime_trades.columns:
                    wins     = (regime_trades["pnl_pct"] > 0).sum()
                    win_rate = wins / n
                    rets     = regime_trades["pnl_pct"].values / 100.0
                    sharpe   = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(52)

                # Use win_rate as proxy for OOS HC accuracy
                # (real HC accuracy requires signal confidence column in trade log)
                hc_acc = win_rate

                if "confidence" in regime_trades.columns:
                    hc_mask = regime_trades["confidence"] > 0.65
                    if hc_mask.sum() >= 5:
                        if "pnl_pct" in regime_trades.columns:
                            hc_wins = (regime_trades.loc[hc_mask, "pnl_pct"] > 0).sum()
                            hc_acc  = hc_wins / hc_mask.sum()

                table.update(ticker, regime_id, {
                    "hc_accuracy":  round(hc_acc,   4),
                    "raw_accuracy": round(win_rate,  4),
                    "hc_trades":    int(n),
                    "win_rate":     round(win_rate,  4),
                    "sharpe":       round(sharpe,    4),
                    "source":       "backtest_oos",
                })

        oos_count = sum(
            1 for ticker in table._table
            for regime in table._table[ticker]
            if table._table[ticker][regime].get("source") == "backtest_oos"
        )
        print(f"  Performance table: {oos_count} regime entries updated from OOS backtest")
        return table

    def save(self, path: str):
        serialisable = {
            ticker: {str(k): v for k, v in regimes.items()}
            for ticker, regimes in self._table.items()
        }
        with open(path, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"  Performance table saved: {path}")

    @classmethod
    def load(cls, path: str) -> "RegimePerformanceTable":
        table = cls()
        with open(path, "r") as f:
            data = json.load(f)
        for ticker, regimes in data.items():
            for k, v in regimes.items():
                table.update(ticker, int(k), v)
        print(f"  Performance table loaded: {path}")
        return table

    def print_summary(self):
        print(f"\n  {'REGIME PERFORMANCE TABLE':^72}")
        print(f"  {'Ticker':<8} {'Regime':<18} {'HC Acc':>8} {'HC Trades':>10} "
              f"{'Sharpe':>8} {'Source'}")
        print("  " + "-" * 72)
        for ticker in sorted(self._table.keys()):
            for regime_id, stats in sorted(self._table[ticker].items()):
                if stats.get("hc_trades", 0) < MIN_REGIME_SAMPLES:
                    continue
                print(
                    f"  {ticker:<8} "
                    f"{REGIME_NAMES.get(regime_id, str(regime_id)):<18} "
                    f"{stats.get('hc_accuracy', 0):>8.4f} "
                    f"{stats.get('hc_trades', 0):>10} "
                    f"{stats.get('sharpe', 0):>8.3f} "
                    f"  {stats.get('source', 'bootstrap')}"
                )


# ════════════════════════════════════════════════════════════════════════════
# TICKER TIER LOOKUP
# ════════════════════════════════════════════════════════════════════════════

def _get_tier(ticker: str) -> int:
    if ticker in BLACKLISTED:
        return 99
    if ticker in TIER_1:
        return 1
    if ticker in TIER_2:
        return 2
    if ticker in TIER_3:
        return 3
    return 2   # default unknown tickers to Tier 2


def _get_conf_gate(ticker: str, regime: int) -> float:
    tier = _get_tier(ticker)
    base = TIER_CONFIDENCE.get(tier, 0.65)
    if regime in (0, 1):   # Bear-Stress, Bear-Stable
        base += BEAR_CONFIDENCE_BOOST
    return base


# ════════════════════════════════════════════════════════════════════════════
# REGIME SELECTOR  (main entry point)
# ════════════════════════════════════════════════════════════════════════════

class RegimeSelector:
    """
    Regime-conditional asset selection.
    Called once per trading day by backtest_engine_v2._run_oos_loop_v2().

    Usage
    -----
    selector = RegimeSelector.build(tickers, signal_cache)
    selection = selector.select(date, bar_probas, bar_confidences, bar_drifts)
    tradeable = [s.ticker for s in selection.selected]
    """

    def __init__(self,
                 tickers:    List[str],
                 perf_table: RegimePerformanceTable):
        self.tickers    = tickers
        self.perf_table = perf_table

    # ── factory methods ───────────────────────────────────────────────────

    @classmethod
    def build(cls,
              tickers:     List[str],
              signal_cache: Dict,
              **kwargs,
              ) -> "RegimeSelector":
        """
        Build selector using bootstrap HC accuracy priors.
        Regime-conditioned ticker ranking is handled by FactorEngine;
        this layer provides quality gates only (drift, confidence, tier).
        """
        perf_table = RegimePerformanceTable.from_bootstrap()
        return cls(tickers, perf_table)

    # ── regime detection ──────────────────────────────────────────────────

    def _get_regime(self,
                    signal_cache: Dict,
                    current_date: str) -> int:
        """
        Read regime from SPY signal cache.
        Falls back to first available ticker, then defaults to 3 (Bull-Trending).
        In new 4-state system: 0=Bear-Stress, 1=Bear-Stable, 2=Bull-Stable, 3=Bull-Trending.
        """
        proxy = "SPY" if "SPY" in signal_cache else \
                next((t for t in signal_cache), None)
        if proxy is None:
            return 3

        sig = signal_cache.get(proxy)
        if sig is None:
            return 3

        if hasattr(sig, "index") and current_date in sig.index:
            if "regime" in sig.columns:
                raw = int(sig.loc[current_date, "regime"])
                return raw if raw in REGIME_NAMES else 3

        return 3

    # ── main selection method ─────────────────────────────────────────────

    def select(self,
               current_date:    str,
               bar_probas:      Dict[str, float],
               bar_confidences: Dict[str, float],
               bar_drifts:      Dict[str, float]  = None,
               signal_cache:    Dict              = None,
               max_positions:   int               = MAX_POSITIONS,
               ) -> DailySelection:
        """
        For a given date, select which tickers to trade.

        Parameters
        ----------
        current_date    : YYYY-MM-DD
        bar_probas      : {ticker: proba_buy}
        bar_confidences : {ticker: confidence}
        bar_drifts      : {ticker: drift_auc}  — optional
        signal_cache    : passed through for regime detection
        max_positions   : cap on concurrent positions

        Returns
        -------
        DailySelection with .selected and .skipped lists
        """
        regime      = self._get_regime(signal_cache or {}, current_date)
        regime_name = REGIME_NAMES.get(regime, "Unknown")
        drifts      = bar_drifts or {}

        # Rank available tickers by historical HC accuracy in this regime
        ranked = self.perf_table.rank_for_regime(
            list(bar_probas.keys()), regime
        )

        # Tickers not in performance table get appended at the end
        ranked_tickers = {t for t, _ in ranked}
        unranked = [
            (t, 0.0) for t in bar_probas
            if t not in ranked_tickers and t not in BLACKLISTED
        ]
        all_candidates = ranked + unranked

        selected:  List[TickerSignal] = []
        skipped:   List[TickerSignal] = []

        for ticker, hist_hc_acc in all_candidates:
            proba      = bar_probas.get(ticker, 0.0)
            confidence = bar_confidences.get(ticker, 0.0)
            drift      = drifts.get(ticker, 0.5)
            tier       = _get_tier(ticker)
            conf_gate  = _get_conf_gate(ticker, regime)
            hc_trades  = self.perf_table.get_hc_trades(ticker, regime)

            # ── quality gates in priority order ──────────────────────────
            skip_reason = None

            if ticker in BLACKLISTED:
                skip_reason = "blacklisted"

            elif tier >= 3:
                skip_reason = f"tier_3_excluded (hc={hist_hc_acc:.3f})"

            elif proba < MIN_PROBA:
                skip_reason = f"low_proba ({proba:.3f} < {MIN_PROBA})"

            elif confidence < conf_gate:
                skip_reason = (
                    f"low_confidence ({confidence:.3f} < {conf_gate:.2f} "
                    f"tier{tier})"
                )

            elif drift > DRIFT_MAX:
                skip_reason = f"regime_drift ({drift:.3f} > {DRIFT_MAX})"

            elif hist_hc_acc < HC_ACC_THRESHOLD:
                skip_reason = (
                    f"low_hc_accuracy ({hist_hc_acc:.3f} < {HC_ACC_THRESHOLD})"
                )

            elif len(selected) >= max_positions:
                skip_reason = f"max_positions ({max_positions})"

            # ── build signal record ───────────────────────────────────────
            signal_rec = TickerSignal(
                ticker         = ticker,
                proba_buy      = round(proba,       4),
                confidence     = round(confidence,  4),
                regime         = regime,
                regime_name    = regime_name,
                tier           = tier,
                hc_accuracy    = round(hist_hc_acc, 4),
                hc_trades      = hc_trades,
                drift_auc      = round(drift,       3),
                label          = "BUY" if proba >= 0.55 else "WAIT/SELL",
                threshold_used = 0.55,
                conf_gate      = round(conf_gate,   2),
                selected       = skip_reason is None,
                skip_reason    = skip_reason,
            )

            if skip_reason is None:
                selected.append(signal_rec)
            else:
                skipped.append(signal_rec)

        return DailySelection(
            date        = current_date,
            regime      = regime,
            regime_name = regime_name,
            selected    = selected,
            skipped     = skipped,
        )

    # ── regime stress report ──────────────────────────────────────────────

    def regime_coverage_report(self) -> pd.DataFrame:
        """
        Shows which tickers are tradeable in each regime based on
        current performance table. Useful for sanity check before backtest.
        """
        rows = []
        for regime_id, regime_name in REGIME_NAMES.items():
            for ticker in sorted(self.tickers):
                if ticker in BLACKLISTED:
                    continue
                hc_acc    = self.perf_table.get_hc_accuracy(ticker, regime_id)
                hc_trades = self.perf_table.get_hc_trades(ticker, regime_id)
                tier      = _get_tier(ticker)
                conf_gate = _get_conf_gate(ticker, regime_id)
                tradeable = (
                    tier < 3 and
                    hc_acc >= HC_ACC_THRESHOLD and
                    hc_trades >= MIN_REGIME_SAMPLES
                )
                rows.append({
                    "regime":     regime_name,
                    "ticker":     ticker,
                    "tier":       tier,
                    "hc_acc":     round(hc_acc,    4),
                    "hc_trades":  hc_trades,
                    "conf_gate":  round(conf_gate, 2),
                    "tradeable":  tradeable,
                })
        return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# INTEGRATION HELPER  (3-line drop-in for backtest_engine_v2)
# ════════════════════════════════════════════════════════════════════════════

def build_bar_metadata(
    tickers_with_signal: List[str],
    signal_cache:        Dict,
    current_date:        str,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Extracts confidence and drift from signal_cache for a single bar.
    Returns (bar_probas, bar_confidences, bar_drifts).

    Drop-in for _run_oos_loop_v2() — replaces manual dict comprehensions.
    """
    bar_confidences: Dict[str, float] = {}
    bar_drifts:      Dict[str, float] = {}

    for ticker in tickers_with_signal:
        sig = signal_cache.get(ticker)
        if sig is None or current_date not in sig.index:
            bar_confidences[ticker] = 0.0
            bar_drifts[ticker]      = 0.5
            continue

        row = sig.loc[current_date]
        bar_confidences[ticker] = float(row.get("confidence", 0.0)) \
                                  if hasattr(row, "get") else 0.0
        bar_drifts[ticker]      = float(row.get("drift_auc", 0.5)) \
                                  if hasattr(row, "get") else 0.5

    return bar_confidences, bar_drifts


# ════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE INTEGRATION  (exact 3-line addition for _run_oos_loop_v2)
# ════════════════════════════════════════════════════════════════════════════

INTEGRATION_SNIPPET = """
# ── Add to BacktestEngineV2.__init__() after self.signal_cache is declared: ──
from regime_selector import RegimeSelector
self.selector: Optional[RegimeSelector] = None   # initialised in run()

# ── Add to BacktestEngineV2.run() immediately after self.prepare(): ──
from regime_selector import RegimeSelector
self.selector = RegimeSelector.build(
    tickers      = self.cfg.tickers,
    signal_cache = self.signal_cache,
)

# ── Add to _run_oos_loop_v2() after bar_probas is populated: ──
from regime_selector import build_bar_metadata
bar_confidences, bar_drifts = build_bar_metadata(
    tickers_with_signal, self.signal_cache, current_date
)
selection = self.selector.select(
    current_date    = current_date,
    bar_probas      = bar_probas,
    bar_confidences = bar_confidences,
    bar_drifts      = bar_drifts,
    signal_cache    = self.signal_cache,
)
tickers_with_signal = [s.ticker for s in selection.selected]
# The rest of the bar loop is unchanged — it now only sees selected tickers
"""


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT  (standalone test)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 65)
    print("  REGIME SELECTOR — BOOTSTRAP VALIDATION")
    print("=" * 65)

    tickers = list(BOOTSTRAP_STATS.keys())
    perf    = RegimePerformanceTable.from_bootstrap()

    selector = RegimeSelector(tickers=tickers, perf_table=perf)

    # Coverage report — which tickers are tradeable in each regime
    report = selector.regime_coverage_report()
    print("\n  REGIME COVERAGE REPORT")
    print(f"  {'Regime':<20} {'Ticker':<8} {'Tier':<6} "
          f"{'HC Acc':>8} {'HC Trades':>10} {'Conf Gate':>10} {'Tradeable'}")
    print("  " + "-" * 75)

    for _, row in report.iterrows():
        flag = "YES" if row["tradeable"] else "no"
        print(
            f"  {row['regime']:<20} {row['ticker']:<8} {row['tier']:<6} "
            f"{row['hc_acc']:>8.4f} {row['hc_trades']:>10} "
            f"{row['conf_gate']:>10.2f}   {flag}"
        )

    # Tier summary
    print(f"\n  TIER SUMMARY (calibrated May 2026)")
    print(f"  {'Ticker':<8} {'Tier':<6} {'HC Acc':>8} {'Conf Gate':>10}")
    print("  " + "-" * 38)
    for ticker in sorted(tickers, key=lambda t: -BOOTSTRAP_STATS[t]["hc_accuracy"]):
        tier      = _get_tier(ticker)
        hc        = BOOTSTRAP_STATS[ticker]["hc_accuracy"]
        conf_gate = TIER_CONFIDENCE.get(tier, 0.65)
        flag      = " (excluded)" if tier >= 3 else ""
        print(f"  {ticker:<8} {tier:<6} {hc:>8.4f} {conf_gate:>10.2f}{flag}")

    print(f"\n  HC_ACC_THRESHOLD : {HC_ACC_THRESHOLD}")
    print(f"  MAX_POSITIONS    : {MAX_POSITIONS}")
    print(f"  DRIFT_MAX        : {DRIFT_MAX}")
    print(f"  BLACKLISTED      : {BLACKLISTED or 'none'}")