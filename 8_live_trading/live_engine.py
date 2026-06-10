"""
live_engine.py
==============
Layer 8 — Live execution engine for Virtual Trading Firm.

Bridges the backtest engine's signals with Alpaca paper trading.

How it works:
  1. LiveDataFeed fetches today's real prices → builds price_data structure
  2. BacktestEngineV2 runs its full signal pipeline (HMM + XGBoost + Kalman)
     using real prices instead of the cached backtest pkl
  3. Engine generates BUY/SELL signals via its existing logic (unchanged)
  4. LiveEngine intercepts those signals via the existing RL callback hook
  5. AlpacaClient places real paper orders
  6. Results logged to live_trade_log.csv

The engine itself is NEVER modified.
The RL callback hook (already in the engine) is reused for signal interception.

Daily workflow (run after market close):
  python 8_live_trading/live_engine.py

Or via run_backtest.py:
  python run_backtest.py --live
"""

import os
import sys
import json
import pickle
import shutil
import traceback
from datetime import datetime, date
from typing import Dict, List, Optional

import time

import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "4_signals"))
sys.path.insert(0, os.path.join(ROOT, "5_backtesting"))
sys.path.insert(0, os.path.join(ROOT, "8_live_trading"))

from alpaca_client import AlpacaClient, TICKERS
from live_data_feed import LiveDataFeed, LIVE_PKL, BACKTEST_PKL

LIVE_DIR      = os.path.join(ROOT, "8_live_trading", "data")
LIVE_TRADE_LOG= os.path.join(LIVE_DIR, "live_trade_log.csv")
LIVE_EQUITY   = os.path.join(LIVE_DIR, "live_equity_curve.csv")
STATE_FILE    = os.path.join(LIVE_DIR, "live_state.json")
REGIME_STATE  = os.path.join(LIVE_DIR, "regime_state.json")
os.makedirs(LIVE_DIR, exist_ok=True)


def _load_regime_state() -> dict:
    if os.path.exists(REGIME_STATE):
        try:
            with open(REGIME_STATE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"prev_regime": "Unknown", "bear_duration_bars": 0,
            "bear_confirm_count": 0}


def _save_regime_state(state: dict):
    with open(REGIME_STATE, "w") as f:
        json.dump(state, f)


# ══════════════════════════════════════════════════════════════════════════
# SIGNAL INTERCEPTOR
# Used as the RL callback hook — captures every BUY/SELL signal
# the engine generates and executes it via Alpaca instead of simulating.
# ══════════════════════════════════════════════════════════════════════════

class SignalInterceptor:
    """
    Plugs into the engine's _rl_callback hook.
    Captures every signal, executes via Alpaca, logs results.
    """

    def __init__(
        self,
        client:        AlpacaClient,
        equity:        float,
        dry_run:       bool = False,
        verbose:       bool = True,
    ):
        self.client   = client
        self.equity   = equity     # starting equity for position sizing
        self.dry_run  = dry_run   # True = log only, no real orders
        self.verbose  = verbose
        self.signals: List[Dict] = []
        self.orders:  List[Dict] = []

    def __call__(self, state: Dict) -> float:
        """
        Called by engine before each BUY execution.
        state = {ticker, confidence, hmm_regime, heat, drawdown,
                 regime_bar_count, n_open, pos_value_base, equity, current_date}

        Returns size multiplier (1.0 = no change, use engine's sizing).
        Side effect: logs the signal for post-run analysis.
        """
        ticker     = state["ticker"]
        pos_value  = state["pos_value_base"]
        confidence = state["confidence"]
        regime     = state["hmm_regime"]

        self.signals.append({
            "date":        str(state.get("current_date", date.today())),
            "ticker":      ticker,
            "action":      "BUY_SIGNAL",
            "confidence":  round(confidence, 4),
            "hmm_regime":  regime,
            "heat":        round(state.get("heat", 0), 4),
            "drawdown":    round(state.get("drawdown", 0), 4),
            "pos_value":   round(pos_value, 2),
            "n_open":      state.get("n_open", 0),
        })

        if self.verbose:
            print(f"  → SIGNAL: BUY {ticker} | "
                  f"conf={confidence:.2f} | "
                  f"regime={regime} | "
                  f"pos_value=${pos_value:,.0f}")

        # Return 1.0 — use engine's sizing as-is
        # (RL model integration comes after live data accumulation)
        return 1.0


# ══════════════════════════════════════════════════════════════════════════
# POSITION RECONCILER
# Syncs engine's intended positions with actual Alpaca positions.
# ══════════════════════════════════════════════════════════════════════════

class PositionReconciler:
    """
    Compares what the engine wants with what Alpaca holds.
    Executes the delta (buys/sells needed to reach target).
    """

    def __init__(self, client: AlpacaClient, verbose: bool = True):
        self.client  = client
        self.verbose = verbose

    def get_target_positions(
        self,
        engine_signals: List[Dict],
        equity:         float,
    ) -> Dict[str, float]:
        """
        Build target portfolio from engine signals.
        Returns {ticker: target_notional_value}.
        """
        targets = {}
        for sig in engine_signals:
            if sig["action"] in ("BUY_SIGNAL", "BUY"):
                ticker = sig["ticker"]
                targets[ticker] = sig["pos_value"]
        return targets

    def execute_sells(
        self,
        engine_sell_signals: List[str],
        dry_run: bool = False,
    ) -> List[Dict]:
        """
        Execute sells for tickers the engine wants to exit.
        """
        orders = []
        current = {p["ticker"]: p for p in self.client.get_positions()}

        for ticker in engine_sell_signals:
            if ticker not in current:
                if self.verbose:
                    print(f"  → SELL {ticker}: not in portfolio, skip")
                continue

            pos = current[ticker]
            if self.verbose:
                print(f"  → SELL {ticker}: "
                      f"{pos['qty']:.4f} shares @ "
                      f"${pos['current_price']:.2f} | "
                      f"P&L: ${pos['unrealized_pl']:+.2f}")

            if not dry_run:
                order = self.client.close_position(ticker)
                if order is None:
                    order = {}
                # Ensure the logger has what it needs
                order.setdefault("ticker", ticker)
                order.setdefault("side", "sell")
                order.setdefault("filled_avg_price", pos["current_price"])
                order.setdefault("qty", pos["qty"])
                orders.append(order)
                print(f"    ✓ SELL order placed: {order.get('order_id','?')}")
            else:
                print(f"    [DRY RUN] would sell {ticker}")
                orders.append({"ticker": ticker, "side": "sell", "dry_run": True})

        return orders

    def execute_buys(
        self,
        buy_signals:    List[Dict],
        available_cash: float,
        dry_run:        bool = False,
    ) -> List[Dict]:
        """
        Execute buys from engine BUY signals.
        Caps at available cash to avoid over-investing.
        """
        orders = []
        current = {p["ticker"] for p in self.client.get_positions()}
        cash_remaining = available_cash * 0.95  # keep 5% cash buffer

        for sig in buy_signals:
            ticker    = sig["ticker"]
            notional  = sig["pos_value"]

            if ticker in current:
                if self.verbose:
                    print(f"  → BUY {ticker}: already held, skip")
                continue

            if notional > cash_remaining:
                if self.verbose:
                    print(f"  → BUY {ticker}: insufficient cash "
                          f"(need ${notional:,.0f}, have ${cash_remaining:,.0f})")
                continue

            if self.verbose:
                print(f"  → BUY {ticker}: ${notional:,.0f} notional")

            if not dry_run:
                order = self.client.place_order(
                    ticker   = ticker,
                    side     = "buy",
                    notional = notional,
                )
                orders.append(order)
                if order:
                    cash_remaining -= notional
                    print(f"    ✓ BUY order placed: {order.get('order_id','?')}")
            else:
                print(f"    [DRY RUN] would buy {ticker} for ${notional:,.0f}")
                orders.append({
                    "ticker": ticker, "side": "buy",
                    "notional": notional, "dry_run": True
                })
                cash_remaining -= notional

        return orders


# ══════════════════════════════════════════════════════════════════════════
# TRADE LOGGER
# Appends live trades to CSV matching backtest trade_log.csv format
# so they feed directly into RL retraining later.
# ══════════════════════════════════════════════════════════════════════════

class LiveTradeLogger:
    """Logs live trades in the same format as backtest trade_log.csv."""

    COLUMNS = [
        "date", "ticker", "action", "price", "shares",
        "proba", "weight", "hmm_regime", "reason",
        "portfolio_value", "order_id", "notional",
    ]

    def __init__(self, log_path: str = LIVE_TRADE_LOG):
        self.path = log_path
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.path):
            pd.DataFrame(columns=self.COLUMNS).to_csv(self.path, index=False)

    def log(
        self,
        date_str:        str,
        ticker:          str,
        action:          str,       # BUY or SELL
        price:           float,
        shares:          float,
        proba:           float,
        weight:          float,
        hmm_regime:      str,
        reason:          str,
        portfolio_value: float,
        order_id:        str = "",
        notional:        float = 0.0,
    ):
        row = pd.DataFrame([{
            "date":            date_str,
            "ticker":          ticker,
            "action":          action,
            "price":           round(price, 4),
            "shares":          round(shares, 6),
            "proba":           round(proba, 4),
            "weight":          round(weight, 4),
            "hmm_regime":      hmm_regime,
            "reason":          reason,
            "portfolio_value": round(portfolio_value, 2),
            "order_id":        order_id,
            "notional":        round(notional, 2),
        }])
        row.to_csv(self.path, mode="a", header=False, index=False)

    def log_order(
        self,
        order:     Dict,
        signal:    Dict,
        portfolio: Dict,
    ):
        """Log a completed order using signal context and portfolio state."""
        if not order or "ticker" not in order:
            return
        self.log(
            date_str        = signal.get("date", str(date.today())),
            ticker          = order["ticker"],
            action          = order["side"].upper(),
            price           = order.get("filled_avg_price", 0),
            shares          = order.get("qty", 0),
            proba           = signal.get("confidence", 0.5),
            weight          = signal.get("pos_value", 0) /
                              max(portfolio.get("equity", 100_000), 1),
            hmm_regime      = signal.get("hmm_regime", "Unknown"),
            reason          = signal.get("reason", "live_signal"),
            portfolio_value = portfolio.get("equity", 0),
            order_id        = order.get("order_id", ""),
            notional        = order.get("notional", 0),
        )


# ══════════════════════════════════════════════════════════════════════════
# EQUITY LOGGER
# Appends daily equity to live_equity_curve.csv
# ══════════════════════════════════════════════════════════════════════════

def log_equity(equity: float, regime: str = "Unknown"):
    """Append today's equity to the live equity curve."""
    today = date.today().strftime("%Y-%m-%d")
    row   = pd.DataFrame([{
        "date":   today,
        "equity": round(equity, 2),
        "regime": regime,
    }])
    if not os.path.exists(LIVE_EQUITY):
        row.to_csv(LIVE_EQUITY, index=False)
    else:
        row.to_csv(LIVE_EQUITY, mode="a", header=False, index=False)
    print(f"  ✓ Equity logged: ${equity:,.2f} | regime: {regime}")


# ══════════════════════════════════════════════════════════════════════════
# LIVE ENGINE — MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════

class LiveEngine:
    """
    Orchestrates the daily live trading run.

    Workflow:
      1. Top-up price data from Alpaca
      2. Copy live pkl to engine's expected path
      3. Run engine in signal-generation mode (last N bars)
      4. Intercept signals via RL callback
      5. Execute paper orders via Alpaca
      6. Log trades and equity
      7. Restore backtest pkl
    """

    def __init__(
        self,
        dry_run: bool = False,
        verbose: bool = True,
    ):
        self.dry_run = dry_run
        self.verbose = verbose
        self.client  = AlpacaClient()
        self.feed    = LiveDataFeed(self.client)
        self.logger  = LiveTradeLogger()

        # Lazy-loaded signal engines (loaded once on first signal generation)
        self._global_engine = None
        self._sector_engine = None
        self._hmm           = None
        self._last_regime   = "Unknown"

    def run(self) -> Dict:
        """Execute one daily live trading cycle."""
        today = date.today().strftime("%Y-%m-%d")
        print(f"\n{'═'*60}")
        print(f"  LIVE ENGINE — {today}")
        print(f"  Mode: {'DRY RUN (no real orders)' if self.dry_run else 'PAPER TRADING'}")
        print(f"{'═'*60}\n")

        results = {
            "date":     today,
            "signals":  [],
            "orders":   [],
            "equity":   0.0,
            "regime":   "Unknown",
            "errors":   [],
        }

        try:
            # Step 1 — Get current account state
            print("  [1] Account state")
            account   = self.client.get_account()
            equity    = account["equity"]
            cash      = account["cash"]
            positions = self.client.get_positions()
            print(f"      Equity:    ${equity:,.2f}")
            print(f"      Cash:      ${cash:,.2f}")
            print(f"      Positions: {len(positions)}")
            results["equity"] = equity

            # Step 2 — Refresh price data
            print("\n  [2] Price data top-up")
            self.feed.top_up(days=5)

            # Step 3 — Run signal generation
            # We use a lightweight signal extraction approach:
            # Load the live pkl and run the signal engine components
            # without running the full backtest loop.
            print("\n  [3] Signal generation")
            signals = self._generate_signals(equity)
            results["signals"] = signals

            if not signals:
                print("      No signals today")
            else:
                buy_signals  = [s for s in signals if s["action"] == "BUY"]
                sell_signals = [s["ticker"] for s in signals if s["action"] == "SELL"]
                print(f"      BUY signals:  {len(buy_signals)}")
                print(f"      SELL signals: {len(sell_signals)}")

                # Step 4 — Execute orders
                print("\n  [4] Order execution")
                reconciler = PositionReconciler(self.client, self.verbose)

                # Sells first (free up cash)
                sell_orders = reconciler.execute_sells(sell_signals, self.dry_run)
                results["orders"].extend(sell_orders)

                # Then buys — wait for sell proceeds to settle into buying power
                time.sleep(3)
                fresh_account = self.client.get_account()
                buy_orders    = reconciler.execute_buys(
                    buy_signals,
                    available_cash = fresh_account["buying_power"],
                    dry_run        = self.dry_run,
                )
                results["orders"].extend(buy_orders)

                # Refresh fill prices before logging — orders are submitted async,
                # so filled_avg_price is 0 at placement time.
                for order in buy_orders + sell_orders:
                    oid = order.get("order_id")
                    if not oid or oid in ("?", "", "manual_backfill"):
                        continue
                    filled_ok = False
                    for attempt in range(6):           # up to ~15s total
                        time.sleep(2.5)
                        try:
                            filled = self.client.get_order(oid)
                            if filled.get("filled_avg_price"):
                                order["filled_avg_price"] = filled["filled_avg_price"]
                                order["qty"] = filled["qty"] or order.get("qty", 0)
                                filled_ok = True
                                break
                            if filled.get("status") in ("filled", "partially_filled"):
                                order["filled_avg_price"] = filled.get("filled_avg_price", 0)
                                order["qty"] = filled.get("qty", 0)
                                filled_ok = True
                                break
                        except Exception as e:
                            print(f"      fill refresh error {oid}: {e}")
                    if not filled_ok:
                        print(f"      ⚠ {order.get('ticker','?')} fill not confirmed "
                              f"after retries — logging may show 0, backfill needed")

                # Step 5 — Log trades
                print("\n  [5] Trade logging")
                logged = 0
                for order in buy_orders + sell_orders:
                    if not order or order.get("dry_run"):
                        continue
                    side = order.get("side", "").upper()
                    sig = next(
                        (s for s in signals
                         if s.get("ticker") == order.get("ticker")
                         and s.get("action", "").upper() == side), {})
                    self.logger.log_order(order=order, signal=sig, portfolio={"equity": equity})
                    logged += 1
                print(f"      {logged} trades logged")

            # Step 6 — Log equity
            print("\n  [6] Equity snapshot")
            regime = getattr(self, "_last_regime", "Unknown")
            log_equity(equity, regime)
            results["regime"] = regime

        except Exception as e:
            print(f"\n  ✗ Live engine error: {e}")
            traceback.print_exc()
            results["errors"].append(str(e))

        # Summary
        print(f"\n{'═'*60}")
        print(f"  LIVE RUN COMPLETE")
        print(f"  Signals:  {len(results['signals'])}")
        print(f"  Orders:   {len(results['orders'])}")
        print(f"  Equity:   ${results['equity']:,.2f}")
        print(f"{'═'*60}\n")

        return results

    def _generate_signals(self, equity: float) -> List[Dict]:
        """
        Generate signals matching the backtest architecture:
        HMM regime → global XGBoost (baseline) → sector XGBoost (overwrites)
        → regime gate → sector-native threshold → buy/sell decision.
        """
        signals = []
        today_str = date.today().strftime("%Y-%m-%d")

        try:
            # Load live price data
            if not os.path.exists(LIVE_PKL):
                print("      ⚠ No live pkl found — run full refresh first")
                return signals
            with open(LIVE_PKL, "rb") as f:
                price_data = pickle.load(f)

            # Use the most recent CLOSED bar for signal lookup — the sector engine
            # does exact date matching, and today's bar doesn't exist until close.
            try:
                last_bar_date = (price_data["SPY"]["close"].index[-1]
                                 .strftime("%Y-%m-%d"))
            except Exception:
                last_bar_date = today_str

            held = {p["ticker"] for p in self.client.get_positions()}

            # ── STEP 1: HMM regime ─────────────────────────────────────
            if self._hmm is None:
                hmm_path = os.path.join(ROOT, "5_backtesting", "results",
                                        "hmm_detector.pkl")
                with open(hmm_path, "rb") as f:
                    self._hmm = pickle.load(f)
                print("      ✓ HMM detector loaded")

            self._hmm._price_data = price_data
            regime = self._hmm.get_regime(today_str)
            print(f"      HMM regime: {regime}")
            self._last_regime = regime

            # ── Regime state persistence for flip detection ──────────────
            _state = _load_regime_state()
            prev_regime = _state.get("prev_regime", "Unknown")

            if regime in ("Bear-Trending", "Bear-Stress"):
                _state["bear_confirm_count"] = _state.get("bear_confirm_count", 0) + 1
            else:
                _state["bear_duration_bars"] = _state.get("bear_confirm_count", 0)
                _state["bear_confirm_count"] = 0

            _just_flipped_bull = (
                regime in ("Bull-Trending", "Bull-Stable") and
                prev_regime in ("Bear-Trending", "Bear-Stable", "Bear-Stress")
            )
            _state["prev_regime"] = regime
            _save_regime_state(_state)

            # ── STEP 2: Load signal engines (once) ─────────────────────
            if self._global_engine is None:
                from signal_engine import SignalEngine
                self._global_engine = SignalEngine(
                    os.path.join(ROOT, "4_signals",
                                 "xgboost_global_model.pkl"))
                print("      ✓ Global engine loaded")

            if self._sector_engine is None:
                from sector_signal_engine import SectorSignalEngine
                self._sector_engine = SectorSignalEngine()
                self._sector_engine.load()
                print("      ✓ Sector engine loaded")

            from sector_signal_engine import REGIME_TO_SECTORS, TICKER_TO_SECTOR

            # ── STEP 3: Global engine — baseline proba for all tickers ─
            bar_probas: Dict[str, float] = {}
            for ticker in TICKERS:
                try:
                    g = self._global_engine.get_state(ticker, today_str)
                    if g.get("status") == "success":
                        bar_probas[ticker] = (g.get("state_vector", {})
                                               .get("proba_buy", 0.0))
                except Exception:
                    pass

            print(f"      [dbg] global probas: "
                  f"{ {k: round(v,3) for k,v in bar_probas.items()} }")

            # ── STEP 4: Sector engine overwrites where active ──────────
            sector_thresholds: Dict[str, float] = {}
            for ticker in TICKERS:
                try:
                    sig = self._sector_engine.get_signals(
                        ticker, last_bar_date, regime,
                        self._price_df(price_data, ticker))
                    _p = sig.get("proba_buy", float("nan"))
                    _thr = sig.get('threshold', 0.55)
                    _sell = _thr * 0.80 if _thr else 0.44
                    if not np.isnan(_p):
                        if _p > _thr:
                            _zone = "BUY"
                        elif _p < _sell:
                            _zone = "SELL"
                        else:
                            _zone = "HOLD"
                    else:
                        _zone = "—"
                    _held = "HELD" if ticker in held else ""
                    print(f"      [dbg] {ticker:5} active={str(sig.get('active')):5} "
                          f"proba={_p if not np.isnan(_p) else 'NaN':>6} "
                          f"buy={_thr:.2f} sell={_sell:.3f} "
                          f"zone={_zone:4} {_held}")
                    if sig.get("active") and not np.isnan(sig["proba_buy"]):
                        bar_probas[ticker]        = sig["proba_buy"]
                        sector_thresholds[ticker] = sig["threshold"]
                except Exception as e:
                    print(f"      [dbg] {ticker:5} EXCEPTION: {e}")

            # ── STEP 5: Regime gate ────────────────────────────────────
            anchors        = {"AAPL", "QQQ"}
            active_sectors = REGIME_TO_SECTORS.get(regime, [])

            filtered: Dict[str, float] = {}
            for tkr, proba in bar_probas.items():
                if tkr in anchors:
                    filtered[tkr] = proba
                elif TICKER_TO_SECTOR.get(tkr) in active_sectors:
                    filtered[tkr] = proba

            print(f"      [dbg] regime={regime} active_sectors={active_sectors}")
            print(f"      [dbg] filtered: "
                  f"{ {k: round(v,3) for k,v in filtered.items()} }")
            print(f"      [dbg] held: {held}")

            # ── STEP 6: SELL check on held positions ───────────────────
            for tkr in held:
                if tkr not in filtered:
                    continue
                # Backtest sell_thr: sector native × 0.80, else default 0.45 (line 1444-1453)
                if tkr in sector_thresholds:
                    sell_thr = sector_thresholds[tkr] * 0.80
                else:
                    sell_thr = 0.45      # cfg.sell_threshold default for anchors
                if filtered[tkr] < sell_thr:
                    signals.append({
                        "date":       today_str,
                        "ticker":     tkr,
                        "action":     "SELL",
                        "confidence": round(1.0 - filtered[tkr], 3),
                        "hmm_regime": regime,
                        "reason":     "xgboost_sell_signal",
                    })
                    print(f"      SELL {tkr}: proba={filtered[tkr]:.3f} < sell_thr={sell_thr:.3f}")

            # ── GAP 1: TSLA Strategy A (every day, regime-independent) ───
            tsla_sigs = self._tsla_signals(price_data, regime, equity, held)
            signals.extend(tsla_sigs)
            for s in tsla_sigs:
                print(f"      TSLA {s['action']}: {s['reason']}")

            # ── STEP 7: BUY decisions ──────────────────────────────────
            _SIZE = {
                "Bull-Trending": 0.18,
                "Bull-Stable":   0.12,
                "Bear-Stable":   0.10,
                "Bear-Stress":   0.08,
            }
            pos_size = _SIZE.get(regime, 0.12)

            for ticker, proba in filtered.items():
                if ticker in held:
                    continue
                if ticker == "TSLA":
                    continue  # handled by _tsla_signals above

                if ticker in sector_thresholds:
                    buy_thr = sector_thresholds[ticker]
                elif regime == "Bull-Trending":
                    buy_thr = 0.50
                elif regime == "Bull-Stable":
                    buy_thr = 0.52
                else:
                    buy_thr = 0.58

                if regime == "Bull-Trending" and ticker == "META":
                    buy_thr = 0.48

                if proba > buy_thr:
                    src = "sector" if ticker in sector_thresholds else "global"
                    signals.append({
                        "date":       today_str,
                        "ticker":     ticker,
                        "action":     "BUY",
                        "confidence": round(proba, 3),
                        "hmm_regime": regime,
                        "pos_value":  round(equity * pos_size, 2),
                        "reason":     f"{src}_xgboost_signal",
                    })
                    print(f"      BUY {ticker}: proba={proba:.3f} "
                          f"thr={buy_thr:.2f} [{src}]")

            # ── GAP 2: Anchor re-entry on Bear→Bull flip ───────────────
            if _just_flipped_bull:
                bear_was_short = _state.get("bear_duration_bars", 99) <= 30
                anchor_size = 0.24 if bear_was_short else 0.22
                for anchor in ("AAPL", "QQQ"):
                    if anchor not in held:
                        signals.append({
                            "date":       today_str,
                            "ticker":     anchor,
                            "action":     "BUY",
                            "confidence": 1.0,
                            "hmm_regime": regime,
                            "pos_value":  round(equity * anchor_size, 2),
                            "reason":     "anchor_reentry",
                        })
                        print(f"      ANCHOR BUY {anchor}: "
                              f"{anchor_size:.0%} on bull flip")

        except Exception as e:
            print(f"      ⚠ Signal generation error: {e}")
            traceback.print_exc()

        return signals

    def _tsla_signals(self, price_data: dict, regime: str,
                      equity: float, held: set) -> List[Dict]:
        """Exact replication of backtest TSLA Strategy A (lines 1292-1356)."""
        out = []
        if "TSLA" not in price_data:
            return out
        tc = price_data["TSLA"]["close"]
        if len(tc) < 15:
            return out
        cls = tc.iloc[-200:].values.astype(float)
        _d  = np.diff(cls[-15:]) if len(cls) >= 15 else np.array([0.0])
        _ag = float(np.mean(np.clip(_d, 0, None))) or 1e-9
        _al = float(np.mean(np.clip(-_d, 0, None))) or 1e-9
        rsi = 100.0 - (100.0 / (1.0 + _ag / _al))
        rsi_hist = []
        n = len(cls)
        for _j in range(max(0, n - 17), n):
            _c2 = cls[max(0, _j - 14):_j + 1]
            if len(_c2) < 2:
                continue
            _d2 = np.diff(_c2)
            _g2 = float(np.mean(np.clip(_d2, 0, None))) or 1e-9
            _l2 = float(np.mean(np.clip(-_d2, 0, None))) or 1e-9
            rsi_hist.append(100.0 - (100.0 / (1.0 + _g2 / _l2)))
        rsi_rising  = len(rsi_hist) >= 2 and rsi_hist[-1] > rsi_hist[-2]
        rsi_falling = len(rsi_hist) >= 3 and rsi_hist[-1] < rsi_hist[-2] < rsi_hist[-3]
        rsi_was_hot = len(rsi_hist) >= 3 and max(rsi_hist[-3:]) > 70
        ma200 = float(np.mean(cls[-200:])) if len(cls) >= 200 else float(np.mean(cls))
        px    = float(cls[-1])
        pct   = (px / ma200 - 1) * 100 if ma200 > 0 else 0.0
        in_pos = "TSLA" in held

        peak_exit = in_pos and rsi_was_hot and rsi_falling
        ext_exit  = in_pos and pct > 120.0
        bear_exit = in_pos and regime in ("Bear-Trending", "Bear-Stress")

        if peak_exit or ext_exit or bear_exit:
            reason = ("tsla_peak_exit" if peak_exit else
                      "tsla_ext_exit"  if ext_exit  else
                      f"tsla_bear|{regime}")
            out.append({
                "date": date.today().strftime("%Y-%m-%d"),
                "ticker": "TSLA", "action": "SELL",
                "confidence": 0.0, "hmm_regime": regime,
                "reason": f"{reason}|rsi={rsi:.1f}|pct={pct:.1f}",
            })
        elif (not in_pos and regime == "Bull-Trending"
              and 42 <= rsi <= 62 and -10 <= pct <= 60 and rsi_rising):
            out.append({
                "date": date.today().strftime("%Y-%m-%d"),
                "ticker": "TSLA", "action": "BUY",
                "confidence": 0.88, "hmm_regime": regime,
                "pos_value": round(equity * 0.12 * 0.80, 2),
                "reason": "tsla_runup_entry",
            })
        return out

    def _price_df(self, price_data: dict, ticker: str):
        """Convert price_data[ticker] series dict to OHLCV DataFrame."""
        if ticker not in price_data:
            return None
        d  = price_data[ticker]
        df = pd.DataFrame({
            "open":   d["open"],
            "high":   d["high"],
            "low":    d["low"],
            "close":  d["close"],
            "volume": d["volume"],
        })
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"          # sector engine reset_index() needs this
        return df


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live Engine")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate signals but place no real orders")
    parser.add_argument("--status",  action="store_true",
                        help="Show account + position status only")
    args = parser.parse_args()

    if args.status:
        client    = AlpacaClient()
        account   = client.get_account()
        positions = client.get_positions()
        hours     = client.get_market_hours()
        print(f"\n  Account:    {account['status']}")
        print(f"  Equity:     ${account['equity']:,.2f}")
        print(f"  Cash:       ${account['cash']:,.2f}")
        print(f"  Positions:  {len(positions)}")
        print(f"  Market:     {'OPEN' if hours['is_open'] else 'CLOSED'}")
        print(f"  Next open:  {hours.get('next_open','?')[:16]}")
        if positions:
            print(f"\n  Open positions:")
            for p in positions:
                print(f"    {p['ticker']:6} {p['qty']:>8.2f} shares | "
                      f"${p['market_value']:>10,.2f} | "
                      f"P&L: {p['unrealized_pct']:>+.1f}%")
    else:
        engine = LiveEngine(dry_run=args.dry_run, verbose=True)
        engine.run()