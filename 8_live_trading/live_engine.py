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
HOLD_STATE    = os.path.join(LIVE_DIR, "position_hold_state.json")
MIN_HOLD_BARS = 3   # must match BacktestConfig.min_hold_bars
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


def _load_hold_state() -> dict:
    if os.path.exists(HOLD_STATE):
        try:
            with open(HOLD_STATE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_hold_state(state: dict):
    with open(HOLD_STATE, "w") as f:
        json.dump(state, f, indent=2)


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
                known_price = pos["current_price"]
                known_qty   = pos["qty"]
                order = self.client.close_position(ticker)
                if not order:
                    order = {}
                order.setdefault("ticker", ticker)
                order.setdefault("side", "sell")
                order["filled_qty"] = known_qty   # authoritative floor; upgraded only if API confirms a real qty
                order["pos_value"]  = known_qty * known_price  # notional at exit; log_order divides by equity for weight
                print(f"    ✓ SELL order submitted: {order.get('order_id','?')}")
                oid = order.get("order_id")
                if oid:
                    fp, fq, confirmed = self._await_fill(oid, ticker)
                    if confirmed and fp and float(fp) > 0:
                        order["filled_avg_price"] = float(fp)
                        if fq and float(fq) > 0:
                            order["filled_qty"] = float(fq)  # upgrade to API-confirmed fill qty
                        order["price_estimated"]  = False
                    else:
                        order["filled_avg_price"] = known_price
                        order["price_estimated"]  = True
                else:
                    order["filled_avg_price"] = known_price
                    order["price_estimated"]  = True
                orders.append(order)
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
                if order:
                    cash_remaining -= notional
                    print(f"    ✓ BUY order submitted: {order.get('order_id','?')}")
                    oid = order.get("order_id")
                    if oid:
                        fp, fq, confirmed = self._await_fill(oid, ticker, notional=notional)
                        if confirmed:
                            order["filled_avg_price"] = fp
                            order["filled_qty"]       = fq
                            order["price_estimated"]  = False
                        else:
                            order["price_estimated"] = True
                    else:
                        order["price_estimated"] = True
                    orders.append(order)
                    # seed bars-held at 0 so next-run increment lands at 1
                    # (entry-bar parity with backtest _bars_held logic)
                    _hs = _load_hold_state()
                    if ticker not in _hs:
                        _hs[ticker] = {"bars": 0, "last_inc_date": date.today().isoformat()}
                        _save_hold_state(_hs)
            else:
                print(f"    [DRY RUN] would buy {ticker} for ${notional:,.0f}")
                orders.append({
                    "ticker": ticker, "side": "buy",
                    "notional": notional, "dry_run": True
                })
                cash_remaining -= notional

        return orders

    def _await_fill(self, order_id: str, ticker: str,
                    retries: int = 5, interval: float = 2.0,
                    notional: float = 0.0):
        """Poll Alpaca until both filled_avg_price AND filled_qty > 0.
        Returns (price, qty, confirmed). Fallback chain ensures qty is never 0.

        Fallback order when retries exhaust with price confirmed but qty=0:
          1. Notional order: qty = notional / filled_avg_price (computed, reliable)
          2. Share-based: get_positions() and read actual held qty for ticker
          3. Last resort: log price with qty=0 (prints warning; needs reconciliation)
        If price never arrives: return (None, None, False) → estimated price path.
        """
        last_price = None
        for attempt in range(retries):
            time.sleep(interval)
            try:
                o  = self.client.get_order(order_id)
                fp = o.get("filled_avg_price")
                fq = o.get("filled_qty") or o.get("qty")  # get_order normalizes Alpaca's filled_qty to "qty"
                if fp:
                    last_price = float(fp)
                if fp and fq and float(fq) > 0:
                    price, qty = float(fp), float(fq)
                    print(f"    ✓ {ticker} fill confirmed: "
                          f"${price:.4f} × {qty:.6f} sh (attempt {attempt+1}/{retries})")
                    return price, qty, True
                # price arrived but qty still 0 — keep polling
            except Exception as e:
                print(f"    fill poll error [{ticker}]: {e}")

        # Fallback chain — price confirmed but qty never propagated
        if last_price:
            # 1. Notional order: compute qty from notional / fill_price
            if notional:
                fallback_qty = round(notional / last_price, 6)
                print(f"    ⚠ {ticker}: qty lag — computed qty={fallback_qty:.6f} "
                      f"from notional/price (price confirmed)")
                return last_price, fallback_qty, True
            # 2. Share-based order: pull actual position qty from Alpaca
            try:
                pos_map = {p["ticker"]: p for p in self.client.get_positions()}
                if ticker in pos_map:
                    pos_qty = float(pos_map[ticker]["qty"])
                    if pos_qty > 0:
                        print(f"    ⚠ {ticker}: qty lag — using position "
                              f"qty={pos_qty:.6f} from Alpaca (price confirmed)")
                        return last_price, pos_qty, True
            except Exception as e:
                print(f"    position fallback error [{ticker}]: {e}")
            # 3. Last resort: price confirmed, qty unknown — flag for reconciliation
            print(f"    ✗ {ticker}: fill qty not confirmed after {retries} polls — "
                  f"execute_sells will use known_qty from position snapshot")
            return last_price, 0.0, True

        print(f"    ⚠ {ticker}: fill not confirmed after {retries} retries "
              f"(~{int(retries * interval)}s) — price will be ESTIMATED")
        return None, None, False


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
        "portfolio_value", "order_id", "notional", "price_estimated",
    ]

    def __init__(self, log_path: str = LIVE_TRADE_LOG):
        self.path = log_path
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.path):
            pd.DataFrame(columns=self.COLUMNS).to_csv(self.path, index=False)
            return
        existing = pd.read_csv(self.path)
        for col in self.COLUMNS:
            if col not in existing.columns:
                existing[col] = None
        existing[self.COLUMNS].to_csv(self.path, index=False)

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
        order_id:        str  = "",
        notional:        float = 0.0,
        price_estimated: bool  = False,
    ):
        row = pd.DataFrame([{
            "date":            date_str,
            "ticker":          ticker,
            "action":          action,
            "price":           round(float(price or 0), 4),
            "shares":          round(float(shares or 0), 6),
            "proba":           round(float(proba or 0), 4),
            "weight":          round(float(weight or 0), 4),
            "hmm_regime":      hmm_regime,
            "reason":          reason,
            "portfolio_value": round(float(portfolio_value or 0), 2),
            "order_id":        order_id,
            "notional":        round(float(notional or 0), 2),
            "price_estimated": price_estimated,
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
        _raw_price  = order.get("filled_avg_price") or order.get("price")
        _fill_price = float(_raw_price) if _raw_price is not None \
                      else float(signal.get("price") or 0)
        _raw_qty    = order.get("filled_qty") or order.get("qty")
        _fill_qty   = float(_raw_qty) if _raw_qty is not None else 0.0
        _estimated  = bool(order.get("price_estimated", False))
        if _estimated:
            print(f"    ⚠ {order['ticker']} price ESTIMATED "
                  f"(${_fill_price:.2f}) — fill not confirmed; flagged in log")
        self.log(
            date_str        = signal.get("date", str(date.today())),
            ticker          = order["ticker"],
            action          = order["side"].upper(),
            price           = _fill_price,
            shares          = _fill_qty,
            proba           = signal.get("confidence", 0.5),
            weight          = (order.get("pos_value") or signal.get("pos_value", 0)) /
                              max(portfolio.get("equity", 100_000), 1),
            hmm_regime      = signal.get("hmm_regime", "Unknown"),
            reason          = signal.get("reason", "live_signal"),
            portfolio_value = portfolio.get("equity", 0),
            order_id        = order.get("order_id", ""),
            notional        = order.get("notional", 0),
            price_estimated = _estimated,
        )


# ══════════════════════════════════════════════════════════════════════════
# EQUITY LOGGER
# Appends daily equity to live_equity_curve.csv
# ══════════════════════════════════════════════════════════════════════════

def log_equity(equity: float, regime: str = "Unknown"):
    """Write today's equity to the live equity curve (upsert — one row per date)."""
    today = date.today().strftime("%Y-%m-%d")
    row   = pd.DataFrame([{
        "date":   today,
        "equity": round(equity, 2),
        "regime": regime,
    }])
    if not os.path.exists(LIVE_EQUITY):
        row.to_csv(LIVE_EQUITY, index=False)
    else:
        existing = pd.read_csv(LIVE_EQUITY)
        existing = existing[existing["date"] != today]   # drop same-day row if present
        pd.concat([existing, row], ignore_index=True).to_csv(LIVE_EQUITY, index=False)
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

                # Brief pause for sell proceeds to settle, then refresh buying power.
                # Fill confirmation already happened inside execute_sells/_await_fill.
                time.sleep(1)
                fresh_account = self.client.get_account()
                buy_orders    = reconciler.execute_buys(
                    buy_signals,
                    available_cash = fresh_account["buying_power"],
                    dry_run        = self.dry_run,
                )
                results["orders"].extend(buy_orders)

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
            if not self.dry_run:
                log_equity(equity, regime)
            else:
                print(f"  ✓ Equity (dry-run, not written): ${equity:,.2f} | regime: {regime}")
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
        signals     = []
        today_str   = date.today().strftime("%Y-%m-%d")
        _hold_state = _load_hold_state()   # safe default; overwritten inside try

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

            _all_positions = self.client.get_positions()
            held           = {p["ticker"] for p in _all_positions}
            positions_map  = {p["ticker"]: p for p in _all_positions}

            # ── bars-held counter (idempotent per calendar day) ────────────
            _today_iso  = date.today().isoformat()
            _hold_state = _load_hold_state()
            for _t in held:
                _rec = _hold_state.get(_t, {"bars": -1, "last_inc_date": None})
                if _rec["last_inc_date"] != _today_iso:
                    _rec["bars"] += 1
                    _rec["last_inc_date"] = _today_iso
                _hold_state[_t] = _rec
            for _t in list(_hold_state.keys()):
                if _t not in held:
                    del _hold_state[_t]   # purge closed positions
            bars_held = {_t: _rec["bars"] for _t, _rec in _hold_state.items()}

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

            if regime in ("Bear-Stress", "Bear-Stable"):
                _state["bear_confirm_count"] = _state.get("bear_confirm_count", 0) + 1
            else:
                _state["bear_duration_bars"] = _state.get("bear_confirm_count", 0)
                _state["bear_confirm_count"] = 0

            _just_flipped_bull = (
                regime in ("Bull-Trending", "Bull-Stable") and
                prev_regime in ("Bear-Stable", "Bear-Stress")
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
                    if not np.isnan(sig["proba_buy"]):
                        if sig.get("active"):
                            # active sectors overwrite the global proba for ENTRIES
                            bar_probas[ticker]        = sig["proba_buy"]
                            sector_thresholds[ticker] = sig["threshold"]
                        elif ticker in held:
                            # held-but-inactive: still record proba + threshold for EXIT eval
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

            # ── STEP 6: SELL check on ALL held positions (gate-independent) ──
            for tkr in held:
                # exits are NEVER gated by sector-active — only entries are
                proba = filtered.get(tkr, bar_probas.get(tkr))
                if proba is None or np.isnan(proba):
                    continue
                # sell threshold: sector-native × 0.80 if we have it, else regime default 0.45
                if tkr in sector_thresholds:
                    sell_thr = sector_thresholds[tkr] * 0.80
                else:
                    sell_thr = 0.45
                if proba < sell_thr:
                    # mirror backtest line 1573: profitable + too new → skip sell
                    _pos   = positions_map.get(tkr)
                    _bheld = bars_held.get(tkr, 99)   # 99 = unknown → fail-safe allow sell
                    if (_pos is not None
                            and _bheld < MIN_HOLD_BARS
                            and _pos["current_price"] > _pos["avg_price"]):
                        print(f"      HOLD {tkr}: min-hold ({_bheld}/{MIN_HOLD_BARS} bars, "
                              f"profitable {_pos['current_price']:.2f}>{_pos['avg_price']:.2f})")
                        continue
                    signals.append({
                        "date":       today_str,
                        "ticker":     tkr,
                        "action":     "SELL",
                        "confidence": round(1.0 - proba, 3),
                        "hmm_regime": regime,
                        "reason":     "xgboost_sell_signal",
                    })
                    print(f"      SELL {tkr}: proba={proba:.3f} < sell_thr={sell_thr:.3f} (held, gate-independent)")

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
                        "price":      float(price_data[ticker]["close"].iloc[-1])
                                      if ticker in price_data else 0.0,
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
                            "price":      float(price_data[anchor]["close"].iloc[-1])
                                          if anchor in price_data else 0.0,
                            "reason":     "anchor_reentry",
                        })
                        print(f"      ANCHOR BUY {anchor}: "
                              f"{anchor_size:.0%} on bull flip")

        except Exception as e:
            print(f"      ⚠ Signal generation error: {e}")
            traceback.print_exc()

        if not self.dry_run:
            _save_hold_state(_hold_state)
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
        bear_exit = in_pos and regime in ("Bear-Trending","Bear-Stress")

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
                "price":     float(cls[-1]),
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