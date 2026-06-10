"""
alpaca_client.py
================
Layer 8 — Alpaca Paper Trading API wrapper.

Clean interface between the virtual trading firm and Alpaca's REST API.
All order placement, position management, and market data goes through here.

Used by:
  live_engine.py      — places BUY/SELL orders from engine signals
  live_data_feed.py   — fetches real-time + historical price data
  position_sync.py    — syncs engine state with real Alpaca positions

Paper trading endpoint: https://paper-api.alpaca.markets
Data endpoint:          https://data.alpaca.markets

All methods return clean dicts — no raw Alpaca objects leak out.
"""

import os
import time
import requests
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════

TRADE_URL = "https://paper-api.alpaca.markets"
DATA_URL  = "https://data.alpaca.markets"

# Tickers the engine trades — must match backtest_engine_v2.py
TICKERS = [
    "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "AVGO", "TSM", "RACE",
    "QQQ",  "XOM",  "CVX", "PG", "WMT", "GLD",
]


# ══════════════════════════════════════════════════════════════════════════
# ALPACA CLIENT
# ══════════════════════════════════════════════════════════════════════════

class AlpacaClient:
    """
    Clean wrapper around Alpaca paper trading REST API.

    Usage:
        client = AlpacaClient()
        account = client.get_account()
        bars    = client.get_bars("NVDA", days=30)
        order   = client.place_order("NVDA", side="buy", notional=5000)
        pos     = client.get_position("NVDA")
        client.close_position("NVDA")
    """

    def __init__(
        self,
        api_key:    Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        self._api_key    = api_key    or os.getenv("ALPACA_API_KEY",    "")
        self._secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")

        if not self._api_key or not self._secret_key:
            raise ValueError(
                "Missing Alpaca API keys. Add to .env:\n"
                "  ALPACA_API_KEY=your_key\n"
                "  ALPACA_SECRET_KEY=your_secret"
            )

        self._headers = {
            "APCA-API-KEY-ID":     self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
            "accept":              "application/json",
            "content-type":        "application/json",
        }
        self._session = requests.Session()
        self._session.headers.update(self._headers)
        print(f"  AlpacaClient ready | key: {self._api_key[:8]}...")

    # ── internal request helper ───────────────────────────────────────────

    def _get(self, base: str, path: str, params: dict = None) -> dict:
        resp = self._session.get(f"{base}{path}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _post(self, base: str, path: str, payload: dict) -> dict:
        resp = self._session.post(f"{base}{path}", json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, base: str, path: str) -> dict:
        resp = self._session.delete(f"{base}{path}", timeout=15)
        if resp.status_code == 204:
            return {}
        resp.raise_for_status()
        return resp.json() if resp.text else {}

    # ── account ───────────────────────────────────────────────────────────

    def get_account(self) -> Dict:
        """Returns account status, equity, buying power, cash."""
        data = self._get(TRADE_URL, "/v2/account")
        return {
            "id":            data["id"],
            "status":        data["status"],
            "equity":        round(float(data["equity"]),        2),
            "cash":          round(float(data["cash"]),          2),
            "buying_power":  round(float(data["buying_power"]),  2),
            "portfolio_val": round(float(data["portfolio_value"]),2),
            "daytrade_count":int(data.get("daytrade_count", 0)),
        }

    def get_portfolio_history(self, days: int = 30) -> List[Dict]:
        """Returns daily equity curve for last N days."""
        try:
            data = self._get(TRADE_URL, "/v2/account/portfolio/history",
                             params={"period": f"{days}D", "timeframe": "1D"})
            timestamps = data.get("timestamp", [])
            equities   = data.get("equity",    [])
            return [
                {
                    "date":   datetime.fromtimestamp(ts).strftime("%Y-%m-%d"),
                    "equity": round(float(eq), 2),
                }
                for ts, eq in zip(timestamps, equities)
                if eq is not None
            ]
        except Exception as e:
            print(f"  ⚠ Portfolio history failed: {e}")
            return []

    # ── positions ─────────────────────────────────────────────────────────

    def get_positions(self) -> List[Dict]:
        """Returns all open positions."""
        data = self._get(TRADE_URL, "/v2/positions")
        positions = []
        for p in data:
            positions.append({
                "ticker":       p["symbol"],
                "qty":          float(p["qty"]),
                "market_value": round(float(p["market_value"]),     2),
                "avg_price":    round(float(p["avg_entry_price"]),  2),
                "current_price":round(float(p["current_price"]),    2),
                "unrealized_pl":round(float(p["unrealized_pl"]),    2),
                "unrealized_pct":round(float(p["unrealized_plpc"]) * 100, 2),
                "side":         p["side"],
            })
        return positions

    def get_position(self, ticker: str) -> Optional[Dict]:
        """Returns position for one ticker, or None if not held."""
        try:
            p = self._get(TRADE_URL, f"/v2/positions/{ticker}")
            return {
                "ticker":        p["symbol"],
                "qty":           float(p["qty"]),
                "market_value":  round(float(p["market_value"]),    2),
                "avg_price":     round(float(p["avg_entry_price"]), 2),
                "current_price": round(float(p["current_price"]),   2),
                "unrealized_pl": round(float(p["unrealized_pl"]),   2),
                "unrealized_pct":round(float(p["unrealized_plpc"]) * 100, 2),
            }
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise

    def close_position(self, ticker: str) -> Dict:
        """Close entire position for ticker. Returns order dict."""
        try:
            resp = self._delete(TRADE_URL, f"/v2/positions/{ticker}")
            return resp
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return {}   # already closed
            raise

    def close_all_positions(self) -> List[Dict]:
        """Close all open positions. Use carefully."""
        try:
            resp = self._session.delete(
                f"{TRADE_URL}/v2/positions",
                params={"cancel_orders": True},
                timeout=15,
            )
            return resp.json() if resp.text else []
        except Exception as e:
            print(f"  ⚠ close_all_positions failed: {e}")
            return []

    # ── orders ────────────────────────────────────────────────────────────

    def place_order(
        self,
        ticker:        str,
        side:          str,              # "buy" or "sell"
        notional:      Optional[float] = None,  # dollar amount (preferred)
        qty:           Optional[float] = None,  # shares (fallback)
        order_type:    str = "market",
        time_in_force: str = "gtc",
    ) -> Dict:
        """
        Place a paper order.

        Preferred: notional (dollar amount) so we match the engine's
        pos_value exactly. Falls back to qty if notional not supported.

        Returns order dict with id, status, filled_avg_price.
        """
        # Alpaca rule: fractional AND notional orders must be DAY orders
        _is_fractional_qty = (qty is not None and float(qty) != int(float(qty)))
        if notional is not None or _is_fractional_qty:
            effective_tif = "day"
        else:
            effective_tif = time_in_force
        payload: Dict = {
            "symbol":        ticker,
            "side":          side,
            "type":          order_type,
            "time_in_force": effective_tif,
        }
        if notional is not None:
            payload["notional"] = str(round(notional, 2))
        elif qty is not None:
            payload["qty"] = str(round(qty, 6))
        else:
            raise ValueError("Either notional or qty required")

        try:
            data = self._post(TRADE_URL, "/v2/orders", payload)
            return {
                "order_id":        data["id"],
                "ticker":          data["symbol"],
                "side":            data["side"],
                "status":          data["status"],
                "qty":             float(data.get("qty") or 0),
                "notional":        float(data.get("notional") or 0),
                "filled_avg_price":float(data.get("filled_avg_price") or 0),
                "submitted_at":    data.get("submitted_at", ""),
            }
        except requests.exceptions.HTTPError as e:
            print(f"  ⚠ Order failed {side} {ticker}: {e.response.text[:200]}")
            return {}

    def get_order(self, order_id: str) -> Dict:
        """Fetch a single order by ID and return fill data."""
        data = self._get(TRADE_URL, f"/v2/orders/{order_id}")
        return {
            "order_id":         data["id"],
            "status":           data["status"],
            "filled_avg_price": float(data.get("filled_avg_price") or 0),
            "qty":              float(data.get("filled_qty") or data.get("qty") or 0),
        }

    def get_orders(self, status: str = "open") -> List[Dict]:
        """Returns orders filtered by status: open, closed, all."""
        data = self._get(TRADE_URL, "/v2/orders",
                         params={"status": status, "limit": 100})
        return [
            {
                "order_id": o["id"],
                "ticker":   o["symbol"],
                "side":     o["side"],
                "status":   o["status"],
                "qty":      float(o.get("qty") or 0),
                "filled_avg_price": float(o.get("filled_avg_price") or 0),
            }
            for o in data
        ]

    def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        try:
            self._session.delete(f"{TRADE_URL}/v2/orders", timeout=15)
        except Exception as e:
            print(f"  ⚠ cancel_all_orders: {e}")

    # ── market data ───────────────────────────────────────────────────────

    def get_bars(
        self,
        ticker:    str,
        days:      int = 252,
        timeframe: str = "1Day",
    ) -> List[Dict]:
        """
        Fetch OHLCV bars for ticker.
        Returns list of dicts: date, open, high, low, close, volume.
        Sorted oldest → newest.
        """
        start = (date.today() - timedelta(days=days + 10)).isoformat()
        end   = date.today().isoformat()
        bars  = []
        url   = f"{DATA_URL}/v2/stocks/{ticker}/bars"
        params = {
            "timeframe": timeframe,
            "start":     start,
            "end":       end,
            "limit":     1000,
            "feed":      "iex",    # free tier feed
        }

        while True:
            try:
                data      = self._get(DATA_URL,
                                      f"/v2/stocks/{ticker}/bars",
                                      params=params)
                raw_bars  = data.get("bars", [])
                for b in raw_bars:
                    bars.append({
                        "date":   b["t"][:10],
                        "open":   round(float(b["o"]), 4),
                        "high":   round(float(b["h"]), 4),
                        "low":    round(float(b["l"]), 4),
                        "close":  round(float(b["c"]), 4),
                        "volume": int(b["v"]),
                    })
                next_token = data.get("next_page_token")
                if not next_token:
                    break
                params["page_token"] = next_token
                time.sleep(0.1)
            except Exception as e:
                print(f"  ⚠ get_bars {ticker}: {e}")
                break

        return sorted(bars, key=lambda x: x["date"])

    def get_bars_multi(
        self,
        tickers:   List[str],
        days:      int = 252,
        timeframe: str = "1Day",
    ) -> Dict[str, List[Dict]]:
        """
        Fetch bars for multiple tickers in one call (efficient).
        Returns {ticker: [bar_dicts]}.
        """
        start  = (date.today() - timedelta(days=days + 10)).isoformat()
        end    = date.today().isoformat()
        result = {tk: [] for tk in tickers}
        params = {
            "symbols":   ",".join(tickers),
            "timeframe": timeframe,
            "start":     start,
            "end":       end,
            "limit":     1000,
            "feed":      "iex",
        }

        while True:
            try:
                data     = self._get(DATA_URL, "/v2/stocks/bars",
                                     params=params)
                raw_bars = data.get("bars", {})
                for tk, bars in raw_bars.items():
                    if tk in result:
                        for b in bars:
                            result[tk].append({
                                "date":   b["t"][:10],
                                "open":   round(float(b["o"]), 4),
                                "high":   round(float(b["h"]), 4),
                                "low":    round(float(b["l"]), 4),
                                "close":  round(float(b["c"]), 4),
                                "volume": int(b["v"]),
                            })
                next_token = data.get("next_page_token")
                if not next_token:
                    break
                params["page_token"] = next_token
                time.sleep(0.2)
            except Exception as e:
                print(f"  ⚠ get_bars_multi: {e}")
                break

        # Sort each ticker oldest → newest
        for tk in result:
            result[tk] = sorted(result[tk], key=lambda x: x["date"])
        return result

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """Returns latest trade price for ticker."""
        try:
            data = self._get(DATA_URL,
                             f"/v2/stocks/{ticker}/trades/latest",
                             params={"feed": "iex"})
            return round(float(data["trade"]["p"]), 4)
        except Exception:
            try:
                data = self._get(DATA_URL,
                                 f"/v2/stocks/{ticker}/quotes/latest",
                                 params={"feed": "iex"})
                q = data["quote"]
                return round((float(q["ap"]) + float(q["bp"])) / 2, 4)
            except Exception as e:
                print(f"  ⚠ get_latest_price {ticker}: {e}")
                return None

    def market_is_open(self) -> bool:
        """Returns True if US market is currently open."""
        try:
            data = self._get(TRADE_URL, "/v2/clock")
            return bool(data.get("is_open", False))
        except Exception:
            return False

    def get_market_hours(self) -> Dict:
        """Returns today's market open/close times."""
        try:
            data = self._get(TRADE_URL, "/v2/clock")
            return {
                "is_open":   data.get("is_open", False),
                "next_open": data.get("next_open", ""),
                "next_close":data.get("next_close", ""),
                "timestamp": data.get("timestamp", ""),
            }
        except Exception as e:
            return {"is_open": False, "error": str(e)}

    def is_trading_day(self) -> bool:
        """Returns True if today is a trading day (not weekend/holiday)."""
        try:
            today = date.today().isoformat()
            data  = self._get(TRADE_URL, "/v2/calendar",
                              params={"start": today, "end": today})
            return len(data) > 0
        except Exception:
            # Fallback: check if weekday
            return date.today().weekday() < 5

    # ── portfolio state (mirrors what engine needs) ───────────────────────

    def get_portfolio_state(self) -> Dict:
        """
        Returns portfolio state dict matching what backtest_engine_v2.py
        expects from portfolio.get_portfolio_state().
        Used by live_engine.py to sync state.
        """
        account   = self.get_account()
        positions = self.get_positions()

        total_market_val = sum(p["market_value"] for p in positions)
        equity           = account["equity"]
        heat             = total_market_val / equity if equity > 0 else 0

        return {
            "equity":       equity,
            "cash":         account["cash"],
            "buying_power": account["buying_power"],
            "heat":         round(heat, 4),
            "n_positions":  len(positions),
            "positions":    {p["ticker"]: p for p in positions},
            "drawdown":     0.0,   # computed by live_engine from history
        }


# ══════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═"*55)
    print("  AlpacaClient — quick test")
    print("═"*55 + "\n")

    client = AlpacaClient()

    # Account
    acc = client.get_account()
    print(f"  Account:       {acc['status']}")
    print(f"  Equity:        ${acc['equity']:,.2f}")
    print(f"  Buying power:  ${acc['buying_power']:,.2f}")

    # Market hours
    hours = client.get_market_hours()
    print(f"  Market open:   {hours['is_open']}")
    print(f"  Next open:     {hours.get('next_open','?')[:16]}")

    # Latest price
    price = client.get_latest_price("NVDA")
    print(f"  NVDA price:    ${price}")

    # Bars for NVDA (last 10 days)
    bars = client.get_bars("NVDA", days=10)
    print(f"  NVDA bars:     {len(bars)} days")
    if bars:
        b = bars[-1]
        print(f"  Latest bar:    {b['date']} close=${b['close']}")

    # Positions
    positions = client.get_positions()
    print(f"  Open positions:{len(positions)}")

    # Portfolio state
    state = client.get_portfolio_state()
    print(f"  Heat:          {state['heat']:.1%}")

    print(f"\n  ✓ AlpacaClient working\n")