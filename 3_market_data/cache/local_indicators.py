import pandas as pd
import pandas_ta as ta
import yfinance as yf
from datetime import datetime, timedelta
from langchain_core.tools import tool
from typing import Annotated


def _fetch_ohlcv(symbol: str, curr_date: str, look_back_days: int) -> pd.DataFrame:
    end = datetime.strptime(curr_date, "%Y-%m-%d")
    start = end - timedelta(days=look_back_days + 50)  # extra buffer for indicators
    df = yf.download(symbol, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df.dropna()


def _compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    df["sma_50"]    = ta.sma(close, length=50)
    df["sma_200"]   = ta.sma(close, length=min(200, len(df)))
    df["ema_10"]    = ta.ema(close, length=10)
    df["rsi_14"]    = ta.rsi(close, length=14)
    df["atr_14"]    = ta.atr(high, low, close, length=14)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd"]        = macd.iloc[:, 0]
        df["macd_signal"] = macd.iloc[:, 1]
        df["macd_hist"]   = macd.iloc[:, 2]

    bb = ta.bbands(close, length=20, std=2)
    if bb is not None:
        df["bb_lower"]  = bb.iloc[:, 0]
        df["bb_mid"]    = bb.iloc[:, 1]
        df["bb_upper"]  = bb.iloc[:, 2]

    df["vwma_20"] = ta.vwma(close, vol, length=20)

    return df


def _format_summary(symbol: str, df: pd.DataFrame, curr_date: str) -> str:
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    def v(col):
        val = last.get(col)
        return f"{val:.4f}" if pd.notna(val) else "N/A"

    price_change = last["close"] - prev["close"]
    pct_change   = (price_change / prev["close"]) * 100 if prev["close"] else 0

    # RSI signal
    rsi = last.get("rsi_14")
    if pd.notna(rsi):
        if rsi > 70:   rsi_signal = "OVERBOUGHT ⚠️"
        elif rsi < 30: rsi_signal = "OVERSOLD 🟢"
        else:          rsi_signal = "NEUTRAL"
    else:
        rsi_signal = "N/A"

    # MACD signal
    macd_val  = last.get("macd")
    macd_sig  = last.get("macd_signal")
    if pd.notna(macd_val) and pd.notna(macd_sig):
        macd_cross = "BULLISH crossover 🟢" if macd_val > macd_sig else "BEARISH crossover 🔴"
    else:
        macd_cross = "N/A"

    # Bollinger position
    bb_u = last.get("bb_upper")
    bb_l = last.get("bb_lower")
    bb_m = last.get("bb_mid")
    close_price = last["close"]
    if pd.notna(bb_u) and pd.notna(bb_l):
        bb_width = bb_u - bb_l
        bb_pct   = ((close_price - bb_l) / bb_width * 100) if bb_width else 0
        if close_price >= bb_u:   bb_pos = "AT/ABOVE upper band — potential overbought"
        elif close_price <= bb_l: bb_pos = "AT/BELOW lower band — potential oversold"
        else:                     bb_pos = f"{bb_pct:.1f}% within bands"
    else:
        bb_pos = "N/A"

    # Trend context
    sma50  = last.get("sma_50")
    sma200 = last.get("sma_200")
    if pd.notna(sma50) and pd.notna(sma200):
        trend = "BULLISH (Golden Cross zone)" if sma50 > sma200 else "BEARISH (Death Cross zone)"
    else:
        trend = "N/A"

    summary = f"""
=== LOCAL TECHNICAL ANALYSIS: {symbol} as of {curr_date} ===
[Computed locally via pandas-ta — no token cost]

PRICE ACTION
  Close      : ${close_price:.2f}
  Change     : ${price_change:+.2f} ({pct_change:+.2f}%)
  High       : ${last['high']:.2f}
  Low        : ${last['low']:.2f}
  Volume     : {int(last['volume']):,}

TREND INDICATORS
  SMA 50     : {v('sma_50')}
  SMA 200    : {v('sma_200')}
  EMA 10     : {v('ema_10')}
  Trend      : {trend}

MOMENTUM
  RSI (14)   : {v('rsi_14')} → {rsi_signal}
  MACD       : {v('macd')}
  MACD Signal: {v('macd_signal')}
  MACD Hist  : {v('macd_hist')}
  Signal     : {macd_cross}

VOLATILITY
  ATR (14)   : {v('atr_14')}
  BB Upper   : {v('bb_upper')}
  BB Mid     : {v('bb_mid')}
  BB Lower   : {v('bb_lower')}
  BB Position: {bb_pos}

VOLUME
  VWMA (20)  : {v('vwma_20')}
  Vol vs VWMA: {"Above avg 📈" if pd.notna(last.get('vwma_20')) and last['volume'] > last.get('vwma_20', 0) else "Below avg 📉"}

LAST {min(5, len(df))} CLOSES
{df[['close','rsi_14','macd']].tail(5).round(2).to_string()}
"""
    return summary.strip()


@tool
def get_indicators_local(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "ignored — all indicators computed locally"],
    curr_date: Annotated[str, "The current trading date YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 90,
) -> str:
    """
    Compute ALL technical indicators locally using pandas-ta.
    Returns a pre-formatted summary — no repeated tool calls needed.
    Covers: SMA50, SMA200, EMA10, RSI, MACD, ATR, Bollinger Bands, VWMA.
    """
    try:
        df = _fetch_ohlcv(symbol, curr_date, look_back_days)
        if df.empty:
            return f"No data found for {symbol} up to {curr_date}"
        df = _compute_all_indicators(df)
        return _format_summary(symbol, df, curr_date)
    except Exception as e:
        return f"Error computing indicators for {symbol}: {e}"


def patch_tradingagents():
    """
    Monkey-patch TradingAgents to use local pandas-ta indicators
    instead of routing through Groq tool calls.
    Call this BEFORE initializing TradingAgentsGraph.
    """
    import tradingagents.agents.utils.agent_utils as agent_utils
    import tradingagents.agents.utils.technical_indicators_tools as ta_tools
    import tradingagents.agents.analysts.market_analyst as market_mod

    agent_utils.get_indicators   = get_indicators_local
    ta_tools.get_indicators      = get_indicators_local

    print("✅ pandas-ta patch applied — indicators computed locally")