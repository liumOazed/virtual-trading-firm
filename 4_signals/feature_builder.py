import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
import importlib.util as _ilu
import os as _os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)


def _load_advanced():
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    spec  = _ilu.spec_from_file_location(
        "advanced_price_features",
        os.path.join(_root, "3_market_data", "advanced_price_features.py")
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def fetch_ohlcv(ticker: str, end_date: str, lookback_days: int = 365) -> pd.DataFrame:
    end   = datetime.strptime(end_date, "%Y-%m-%d")
    start = end - timedelta(days=lookback_days)
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df.dropna()
    return df


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # Trend
    df["sma_20"]      = ta.sma(close, length=20)
    df["sma_50"]      = ta.sma(close, length=50)
    df["ema_10"]      = ta.ema(close, length=10)
    df["ema_20"]      = ta.ema(close, length=20)

    # Momentum
    df["rsi_14"]      = ta.rsi(close, length=14)
    df["rsi_7"]       = ta.rsi(close, length=7)
    df["mom_10"]      = ta.mom(close, length=10)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd"]        = macd.iloc[:, 0]
        df["macd_signal"] = macd.iloc[:, 1]
        df["macd_hist"]   = macd.iloc[:, 2]

    # Volatility
    df["atr_14"]      = ta.atr(high, low, close, length=14)
    df["atr_7"]       = ta.atr(high, low, close, length=7)

    bb = ta.bbands(close, length=20, std=2)
    if bb is not None:
        df["bb_lower"]    = bb.iloc[:, 0]
        df["bb_mid"]      = bb.iloc[:, 1]
        df["bb_upper"]    = bb.iloc[:, 2]
        df["bb_width"]    = (bb.iloc[:, 2] - bb.iloc[:, 0]) / bb.iloc[:, 1]
        df["bb_pct"]      = (close - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0])

    # Volume
    df["vwma_20"]     = ta.vwma(close, vol, length=20)
    df["vol_ratio"]   = vol / vol.rolling(20).mean()

    # Price derived
    df["return_1d"]   = close.pct_change(1)
    df["return_5d"]   = close.pct_change(5)
    df["return_10d"]  = close.pct_change(10)
    df["return_20d"]  = close.pct_change(20)
    df["hl_ratio"]    = (high - low) / close
    df["gap"]         = (close - close.shift(1)) / close.shift(1)

    # Trend strength
    df["above_sma20"] = (close > df["sma_20"]).astype(int)
    df["above_sma50"] = (close > df["sma_50"]).astype(int)
    df["golden_cross"] = (df["sma_20"] > df["sma_50"]).astype(int)

    return df


def add_fundamental_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Safe fundamental features - handles ETFs and missing data gracefully"""
    try:
        info = yf.Ticker(ticker).info
        
        fundamentals = {
            "pe_ratio":    info.get("trailingPE"),
            "forward_pe":  info.get("forwardPE"),
            "pb_ratio":    info.get("priceToBook"),
            "debt_equity": info.get("debtToEquity"),
            "roe":         info.get("returnOnEquity"),
            "beta":        info.get("beta"),
            "short_ratio": info.get("shortRatio"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow":  info.get("fiftyTwoWeekLow"),
        }
        
        for col, value in fundamentals.items():
            if value is not None:
                df[col] = value
            else:
                df[col] = np.nan
                
        # Safe 52w percent
        if fundamentals["fiftyTwoWeekHigh"] and fundamentals["fiftyTwoWeekLow"]:
            high = fundamentals["fiftyTwoWeekHigh"]
            low  = fundamentals["fiftyTwoWeekLow"]
            df["pct_from_52h"] = (df["close"] - high) / high
            df["pct_from_52l"] = (df["close"] - low) / low
        else:
            df["pct_from_52h"] = 0.0
            df["pct_from_52l"] = 0.0
            
    except Exception as e:
        print(f"   ⚠️  Fundamentals failed for {ticker}: {e}")
        # Fill with NaNs instead of crashing
        for col in ["pe_ratio", "forward_pe", "pb_ratio", "debt_equity", 
                   "roe", "beta", "short_ratio", "pct_from_52h", "pct_from_52l"]:
            df[col] = np.nan
            
    return df


def add_labels(df: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
    future_return = df["close"].pct_change(forward_days).shift(-forward_days)
    df["label"]        = (future_return > 0).astype(int)   # 1=BUY 0=SELL/HOLD
    df["future_return"] = future_return
    return df


def build_features(
    ticker: str,
    end_date: str,
    lookback_days: int = 730,
    forward_days: int = 5,
    sentiment_score: float = 0.0,
) -> pd.DataFrame:
    
    print(f"   → Fetching OHLCV for {ticker}...")
    df = fetch_ohlcv(ticker, end_date, lookback_days)

    if df.empty:
        print(f"   ❌ No OHLCV data for {ticker}")
        return pd.DataFrame()

    print(f"   → Computing technical features...")
    df = add_technical_features(df)

    print(f"   → Adding fundamental features...")
    df = add_fundamental_features(df, ticker)
    
    print(f"   → Computing advanced price features...")
    apf = _load_advanced()
    df  = apf.add_advanced_price_features(df)

    print(f"   → Adding labels...")
    df = add_labels(df, forward_days)

    df["sentiment"] = sentiment_score

    # <<<--- THIS IS THE KEY FIX --->
    # Drop ONLY rows that have critical NaNs (technical + label), keep fundamentals as NaN
    critical_cols = ["close", "rsi_14", "atr_14", "label"]
    df = df.dropna(subset=critical_cols)
    
    # Fill remaining NaNs (fundamentals) with 0 or median
    df = df.fillna(0.0)

    print(f"   ✅ Feature matrix: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def get_feature_columns() -> list:
    return [
        "rsi_14", "rsi_7", "mom_10",
        "macd", "macd_signal", "macd_hist",
        "atr_14", "atr_7",
        "bb_width", "bb_pct",
        "vol_ratio",
        "return_1d", "return_5d", "return_10d", "return_20d",
        "hl_ratio", "gap",
        "above_sma20", "above_sma50", "golden_cross",
        "pe_ratio", "forward_pe", "pb_ratio",
        "debt_equity", "roe", "beta", "short_ratio",
        "pct_from_52h", "pct_from_52l",
        "sentiment",
    ]


if __name__ == "__main__":
    from datetime import date
    TODAY = date.today().strftime("%Y-%m-%d")

    print("=" * 55)
    print("  FEATURE BUILDER TEST")
    print("=" * 55)

    df = build_features("AAPL", TODAY, lookback_days=365, forward_days=5)

    feature_cols = get_feature_columns()
    available    = [c for c in feature_cols if c in df.columns]

    print(f"\n📊 Feature sample (last 3 rows):")
    print(df[available].tail(3).round(4).to_string())
    print(f"\n🏷️  Label distribution:")
    print(df["label"].value_counts())
    print(f"\n✅ Feature builder ready — {len(available)} features")
    