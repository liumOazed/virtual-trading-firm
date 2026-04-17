import os
from dotenv import load_dotenv
from groq import Groq
from newsapi import NewsApiClient
import yfinance as yf
import pandas as pd

load_dotenv()

def test_groq():
    print("\n🔧 Testing Groq...")
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Say 'Groq connected.' only."}],
            max_tokens=10
        )
        print(f"✅ Groq OK → {response.choices[0].message.content.strip()}")
        return True
    except Exception as e:
        print(f"❌ Groq failed → {e}")
        return False

def test_newsapi():
    print("\n🔧 Testing NewsAPI...")
    try:
        client = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))
        result = client.get_top_headlines(category="business", language="en", page_size=1)
        title = result["articles"][0]["title"] if result["articles"] else "No articles"
        print(f"✅ NewsAPI OK → {title[:60]}...")
        return True
    except Exception as e:
        print(f"❌ NewsAPI failed → {e}")
        return False

def test_yfinance():
    print("\n🔧 Testing yfinance...")
    try:
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="2d")
        if hist.empty:
            raise ValueError("Empty data returned")
        latest_close = round(hist["Close"].iloc[-1], 2)
        print(f"✅ yfinance OK → AAPL latest close: ${latest_close}")
        return True
    except Exception as e:
        print(f"❌ yfinance failed → {e}")
        return False

def test_pandas_ta():
    print("\n🔧 Testing pandas-ta...")
    try:
        import pandas_ta as ta
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="30d")
        hist.ta.rsi(length=14, append=True)
        rsi_col = [c for c in hist.columns if "RSI" in c][0]
        latest_rsi = round(hist[rsi_col].dropna().iloc[-1], 2)
        print(f"✅ pandas-ta OK → AAPL RSI(14): {latest_rsi}")
        return True
    except Exception as e:
        print(f"❌ pandas-ta failed → {e}")
        return False

def test_filterpy():
    print("\n🔧 Testing filterpy...")
    try:
        from filterpy.kalman import EnsembleKalmanFilter
        import numpy as np

        N = 20       # ensemble members
        dim_x = 1    # state dimension
        dim_z = 1    # observation dimension

        def hx(x): return x
        def fx(x, dt): return x

        ekf = EnsembleKalmanFilter(x=np.zeros(dim_x), P=np.eye(dim_x),
                                   dim_z=dim_z, dt=1.0, N=N, hx=hx, fx=fx)
        ekf.update(np.array([1.0]))
        print("✅ filterpy OK → EnKF initialized and updated successfully")
        return True
    except Exception as e:
        print(f"❌ filterpy failed → {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("   VIRTUAL TRADING FIRM — CONNECTION TEST")
    print("=" * 50)

    results = {
        "Groq":      test_groq(),
        "NewsAPI":   test_newsapi(),
        "yfinance":  test_yfinance(),
        "pandas-ta": test_pandas_ta(),
        "filterpy":  test_filterpy(),
    }

    print("\n" + "=" * 50)
    print("   SUMMARY")
    print("=" * 50)
    passed = sum(results.values())
    total = len(results)
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\n  {passed}/{total} connections passed")
    if passed == total:
        print("\n  🚀 All systems go. Ready to build.")
    else:
        print("\n  ⚠️  Fix the failed ones before proceeding.")
    print("=" * 50)