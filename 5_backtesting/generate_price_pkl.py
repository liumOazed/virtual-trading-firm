"""
generate_price_pkl.py
=====================
One-off script — generates price_data.pkl for rl_agent.py
WITHOUT re-running the full backtest engine.

Run from your project root:
    python 5_backtesting/generate_price_pkl.py
"""

import os, sys, pickle

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "5_backtesting"))

from backtest_engine_v2 import BacktestConfig, DataLoader

cfg = BacktestConfig(
    tickers    = ["AAPL", "NVDA", "MSFT", "SPY", "QQQ", "TSLA"],
    start_date = "2023-01-01",
    end_date   = "2025-04-01",
)

print("📥 Downloading price data …")
loader = DataLoader(cfg)
loader.load()

out_path = "5_backtesting/results/price_data.pkl"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "wb") as f:
    pickle.dump(loader.price_data, f)

print(f"✅ price_data.pkl saved → {out_path}")
print(f"   Tickers: {list(loader.price_data.keys())}")
print(f"   Now run rl_agent.py")