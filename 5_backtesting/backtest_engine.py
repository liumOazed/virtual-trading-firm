import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf

# Add paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, '4_signals'))

from signal_engine import SignalEngine
from portfolio import Portfolio


class BacktestEngine:
    def __init__(self, tickers, start_date, end_date, initial_capital=100000.0):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

        self.engine = SignalEngine()
        self.portfolio = Portfolio(initial_capital=initial_capital)

        self.price_data = {}
        self.signal_cache = {}

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    def preload_data(self):
        print("📥 Pre-loading price data...")
        
        # Calculate extended date range for feature calculation
        # We need extra lookback for features like Hurst (252 days) and ESN (100 days)
        from datetime import datetime, timedelta
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        extended_start = (start_dt - timedelta(days=400)).strftime("%Y-%m-%d")

        for ticker in self.tickers:
            try:
                df = yf.download(
                    ticker,
                    start=extended_start,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=True
                )

                if df.empty:
                    print(f"⚠️ No data for {ticker}")
                    continue

                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.columns = ['open', 'high', 'low', 'close', 'volume']

                self.price_data[ticker] = df
                print(f"✅ {ticker}: {len(df)} rows")

            except Exception as e:
                print(f"❌ {ticker} failed: {e}")

    # -----------------------------
    # PRECOMPUTE SIGNALS (FAST)
    # -----------------------------
    def precompute_signals(self):
        print("\n🧠 Precomputing signals ONCE...")

        for ticker in self.tickers:
            if ticker not in self.price_data:
                continue

            df = self.price_data[ticker].copy()
            df = df.reset_index()

            # FIX: correct date column
            df['date'] = df['Date'].dt.strftime('%Y-%m-%d')

            try:
                state_df = self.engine.get_full_signals(df, ticker)

                # MUST include proba_buy
                if 'proba_buy' not in state_df.columns:
                    print(f"❌ {ticker} missing proba_buy → model broken")
                    continue

                self.signal_cache[ticker] = state_df.set_index('date')

                print(f"✅ Cached: {ticker}")

            except Exception as e:
                print(f"❌ Signal fail {ticker}: {e}")

    # -----------------------------
    # MAIN LOOP
    # -----------------------------
    def run(self, fixed_size=0.10):

        if not self.price_data:
            self.preload_data()
            self.precompute_signals()

        print(f"\n🚀 Backtest: {self.start_date} → {self.end_date}\n")

        sample = next(iter(self.price_data.values()))
        all_dates = sample.index.strftime('%Y-%m-%d').tolist()

        for i, current_date in enumerate(all_dates):

            daily_prices = {}

            for ticker in self.tickers:

                if ticker not in self.price_data:
                    continue

                df = self.price_data[ticker]
                if current_date not in df.index.strftime('%Y-%m-%d'):
                    continue

                price = float(df.loc[current_date]['close'])
                daily_prices[ticker] = price

                # -----------------------
                # GET SIGNAL (FROM CACHE)
                # -----------------------
                if ticker not in self.signal_cache:
                    continue

                cache_df = self.signal_cache[ticker]

                if current_date not in cache_df.index:
                    continue

                row = cache_df.loc[current_date]

                proba = float(row.get("proba_buy", np.nan))

                # 🚨 CRITICAL FIX
                if np.isnan(proba):
                    continue

                print(f"{current_date} | {ticker} | proba={proba:.3f}")

                # -----------------------
                # POSITION SIZING FIX
                # -----------------------
                equity = self.portfolio.get_portfolio_state()['total_equity']
                position_value = equity * fixed_size

                shares = position_value / price

                if shares < 0.01:
                    continue  # avoid 0-share trades

                # -----------------------
                # TRADING LOGIC
                # -----------------------
                if proba > 0.55:

                    print(f"🔥 BUY {ticker} ({proba:.3f})")

                    self.portfolio.execute_trade(
                        ticker,
                        "BUY",
                        shares,
                        price,
                        current_date
                    )

                elif proba < 0.45 and ticker in self.portfolio.positions:

                    print(f"🛑 SELL {ticker} ({proba:.3f})")

                    self.portfolio.execute_trade(
                        ticker,
                        "SELL",
                        self.portfolio.positions[ticker]['shares'],
                        price,
                        current_date
                    )

            # update portfolio
            self.portfolio.update_prices(daily_prices)
            self.portfolio.record_snapshot(current_date)

            if i % 20 == 0 or i == len(all_dates) - 1:
                state = self.portfolio.get_portfolio_state()
                print(
                    f"📅 {current_date} | Equity: ${state['total_equity']:,.0f} | "
                    f"Heat: {state['heat']:.1%} | DD: {state['drawdown']:.1%}"
                )

        print("\n🏁 DONE")
        self.finalize()

    # -----------------------------
    # RESULTS
    # -----------------------------
    def finalize(self):
        equity_df = pd.DataFrame(self.portfolio.equity_history)
        trades_df = pd.DataFrame(self.portfolio.trade_history)

        os.makedirs("5_backtesting", exist_ok=True)

        equity_df.to_csv("5_backtesting/equity_curve.csv", index=False)
        trades_df.to_csv("5_backtesting/trade_log.csv", index=False)

        perf = self.portfolio.get_performance_summary()

        print("\n📊 RESULTS:")
        print(f"Return : {perf.get('total_return', 0):.2f}%")
        print(f"Sharpe : {perf.get('sharpe_ratio', 0):.3f}")
        print(f"DD     : {perf.get('max_drawdown', 0):.2f}%")
        print(f"Trades : {perf.get('num_trades', 0)}")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":

    bt = BacktestEngine(
        tickers=["AAPL", "NVDA", "MSFT", "SPY", "QQQ", "TSLA"],
        start_date="2025-01-01",   # ✅ FIXED (3 months)
        end_date="2025-04-01",
        initial_capital=100000
    )

    bt.run(fixed_size=0.10)