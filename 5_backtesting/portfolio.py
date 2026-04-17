import pandas as pd
import numpy as np
from datetime import datetime

class Portfolio:
    def __init__(self, initial_capital: float = 100000.0, commission_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        
        self.positions = {}          # {ticker: {'shares': float, 'avg_price': float, 'entry_date': str}}
        self.trade_history = []
        self.equity_history = []
        self.peak_equity = initial_capital

    def get_total_equity(self) -> float:
        holdings_value = sum(
            pos['shares'] * pos.get('current_price', pos['avg_price']) 
            for pos in self.positions.values()
        )
        return self.cash + holdings_value

    def update_prices(self, price_dict: dict):
        """Update current prices for all positions. price_dict = {'AAPL': 172.45, 'TSLA': 245.3, ...}"""
        for ticker, price in price_dict.items():
            if ticker in self.positions:
                self.positions[ticker]['current_price'] = price

    def execute_trade(self, ticker, action, size, price, date_str):
        """Execute a trade for a given ticker and action (BUY or SELL)."""
        if action.upper() == "BUY":
            if ticker not in self.positions:
                self.positions[ticker] = {"shares": 0.0, "avg_price": 0.0, "current_price": 0.0, "last_updated": None}
            old_shares = self.positions[ticker]["shares"]
            old_avg = self.positions[ticker]["avg_price"]
            new_shares = size / price
            self.positions[ticker]["shares"] = old_shares + new_shares
            self.positions[ticker]["avg_price"] = (old_avg * old_shares + new_shares * price) / (old_shares + new_shares)
            commission = size * self.commission_rate
            self.cash -= (size + commission)
            self.positions[ticker]["last_updated"] = date_str
            
            # Record trade in history
            self.trade_history.append({
                'date': date_str,
                'ticker': ticker,
                'action': 'BUY',
                'shares': new_shares,
                'price': price,
                'total': size
            })
            
            print(f"   📊 {date_str} | {ticker:<6} | Executed BUY {new_shares:.2f} shares @ ${price:.2f} | Cash: ${self.cash:.2f} | Total Equity: ${self.cash + sum([pos['shares'] * pos['avg_price'] for pos in self.positions.values()]):.2f}")

        elif action.upper() == "SELL":
            if ticker not in self.positions:
                return False
            current_shares = self.positions[ticker]["shares"]
            if current_shares > 0:
                # Calculate proceeds from sale
                sale_proceeds = current_shares * price
                commission = sale_proceeds * self.commission_rate
                self.cash += (sale_proceeds - commission)
                
                # Record trade in history before deleting
                self.trade_history.append({
                    'date': date_str,
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': current_shares,
                    'price': price,
                    'total': sale_proceeds
                })
                
                # Remove the position entirely (selling all shares)
                del self.positions[ticker]
            else:
                return False
        else:
            return False

        return True

    def get_portfolio_state(self) -> dict:
        total_equity = self.get_total_equity()
        exposure = sum(
            pos['shares'] * pos.get('current_price', pos['avg_price'])
            for pos in self.positions.values()
        )
        heat = exposure / total_equity if total_equity > 0 else 0
        
        drawdown = (self.peak_equity - total_equity) / self.peak_equity if total_equity < self.peak_equity else 0.0

        return {
            "total_equity": round(total_equity, 2),
            "cash": round(self.cash, 2),
            "heat": round(heat, 4),
            "drawdown": round(drawdown, 4),
            "active_positions": len(self.positions),
            "positions": list(self.positions.keys())
        }

    def record_snapshot(self, date_str: str):
        """Record equity for performance analysis"""
        equity = self.get_total_equity()
        self.equity_history.append({'date': date_str, 'equity': equity})
        
        if equity > self.peak_equity:
            self.peak_equity = equity

    def get_performance_summary(self) -> dict:
        """Return basic performance stats"""
        if not self.equity_history:
            return {}
            
        equity_curve = pd.DataFrame(self.equity_history)
        returns = equity_curve['equity'].pct_change().dropna()
        
        total_return = (equity_curve['equity'].iloc[-1] / self.initial_capital) - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        
        return {
            "total_return": round(total_return * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown": round(max((self.peak_equity - e['equity']) / self.peak_equity 
                                    for e in self.equity_history) * 100, 2),
            "num_trades": len(self.trade_history)
        }
        
if __name__ == "__main__":
    port = Portfolio(initial_capital=100000, commission_rate=0.001)
    
    port.execute_trade("AAPL", "BUY", size_pct=0.10, price=170.5, date_str="2026-04-15")
    port.update_prices({"AAPL": 175.2})
    port.record_snapshot("2026-04-16")
    
    port.execute_trade("AAPL", "SELL", size_pct=1.0, price=178.0, date_str="2026-04-18")
    port.record_snapshot("2026-04-18")
    
    print(port.get_portfolio_state())
    print(port.get_performance_summary())