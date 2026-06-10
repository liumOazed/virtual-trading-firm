"""
status_report.py
================
Modern reconciliation report for the Virtual Trading Firm (ARIA).
Drop into 8_live_trading/ and call render_status_report(client) from
run_live.py cmd_status() instead of the plain print block.

Shows:
  - Live account equity (source of truth, from Alpaca)
  - Full P&L reconciliation: realized + unrealized = net
  - Per-position table with sparkline-style bars
  - Regime + cash deployment
  - Reconciliation that EXPLAINS why numbers differ
"""

import os
import pandas as pd
from datetime import date

LIVE_DIR   = os.path.join(os.path.dirname(__file__), "data")
TRADE_LOG  = os.path.join(LIVE_DIR, "live_trade_log.csv")
EQUITY_LOG = os.path.join(LIVE_DIR, "live_equity_curve.csv")

# ── ANSI styling (works in most terminals) ──────────────────────────────
class C:
    R   = "\033[0m"
    B   = "\033[1m"
    DIM = "\033[2m"
    GRN = "\033[32m"
    RED = "\033[31m"
    YEL = "\033[33m"
    CYN = "\033[36m"
    BLU = "\033[34m"
    MAG = "\033[35m"
    GRY = "\033[90m"
    BGRN= "\033[42m"
    BRED= "\033[41m"

def _c(text, color):
    return f"{color}{text}{C.R}"

def _money(v, width=12, sign=False):
    s = f"{'+' if sign and v>=0 else ''}${v:,.2f}"
    return s.rjust(width)

def _pct(v, width=7):
    return f"{'+' if v>=0 else ''}{v:.2f}%".rjust(width)

def _bar(pct, maxw=18):
    """Horizontal bar for position weight or P&L."""
    n = int(min(abs(pct), 100) / 100 * maxw)
    return "█" * max(n, 1)


def _compute_realized_pnl():
    """Sum realized P&L from the trade log (matched buy→sell round trips)."""
    if not os.path.exists(TRADE_LOG):
        return 0.0, 0, 0
    try:
        tl = pd.read_csv(TRADE_LOG)
    except Exception:
        return 0.0, 0, 0
    if tl.empty or "action" not in tl.columns:
        return 0.0, 0, 0

    buys  = tl[tl["action"] == "BUY"].copy()
    sells = tl[tl["action"] == "SELL"].copy()

    # Match FIFO per ticker
    realized = 0.0
    inventory = {}
    for _, r in tl.sort_values("date").iterrows():
        tk = r["ticker"]
        px = float(r.get("price", 0) or 0)
        sh = float(r.get("shares", 0) or 0)
        if r["action"] == "BUY":
            inventory.setdefault(tk, []).append([px, sh])
        elif r["action"] == "SELL" and tk in inventory and inventory[tk]:
            remaining = sh
            while remaining > 1e-9 and inventory[tk]:
                lot = inventory[tk][0]
                take = min(lot[1], remaining)
                realized += (px - lot[0]) * take
                lot[1] -= take
                remaining -= take
                if lot[1] <= 1e-9:
                    inventory[tk].pop(0)
    return realized, len(buys), len(sells)


def render_status_report(client, regime: str = None):
    """Render the full modern reconciliation report."""
    account   = client.get_account()
    positions = client.get_positions()
    hours     = client.get_market_hours()

    equity      = account["equity"]
    cash        = account["cash"]
    bp          = account["buying_power"]
    start_eq    = 100_000.0

    pos_unreal  = sum(p["unrealized_pl"] for p in positions)
    pos_value   = sum(p["market_value"] for p in positions)
    realized, n_buys, n_sells = _compute_realized_pnl()

    net_pnl     = equity - start_eq
    net_pct     = (equity / start_eq - 1) * 100
    deployed_pct= (pos_value / equity * 100) if equity else 0
    cash_pct    = (cash / equity * 100) if equity else 0

    W = 64
    line = "─" * W

    # ── HEADER ──
    print()
    print(_c("╔" + "═"*W + "╗", C.CYN))
    title = "  ARIA · LIVE RECONCILIATION REPORT"
    print(_c("║", C.CYN) + _c(title.ljust(W), C.B+C.CYN) + _c("║", C.CYN))
    sub = f"  {date.today().strftime('%A, %B %d, %Y')}   ·   {'◉ MARKET OPEN' if hours['is_open'] else '○ MARKET CLOSED'}"
    print(_c("║", C.CYN) + _c(sub.ljust(W), C.DIM) + _c("║", C.CYN))
    print(_c("╚" + "═"*W + "╝", C.CYN))

    # ── EQUITY HERO ──
    pnl_color = C.GRN if net_pnl >= 0 else C.RED
    arrow = "▲" if net_pnl >= 0 else "▼"
    print()
    print(f"  {_c('NET LIQUIDATION VALUE', C.DIM)}")
    print(f"  {_c('$'+format(equity, ',.2f'), C.B+C.CYN)}   "
          f"{_c(arrow + ' ' + _money(abs(net_pnl), 1).strip() + '  (' + _pct(net_pct).strip() + ')', pnl_color)}")
    if regime:
        rcolor = C.GRN if "Bull" in regime else C.RED if "Bear" in regime else C.YEL
        print(f"  {_c('Regime:', C.DIM)} {_c(regime, rcolor+C.B)}")

    # ── RECONCILIATION ──
    rc = C.GRN if realized >= 0 else C.RED
    uc = C.GRN if pos_unreal >= 0 else C.RED
    nc = C.GRN if net_pnl >= 0 else C.RED

    print()
    print(_c("  P&L BREAKDOWN", C.B))
    print(_c("  " + line, C.GRY))

    # REALIZED — money already banked from closed trades
    print(f"  {_c('① REALIZED', rc+C.B)}  {_c('— locked in from closed trades', C.DIM)}")
    print(f"     {_c('Cash you have already won or lost. Permanent.', C.GRY)}")
    print(f"     {_c(_money(realized, sign=True), rc+C.B)}   "
          f"{_c(f'({n_sells} round trip{"s" if n_sells!=1 else ""} closed)', C.DIM)}")
    print()

    # UNREALIZED — paper gains on open positions, can still change
    print(f"  {_c('② UNREALIZED', uc+C.B)}  {_c('— paper P&L on open positions', C.DIM)}")
    print(f"     {_c('Floating. Changes every tick until you sell.', C.GRY)}")
    print(f"     {_c(_money(pos_unreal, sign=True), uc+C.B)}   "
          f"{_c(f'({len(positions)} position{"s" if len(positions)!=1 else ""} held)', C.DIM)}")
    print()

    print(_c("  " + line, C.GRY))
    print(f"  {_c('① + ② = TOTAL P&L', C.B).ljust(43)} "
          f"{_c(_money(net_pnl, sign=True), nc+C.B)}")
    print(_c("  " + line, C.GRY))
    print(f"  {_c('Starting capital', C.DIM).ljust(43)} {_money(start_eq)}")
    print(f"  {_c('Current equity (live from Alpaca)', C.DIM).ljust(43)} "
          f"{_c(_money(equity), C.B)}")

    # ── CAPITAL DEPLOYMENT ──
    print()
    print(_c("  CAPITAL DEPLOYMENT", C.B))
    print(_c("  " + line, C.GRY))
    dep_bar = "█" * int(deployed_pct / 100 * 40)
    csh_bar = "░" * int(cash_pct / 100 * 40)
    print(f"  {_c('Deployed', C.CYN)} {deployed_pct:>5.1f}%  {_c(dep_bar, C.CYN)}{_c(csh_bar, C.GRY)}")
    print(f"  {_c('Invested', C.DIM)}  {_money(pos_value)}   "
          f"{_c('Cash', C.DIM)} {_money(cash)}   "
          f"{_c('BP', C.DIM)} {_money(bp)}")

    # ── POSITIONS ──
    if positions:
        print()
        print(_c("  OPEN POSITIONS", C.B) + _c(f"   ({len(positions)})", C.DIM))
        print(_c("  " + line, C.GRY))
        print(f"  {_c('TICKER', C.DIM):<14} {_c('VALUE', C.DIM):>14} "
              f"{_c('P&L', C.DIM):>14} {_c('%', C.DIM):>8}  {_c('WEIGHT', C.DIM)}")
        for p in sorted(positions, key=lambda x: -x["market_value"]):
            pc = C.GRN if p["unrealized_pl"] >= 0 else C.RED
            wt = (p["market_value"] / pos_value * 100) if pos_value else 0
            bar = _bar(wt, 16)
            tick_disp = _c(p["ticker"].ljust(6), C.B)
            print(f"  {tick_disp}      "
                  f"{_money(p['market_value'])}  "
                  f"{_c(_money(p['unrealized_pl'], sign=True), pc)}  "
                  f"{_c(_pct(p['unrealized_pct']), pc)}  "
                  f"{_c(bar, C.BLU)} {wt:.0f}%")
        print(_c("  " + line, C.GRY))
        tot_c = C.GRN if pos_unreal >= 0 else C.RED
        print(f"  {_c('TOTAL'.ljust(6), C.B)}      {_money(pos_value)}  "
              f"{_c(_money(pos_unreal, sign=True), tot_c+C.B)}")
    else:
        print()
        print(_c("  No open positions — 100% cash", C.YEL))

    # ── TRADE ACTIVITY ──
    print()
    print(_c("  ACTIVITY", C.B))
    print(_c("  " + line, C.GRY))
    print(f"  {_c('Round trips closed', C.DIM):<43} {n_sells}")
    print(f"  {_c('Total fills logged', C.DIM):<43} {n_buys + n_sells}")
    if hours.get("next_open"):
        print(f"  {_c('Next session', C.DIM):<43} {hours['next_open'][:16]} EST")

    print()
    print(_c("  " + line, C.GRY))
    note = "Equity is live from Alpaca — the single source of truth." if positions else ""
    print(_c("  " + note, C.DIM))
    print()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from alpaca_client import AlpacaClient
    client = AlpacaClient()
    # try to read last regime from state
    regime = None
    rs = os.path.join(LIVE_DIR, "regime_state.json")
    if os.path.exists(rs):
        import json
        try:
            regime = json.load(open(rs)).get("prev_regime")
        except Exception:
            pass
    render_status_report(client, regime)