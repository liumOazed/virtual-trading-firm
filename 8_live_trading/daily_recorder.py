"""
daily_recorder.py
=================
Single-sheet daily record-keeper for ARIA. Run once after US close.
Takes NO trading action — snapshots the account, appends one dated row
to data/daily_history.csv so you can see over weeks whether you're
actually BEATING THE MARKET (alpha vs SPY/QQQ since inception).

Run daily:   python daily_recorder.py
With regime: python daily_recorder.py --regime Bull-Stable
Backfill:    python daily_recorder.py --backfill
             (reconstructs missing days since inception from the equity
              curve + historical SPY/QQQ; per-position detail unavailable
              for past days, but equity/returns/alpha are exact)
"""

import os, sys, csv, json, argparse
import requests
from datetime import date, datetime, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
TRADE_URL  = "https://paper-api.alpaca.markets"
DATA_URL   = "https://data.alpaca.markets"
HEADERS = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": SECRET_KEY,
           "accept": "application/json"}

DATA_DIR   = os.path.join(ROOT, "8_live_trading", "data")
HIST_CSV   = os.path.join(DATA_DIR, "daily_history.csv")
EQUITY_CSV = os.path.join(DATA_DIR, "live_equity_curve.csv")
TRADE_LOG  = os.path.join(DATA_DIR, "live_trade_log.csv")
STATE      = os.path.join(DATA_DIR, "regime_state.json")

INCEPTION_EQUITY = 100_000.0
INCEPTION_DATE   = "2026-06-01"

COLS = ["date","equity","cash","deployed_pct","n_positions",
        "daily_pnl","daily_pnl_pct","total_return_pct",
        "realized_pnl","unrealized_pnl","regime",
        "spy_price","qqq_price","spy_ret_since_incept",
        "qqq_ret_since_incept","alpha_vs_spy","alpha_vs_qqq","positions"]


def _get(url, path, params=None):
    r = requests.get(url+path, headers=HEADERS, params=params or {}, timeout=20)
    r.raise_for_status(); return r.json()

def _ensure():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(HIST_CSV):
        with open(HIST_CSV,"w",newline="") as f: csv.writer(f).writerow(COLS)

def _append(row):
    import pandas as pd
    today = row.get("date", "")
    if not os.path.exists(HIST_CSV):
        with open(HIST_CSV, "w", newline="") as f:
            csv.writer(f).writerow(COLS)
    df = pd.read_csv(HIST_CSV, encoding="utf-8-sig")
    df = df[df["date"].astype(str) != today]   # drop same-day row if present
    new_row = pd.DataFrame([[row.get(c, "") for c in COLS]], columns=COLS)
    pd.concat([df, new_row], ignore_index=True).to_csv(HIST_CSV, index=False)

def _close_on(ticker, on_date):
    """Daily close on/just before a date (handles weekends/holidays)."""
    try:
        start=(datetime.strptime(on_date,"%Y-%m-%d")-timedelta(days=6)).strftime("%Y-%m-%d")
        bars=_get(DATA_URL,f"/v2/stocks/{ticker}/bars",
                  {"timeframe":"1Day","start":start,"end":on_date,"limit":10})
        return float(bars["bars"][-1]["c"]) if bars.get("bars") else None
    except Exception:
        return None

def _latest_close(ticker):
    try:
        bars=_get(DATA_URL,f"/v2/stocks/{ticker}/bars",{"timeframe":"1Day","limit":1})
        return float(bars["bars"][-1]["c"])
    except Exception: return None

def _realized_through(through_date=None):
    """FIFO realized P&L, optionally only counting trades up to a date."""
    if not os.path.exists(TRADE_LOG): return 0.0
    import pandas as pd
    try: tl=pd.read_csv(TRADE_LOG)
    except Exception: return 0.0
    if tl.empty or "action" not in tl: return 0.0
    if through_date: tl=tl[tl["date"]<=through_date]
    realized=0.0; inv={}
    for _,r in tl.sort_values("date").iterrows():
        tk=r["ticker"]; px=float(r.get("price",0) or 0); sh=float(r.get("shares",0) or 0)
        if r["action"]=="BUY": inv.setdefault(tk,[]).append([px,sh])
        elif r["action"]=="SELL" and inv.get(tk):
            rem=sh
            while rem>1e-9 and inv[tk]:
                lot=inv[tk][0]; take=min(lot[1],rem)
                realized+=(px-lot[0])*take; lot[1]-=take; rem-=take
                if lot[1]<=1e-9: inv[tk].pop(0)
    return realized

def _existing_dates():
    if not os.path.exists(HIST_CSV): return set()
    import pandas as pd
    try: return set(pd.read_csv(HIST_CSV, encoding="latin-1")["date"].astype(str))
    except Exception: return set()


def record_today(regime_arg=None):
    _ensure()
    today=date.today().isoformat()
    acct=_get(TRADE_URL,"/v2/account")
    equity=float(acct["equity"]); cash=float(acct["cash"])
    positions=_get(TRADE_URL,"/v2/positions")
    pos_value=sum(float(p["market_value"]) for p in positions)
    unreal=sum(float(p["unrealized_pl"]) for p in positions)
    realized=_realized_through()
    deployed=(pos_value/equity*100) if equity else 0

    # prev equity from history
    prev_eq=None
    import pandas as pd
    if os.path.exists(HIST_CSV):
        h=pd.read_csv(HIST_CSV, encoding="latin-1")
        if len(h): prev_eq=float(h.iloc[-1]["equity"])
    daily_pnl=(equity-prev_eq) if prev_eq else 0.0
    daily_pct=(daily_pnl/prev_eq*100) if prev_eq else 0.0
    total_ret=(equity/INCEPTION_EQUITY-1)*100

    regime=regime_arg or "Unknown"
    if not regime_arg and os.path.exists(STATE):
        try: regime=json.load(open(STATE)).get("prev_regime","Unknown")
        except Exception: pass

    spy_now=_latest_close("SPY"); qqq_now=_latest_close("QQQ")
    spy_inc=_close_on("SPY",INCEPTION_DATE); qqq_inc=_close_on("QQQ",INCEPTION_DATE)
    spy_ret=((spy_now/spy_inc-1)*100) if (spy_now and spy_inc) else 0
    qqq_ret=((qqq_now/qqq_inc-1)*100) if (qqq_now and qqq_inc) else 0

    pos_str=" ".join(f"{p['symbol']}:{float(p['unrealized_plpc'])*100:+.1f}%" for p in positions)
    row={"date":today,"equity":round(equity,2),"cash":round(cash,2),
         "deployed_pct":round(deployed,1),"n_positions":len(positions),
         "daily_pnl":round(daily_pnl,2),"daily_pnl_pct":round(daily_pct,2),
         "total_return_pct":round(total_ret,2),"realized_pnl":round(realized,2),
         "unrealized_pnl":round(unreal,2),"regime":regime,
         "spy_price":round(spy_now,2) if spy_now else "",
         "qqq_price":round(qqq_now,2) if qqq_now else "",
         "spy_ret_since_incept":round(spy_ret,2),"qqq_ret_since_incept":round(qqq_ret,2),
         "alpha_vs_spy":round(total_ret-spy_ret,2),
         "alpha_vs_qqq":round(total_ret-qqq_ret,2),"positions":pos_str}
    _append(row)
    _print_row(row)


def backfill():
    """Reconstruct missing days from the equity curve + historical prices.
    Per-position detail unavailable for past days; equity/returns/alpha exact."""
    _ensure()
    import pandas as pd
    if not os.path.exists(EQUITY_CSV):
        print("  ✗ No live_equity_curve.csv — cannot backfill."); return
    eq=pd.read_csv(EQUITY_CSV)
    # expect columns date, equity, regime (adapt to actual)
    datecol=next((c for c in eq.columns if "date" in c.lower()), eq.columns[0])
    eqcol=next((c for c in eq.columns if "equit" in c.lower()), None)
    regcol=next((c for c in eq.columns if "regime" in c.lower()), None)
    if eqcol is None:
        print(f"  ✗ Can't find equity column in {list(eq.columns)}"); return
    eq=eq.dropna(subset=[eqcol])
    eq["d"]=pd.to_datetime(eq[datecol]).dt.date.astype(str)
    eq=eq.drop_duplicates(subset="d", keep="last").sort_values("d")

    spy_inc=_close_on("SPY",INCEPTION_DATE); qqq_inc=_close_on("QQQ",INCEPTION_DATE)
    existing=_existing_dates()
    prev_eq=None; added=0

    for _,r in eq.iterrows():
        d=r["d"]
        if d in existing:
            prev_eq=float(r[eqcol]); continue
        equity=float(r[eqcol])
        daily_pnl=(equity-prev_eq) if prev_eq else 0.0
        daily_pct=(daily_pnl/prev_eq*100) if prev_eq else 0.0
        total_ret=(equity/INCEPTION_EQUITY-1)*100
        realized=_realized_through(d)
        spy_d=_close_on("SPY",d); qqq_d=_close_on("QQQ",d)
        spy_ret=((spy_d/spy_inc-1)*100) if (spy_d and spy_inc) else 0
        qqq_ret=((qqq_d/qqq_inc-1)*100) if (qqq_d and qqq_inc) else 0
        regime=str(r[regcol]) if regcol else "Unknown"
        row={"date":d,"equity":round(equity,2),"cash":"","deployed_pct":"",
             "n_positions":"","daily_pnl":round(daily_pnl,2),
             "daily_pnl_pct":round(daily_pct,2),"total_return_pct":round(total_ret,2),
             "realized_pnl":round(realized,2),"unrealized_pnl":"",
             "regime":regime,"spy_price":round(spy_d,2) if spy_d else "",
             "qqq_price":round(qqq_d,2) if qqq_d else "",
             "spy_ret_since_incept":round(spy_ret,2),"qqq_ret_since_incept":round(qqq_ret,2),
             "alpha_vs_spy":round(total_ret-spy_ret,2),
             "alpha_vs_qqq":round(total_ret-qqq_ret,2),
             "positions":"(backfilled — no position detail)"}
        _append(row); prev_eq=equity; added+=1
        print(f"  + backfilled {d}: equity ${equity:,.0f}  "
              f"total {total_ret:+.1f}%  alpha/SPY {total_ret-spy_ret:+.1f}%")
    print(f"\n  + Backfilled {added} missing day(s). Re-run sorted? Open CSV.")


def _print_row(row):
    print("="*58); print(f"  DAILY RECORD — {row['date']}"); print("="*58)
    print(f"  Equity:        ${row['equity']:,.2f}  ({row['daily_pnl_pct']:+.2f}% today)")
    print(f"  Total return:  {row['total_return_pct']:+.2f}%  since {INCEPTION_DATE}")
    print(f"  Regime:        {row['regime']}")
    print(f"  Deployed:      {row['deployed_pct']}%   Cash: ${row['cash']:,}")
    print(f"  Realized:      ${row['realized_pnl']:+,.2f}   "
          f"Unrealized: ${row['unrealized_pnl']:+,.2f}")
    print(f"\n  vs MARKET since inception:")
    print(f"    You:  {row['total_return_pct']:+.2f}%")
    print(f"    SPY:  {row['spy_ret_since_incept']:+.2f}%   -> alpha {row['alpha_vs_spy']:+.2f}%")
    print(f"    QQQ:  {row['qqq_ret_since_incept']:+.2f}%   -> alpha {row['alpha_vs_qqq']:+.2f}%")
    v="BEATING" if row['alpha_vs_spy']>0 else "TRAILING"
    print(f"    -> {v} SPY by {abs(row['alpha_vs_spy']):.2f}%")
    print(f"\n  + Appended to daily_history.csv")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--regime", default=None)
    ap.add_argument("--backfill", action="store_true")
    args=ap.parse_args()
    if args.backfill: backfill()
    else: record_today(args.regime)