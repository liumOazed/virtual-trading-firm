"""
alpaca_ledger.py
================
Alpaca is the SOURCE OF TRUTH for the live ARIA-Momentum book. This module
rebuilds the three live data files from Alpaca's own records, idempotently,
every time it runs (the engine calls it after each run; daily_recorder.py
calls it standalone):

  live_trade_log.csv     one row per FILLED order, from Alpaca FILL activities
                         (exact qty, qty-weighted avg fill price across partial
                         fills, New York fill date/time, real order_id). Columns
                         only the engine knows (proba, weight, hmm_regime,
                         reason, portfolio_value, notional) are kept by order_id.
  daily_history.csv      one row per CLOSED trading day (Alpaca calendar),
                         equity = Alpaca's official end-of-day equity
                         (portfolio history 1D; if Alpaca hasn't posted a
                         day yet, the flagged reconstruction stands in until
                         the next sync). Never a midday snapshot,
                         never today's still-open session.
  live_equity_curve.csv  the same closes, plus today's provisional engine
                         snapshot (written by live_engine.log_equity) until
                         that day closes and gets finalized.

Everything else in daily_history is derived from Alpaca records + official
closing prices, with exact accounting identities:
  cash       = deposits + fills (sells − buys) + other cash activity
               (fees −, dividends/interest +)
  equity     = deposits + realized + other_pnl + unrealized     (identity)
  recon_diff = (cash + Σ qty × official close) − Alpaca equity  → should be ~0

Every run checks recon_diff and warns on any day off by more than RECON_TOL,
so a divergence between our books and Alpaca's shows up immediately.

Benchmarks (SPY/QQQ) and position marks use official consolidated closes
from yfinance; Alpaca's free IEX feed is the fallback (can differ slightly).
Since-inception benchmark returns are measured from the OPEN of the
inception day (2026-06-15) — the moment the $100k went live — so book and
benchmarks cover exactly the same period.

Stale Alpaca bars: on rare days Alpaca's 1D portfolio history repeats the
previous day's equity unchanged (e.g. 2026-07-29) even though positions were
held and prices moved. When Alpaca's value is identical to the prior day's
AND our reconstruction disagrees by more than RECON_TOL, the reconstructed
equity (cash + positions at official close) is used and the row is marked
equity_source = 'rebuilt (alpaca 1D stale)'. Everything else is Alpaca's.
"""

import os
import json
from collections import defaultdict, deque
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from market_clock import NY, trading_date

load_dotenv()

API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
TRADE_URL  = "https://paper-api.alpaca.markets"
DATA_URL   = "https://data.alpaca.markets"
HEADERS    = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": SECRET_KEY,
              "accept": "application/json"}

ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR   = os.path.join(ROOT, "8_live_trading", "data")
HIST_CSV   = os.path.join(DATA_DIR, "daily_history.csv")
EQUITY_CSV = os.path.join(DATA_DIR, "live_equity_curve.csv")
TRADE_LOG  = os.path.join(DATA_DIR, "live_trade_log.csv")
STATE      = os.path.join(DATA_DIR, "regime_state.json")

INCEPTION_DATE = "2026-06-15"          # first live fill (QQQ BUY 09:37 ET)
BENCHMARKS     = ["SPY", "QQQ"]
RECON_TOL      = 0.05                  # $ — max tolerated books-vs-Alpaca gap
# Transfers in/out are CAPITAL, not P&L.
DEPOSIT_TYPES  = {"JNLC", "JNLS", "CSD", "CSW", "JNL", "TRANS"}

HIST_COLS = ["date", "equity", "cash", "deployed_pct", "n_positions",
             "daily_pnl", "daily_pnl_pct", "total_return_pct",
             "realized_pnl", "unrealized_pnl", "regime",
             "spy_price", "qqq_price", "spy_ret_since_incept",
             "qqq_ret_since_incept", "alpha_vs_spy", "alpha_vs_qqq", "positions",
             "other_pnl_cum", "recon_diff", "equity_source"]
TRADE_COLS = ["date", "ticker", "action", "price", "shares",
              "proba", "weight", "hmm_regime", "reason",
              "portfolio_value", "order_id", "notional", "price_estimated",
              "filled_at"]
ENGINE_ONLY = ["proba", "weight", "hmm_regime", "reason", "portfolio_value", "notional"]


# ── Alpaca I/O ────────────────────────────────────────────────────────────────
def _get(path, params=None, base=TRADE_URL):
    r = requests.get(base + path, headers=HEADERS, params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_activities() -> pd.DataFrame:
    """Every account activity (fills, fees, transfers, dividends…) since
    shortly before inception, oldest first."""
    after = (pd.Timestamp(INCEPTION_DATE) - timedelta(days=10)).strftime("%Y-%m-%d")
    out, token = [], None
    while True:
        p = {"direction": "asc", "page_size": 100, "after": after}
        if token:
            p["page_token"] = token
        page = _get("/v2/account/activities", p)
        if not page:
            break
        out += page
        token = page[-1]["id"]
        if len(page) < 100:
            break
    return pd.DataFrame(out)


def trading_days(start: str, end: str) -> list[str]:
    return [d["date"] for d in _get("/v2/calendar", {"start": start, "end": end})]


def portfolio_closes(start: str, end: str) -> dict[str, float]:
    """Alpaca official end-of-day equity per NY date. start AND end are
    required — with start alone Alpaca silently returns only ~1 month."""
    ph = _get("/v2/account/portfolio/history", {
        "timeframe": "1D",
        "start": f"{start}T00:00:00-04:00",
        "end":   f"{end}T23:59:59-04:00",
    })
    out = {}
    for ts, eq in zip(ph["timestamp"], ph["equity"]):
        if eq:
            d = pd.Timestamp(ts, unit="s", tz="UTC").tz_convert(NY).date().isoformat()
            out[d] = float(eq)
    return out


def official_closes(symbols, start: str, end: str, field: str = "Close") -> tuple[pd.DataFrame, str]:
    """Daily closes (or opens, field='Open'), unadjusted, index = 'YYYY-MM-DD'.
    yfinance first (official consolidated prices); Alpaca IEX bars as fallback."""
    symbols = sorted(set(symbols))
    end_x = (pd.Timestamp(end) + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        import yfinance as yf
        raw = yf.download(symbols, start=start, end=end_x, auto_adjust=False,
                          progress=False, threads=False)[field]
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(symbols[0])
        raw.index = pd.to_datetime(raw.index).strftime("%Y-%m-%d")
        if raw.reindex(columns=symbols).notna().any().all():
            return raw.reindex(columns=symbols), "yfinance (official close)"
    except Exception as e:
        print(f"  ! yfinance closes failed ({e}) - falling back to Alpaca IEX")
    frames = {}
    for s in symbols:
        bars = _get(f"/v2/stocks/{s}/bars", {"timeframe": "1Day", "start": start,
                    "end": end, "limit": 10000, "adjustment": "raw"}, base=DATA_URL)
        frames[s] = pd.Series({pd.Timestamp(b["t"]).tz_convert(NY).date().isoformat(): b["o" if field == "Open" else "c"]
                               for b in bars.get("bars") or []})
    return pd.DataFrame(frames), "Alpaca IEX (fallback — may differ from official close)"


# ── derivations ───────────────────────────────────────────────────────────────
def filled_orders(acts: pd.DataFrame) -> pd.DataFrame:
    """One row per order from its FILL/partial_fill executions."""
    f = acts[acts["activity_type"] == "FILL"].copy()
    if f.empty:
        return pd.DataFrame(columns=["order_id", "date", "filled_at", "ticker",
                                     "action", "shares", "price"])
    f["t"]   = pd.to_datetime(f["transaction_time"], utc=True).dt.tz_convert(NY)
    f["qty"] = f["qty"].astype(float)
    f["px"]  = f["price"].astype(float)
    g = f.groupby("order_id")
    o = pd.DataFrame({
        "first":  g["t"].min(),
        "last":   g["t"].max(),
        "ticker": g["symbol"].first(),
        "action": g["side"].first().str.upper(),
        "shares": g["qty"].sum(),
        "price":  g.apply(lambda x: (x["qty"] * x["px"]).sum() / x["qty"].sum()),
    }).reset_index()
    o["date"]      = o["first"].dt.date.astype(str)
    o["filled_at"] = o["last"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return o.sort_values("first").reset_index(drop=True)


def cash_flows(acts: pd.DataFrame) -> pd.DataFrame:
    """Non-fill cash activity per NY date, split into capital vs P&L."""
    nf = acts[acts["activity_type"] != "FILL"].copy()
    if nf.empty:
        return pd.DataFrame(columns=["date", "deposit", "other"])
    nf["amt"]     = nf["net_amount"].astype(float)
    nf["deposit"] = np.where(nf["activity_type"].isin(DEPOSIT_TYPES), nf["amt"], 0.0)
    nf["other"]   = np.where(nf["activity_type"].isin(DEPOSIT_TYPES), 0.0, nf["amt"])
    return nf.groupby("date")[["deposit", "other"]].sum().reset_index()


def _regime_map() -> dict[str, str]:
    """Engine-recorded regime per date (live_equity_curve.csv is written by
    the engine at run time, so its regime column is the regime that was in
    force that session)."""
    m = {}
    for path in (HIST_CSV, EQUITY_CSV):          # curve wins over history
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, dtype={"date": str})
                if "regime" in df:
                    m.update({d: r for d, r in zip(df["date"], df["regime"])
                              if isinstance(r, str) and r and r != "Unknown"})
            except Exception:
                pass
    return m


def _fallback_regime() -> str:
    try:
        return json.load(open(STATE)).get("prev_regime", "Unknown")
    except Exception:
        return "Unknown"


# ── rebuilds ──────────────────────────────────────────────────────────────────
def rebuild_trade_log(orders: pd.DataFrame, today: str) -> tuple[pd.DataFrame, list]:
    old = pd.read_csv(TRADE_LOG, dtype={"order_id": str}) if os.path.exists(TRADE_LOG) \
        else pd.DataFrame(columns=TRADE_COLS)
    old = old.reset_index(drop=True)
    known = set(orders["order_id"])

    # Rows logged without a real order id (e.g. an old 'backfill' row): match
    # to an Alpaca order with the same ticker, side and share count.
    claimed = set(old["order_id"].dropna()) & known
    for i, r in old[~old["order_id"].isin(known)].iterrows():
        cand = orders[(orders["ticker"] == r["ticker"]) & (orders["action"] == r["action"])
                      & ((orders["shares"] - float(r["shares"] or 0)).abs() < 1e-4)
                      & ~orders["order_id"].isin(claimed)]
        if len(cand):
            old.at[i, "order_id"] = cand.iloc[0]["order_id"]
            claimed.add(cand.iloc[0]["order_id"])

    enrich = old.drop_duplicates("order_id", keep="last").set_index("order_id")
    rows = []
    for _, o in orders.iterrows():
        e = enrich.loc[o["order_id"]] if o["order_id"] in enrich.index else None
        row = {
            "date": o["date"], "ticker": o["ticker"], "action": o["action"],
            "price": round(o["price"], 4), "shares": round(o["shares"], 6),
            "order_id": o["order_id"], "price_estimated": False,
            "filled_at": o["filled_at"],
        }
        for c in ENGINE_ONLY:
            row[c] = e[c] if e is not None and c in e else ""
        if e is None:
            row["reason"] = "external_or_manual (not logged by engine)"
        rows.append(row)

    # Engine rows with no Alpaca fill: keep today's (fill may not be posted
    # yet — next run finalizes it); drop older ones (never filled).
    unmatched = old[~old["order_id"].isin(known)]
    keep    = unmatched[unmatched["date"].astype(str) >= today]
    dropped = unmatched[unmatched["date"].astype(str) < today]
    new = pd.concat([pd.DataFrame(rows), keep], ignore_index=True)
    new = new.reindex(columns=TRADE_COLS)
    return new, dropped.to_dict("records")


def rebuild_history(acts, orders, today, verbose=True):
    days_all = trading_days((pd.Timestamp(INCEPTION_DATE) - timedelta(days=10)).strftime("%Y-%m-%d"),
                            today)
    pre     = [d for d in days_all if d < INCEPTION_DATE]
    base_day = pre[-1]                              # close before inception
    closed  = [d for d in days_all if INCEPTION_DATE <= d < today]
    if not closed:
        return pd.DataFrame(columns=HIST_COLS), {}

    eq_close = portfolio_closes(base_day, today)
    # Alpaca posts a day's 1D equity with a lag. A missing day is filled with
    # the reconstruction (cash + positions at official close — matches Alpaca
    # to the cent on posted days) and flagged; the next sync swaps in Alpaca's
    # figure. (account.last_equity is NOT used: just after NY midnight it can
    # still hold the day BEFORE, which would silently mis-state equity.)

    syms = sorted(set(orders["ticker"]) | set(BENCHMARKS))
    px, px_src = official_closes(syms, base_day, closed[-1])
    opens, _   = official_closes(BENCHMARKS, INCEPTION_DATE, INCEPTION_DATE, field="Open")
    flows = cash_flows(acts).set_index("date") if len(acts) else pd.DataFrame()
    regimes, fallback = _regime_map(), _fallback_regime()

    lots = defaultdict(deque)       # ticker -> deque[[qty, px]] FIFO
    cash = deposits = other = realized = 0.0
    prev_eq = eq_close.get(base_day)
    last_regime = None
    bench_base = {b: opens.at[INCEPTION_DATE, b] for b in BENCHMARKS}
    prev_alpaca = eq_close.get(base_day)
    oi = 0
    orders = orders.sort_values("first").reset_index(drop=True)
    rows, flow_dates = [], sorted(flows.index) if len(flows) else []
    fi = 0

    for d in closed:
        while fi < len(flow_dates) and flow_dates[fi] <= d:
            fd = flow_dates[fi]
            deposits += flows.at[fd, "deposit"]; other += flows.at[fd, "other"]
            cash     += flows.at[fd, "deposit"] + flows.at[fd, "other"]
            fi += 1
        while oi < len(orders) and orders.at[oi, "date"] <= d:
            o = orders.iloc[oi]; q, p = float(o["shares"]), float(o["price"])
            if o["action"] == "BUY":
                lots[o["ticker"]].append([q, p]); cash -= q * p
            else:
                cash += q * p; rem = q
                while rem > 1e-9 and lots[o["ticker"]]:
                    lot = lots[o["ticker"]][0]; take = min(lot[0], rem)
                    realized += (p - lot[1]) * take
                    lot[0] -= take; rem -= take
                    if lot[0] <= 1e-9:
                        lots[o["ticker"]].popleft()
            oi += 1

        if d not in px.index:
            raise RuntimeError(f"No official close for {d} — price source incomplete")
        held = {t: (sum(l[0] for l in ls), sum(l[0] * l[1] for l in ls))
                for t, ls in lots.items() if sum(l[0] for l in ls) > 1e-6}
        mv = sum(q * px.at[d, t] for t, (q, _) in held.items())
        alpaca_eq, rebuilt_eq = eq_close.get(d), cash + mv
        if alpaca_eq is None:
            source, equity = "rebuilt (alpaca 1D not posted yet)", rebuilt_eq
        elif held and alpaca_eq == prev_alpaca and abs(rebuilt_eq - alpaca_eq) > RECON_TOL:
            source, equity = "rebuilt (alpaca 1D stale)", rebuilt_eq
        else:
            source, equity = "alpaca", alpaca_eq
        if alpaca_eq is not None:
            prev_alpaca = alpaca_eq
        pos_str = " ".join(f"{t}:{(px.at[d, t] / (c / q) - 1) * 100:+.1f}%"
                           for t, (q, c) in sorted(held.items()))
        regime = regimes.get(d) or last_regime or fallback
        last_regime = regime
        total = (equity / deposits - 1) * 100
        bret = {b: (px.at[d, b] / bench_base[b] - 1) * 100 for b in BENCHMARKS}
        rows.append({
            "date": d, "equity": round(equity, 2), "cash": round(cash, 2),
            "deployed_pct": round((equity - cash) / equity * 100, 2),
            "n_positions": len(held),
            "daily_pnl": round(equity - prev_eq, 2),
            "daily_pnl_pct": round((equity / prev_eq - 1) * 100, 4),
            "total_return_pct": round(total, 4),
            "realized_pnl": round(realized, 2),
            "unrealized_pnl": round(equity - deposits - realized - other, 2),
            "regime": regime,
            "spy_price": round(px.at[d, "SPY"], 2), "qqq_price": round(px.at[d, "QQQ"], 2),
            "spy_ret_since_incept": round(bret["SPY"], 4),
            "qqq_ret_since_incept": round(bret["QQQ"], 4),
            "alpha_vs_spy": round(total - bret["SPY"], 4),
            "alpha_vs_qqq": round(total - bret["QQQ"], 4),
            "positions": pos_str,
            "other_pnl_cum": round(other, 2),
            "recon_diff": round(rebuilt_eq - alpaca_eq, 2) if alpaca_eq is not None else np.nan,
            "equity_source": source,
        })
        prev_eq = equity

    hist = pd.DataFrame(rows, columns=HIST_COLS)
    return hist, {"price_source": px_src, "base_day": base_day}


def sync(verbose: bool = True) -> dict:
    """Rebuild all three live files from Alpaca. Safe to run any time."""
    today  = trading_date()
    acts   = fetch_activities()
    orders = filled_orders(acts)

    trades, dropped = rebuild_trade_log(orders, today)
    hist, meta      = rebuild_history(acts, orders, today, verbose)

    trades.to_csv(TRADE_LOG, index=False)
    hist.to_csv(HIST_CSV, index=False)

    # equity curve: official closes + today's provisional engine snapshot
    curve = pd.DataFrame({"date": hist["date"], "equity": hist["equity"],
                          "regime": hist["regime"]})
    if os.path.exists(EQUITY_CSV):
        old = pd.read_csv(EQUITY_CSV, dtype={"date": str})
        curve = pd.concat([curve, old[old["date"] >= today]], ignore_index=True)
    curve.to_csv(EQUITY_CSV, index=False)

    stale = hist[hist["equity_source"] != "alpaca"]
    bad   = hist[(hist["recon_diff"].abs() > RECON_TOL) & (hist["equity_source"] == "alpaca")]
    if verbose:
        last = hist.iloc[-1] if len(hist) else None
        print(f"  OK Alpaca ledger synced - {len(trades)} filled orders, "
              f"{len(hist)} closed trading days"
              + (f" through {last['date']} (equity ${last['equity']:,.2f})" if last is not None else ""))
        print(f"    prices: {meta.get('price_source', 'n/a')}")
        if dropped:
            print(f"  ! Dropped {len(dropped)} logged order(s) with no Alpaca fill:")
            for r in dropped:
                print(f"      {r.get('date')} {r.get('ticker')} {r.get('action')} id={r.get('order_id')}")
        for _, r in stale.iterrows():
            if "stale" in r["equity_source"]:
                print(f"    note: {r['date']} Alpaca 1D equity was stale (repeated prior day); "
                      f"used reconstruction (Alpaca off by ${-r['recon_diff']:+,.2f})")
            else:
                print(f"    note: {r['date']} Alpaca has not posted this close yet; using "
                      f"reconstruction (replaced automatically on the next sync)")
        if len(bad):
            print(f"  ! RECONCILIATION: {len(bad)} day(s) where our books differ from "
                  f"Alpaca by > ${RECON_TOL:.2f}:")
            for _, r in bad.iterrows():
                print(f"      {r['date']}  diff ${r['recon_diff']:+,.2f}")
        else:
            print(f"    reconciliation: every day within ${RECON_TOL:.2f} of Alpaca OK")
    return {"trades": trades, "hist": hist, "recon_bad": bad, "stale": stale,
            "dropped": dropped, **meta}


if __name__ == "__main__":
    sync()
