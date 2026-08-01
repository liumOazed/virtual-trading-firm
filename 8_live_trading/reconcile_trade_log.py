"""
reconcile_trade_log.py
=======================
Safety net for live_trade_log.csv. Any row logged with price_estimated=True
(fill wasn't confirmed at the time — see live_engine._await_fill) is a
candidate for drift between what was logged and what Alpaca actually filled.

For each such row with a real order_id, re-fetches the order from Alpaca and,
if it has since settled (status == "filled"), overwrites price/shares with
the true fill and clears price_estimated. Safe to run any time — rows that
are already correct or still unfilled are left untouched.

Run manually:  python reconcile_trade_log.py
Wired in:      run_live.py cmd_run(), after the daily trading step.
"""
import os
import pandas as pd

LIVE_DIR  = os.path.join(os.path.dirname(__file__), "data")
TRADE_LOG = os.path.join(LIVE_DIR, "live_trade_log.csv")


def reconcile(client=None, verbose: bool = True) -> int:
    """Returns the number of rows corrected."""
    if not os.path.exists(TRADE_LOG):
        return 0
    if client is None:
        from alpaca_client import AlpacaClient
        client = AlpacaClient()

    df = pd.read_csv(TRADE_LOG)
    if df.empty or "price_estimated" not in df.columns:
        return 0

    is_estimated = df["price_estimated"].astype(str).str.lower() == "true"
    has_order    = df["order_id"].notna() & (df["order_id"] != "") & (df["order_id"] != "backfill")
    candidates   = df[is_estimated & has_order]

    fixed = 0
    for i, row in candidates.iterrows():
        try:
            o = client.get_order(row["order_id"])
        except Exception as e:
            if verbose:
                print(f"    reconcile: couldn't fetch order {row['order_id']} [{row['ticker']}]: {e}")
            continue
        if o.get("status") != "filled":
            continue  # still pending/partial — leave as-is, try again next run
        fp, fq = o.get("filled_avg_price"), o.get("qty")
        if not fp or not fq:
            continue
        old_price, old_shares = row["price"], row["shares"]
        df.at[i, "price"]           = round(float(fp), 4)
        df.at[i, "shares"]          = round(float(fq), 6)
        df.at[i, "price_estimated"] = False
        fixed += 1
        if verbose:
            print(f"    + reconciled {row['date']} {row['ticker']} {row['action']}: "
                  f"{old_shares}@{old_price} -> {round(float(fq),6)}@{round(float(fp),4)}")

    if fixed:
        df.to_csv(TRADE_LOG, index=False)
        if verbose:
            print(f"  + reconcile_trade_log: corrected {fixed} row(s)")
    return fixed


if __name__ == "__main__":
    reconcile()
