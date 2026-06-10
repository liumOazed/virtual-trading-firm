import yfinance as yf
import pandas as pd

SECTOR_TICKERS = {
    "Hardware":   ["NVDA", "AVGO", "ASML", "AMD", "TSM"],
    "Hypercloud": ["MSFT", "GOOGL", "AMZN", "META"],
    "Software":   ["CRM", "ORCL", "FTNT", "SNOW", "DOCN"],
    "Autos":      ["GM", "F", "TSLA", "TM", "RACE"],
    "Defensive":  ["XOM", "CVX", "PG", "WMT", "GLD"],
}

TARGET_STARTS = {
    "Hardware":   "2009-01-01",
    "Hypercloud": "2012-06-01",
    "Software":   "2021-01-01",
    "Autos":      "2015-01-01",
    "Defensive":  "2004-11-01",
}

END_DATE = "2026-05-19"

print("=" * 72)
print("  yfinance DATA AVAILABILITY CHECK")
print("=" * 72)

results = []

for sector, tickers in SECTOR_TICKERS.items():
    target_start = TARGET_STARTS[sector]
    print(f"\n── {sector} (target start: {target_start}) ──")

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start       = "2000-01-01",
                end         = END_DATE,
                progress    = False,
                auto_adjust = True
            )

            if df.empty:
                print(f"  ❌ {ticker:<6} NO DATA")
                results.append({
                    "sector": sector, "ticker": ticker,
                    "status": "NO DATA", "first_date": None,
                    "total_rows": 0, "rows_from_target": 0,
                    "hits_target": False,
                })
                continue

            first_date    = df.index.min().strftime("%Y-%m-%d")
            total_rows    = len(df)
            target_dt     = pd.Timestamp(target_start)
            rows_from_tgt = len(df[df.index >= target_dt])
            hits_target   = df.index.min() <= target_dt
            gap_flag      = "✅" if hits_target else "⚠️ LATE"

            print(f"  {gap_flag} {ticker:<6} "
                  f"first={first_date} | "
                  f"total={total_rows:>5} rows | "
                  f"from_target={rows_from_tgt:>5} rows")

            results.append({
                "sector":           sector,
                "ticker":           ticker,
                "status":           "OK",
                "first_date":       first_date,
                "total_rows":       total_rows,
                "rows_from_target": rows_from_tgt,
                "hits_target":      hits_target,
            })

        except Exception as e:
            print(f"  ❌ {ticker:<6} ERROR: {e}")
            results.append({
                "sector":           sector,
                "ticker":           ticker,
                "status":           f"ERROR: {e}",
                "first_date":       None,
                "total_rows":       0,
                "rows_from_target": 0,
                "hits_target":      False,
            })

# ── summary table ──────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SUMMARY")
print("=" * 72)

df_res = pd.DataFrame(results)

for sector in SECTOR_TICKERS.keys():
    sec_df       = df_res[df_res["sector"] == sector]
    ok_df        = sec_df[sec_df["status"] == "OK"]
    late_tickers = sec_df[
        ~sec_df["hits_target"].fillna(False)
    ]["ticker"].tolist()

    if not ok_df.empty:
        min_rows    = ok_df["rows_from_target"].min()
        min_idx     = ok_df["rows_from_target"].idxmin()
        bottleneck  = ok_df.loc[min_idx, "ticker"]
        total_rows  = ok_df["rows_from_target"].sum()
    else:
        min_rows   = 0
        bottleneck = "N/A"
        total_rows = 0

    print(f"\n  {sector:<15} target={TARGET_STARTS[sector]}")
    print(f"    Tickers OK        : {ok_df['ticker'].tolist()}")
    print(f"    Bottleneck ticker : {bottleneck}")
    print(f"    Min rows (single) : {min_rows}")
    print(f"    Total rows (sum)  : {total_rows}")
    print(f"    Late IPO tickers  : {late_tickers if late_tickers else 'None'}")

# save to csv
out_path = "4_signals/data_availability_check.csv"
df_res.to_csv(out_path, index=False)
print(f"\n✅ Saved: {out_path}")
print("✅ Check complete")
