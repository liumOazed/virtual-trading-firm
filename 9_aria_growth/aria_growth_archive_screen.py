"""
ARIA-Growth — Screen Archiver
==============================
Banks a DATED snapshot of each growth screen so that, over time, you build a
point-in-time fundamentals dataset. THIS is what eventually makes an honest
backtest possible: every snapshot is stamped with the date it was knowable, so
a future backtest can replay history with zero look-ahead.

Run this right after each `growth_screener.py` run (monthly is plenty):
  python growth_screener.py                       # produces growth_screen_results.csv
  python aria_growth_archive_screen.py            # banks a dated copy

Output:
  9_aria_growth/screens/growth_screen_<YYYY-MM-DD>.csv
  9_aria_growth/screens/_manifest.csv             (index of every snapshot)

The dated snapshots are append-only history — never edit them. The whole point
is that snapshot N reflects ONLY what was known on date N.
"""

import argparse
import shutil
from datetime import date
from pathlib import Path
import pandas as pd

OUT_DIR = Path("9_aria_growth/screens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="growth_screen_results.csv",
                    help="path to the latest screen (.csv or .xlsx)")
    ap.add_argument("--date", default=date.today().isoformat(),
                    help="snapshot date (default: today)")
    args = ap.parse_args()

    src = Path(args.screen)
    if not src.exists():
        # try the xlsx the user uploaded
        alt = Path("growth_screen_results.xlsx")
        if alt.exists():
            src = alt
        else:
            raise SystemExit(f"Screen not found: {args.screen} (or growth_screen_results.xlsx)")

    df = pd.read_excel(src) if src.suffix == ".xlsx" else pd.read_csv(src)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / f"growth_screen_{args.date}.csv"
    if dest.exists():
        print(f"  ⚠ Snapshot for {args.date} already exists: {dest}")
        print(f"    Refusing to overwrite (snapshots are append-only history).")
        return
    df.to_csv(dest, index=False)

    # Update manifest
    manifest = OUT_DIR / "_manifest.csv"
    row = pd.DataFrame([{
        "date": args.date, "n_stocks": len(df), "file": dest.name,
        "n_sectors": df["sector"].nunique() if "sector" in df else None,
    }])
    if manifest.exists():
        m = pd.read_csv(manifest)
        m = pd.concat([m[m["date"] != args.date], row], ignore_index=True)
    else:
        m = row
    m = m.sort_values("date").reset_index(drop=True)
    m.to_csv(manifest, index=False)

    print(f"  ✓ Archived {len(df)} stocks → {dest}")
    print(f"  ✓ Manifest now has {len(m)} snapshot(s):")
    for _, r in m.iterrows():
        print(f"     {r['date']}  {int(r['n_stocks'])} stocks")
    if len(m) >= 12:
        print(f"\n  📊 You now have {len(m)} monthly snapshots — enough to start a")
        print(f"     point-in-time backtest of the regime rotation. Worth doing.")
    else:
        print(f"\n  ({12 - len(m)} more monthly snapshots → enough for a point-in-time backtest)")


if __name__ == "__main__":
    main()