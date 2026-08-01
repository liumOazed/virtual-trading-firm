"""
ARIA-Growth — Screen Archiver
==============================
Banks a DATED snapshot of each growth screen so that, over time, you build a
point-in-time fundamentals dataset. THIS is what eventually makes an honest
backtest possible: every snapshot is stamped with the date it was knowable, so
a future backtest can replay history with zero look-ahead.

Run this right after each `growth_screener.py` run (monthly is plenty):
  python growth_screener.py                       # produces growth_screen_results_YYYY-MM.csv
  python aria_growth_archive_screen.py            # banks a dated copy (auto-finds the latest screen)

Output:
  9_aria_growth/screens/growth_screen_<YYYY-MM-DD>.csv
  9_aria_growth/screens/_manifest.csv             (index of every snapshot)

The dated snapshots are append-only history — never edit them. The whole point
is that snapshot N reflects ONLY what was known on date N.
"""

import argparse
import shutil
import sys
from datetime import date
from pathlib import Path
import pandas as pd

# Windows console defaults to cp1252, which can't encode the ✓/⚠ glyphs below.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "screens"


def _find_latest_screen():
    """Most recent growth_screen_results_*.csv/.xlsx in this folder, else the legacy fixed name."""
    dated = sorted(HERE.glob("growth_screen_results_*.csv")) + sorted(HERE.glob("growth_screen_results_*.xlsx"))
    if dated:
        return sorted(dated, key=lambda p: p.stem)[-1]
    for name in ("growth_screen_results.csv", "growth_screen_results.xlsx"):
        if (HERE / name).exists():
            return HERE / name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default=None,
                    help="path to the screen (.csv or .xlsx); default: latest "
                         "growth_screen_results_*.csv in this folder")
    ap.add_argument("--date", default=date.today().isoformat(),
                    help="snapshot date (default: today)")
    args = ap.parse_args()

    src = Path(args.screen) if args.screen else _find_latest_screen()
    if not src or not src.exists():
        raise SystemExit(
            f"Screen not found: {args.screen or '(none passed)'}\n"
            f"   Run growth_screener.py first, or pass --screen with the full path.")

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