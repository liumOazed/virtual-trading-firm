"""
sector_model_trainer.py
=======================
VIRTUAL TRADING FIRM | Stage 4 — Sector Model Training v4

Changes from v3:
  - Removed duplicate sector_training_summary.csv
  - Single output: sector_diagnostic_summary.csv with quality flags
    and deploy recommendations per sector
  - Regime filter confirmed working (HMM pre-labeling before train_xgboost)
  - Guard in xgboost_model.py prevents regime column overwrite

Regime filter per sector (HMM label strings):
  Hardware   → Bull-Trending only
  Hypercloud → Bull-Trending + Bull-Stable
  Software   → Bull-Stable only
  Autos      → Bull-Trending only
  Defensive  → Bear-Stress + Bear-Stable

Quality flags for deploy recommendation:
  Thr F1    >= 0.45  model has meaningful signal post-threshold
  Threshold >= 0.45  model not firing on noise
  Best iter >= 50    model learned enough before early stop

Output:
  4_signals/models/hardware_model.pkl
  4_signals/models/hypercloud_model.pkl
  4_signals/models/software_model.pkl
  4_signals/models/autos_model.pkl
  4_signals/models/defensive_model.pkl
  4_signals/models/sector_diagnostic_summary.csv
"""

import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
from datetime import date

# ── path setup ──────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "4_signals"))

from xgboost_model import (
    build_multi_ticker_dataset,
    train_xgboost,
    GPU_CFG,
)


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

MIN_ROWS_THRESHOLD = 1000

HMM_BULL_TRENDING = "Bull-Trending"
HMM_BULL_STABLE   = "Bull-Stable"
HMM_BEAR_STRESS   = "Bear-Stress"
HMM_BEAR_STABLE   = "Bear-Stable"


# ═══════════════════════════════════════════════════════════════════════
# SECTOR DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

SECTORS = {
    "hardware": {
        "name":          "Hardware / Semiconductor",
        "tickers":       ["NVDA", "AVGO", "ASML", "AMD", "TSM"],
        "start_date":    "2009-08-06",
        "regime_filter": [HMM_BULL_TRENDING],
        "regime_role":   "Bull-Trending primary",
        "pkl":           "4_signals/models/hardware_model.pkl",
        "n_trials":      15,
        "wf_config": {
            "train_window": 1000,
            "step":         100,
            "wf_step":      100,
        },
    },
    "hypercloud": {
        "name":          "Hypercloud",
        "tickers":       ["MSFT", "GOOGL", "AMZN", "META"],
        "start_date":    "2012-06-01",
        "regime_filter": [HMM_BULL_TRENDING, HMM_BULL_STABLE],
        "regime_role":   "Bull-Trending + Bull-Stable",
        "pkl":           "4_signals/models/hypercloud_model.pkl",
        "n_trials":      15,
        "wf_config": {
            "train_window": 1000,
            "step":         100,
            "wf_step":      100,
        },
    },
    "autos": {
        "name":          "Autos / EV",
        "tickers":       ["GM", "F", "TSLA", "TM", "RACE"],
        "start_date":    "2015-10-21",
        "regime_filter": [HMM_BULL_TRENDING],
        "regime_role":   "Bull-Trending",
        "pkl":           "4_signals/models/autos_model.pkl",
        "n_trials":      15,
        "wf_config": {
            "train_window": 1000,
            "step":         100,
            "wf_step":      100,
        },
    },
    "defensive": {
        "name":          "Defensive",
        "tickers":       ["XOM", "CVX", "PG", "WMT", "GLD"],
        "start_date":    "2004-11-18",
        "regime_filter": [HMM_BEAR_STRESS, HMM_BEAR_STABLE],
        "regime_role":   "Bear-Stress + Bear-Stable",
        "pkl":           "4_signals/models/defensive_model.pkl",
        "n_trials":      15,
        "wf_config": {
            "train_window": 600,
            "step":         100,
            "wf_step":      100,
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════
# REGIME FILTER
# ═══════════════════════════════════════════════════════════════════════

def filter_by_regime(df: pd.DataFrame,
                     regime_filter: list,
                     sector_name: str) -> pd.DataFrame:
    """
    Filter dataset to only include rows matching target HMM regimes.
    Called AFTER regime column is pre-labeled in train_sector().
    """
    if "regime" not in df.columns:
        print(f"\n  ⚠️  WARNING: 'regime' column not found")
        print(f"      Sector : {sector_name}")
        print(f"      Columns: {list(df.columns)}")
        print(f"      Regime filtering SKIPPED — training on all bars")
        return df

    before = len(df)
    dist   = df["regime"].value_counts().to_dict()

    print(f"\n  Regime distribution before filter ({before:,} rows):")
    for regime in sorted(dist.keys()):
        count = dist[regime]
        pct   = count / before * 100
        flag  = " ← KEEPING" if regime in regime_filter else ""
        print(f"    {regime:<20}: {count:>6,} rows "
              f"({pct:>5.1f}%){flag}")

    filtered = df[df["regime"].isin(regime_filter)].copy()
    after    = len(filtered)
    kept_pct = after / before * 100 if before > 0 else 0

    print(f"\n  After filter : {after:,} rows kept "
          f"({kept_pct:.1f}% of {before:,})")
    print(f"  Regimes kept : {regime_filter}")

    if after == 0:
        print(f"\n  ❌ Zero rows after filter.")
        print(f"     HMM labels seen : {list(dist.keys())}")
        print(f"     Filter strings  : {regime_filter}")

    return filtered


# ═══════════════════════════════════════════════════════════════════════
# SINGLE SECTOR TRAINER
# ═══════════════════════════════════════════════════════════════════════

def train_sector(sector_key: str, sector_cfg: dict) -> dict:
    """
    Train one sector model from scratch:
      1. Build dataset via build_multi_ticker_dataset()
      2. Pre-label regime column using HMM (same as xgboost_model.py)
      3. Filter to target regime bars only
      4. Train full stacked ensemble via train_xgboost()
      5. Save PKL and return metrics
    """
    name          = sector_cfg["name"]
    tickers       = sector_cfg["tickers"]
    start_date    = sector_cfg["start_date"]
    regime_filter = sector_cfg["regime_filter"]
    pkl           = sector_cfg["pkl"]
    trials        = sector_cfg["n_trials"]
    wf_cfg        = sector_cfg["wf_config"]
    end_date      = date.today().strftime("%Y-%m-%d")

    start_dt      = pd.Timestamp(start_date)
    end_dt        = pd.Timestamp(end_date)
    lookback_days = (end_dt - start_dt).days

    print("\n" + "═" * 62)
    print(f"  SECTOR        : {name}")
    print(f"  Tickers       : {tickers}")
    print(f"  Date range    : {start_date} → {end_date}")
    print(f"  Lookback days : {lookback_days}")
    print(f"  Regime filter : {regime_filter}")
    print(f"  WF step       : {wf_cfg['step']}")
    print(f"  Optuna trials : {trials}")
    print(f"  Output        : {pkl}")
    print("═" * 62)

    os.makedirs(
        os.path.dirname(os.path.abspath(pkl)),
        exist_ok=True
    )

    t_start = time.time()

    try:
        # ── Step 1: build dataset ──────────────────────────────────────
        print(f"\n📥 Building dataset for {name}...")
        df = build_multi_ticker_dataset(
            tickers       = tickers,
            end_date      = end_date,
            lookback_days = lookback_days,
            forward_days  = 5,
            sector_key    = sector_key,
        )

        if df is None or df.empty:
            return {
                "sector":      sector_key,
                "name":        name,
                "status":      "FAILED",
                "reason":      "Empty dataset — all tickers failed to load",
                "elapsed_min": 0,
            }

        actual_tickers = df["ticker"].unique().tolist()
        raw_rows       = len(df)

        print(f"  ✅ Raw dataset : {raw_rows:,} rows | "
              f"Tickers: {actual_tickers}")

        if len(actual_tickers) < 2:
            return {
                "sector":      sector_key,
                "name":        name,
                "status":      "FAILED",
                "reason":      f"Only {len(actual_tickers)} ticker loaded "
                               f"— need at least 2",
                "elapsed_min": round((time.time() - t_start) / 60, 1),
            }

        # ── Step 2: pre-label regime column ───────────────────────────
        # Replicates HMM logic from xgboost_model.py lines 818-869.
        # Must happen BEFORE filter and BEFORE train_xgboost().
        # Guard in xgboost_model.py prevents overwrite of this column.
        print(f"\n🗺️  Pre-labeling regime column via HMM...")
        try:
            import yfinance as yf
            from hmm_regime import GaussianHMMRegimeDetector

            hmm_start = "2015-01-01"

            spy_full = yf.download("SPY", start=hmm_start,
                                   end=end_date, progress=False,
                                   auto_adjust=False)
            ief_full = yf.download("IEF", start=hmm_start,
                                   end=end_date, progress=False,
                                   auto_adjust=False)

            for _df in [spy_full, ief_full]:
                if isinstance(_df.columns, pd.MultiIndex):
                    _df.columns = _df.columns.get_level_values(0)

            spy_full = spy_full.rename(columns={"Close": "close"})
            ief_full = ief_full.rename(columns={"Close": "close"})

            price_data_for_hmm = {"SPY": spy_full, "IEF": ief_full}

            for tkr in actual_tickers:
                try:
                    t_full = yf.download(tkr, start=hmm_start,
                                         end=end_date, progress=False,
                                         auto_adjust=False)
                    if t_full.empty:
                        continue
                    if isinstance(t_full.columns, pd.MultiIndex):
                        t_full.columns = t_full.columns.get_level_values(0)
                    t_full = t_full.rename(columns={"Close": "close"})
                    price_data_for_hmm[tkr] = t_full
                except Exception:
                    pass

            hmm = GaussianHMMRegimeDetector()
            hmm.fit_initial(price_data_for_hmm)

            if "date" in df.columns:
                dates_dt     = pd.to_datetime(df["date"])
                df["regime"] = [hmm.get_regime(d) for d in dates_dt]
                dist         = df["regime"].value_counts().to_dict()
                print(f"  ✅ Regime column added | distribution: {dist}")
            else:
                print(f"  ⚠️  No date column — defaulting to Bull-Stable")
                df["regime"] = HMM_BULL_STABLE

        except Exception as e:
            print(f"  ⚠️  HMM pre-label failed: {e}")
            print(f"      Defaulting all rows to Bull-Stable")
            df["regime"] = HMM_BULL_STABLE

        # ── Step 3: regime filter ──────────────────────────────────────
        print(f"\n🔍 Applying regime filter: {regime_filter}")
        df_filtered   = filter_by_regime(df, regime_filter, name)
        filtered_rows = len(df_filtered)

        if filtered_rows < MIN_ROWS_THRESHOLD:
            return {
                "sector":        sector_key,
                "name":          name,
                "status":        "FAILED",
                "reason":        f"Only {filtered_rows:,} rows after regime "
                                 f"filter — below minimum {MIN_ROWS_THRESHOLD}.",
                "elapsed_min":   round((time.time() - t_start) / 60, 1),
                "raw_rows":      raw_rows,
                "filtered_rows": filtered_rows,
            }

        print(f"\n  Per-ticker rows after filter:")
        for tkr in actual_tickers:
            tkr_rows = len(df_filtered[df_filtered["ticker"] == tkr])
            flag     = " ⚠️  THIN" if tkr_rows < 300 else ""
            print(f"    {tkr:<6} : {tkr_rows:>6,} rows{flag}")

        # ── Step 4: train ──────────────────────────────────────────────
        print(f"\n🧠 Training stacked ensemble for {name}...")
        print(f"   Filtered rows  : {filtered_rows:,}")
        print(f"   Optuna trials  : {trials}")
        print(f"   WF step        : {wf_cfg['step']}")
        print(f"   Fresh training : NOT loading any existing pkl\n")

        _, _, metrics = train_xgboost(
            df         = df_filtered,
            save_path  = pkl,
            n_trials   = trials,
            wf_step    = wf_cfg["wf_step"],
            sector_key = sector_key,
        )

        elapsed     = time.time() - t_start
        pkl_size_mb = round(os.path.getsize(pkl) / 1e6, 1) \
                      if os.path.exists(pkl) else 0

        result = {
            "sector":         sector_key,
            "name":           name,
            "status":         "OK",
            "pkl":            pkl,
            "pkl_size_mb":    pkl_size_mb,
            "tickers_target": tickers,
            "tickers_loaded": actual_tickers,
            "n_tickers":      len(actual_tickers),
            "start_date":     start_date,
            "regime_filter":  str(regime_filter),
            "raw_rows":       raw_rows,
            "filtered_rows":  filtered_rows,
            "filter_pct":     round(filtered_rows / raw_rows * 100, 1),
            "n_samples":      metrics.get("n_samples", 0),
            "n_windows":      metrics.get("n_windows", 0),
            "wf_acc":         metrics.get("wf_accuracy_mean", 0),
            "wf_acc_std":     metrics.get("wf_accuracy_std", 0),
            "wf_f1":          metrics.get("wf_f1_mean", 0),
            "wf_f1_std":      metrics.get("wf_f1_std", 0),
            "threshold":      metrics.get("optimal_threshold", 0.5),
            "thr_acc":        metrics.get("threshold_acc", 0),
            "thr_f1":         metrics.get("threshold_f1", 0),
            "best_iter":      metrics.get("best_iteration", 0),
            "optuna_score":   metrics.get("meta_optuna_score", 0),
            "gpu":            metrics.get("gpu_used", False),
            "elapsed_min":    round(elapsed / 60, 1),
        }

        print(f"\n✅ {name} — Training Complete")
        print(f"   {'─' * 50}")
        print(f"   Tickers loaded  : {result['tickers_loaded']}")
        print(f"   Date range      : {start_date} → {end_date}")
        print(f"   Raw rows        : {result['raw_rows']:,}")
        print(f"   Filtered rows   : {result['filtered_rows']:,} "
              f"({result['filter_pct']}% kept)")
        print(f"   Final samples   : {result['n_samples']:,}")
        print(f"   WF windows      : {result['n_windows']}")
        print(f"   PKL size        : {result['pkl_size_mb']} MB")
        print(f"   {'─' * 50}")
        print(f"   WF  Acc         : {result['wf_acc']:.4f} "
              f"± {result['wf_acc_std']:.4f}")
        print(f"   WF  F1          : {result['wf_f1']:.4f} "
              f"± {result['wf_f1_std']:.4f}")
        print(f"   {'─' * 50}")
        print(f"   Threshold       : {result['threshold']:.3f}")
        print(f"   Thr Acc         : {result['thr_acc']:.4f}")
        print(f"   Thr F1          : {result['thr_f1']:.4f}")
        print(f"   {'─' * 50}")
        print(f"   Best iteration  : {result['best_iter']}")
        print(f"   Optuna score    : {result['optuna_score']:.4f}")
        print(f"   GPU             : "
              f"{'Yes (CUDA)' if result['gpu'] else 'No (CPU)'}")
        print(f"   Time            : {result['elapsed_min']} min")

        return result

    except Exception as e:
        elapsed = time.time() - t_start
        print(f"\n❌ {name} FAILED after "
              f"{round(elapsed / 60, 1)} min")
        print(f"   Error: {e}")
        traceback.print_exc()
        return {
            "sector":      sector_key,
            "name":        name,
            "status":      "FAILED",
            "reason":      str(e),
            "elapsed_min": round(elapsed / 60, 1),
        }


# ═══════════════════════════════════════════════════════════════════════
# TRAIN ALL SECTORS
# ═══════════════════════════════════════════════════════════════════════

def train_all_sectors(sectors: dict = None) -> pd.DataFrame:
    """
    Train all sector models sequentially.
    Each sector is fully independent — own tickers, date range,
    regime filter, and fresh PKL.

    Single sector retrain:
        train_all_sectors({"hardware": SECTORS["hardware"]})
    """
    sectors       = sectors or SECTORS
    total         = len(sectors)
    results       = []
    session_start = time.time()

    print("\n" + "█" * 62)
    print(f"  SECTOR MODEL TRAINING SESSION v4")
    print(f"  Sectors   : {total}")
    print(f"  Trials    : 15 per sector")
    print(f"  GPU       : "
          f"{'Yes (CUDA)' if 'cuda' in GPU_CFG.get('device','') else 'No (CPU)'}")
    print(f"  Global pkl: NOT used or modified")
    print("█" * 62)

    print(f"\n  Sector plan:")
    for key, cfg in sectors.items():
        print(f"    {cfg['name']:<25} | "
              f"start={cfg['start_date']} | "
              f"filter={cfg['regime_filter']} | "
              f"trials={cfg['n_trials']} | "
              f"step={cfg['wf_config']['step']}")

    for i, (key, cfg) in enumerate(sectors.items(), 1):
        print(f"\n[{i}/{total}] Starting: {cfg['name']}...")
        result = train_sector(key, cfg)
        results.append(result)

        print(f"\n── Checkpoint [{i}/{total}] ──")
        for r in results:
            if r["status"] == "OK":
                print(f"  ✅ {r['name']:<25} | "
                      f"filtered={r.get('filtered_rows',0):,} "
                      f"({r.get('filter_pct',0)}%) | "
                      f"WF Acc={r['wf_acc']:.4f} | "
                      f"WF F1={r['wf_f1']:.4f} | "
                      f"Thr={r['threshold']:.3f} | "
                      f"Thr Acc={r['thr_acc']:.4f} | "
                      f"Thr F1={r['thr_f1']:.4f} | "
                      f"{r['elapsed_min']}min")
            else:
                print(f"  ❌ {r['name']:<25} | "
                      f"FAILED: {r.get('reason','unknown')}")

    # ── final summary ──────────────────────────────────────────────────
    session_elapsed = round((time.time() - session_start) / 60, 1)

    print("\n" + "═" * 62)
    print("  FINAL SUMMARY — ALL SECTORS")
    print("═" * 62)
    for r in results:
        if r["status"] == "OK":
            print(f"  ✅ {r['name']:<25} | "
                  f"filtered={r.get('filtered_rows',0):,} "
                  f"({r.get('filter_pct',0)}%) | "
                  f"WF Acc={r['wf_acc']:.4f} | "
                  f"WF F1={r['wf_f1']:.4f} | "
                  f"Thr={r['threshold']:.3f} | "
                  f"Thr Acc={r['thr_acc']:.4f} | "
                  f"Thr F1={r['thr_f1']:.4f} | "
                  f"{r['elapsed_min']}min")
        else:
            print(f"  ❌ {r['name']:<25} | "
                  f"FAILED: {r.get('reason','unknown')}")

    print(f"\n  Total session time: {session_elapsed} min")

    # ── diagnostic summary ─────────────────────────────────────────────
    diag_rows = []
    for r in results:
        if r["status"] == "OK":
            diag_rows.append({
                "sector":          r["sector"],
                "name":            r["name"],
                "status":          r["status"],
                "tickers_loaded":  str(r.get("tickers_loaded", [])),
                "start_date":      r.get("start_date", ""),
                "regime_filter":   r.get("regime_filter", ""),
                "raw_rows":        r.get("raw_rows", 0),
                "filtered_rows":   r.get("filtered_rows", 0),
                "filter_pct":      r.get("filter_pct", 0),
                "n_samples":       r.get("n_samples", 0),
                "n_windows":       r.get("n_windows", 0),
                "wf_acc":          r.get("wf_acc", 0),
                "wf_acc_std":      r.get("wf_acc_std", 0),
                "wf_f1":           r.get("wf_f1", 0),
                "wf_f1_std":       r.get("wf_f1_std", 0),
                "threshold":       r.get("threshold", 0),
                "thr_acc":         r.get("thr_acc", 0),
                "thr_f1":          r.get("thr_f1", 0),
                "threshold_ok":    r.get("threshold", 0) >= 0.45,
                "thr_f1_ok":       r.get("thr_f1", 0) >= 0.45,
                "wf_std_ok":       r.get("wf_f1_std", 1) <= 0.15,
                "best_iter_ok":    r.get("best_iter", 0) >= 50,
                "deploy_recommend": (
                    r.get("threshold", 0) >= 0.45 and
                    r.get("thr_f1", 0) >= 0.45 and
                    r.get("best_iter", 0) >= 50
                ),
                "best_iter":       r.get("best_iter", 0),
                "optuna_score":    r.get("optuna_score", 0),
                "pkl":             r.get("pkl", ""),
                "pkl_size_mb":     r.get("pkl_size_mb", 0),
                "elapsed_min":     r.get("elapsed_min", 0),
            })
        else:
            diag_rows.append({
                "sector":           r["sector"],
                "name":             r["name"],
                "status":           r["status"],
                "deploy_recommend": False,
                "reason":           r.get("reason", "unknown"),
            })

    diag_df   = pd.DataFrame(diag_rows)
    diag_path = "4_signals/models/sector_diagnostic_summary.csv"
    os.makedirs("4_signals/models", exist_ok=True)
    diag_df.to_csv(diag_path, index=False)

    # ── deploy recommendation table ────────────────────────────────────
    print(f"\n{'═' * 62}")
    print(f"  DIAGNOSTIC SUMMARY — DEPLOY RECOMMENDATIONS")
    print(f"{'═' * 62}")
    print(f"  {'Sector':<25} | {'Thr F1':<9} | {'Thr':<7} | "
          f"{'F1 std':<8} | {'Iter':<6} | Deploy")
    print(f"  {'-'*25}-+-{'-'*9}-+-{'-'*7}-+"
          f"-{'-'*8}-+-{'-'*6}-+-------")
    for r in diag_rows:
        if r["status"] == "OK":
            deploy    = "✅ YES" if r["deploy_recommend"] else "⚠️  NO"
            f1_flag   = "✅" if r["thr_f1_ok"]    else "❌"
            thr_flag  = "✅" if r["threshold_ok"]  else "❌"
            std_flag  = "✅" if r["wf_std_ok"]     else "❌"
            iter_flag = "✅" if r["best_iter_ok"]  else "❌"
            print(f"  {r['name']:<25} | "
                  f"{f1_flag} {r['thr_f1']:.4f}  | "
                  f"{thr_flag} {r['threshold']:.3f}  | "
                  f"{std_flag} {r['wf_f1_std']:.4f} | "
                  f"{iter_flag} {r['best_iter']:<4} | "
                  f"{deploy}")
        else:
            print(f"  {r['name']:<25} | "
                  f"FAILED: {r.get('reason','?')}")

    print(f"\n  Quality flags:")
    print(f"    Thr F1  >= 0.45 : meaningful signal post-threshold")
    print(f"    Thr     >= 0.45 : not firing on noise")
    print(f"    F1 std  <= 0.15 : stable across WF windows")
    print(f"    Iter    >= 50   : learned enough before early stop")
    print(f"\n  Saved → {diag_path}")
    print(f"  Total session time: {session_elapsed} min")

    return diag_df


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train regime-filtered sector models v4"
    )
    parser.add_argument(
        "--sector",
        type    = str,
        default = "all",
        choices = ["all"] + list(SECTORS.keys()),
        help    = "Sector to train. Default: all. "
                  "Options: hardware, hypercloud, "
                  "software, autos, defensive"
    )
    args = parser.parse_args()

    if args.sector == "all":
        summary = train_all_sectors()
    else:
        summary = train_all_sectors(
            {args.sector: SECTORS[args.sector]}
        )
