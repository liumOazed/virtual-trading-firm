"""
retrain.py — ARIA Model Retraining Orchestrator
================================================
Safe, repeatable retrain of all ARIA models with backup, validation
gate, and auto-rollback. MANUAL command for now (run it and watch);
designed to be schedulable later once trusted.

Full retrain = 2 trainers (confirmed interfaces):
    python 4_signals/xgboost_model.py            → global model + HMM
    python 4_signals/sector_model_trainer.py --sector all → 5 sector models

Procedure:
    1. Backup all 7 .pkl files (dated)
    2. Capture OLD backtest metrics (baseline)
    3. Train global + HMM
    4. Train 5 sectors
    5. Run backtest → capture NEW metrics
    6. GATE: new vs old within tolerances?
    7. PASS → keep + manifest  |  FAIL → rollback + manifest
    8. Write manifest log

Usage:
    python retrain.py                # full retrain with gate
    python retrain.py --dry-run      # backup + show plan, no training
    python retrain.py --skip-gate    # train + keep, NO validation (risky)

╔═══════════════════════════════════════════════════════════════════╗
║  TUNING MAP — REVIEW AFTER ~1 MONTH OF LIVE DATA                   ║
║  ----------------------------------------------------------------  ║
║  Everything you'll want to adjust once you have mature live data   ║
║  is in the CONFIG block below, clearly labelled. As of now (10     ║
║  days live) these are STARTING values, not validated. After a      ║
║  month of observation, revisit each one marked [TUNE].             ║
╚═══════════════════════════════════════════════════════════════════╝
"""

import os, sys, json, shutil, subprocess
from datetime import datetime
from pathlib import Path

ROOT = os.path.abspath(os.path.dirname(__file__))


# ════════════════════════════════════════════════════════════════════════
#  CONFIG — everything tunable lives here. After 1 month, revisit [TUNE].
# ════════════════════════════════════════════════════════════════════════

# ── Model artifacts to back up (6 files) ────────────────────────────────
# [STABLE] Verified against actual files on disk and what each trainer writes.
# software_model.pkl is a legacy artifact — not trained or loaded; excluded.
# hmm_detector.pkl is a cache written by backtest_engine_v2.py on first fit;
#   TRAIN_COMMANDS do not rewrite it, but it's backed up here for rollback safety.
MODEL_PATHS = [
    "4_signals/xgboost_global_model.pkl",       # written by xgboost_model.py
    "4_signals/models/hardware_model.pkl",       # written by sector_model_trainer.py
    "4_signals/models/hypercloud_model.pkl",     # written by sector_model_trainer.py
    "4_signals/models/autos_model.pkl",          # written by sector_model_trainer.py
    "4_signals/models/defensive_model.pkl",      # written by sector_model_trainer.py (gold reuses this)
    "5_backtesting/results/hmm_detector.pkl",    # cache; backed up but not retrained
]

# ── Training commands ───────────────────────────────────────────────────
# [STABLE] Confirmed interfaces. Both use end_date=today automatically.
# xgboost_model.py writes global model only (NOT hmm_detector.pkl).
# sector_model_trainer.py --sector all writes 4 sectors: hardware, hypercloud,
#   autos, defensive. (software sector was removed from trainer; legacy pkl ignored.)
TRAIN_COMMANDS = [
    ["python", "4_signals/xgboost_model.py"],                            # global model only
    ["python", "4_signals/sector_model_trainer.py", "--sector", "all"],  # 4 sectors
]

# ── Backtest command (for the validation gate) ──────────────────────────
# [STABLE] Adjust if run_backtest.py gains/changes flags.
BACKTEST_COMMAND = ["python", "run_backtest.py"]

# ── VALIDATION GATE TOLERANCES ──────────────────────────────────────────
# [TUNE ★★★] THE most important values to revisit after a month live.
# Right now (10 days live) these are educated starting points. After you
# have real data + a few observed retrains, tighten or loosen them.
#
# Logic: new models PASS only if ALL three hold vs the OLD backtest.
# Asymmetric on purpose — DD (your #1 concern) gets the tightest leash,
# return (noisiest) gets the loosest.
GATE = {
    "sharpe_min_ratio":  0.90,   # [TUNE] new Sharpe >= old × 0.90 (within 10%)
    "maxdd_max_ratio":   1.20,   # [TUNE] new |MaxDD| <= old × 1.20 (≤20% worse)
    "return_min_ratio":  0.85,   # [TUNE] new return >= old × 0.85 (within 15%)
}
# After 1 month, consider:
#   - If retrains keep passing easily → tighten (e.g. sharpe 0.95, dd 1.10)
#   - If good retrains get rejected on noise → loosen slightly
#   - If live DD diverges from backtest DD → reconsider dd ratio entirely

# ── METRIC KEYS — how to read new/old metrics from backtest output ──────
# [CONFIRMED] These keys match 5_backtesting/metrics.py (lines 199-207).
# metrics.py must write metrics.json (5-line add — see retrain notes).
# If metrics.json is absent, the gate falls back to computing these 3
# metrics directly from equity_curve.csv (robust, no stdout parsing).
METRICS_FILE = "5_backtesting/results/metrics.json"      # canonical (preferred)
EQUITY_CURVE_CSV = "5_backtesting/results/equity_curve.csv"  # fallback source
METRIC_KEYS = {
    "sharpe":       "sharpe",          # confirmed: metrics.py:202 (sharpe_raw)
    "max_drawdown": "max_drawdown",    # confirmed: metrics.py:207 (percent, negative)
    "total_return": "total_return",    # confirmed: metrics.py:199 (percent)
}
# NOTE: metrics.py stores these as PERCENTAGES (total_return×100, max_dd×100).
# The equity_curve fallback also returns percentages, so the gate ratios
# compare like-for-like. [TUNE] only matters that old & new use same units.

# ── Output locations ────────────────────────────────────────────────────
# [STABLE]
BACKUP_ROOT   = "model_backups"
MANIFEST_LOG  = "model_backups/retrain_manifest.jsonl"   # one line per retrain


# ════════════════════════════════════════════════════════════════════════
#  ORCHESTRATION
# ════════════════════════════════════════════════════════════════════════

def _ts():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def log(msg, sym="•"):
    print(f"  {sym} {msg}")

def backup_models(stamp):
    """Copy all current .pkl files to a dated backup folder."""
    dest = os.path.join(ROOT, BACKUP_ROOT, stamp)
    os.makedirs(dest, exist_ok=True)
    backed = []
    for rel in MODEL_PATHS:
        src = os.path.join(ROOT, rel)
        if os.path.exists(src):
            # preserve subfolder structure in backup
            flat = rel.replace("/", "__").replace("\\", "__")
            shutil.copy2(src, os.path.join(dest, flat))
            backed.append(rel)
        else:
            log(f"WARNING: {rel} not found — skipping backup", "⚠")
    log(f"Backed up {len(backed)}/{len(MODEL_PATHS)} models → {dest}", "✓")
    return dest, backed

def restore_models(backup_dir, backed):
    """Roll back: copy backup .pkl files over the live ones."""
    for rel in backed:
        flat = rel.replace("/", "__").replace("\\", "__")
        src = os.path.join(backup_dir, flat)
        dst = os.path.join(ROOT, rel)
        if os.path.exists(src):
            shutil.copy2(src, dst)
    log(f"Restored {len(backed)} models from {backup_dir}", "↩")

def read_metrics():
    """Read backtest metrics. Prefers canonical metrics.json; falls back
    to computing the 3 gate metrics directly from equity_curve.csv.
    Robust to metrics.py print-format changes."""
    # ── primary: canonical metrics.json (metrics.py writes this) ──
    jpath = os.path.join(ROOT, METRICS_FILE)
    if os.path.exists(jpath):
        try:
            with open(jpath) as f:
                raw = json.load(f)
            return {
                "sharpe":       float(raw[METRIC_KEYS["sharpe"]]),
                "max_drawdown": float(raw[METRIC_KEYS["max_drawdown"]]),
                "total_return": float(raw[METRIC_KEYS["total_return"]]),
                "_source":      "metrics.json",
            }
        except Exception as e:
            log(f"metrics.json unreadable ({e}) — trying equity_curve fallback", "⚠")

    # ── fallback: compute from equity_curve.csv ──
    cpath = os.path.join(ROOT, EQUITY_CURVE_CSV)
    if not os.path.exists(cpath):
        log(f"no metrics.json and no {EQUITY_CURVE_CSV} — cannot read metrics", "⚠")
        return None
    try:
        import pandas as pd, numpy as np
        df = pd.read_csv(cpath)
        # find the equity column (adapt if named differently)
        eqcol = next((c for c in df.columns if "equit" in c.lower()), df.columns[-1])
        eq = df[eqcol].astype(float).dropna()
        ret = eq.pct_change().dropna()
        total_return = (eq.iloc[-1] / eq.iloc[0] - 1) * 100          # percent
        sharpe = (ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0
        max_dd = (eq / eq.cummax() - 1).min() * 100                   # percent, negative
        return {
            "sharpe":       round(float(sharpe), 3),
            "max_drawdown": round(float(max_dd), 2),
            "total_return": round(float(total_return), 2),
            "_source":      "equity_curve.csv (computed)",
        }
    except Exception as e:
        log(f"failed to compute metrics from equity_curve: {e}", "⚠")
        return None

def run_cmd(cmd, label):
    log(f"running: {' '.join(cmd)}", "▶")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        log(f"{label} FAILED (exit {result.returncode})", "✗")
        return False
    log(f"{label} complete", "✓")
    return True

def evaluate_gate(old, new):
    """Compare new vs old. Returns (passed: bool, report: list[str])."""
    report = []
    if old is None:
        report.append("No OLD metrics — cannot gate. Treating as PASS (first run).")
        return True, report
    if new is None:
        report.append("No NEW metrics — backtest failed. ROLLBACK.")
        return False, report

    checks = []
    # Sharpe: new >= old × ratio
    sharpe_ok = new["sharpe"] >= old["sharpe"] * GATE["sharpe_min_ratio"]
    checks.append(sharpe_ok)
    report.append(
        f"Sharpe:  {new['sharpe']:.3f} vs old {old['sharpe']:.3f} "
        f"(need ≥ {old['sharpe']*GATE['sharpe_min_ratio']:.3f}) "
        f"{'✓' if sharpe_ok else '✗'}")

    # Max DD: |new| <= |old| × ratio  (DD stored negative)
    dd_ok = abs(new["max_drawdown"]) <= abs(old["max_drawdown"]) * GATE["maxdd_max_ratio"]
    checks.append(dd_ok)
    report.append(
        f"Max DD:  {new['max_drawdown']:.3f} vs old {old['max_drawdown']:.3f} "
        f"(allow ≤ {abs(old['max_drawdown'])*GATE['maxdd_max_ratio']:.3f}) "
        f"{'✓' if dd_ok else '✗'}")

    # Return: new >= old × ratio
    ret_ok = new["total_return"] >= old["total_return"] * GATE["return_min_ratio"]
    checks.append(ret_ok)
    report.append(
        f"Return:  {new['total_return']:.3f} vs old {old['total_return']:.3f} "
        f"(need ≥ {old['total_return']*GATE['return_min_ratio']:.3f}) "
        f"{'✓' if ret_ok else '✗'}")

    return all(checks), report

def write_manifest(entry):
    path = os.path.join(ROOT, MANIFEST_LOG)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    log(f"manifest updated → {MANIFEST_LOG}", "✓")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="backup + plan, no training")
    ap.add_argument("--skip-gate", action="store_true", help="train+keep, NO validation")
    args = ap.parse_args()

    stamp = _ts()
    print("="*64)
    print(f"  ARIA RETRAIN ORCHESTRATOR — {stamp}")
    print(f"  mode: {'DRY RUN' if args.dry_run else ('NO-GATE' if args.skip_gate else 'FULL + GATE')}")
    print("="*64)

    # 1. Backup
    backup_dir, backed = backup_models(stamp)

    # 2. Old metrics (baseline)
    old_metrics = read_metrics()
    log(f"baseline metrics: {old_metrics}", "•")

    if args.dry_run:
        print("\n  DRY RUN — would now train:")
        for c in TRAIN_COMMANDS: print(f"    {' '.join(c)}")
        print("  then run backtest + gate. No changes made.")
        return

    # 3-4. Train
    for cmd in TRAIN_COMMANDS:
        if not run_cmd(cmd, "train"):
            log("training failed — rolling back", "✗")
            restore_models(backup_dir, backed)
            write_manifest({"stamp": stamp, "result": "TRAIN_FAILED",
                            "backup": backup_dir})
            return

    # 5. Backtest → new metrics
    if not args.skip_gate:
        if not run_cmd(BACKTEST_COMMAND, "backtest"):
            log("backtest failed — rolling back", "✗")
            restore_models(backup_dir, backed)
            write_manifest({"stamp": stamp, "result": "BACKTEST_FAILED",
                            "backup": backup_dir})
            return
        new_metrics = read_metrics()

        # 6. Gate
        passed, report = evaluate_gate(old_metrics, new_metrics)
        print("\n  VALIDATION GATE:")
        for line in report: print(f"    {line}")

        # 7. Keep or rollback
        if passed:
            log("GATE PASSED — keeping new models", "✓")
            result = "PASS_KEPT_NEW"
        else:
            log("GATE FAILED — rolling back to old models", "✗")
            restore_models(backup_dir, backed)
            result = "FAIL_ROLLED_BACK"
    else:
        log("GATE SKIPPED (--skip-gate) — new models kept unvalidated", "⚠")
        new_metrics = read_metrics()
        result = "NO_GATE_KEPT_NEW"

    # 8. Manifest
    write_manifest({
        "stamp": stamp, "result": result, "backup": backup_dir,
        "old_metrics": old_metrics, "new_metrics": new_metrics,
        "gate": GATE,
    })
    print("\n" + "="*64)
    print(f"  RETRAIN COMPLETE — {result}")
    print("="*64)


if __name__ == "__main__":
    main()