"""
scheduler.py
============
VIRTUAL_TRADING_FIRM — Production Scheduler

What runs here (local CPU)
--------------------------
  1st of month 02:00  → backtest retrain (XGBoost walk-forward)
  1st of month 02:00  → price_data.pkl save
  1st of month 02:00  → MODEL_CUTOFF update in groq_explainer.py
  1st of month 02:00  → Colab RL retrain reminder printed to log
  Weekdays     17:30  → metrics tearsheet refresh
  Weekdays     18:00  → Groq daily explainer

What runs on Colab T4 (manual, monthly)
----------------------------------------
  After local backtest finishes → open Colab → run rl_retrain_colab.py
  Files are synced via Google Drive automatically.

Run
---
  python scheduler.py                    # production mode (runs forever)
  python scheduler.py --retrain-now      # force backtest retrain now
  python scheduler.py --explain-now      # force explainer now
  python scheduler.py --metrics-now      # force metrics now

Install
-------
  pip install schedule
"""

import os
import sys
import re
import time
import pickle
import argparse
import logging
import traceback
from datetime import date, datetime, timedelta
from typing import List

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "4_signals"))
sys.path.insert(0, os.path.join(ROOT, "5_backtesting"))
sys.path.insert(0, os.path.join(ROOT, "6_rl_agent"))
sys.path.insert(0, os.path.join(ROOT, "7_explainer"))

try:
    import schedule
except ImportError:
    print("❌ schedule not installed. Run: pip install schedule")
    sys.exit(1)

# ── logging ───────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s | %(levelname)s | %(message)s",
    handlers = [
        logging.FileHandler(f"logs/scheduler_{date.today()}.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("scheduler")

# ── config ────────────────────────────────────────────────────────────────
TICKERS        = ["AAPL", "NVDA", "MSFT", "SPY", "QQQ", "TSLA"]
RESULTS_DIR    = "5_backtesting/results"
RL_DIR         = "6_rl_agent"
EXPLAINER_DIR  = "7_explainer"
EXPLAINER_FILE = "7_explainer/groq_explainer.py"
BACKTEST_START = "2020-01-01"


# ════════════════════════════════════════════════════════════════════════════
# 1.  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _update_model_cutoff(new_date: str):
    if not os.path.exists(EXPLAINER_FILE):
        log.warning(f"Explainer not found: {EXPLAINER_FILE}")
        return
    with open(EXPLAINER_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    updated = re.sub(
        r'MODEL_CUTOFF\s*=\s*"[0-9]{4}-[0-9]{2}-[0-9]{2}"',
        f'MODEL_CUTOFF = "{new_date}"',
        content,
    )
    with open(EXPLAINER_FILE, "w", encoding="utf-8") as f:
        f.write(updated)
    log.info(f"MODEL_CUTOFF updated → {new_date}")


def _write_colab_reminder(end_date: str):
    """Writes a reminder file so you know Colab RL retrain is needed."""
    path = f"logs/COLAB_RETRAIN_NEEDED_{end_date}.txt"
    with open(path, "w") as f:
        f.write(f"RL RETRAIN REQUIRED\n")
        f.write(f"{'='*50}\n")
        f.write(f"Backtest completed: {datetime.now()}\n")
        f.write(f"New data end date:  {end_date}\n\n")
        f.write(f"Steps to retrain RL on Colab T4:\n")
        f.write(f"  1. Open Colab notebook\n")
        f.write(f"  2. Mount Google Drive (files already synced)\n")
        f.write(f"  3. Run: python 6_rl_agent/rl_agent.py\n")
        f.write(f"     OR use the Colab cells from your last session\n")
        f.write(f"  4. best_model.zip saves to Drive automatically\n")
        f.write(f"  5. Delete this file when done\n\n")
        f.write(f"Files ready for Colab:\n")
        f.write(f"  {RESULTS_DIR}/equity_curve_v2.csv\n")
        f.write(f"  {RESULTS_DIR}/trade_log_v2.csv\n")
        f.write(f"  {RESULTS_DIR}/price_data.pkl\n")
    log.info(f"⚠️  COLAB RL RETRAIN NEEDED → {path}")


def _is_weekday() -> bool:
    return date.today().weekday() < 5


def _safe_run(fn, label: str):
    try:
        log.info(f"▶  {label}")
        fn()
        log.info(f"✅ {label} done")
    except Exception:
        log.error(f"❌ {label} failed\n{traceback.format_exc()}")


# ════════════════════════════════════════════════════════════════════════════
# 2.  JOB: BACKTEST RETRAIN  (local CPU — XGBoost only)
# ════════════════════════════════════════════════════════════════════════════

def job_backtest_retrain():
    """
    Monthly backtest retrain — runs locally on CPU.
    Retrains XGBoost walk-forward on all data up to today.
    RL retrain is SEPARATE — must be done manually on Colab T4.
    """
    end_date = (date.today() - timedelta(days=3)).strftime("%Y-%m-%d")
    log.info(f"=== BACKTEST RETRAIN START | end_date={end_date} ===")

    # ── Step 1: backtest engine ───────────────────────────────────────────
    log.info("Step 1/3 — Running backtest engine (XGBoost walk-forward)")
    try:
        from backtest_engine_v2 import BacktestEngineV2, BacktestConfig

        cfg = BacktestConfig(
            tickers      = TICKERS,
            start_date   = BACKTEST_START,
            end_date     = end_date,
            retrain      = True,
            train_months = 12,
            oos_months   = 2,
            fixed_size   = 0.10,
            base_slippage= 0.001,
        )

        bt = BacktestEngineV2(cfg)
        bt.prepare()
        bt.run()
        log.info("Step 1/3 ✅ Backtest + XGBoost retrain complete")

    except Exception:
        log.error(f"Step 1/3 ❌ Backtest failed\n{traceback.format_exc()}")
        return

    # ── Step 2: save price_data.pkl ───────────────────────────────────────
    log.info("Step 2/3 — Saving price_data.pkl")
    try:
        pkl_path = f"{RESULTS_DIR}/price_data.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(bt.loader.price_data, f)
        log.info(f"Step 2/3 ✅ price_data.pkl → {pkl_path}")
    except Exception:
        log.error(f"Step 2/3 ❌ pkl save failed\n{traceback.format_exc()}")
        return

    # ── Step 3: update cutoff + remind about Colab ───────────────────────
    log.info("Step 3/3 — Updating MODEL_CUTOFF + Colab reminder")
    _update_model_cutoff(end_date)
    _write_colab_reminder(end_date)

    log.info("=== BACKTEST RETRAIN COMPLETE ===")
    log.info("⚠️  ACTION REQUIRED: Retrain RL agent on Colab T4")
    log.info(f"    Check: logs/COLAB_RETRAIN_NEEDED_{end_date}.txt")


# ════════════════════════════════════════════════════════════════════════════
# 3.  JOB: DAILY METRICS
# ════════════════════════════════════════════════════════════════════════════

def job_daily_metrics():
    if not _is_weekday():
        return
    try:
        from metrics import run_metrics
        results = run_metrics(
            equity_file = f"{RESULTS_DIR}/equity_curve_v2.csv",
            trade_file  = f"{RESULTS_DIR}/trade_log_v2.csv",
            save_chart  = True,
        )
        core = results.get("core", {})
        log.info(
            f"Metrics | Return={core.get('total_return',0):.2f}% | "
            f"Sharpe={core.get('sharpe',0):.3f} | "
            f"Sortino={core.get('sortino',0):.3f} | "
            f"DD={core.get('max_drawdown',0):.2f}%"
        )
    except Exception:
        log.error(f"Metrics failed\n{traceback.format_exc()}")


# ════════════════════════════════════════════════════════════════════════════
# 4.  JOB: DAILY EXPLAINER
# ════════════════════════════════════════════════════════════════════════════

def job_daily_explainer():
    if not _is_weekday():
        return
    try:
        from groq_explainer import SystemStateCollector, TradingExplainer
        from rl_agent import RLTradingAgent, load_backtest_outputs

        TODAY = date.today().strftime("%Y-%m-%d")

        # RL weights from trained model
        RL_WEIGHTS = {}
        try:
            obs_matrix, _, _, _, _ = load_backtest_outputs(
                tickers     = TICKERS,
                equity_file = f"{RESULTS_DIR}/equity_curve_v2.csv",
                trade_file  = f"{RESULTS_DIR}/trade_log_v2.csv",
                price_pkl   = f"{RESULTS_DIR}/price_data.pkl",
            )
            agent = RLTradingAgent(tickers=TICKERS, device="cpu")
            agent.load(f"{RL_DIR}/best_model/best_model")
            RL_WEIGHTS = agent.predict_weights(obs_matrix[-agent.seq_len:])
            log.info(f"RL weights loaded: "
                     f"{ {t: f'{w:.1%}' for t,w in RL_WEIGHTS.items()} }")
        except Exception:
            log.warning("RL weights unavailable — equal weights")
            RL_WEIGHTS = {t: 1/len(TICKERS) for t in TICKERS}

        # signal engine
        signal_engine = None
        try:
            from signal_engine import SignalEngine
            signal_engine = SignalEngine()
        except Exception as e:
            log.warning(f"SignalEngine unavailable: {e}")

        # metrics
        metrics_result = None
        try:
            from metrics import run_metrics
            metrics_result = run_metrics(save_chart=False)
        except Exception as e:
            log.warning(f"Metrics unavailable: {e}")

        # collect + explain
        collector = SystemStateCollector(TICKERS)
        state     = collector.collect(
            target_date    = TODAY,
            signal_engine  = signal_engine,
            rl_weights     = RL_WEIGHTS,
            equity_file    = f"{RL_DIR}/rl_equity_curve.csv",
            trade_file     = f"{RESULTS_DIR}/trade_log_v2.csv",
            metrics_result = metrics_result,
        )

        explainer = TradingExplainer()
        explainer.run_daily(state, verbose=False)

        if state["recent_trades"]:
            explainer.trade_audit(
                state["recent_trades"][-1], state, verbose=False
            )

        explainer.weekly_summary(state, verbose=False)
        explainer.save_log()

        log.info(f"Explainer done → "
                 f"{EXPLAINER_DIR}/daily_report_{TODAY}.txt")

    except Exception:
        log.error(f"Explainer failed\n{traceback.format_exc()}")


# ════════════════════════════════════════════════════════════════════════════
# 5.  SCHEDULE SETUP
# ════════════════════════════════════════════════════════════════════════════

def setup_schedule():

    # monthly backtest retrain: 1st of month at 02:00
    def _monthly_gate():
        if date.today().day == 1:
            _safe_run(job_backtest_retrain, "Backtest retrain")

    schedule.every().day.at("02:00").do(_monthly_gate)

    # daily metrics: 17:30
    schedule.every().day.at("17:30").do(
        lambda: _safe_run(job_daily_metrics, "Daily metrics")
    )

    # daily explainer: 18:00
    schedule.every().day.at("18:00").do(
        lambda: _safe_run(job_daily_explainer, "Daily explainer")
    )

    log.info("Schedule configured:")
    log.info("  02:00 daily  → monthly backtest retrain gate (1st of month)")
    log.info("  17:30 daily  → metrics tearsheet")
    log.info("  18:00 daily  → Groq explainer")
    log.info("")
    log.info("  ⚠️  RL retrain: manual on Colab T4 after each backtest")
    log.info("      Check logs/COLAB_RETRAIN_NEEDED_*.txt for reminders")


# ════════════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VIRTUAL_TRADING_FIRM Scheduler"
    )
    parser.add_argument("--retrain-now", action="store_true",
                        help="Force immediate backtest retrain")
    parser.add_argument("--explain-now", action="store_true",
                        help="Force immediate explainer")
    parser.add_argument("--metrics-now", action="store_true",
                        help="Force immediate metrics refresh")
    args = parser.parse_args()

    log.info("=" * 55)
    log.info("  VIRTUAL TRADING FIRM — SCHEDULER")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 55)

    if args.retrain_now:
        _safe_run(job_backtest_retrain, "Force backtest retrain")
        return

    if args.explain_now:
        _safe_run(job_daily_explainer, "Force explainer")
        return

    if args.metrics_now:
        _safe_run(job_daily_metrics, "Force metrics")
        return

    setup_schedule()

    log.info("⏰ Running. Ctrl+C to stop.")
    log.info(f"   Today is day {date.today().day} of the month")
    log.info(f"   Next retrain: 1st of next month at 02:00\n")

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()