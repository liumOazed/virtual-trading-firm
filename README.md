# Virtual Trading Firm — Documentation Index

**Project:** Virtual Trading Firm
**Repository root:** `D:\__A Google Drive Project\virtual_trading_firm\`
**Docs location:** `10_docs_v2/`
**Last updated:** June 2026
**Status:** LIVE — paper trading on Alpaca since 2026-06-01

---

## Current State (June 2026)

The system has moved from backtesting into **live paper trading on Alpaca**.
ARIA (the momentum strategy) runs daily against a live $100k paper account.
The focus has shifted from model development to live validation, execution
reliability, and building complementary strategies.

**What's live:**

- ARIA momentum engine — daily `run_live.py`, live Alpaca paper account
- HMM regime detection, XGBoost sector models, ESN — all frozen and serving
  inference in production (no retraining during live runs)
- Daily recording + reconciliation reporting for performance tracking
- **ARIA-Growth — regime-switching growth rotation, live paper on a SEPARATE
  Alpaca account (Zed2), since 2026-06-09** (forward tool, not yet validated)

**What's in development:**

- Growth point-in-time backtest — accumulating monthly screen snapshots toward
  an honest SEC-EDGAR-based validation (see Section 9b)

**What's shelved:**

- RL agent (see "Why RL Is Shelved" below)
- ARIA-Δ options strategy (tested, abandoned — bear-timing proved unprofitable)
- ARIA-MR (pairs trading / mean reversion) — researched extensively, **killed**
  after seven tests showed no accessible edge in liquid US equity pairs at
  daily OR hourly frequency after costs (see Section 9c)

---

## Contents

| File                      | Section                              | Status                |
| ------------------------- | ------------------------------------ | --------------------- |
| 01_core_infrastructure.md | 1. Core Infrastructure Setup         | Complete              |
| 02_agent_architecture.md  | 2. Agent Architecture                | Complete (superseded) |
| 03_market_data_layer.md   | 3. Market Data Layer                 | Complete              |
| 04_signal_generation.md   | 4. Signal Generation                 | Complete              |
| 05_backtesting.md         | 5. Backtesting Engine                | Complete              |
| 06_rl_agent.md            | 6. Reinforcement Learning Agent      | Shelved (see below)   |
| 07_explainer.md           | 7. Groq Explainer                    | Complete              |
| 08_live_trading.md        | 8. Live Trading (Alpaca)             | Active                |
| 09_strategies.md          | 9. Strategy Roadmap (ARIA / ARIA-MR) | Ongoing               |
| 09b_growth.md             | 9b. ARIA-Growth (regime rotation)    | Active (live paper)   |

---

## Strategy Roadmap

The project is building toward a multi-strategy, low-correlation portfolio.
Each strategy is an independent return stream with a different market exposure.

| Strategy        | Type                      | Beta  | Status                    | Target                 |
| --------------- | ------------------------- | ----- | ------------------------- | ---------------------- |
| **ARIA**        | Long-biased momentum (ML) | ~0.28 | LIVE paper                | 13-15% (102% backtest) |
| **ARIA-Growth** | Regime-switching growth   | long  | LIVE paper (Zed2 acct)    | Beat SPY/QQQ (TBD)     |
| **ARIA-MR**     | Market-neutral pairs      | ~0.05 | **Killed** (no edge)      | —                      |
| Leverage        | 1.5x on ARIA              | —     | Planned (post-validation) | Amplify proven engine  |

The thesis: ARIA captures bull momentum; ARIA-Growth rotates the growth book by
market regime (aggressive growth in calm bulls, quality-defensive in stress) on
its own isolated account so its contribution can be measured cleanly against
ARIA. ARIA-MR was intended as the uncorrelated all-weather sleeve but was killed
after honest testing (Section 9c).

**Abandoned: ARIA-Δ (bidirectional options).** A bidirectional regime
strategy using long calls/puts was built and backtested, then abandoned.
Diagnostic testing proved that in the 2020-2026 sample, fear signals
(VIX spikes, credit breakdown) _precede market rises, not falls_ — every
bear-timing signal was contrarian. Bear-shorting lost by construction.
The honest conclusion: market direction (especially bear timing) is not
reliably predictable with these signals, so the strategy was killed rather
than forced. Pairs trading (ARIA-MR) was meant to replace it as the
bidirectional profit engine — it was then also killed (Section 9c).

---

## Why RL Is Shelved (and How It Returns Later)

The RL agent (SAC + GRU, Section 6) was developed and benchmarked, but is
**deliberately shelved during live validation.** This is a sequencing
decision, not an abandonment.

**Why it's shelved now:**

- **Live data is the missing ingredient.** RL learns position-sizing and
  timing policy from experience. Training it on backtest data risks
  overfitting to historical paths. The system needs real live-execution
  data — actual fills, slippage, regime transitions experienced in
  production — before RL has something genuine to learn from.
- **Validate the base engine first.** ARIA's rules-based logic must prove
  itself live before adding a learned layer on top. If the foundation has
  issues (it's mid-validation), they must surface and be fixed against the
  deterministic engine, where behavior is interpretable — not masked under
  an RL policy.
- **Interpretability during debugging.** Live trading surfaced real bugs
  (logging, regime gating, trapped exits). These are far easier to diagnose
  with deterministic rules than with a black-box policy making the decisions.
- **Manual retrain friction.** RL requires Colab T4 GPU retraining (~3h,
  manual). During rapid live iteration, that cycle is too slow to be useful.

**How RL returns later (the plan):**

1. Accumulate 6+ months of live Alpaca execution data — real fills,
   slippage, regime transitions, the daily history now being recorded.
2. Use that live data (not backtest paths) as the RL training environment,
   so the agent learns from genuine market interaction.
3. Apply RL where it's strongest: **position sizing and entry/exit timing**
   on top of the proven ARIA signal — not signal generation itself (the
   XGBoost/HMM stack already does that well).
4. Benchmark the RL-sized portfolio against rules-based ARIA out-of-sample.
   Promote RL only if it clearly beats the deterministic version live.

In short: RL is paused until there's real live data to learn from and a
validated base engine to improve. It's the capstone, not the foundation.

---

## System Architecture

| Layer                      | Components                                                                         |
| -------------------------- | ---------------------------------------------------------------------------------- |
| **Infrastructure**         | venv + Google Drive + Colab GPU + Groq-patched TradingAgents                       |
| **Market Data**            | pandas-ta + FinBERT sentiment + Kalman risk + advanced features (RWI, OU, QV, HJB) |
| **Signal Gen**             | Echo State Network + XGBoost (Optuna/SHAP) + sentiment overlay                     |
| **Backtesting**            | Walk-forward, regime detection, Kalman ensemble, champion selection, kill-switch   |
| **Live Trading**           | Alpaca paper account, daily run_live.py, fill reconciliation, daily recorder       |
| **Reinforcement Learning** | SAC + GRU (50-feature state) — SHELVED pending live data                           |
| **Explainer**              | Groq-powered daily briefings, trade audits, regime warnings                        |

---

## Live Trading Layer (Section 8 — Active)

The production layer added since going live on Alpaca:

```
8_live_trading/
    alpaca_client.py        Alpaca REST wrapper (orders, positions, fills)
                            - get_order() for fill-price confirmation
                            - close_position() for clean full exits
                            - fractional/notional orders forced to DAY tif
    live_engine.py          daily orchestration: data → signals → orders → log
                            - regime gating (REGIME_TO_SECTORS)
                            - three-zone hysteresis (buy / hold / sell)
                            - fill-price retry loop (polls until confirmed)
    live_data_feed.py       daily price top-up, 18 tickers
    status_report.py        reconciliation report (realized/unrealized split,
                            FIFO matching, live equity from Alpaca)
    daily_recorder.py       pure record-keeper: one row/day, equity, P&L,
                            regime, and ALPHA vs SPY/QQQ since inception
    data/                   live_trade_log.csv, live_equity_curve.csv,
                            daily_history.csv, regime_state.json
```

**Regime → sector gating (REGIME_TO_SECTORS):**

| Regime                      | Active Sectors        | Tradeable Tickers                |
| --------------------------- | --------------------- | -------------------------------- |
| Bull-Trending               | hardware, autos, gold | NVDA, AVGO, TSM, TSLA, RACE, GLD |
| Bull-Stable                 | hypercloud, gold      | MSFT, GOOGL, AMZN, META, GLD     |
| Bear-Trending/Stable/Stress | defensive             | XOM, CVX, PG, WMT                |

Anchors (AAPL, QQQ) bypass the gate and use the global model in all regimes.

---

## ARIA-Growth Layer (Section 9b — Active, live paper)

A regime-switching, long-only growth rotation. It detects the market regime
from SPY and holds the growth basket whose characteristics suit that regime,
rotating as the regime flips. Runs on its **own isolated Alpaca paper account
(Zed2, `PA3N0JK22RWE`)** — completely separate from ARIA's account
(`PA304AGCZBF4`) so the two books never collide and each strategy's P&L is
measured cleanly. Went live 2026-06-09 with $100k.

**Honest status:** this is a FORWARD, rules-based tool, **not yet a validated
edge.** The stock screen is a current snapshot of fundamentals, so we cannot
prove it beat the market historically. Paper-trading it tells us whether the
regime calls and rotations behave sensibly in real time; the monthly screen
archive (below) is building the point-in-time dataset for an eventual honest
backtest vs SPY/QQQ and vs ARIA.

```
9_aria_growth/
    growth_screener.py            screens S&P500 + Nasdaq100, scores ~516 stocks
                                  on a growth composite → growth_screen_results.csv
    aria_growth_regime_allocator.py
                                  regime detection (SPY trend/vol/drawdown) +
                                  regime-conditional basket construction
    aria_growth_executor.py       syncs the Zed2 account to the regime basket;
                                  DRY-RUN by default, --execute to place orders;
                                  15% stop-loss + even redistribution to survivors;
                                  HARD account guard (refuses any account != Zed2)
    aria_growth_archive_screen.py banks a DATED monthly screen snapshot (builds
                                  the point-in-time dataset for a real backtest)
    aria_growth_daily_log.py      READ-ONLY daily snapshot logger for pattern
                                  analysis; never places orders
    .env                          Zed2 account keys (isolated from ARIA)
    screens/                      growth_screen_<date>.csv + _manifest.csv
    data/ + logs                  executor_state.json, growth_trade_log.csv,
                                  daily_positions.csv, daily_portfolio.csv,
                                  daily_log_state.json
```

**The three regimes (detected from SPY each run):**

| Regime   | Detected when                                            | Style                                                                   |
| -------- | -------------------------------------------------------- | ----------------------------------------------------------------------- |
| RISK_ON  | SPY > 200dma, vol < 18%, drawdown > -10% (calm bull)     | Aggressive growth — high revenue growth, hypergrowth tolerated          |
| RISK_OFF | SPY < 200dma, OR vol > 28%, OR drawdown < -10% (stress)  | Quality defensive — profitable, large-cap, sane valuation (banks, etc.) |
| NEUTRAL  | mixed signals (above 200dma but high vol, or below+calm) | Balanced GARP — Rule-of-40 ≥ 40, profitable, reasonable multiple        |

**Portfolio construction:** top 20 names by regime-fit score, max 4 per sector,
equal-weight, deploy 95% of equity (5% cash buffer).

**Exit logic (three triggers, only fire when the executor is run):**

1. **Stop-loss** — a holding down ≥ 15% from entry is closed; its cash is
   redistributed EVENLY across survivors (basket shrinks 20→19→…); stopped
   names are blocklisted until the next monthly re-screen (`--reset-stops`).
2. **Regime flip** — RISK_ON ↔ NEUTRAL ↔ RISK_OFF rotates the whole basket.
3. **Falls out of basket** — on a monthly re-screen, a name no longer ranking
   in the regime's top 20 is closed and replaced.

There is no take-profit and no trailing stop (entry-based stop only, to avoid
churning winners that merely pull back from a high). Responsiveness equals run
cadence — the script acts only when run, so stops/flips are caught as often as
it is run.

**Known limitation:** regime detection uses the 200-day average, which lags —
slow to flip into RISK_OFF in a fast crash. The 15% per-stock stop is the
faster backstop for that gap. Regime thresholds are reasonable priors, not yet
optimized; they're editable in the allocator and are exactly what the
point-in-time backtest would tune.

---

## ARIA-MR Layer (Section 9c — Killed)

ARIA-MR (market-neutral pairs trading / mean reversion) was intended as the
uncorrelated, all-weather sleeve. After extensive honest testing it was
**killed** — a successful research outcome, not a failure.

**What was tested (seven backtests + four papers):** daily megacap pairs, daily
sector pairs, an OU-MLE signal engine, tightness (Euclidean-distance) selection,
entry confirmation (reverting-threshold logic), and finally HOURLY data (74
tickers, ~5,000 bars/pair). Every test converged on the same null result:
Sharpe ≈ 0, average loss ≥ average win, no edge after realistic retail costs.

**Why it has no accessible edge:** profitable equity-pairs MR requires breadth
(hundreds-to-thousands of pairs), institutional costs (1-2 bps vs ~20 bps
retail), and factor-neutralized residual signals — structural advantages, not
parameters we can add. The literature's winning results live at MINUTE
frequency, in less-arbitraged/emerging markets, or pre-2010 regimes; the
hourly test confirmed large-cap spreads revert on the same slow multi-week
timescale as daily (no faster idiosyncratic dislocations to capture).

**Reusable assets retained:** the validated cointegration pipeline, OU signal
engine, market-neutral backtester (hand-checked P&L), and the Alpaca hourly
data fetcher — all available if a genuinely different market-neutral edge
(ETF/NAV arb, less-arbitraged small-caps) is pursued later.

---

## Performance Summary

**Backtest** (2021-01-04 to 2025-03-28, 1,544 trading days):

| Metric               | Backtest | RL Agent OOS |
| -------------------- | -------- | ------------ |
| Total return         | 78.8%    | 21.72%       |
| Annualised return    | 14.74%   | 39.75%       |
| Sharpe               | 1.342    | 1.348        |
| Sortino              | 2.155    | 2.301        |
| Calmar               | 1.067    | 2.562        |
| Max drawdown         | -13.82%  | -15.51%      |
| Win rate             | 80.2%    | —            |
| Bull-Trending Sharpe | 2.916    | —            |
| Bear-Trending Sharpe | 2.065    | —            |

**Live paper** (since 2026-06-01): in validation. Tracked daily via
`daily_recorder.py` with alpha measured against SPY/QQQ. Live results
will be summarized here after the initial validation window completes.

**ARIA-Growth live paper** (since 2026-06-09, Zed2 account): in validation.
Tracked daily via `aria_growth_daily_log.py` with book performance measured
against SPY and QQQ since go-live. Results summarized here once a meaningful
window accumulates.

---

## Key Design Decisions

**Groq as explainer only.** Free tier (100k tokens/day) was consumed in a
single multi-ticker training run. All signal generation, regime detection,
and allocation are local ML; Groq is reserved for plain-English explanation
(~300 tokens/ticker/day).

**Local-first ML.** pandas-ta, FinBERT, ESN, XGBoost all run locally, no GPU.
Only the RL agent needed GPU (Colab T4) — and it's now shelved.

**Sentiment in training vs inference.** Training uses `sentiment = 0.0`;
FinBERT applied as a post-prediction overlay at inference only, eliminating
look-ahead leakage from NewsAPI's 30-day window. (Note: FinBERT is currently
disabled in live trading — generic non-ticker news created OOD noise; a
ticker-specific feed + retrain is the planned proper fix.)

**Triple Barrier labels.** ATR-scaled barriers (1.1x profit / 1.0x stop)
assign labels only on meaningful moves, improving label quality.

**Walk-forward validation.** Expanding-window walk-forward CV; no k-fold
(it leaks future data into earlier training windows).

**Kill-switch.** Halts new trades when 20-bar rolling Sharpe < -1.25 AND
drawdown > 4% AND Sharpe slope negative — triple condition avoids false
positives. Equity recording continues so the system can resume.

**Sortino over Sharpe as primary metric.** Given the asymmetric return
profile (high win rate, occasional large winners), Sortino better reflects
quality. RL reward (when reactivated) weights Sortino 70% / Sharpe 30%.

**Look-ahead bias is the #1 enemy.** Reinforced through live development:
strategies must select on past data and trade on future data; never use
today's known winners. This principle killed a growth-stock-selection idea
and shaped the ARIA-MR design (select cointegrated pairs on 2018-2020,
trade on 2021-2026). It also defines ARIA-Growth's honest boundary: the
current screen is a snapshot, so the live book is a forward tool — the
monthly screen archive accumulates point-in-time data so a future backtest
can validate it without look-ahead.

**Isolated accounts per strategy.** ARIA-Growth runs on a separate Alpaca
paper account (Zed2) from ARIA, with a hard account guard in the executor that
refuses to run on any other account. This prevents the two books from
liquidating each other's positions and keeps per-strategy P&L clean for
measuring whether growth genuinely adds to ARIA.

**Honesty over attachment.** ARIA-Δ was killed after the data disproved
its thesis; ARIA-MR was killed after seven tests showed no accessible edge.
A strategy that doesn't survive honest diagnostic testing is abandoned, not
forced. Killing a bad strategy is a successful outcome.

---

## Live Operations

| Task                       | Who                                    | When                              | Duration |
| -------------------------- | -------------------------------------- | --------------------------------- | -------- |
| Daily live run (ARIA)      | Manual, `run_live.py`                  | Weekdays ~19:30 Dhaka (US open)   | ~3-4 min |
| Daily status check (ARIA)  | Manual, `run_live.py --status`         | After run                         | seconds  |
| Daily recording (ARIA)     | Manual, `daily_recorder.py`            | After US close                    | ~30 sec  |
| Growth rebalance check     | Manual, `aria_growth_executor.py`      | Weekly (dry-run; --execute)       | ~30 sec  |
| Growth daily log           | Manual, `aria_growth_daily_log.py`     | Daily after US close (~2am Dhaka) | ~30 sec  |
| Growth monthly re-screen   | Manual, `growth_screener.py` + archive | 1st of month, then --reset-stops  | ~5 min   |
| Backtest retrain (XGBoost) | scheduler.py auto                      | 1st of month, 02:00               | ~8 hours |
| RL agent retrain           | SHELVED                                | —                                 | —        |

---

## Known Issues (current)

| Issue                                                                                                           | Status                                                                                          |
| --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Trapped exits: held positions can't sell when their sector is gated out (e.g. TSLA in autos during Bull-Stable) | Open — fix: evaluate held positions vs sell threshold regardless of sector-active               |
| Fill price logs as 0 if order fills slower than retry window                                                    | Mitigated — retry loop + loud warning; next-run reconciliation is the bulletproof fix if needed |
| FinBERT returns identical sentiment across tickers                                                              | Disabled in live; ticker-specific feed + retrain planned                                        |
| Hypercloud round-trip churn in Bull-Stable (buy ~0.55, sell ~0.40 days later)                                   | Monitoring — surfaced during live validation                                                    |
| COVID crash / 2018 Q4 stress tests return NaN                                                                   | Outside current backtest date range                                                             |
| ARIA-Growth not historically validated (current-snapshot screen)                                                | By design — monthly archive building point-in-time data for an EDGAR backtest                   |
| ARIA-Growth regime detection lags (200dma) — slow to de-risk in fast crash                                      | Mitigated by 15% per-stock stop; thresholds editable, to be tuned in backtest                   |

---

## Documentation History

- **v1** (`8_docs_v1/`): original development docs (Sections 1-7), through
  the backtesting and RL-development phase.
- **v2** (`10_docs_v2/`): current docs, reflecting the move to live Alpaca
  trading, RL shelving, and the multi-strategy roadmap.
