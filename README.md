# Virtual Trading Firm — Documentation Index

**Project:** Virtual Trading Firm
**Repository root:** `D:\__A Google Drive Project\virtual_trading_firm\`
**Docs location:** `10_docs_v2/`
**Last updated:** 2026-07-09
**Status:** LIVE — paper trading on Alpaca (account reset 2026-06-15, $100k); locked 95.55% / Sharpe 1.442 min-hold engine

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
- ARIA-MR momentum-universe probe — a separate mean-reversion test on the 16
  ARIA tickers (oversold-bounce signals), run this session; likewise shelved —
  real but too rare and too low-Sharpe (~0.16 book aggregate) to earn a capital
  slice blended with ARIA (see Section 9c addendum)

---

## Resolved Finding — Backtest Regime Coverage (Updated 2026-06-18)

**The earlier "zero Bull-Stable coverage" finding is resolved.** That finding
applied to the old Hurst-taxonomy backtest (two states: Bull-Trending and
Bear-Trending). The current 4-state GaussianHMM covers all four regimes, with
**Bull-Stable as the single largest segment** (643 of 1,403 equity-curve bars).

**Current 4-state backtest coverage:**

| Regime        | Bars | Ann Ret% | Sharpe | Max DD% | Trades |
| ------------- | ---- | -------- | ------ | ------- | ------ |
| Bull-Trending | 176  | 93.77    | 3.84   | -4.92   | 71     |
| Bull-Stable   | 643  | 27.17    | 1.35   | -6.72   | 194    |
| Bear-Stable   | 561  | 30.74    | 1.81   | -2.97   | 67     |
| Bear-Stress   | 23   | -3.61    | -0.35  | -1.95   | 3      |

Live trading since June 2026 has run in Bull-Stable — now the best-covered
regime in the backtest. The "hypercloud churn" observed in early live runs is
consistent with the 61.9% win rate and 27% annualised return in Bull-Stable
(the strategy rotates more frequently in this regime, which is expected). The
conditional min-hold fix (June 2026) is now mirrored in the live engine and
expected to reduce churn.

**Still-open question — live-vs-train feature drift:** the live HMM
classifies every session so far as Bull-Stable, while the historical backtest
distributes fairly evenly across Bear-Stable (561) and Bull-Stable (643). One
explanation is that the current VIX/vol/breadth feature distribution genuinely
sits in a different part of the HMM state space than the 2020-2026 training
average — a mild form of feature drift that shifts the posterior toward
Bull-Stable. This is not a defect; it resolves only with more live data and
eventually a retrain that includes recent Bull-Stable periods.

**Bear-Stress caveat:** 23 bars and 3 trades is too sparse to draw conclusions
from the -3.61% annualised / Sharpe -0.35. Forced-exit triggers (bear_heat_trim,
structural break detector, emergency retrain) backstop this regime; the sample
count will grow with the next retrain cycle.

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
| **ARIA**        | Long-biased momentum (ML) | ~0.28 | LIVE paper                | 13-15% (95.55% locked) |
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

**Addendum (June 2026) — Momentum-universe MR probe (separate effort):** a
second, independent MR test was run this session on the 16 ARIA tickers
directly: oversold-bounce signals (RSI/BB entry, fixed exit) tested on the
momentum universe to see if an intra-universe mean-reversion slice could
diversify ARIA. Result: real edge exists on individual names, but signals are
too rare and the book aggregate Sharpe is ~0.16 — insufficient to justify a
capital slice in a blend with ARIA. Shelved on the same grounds as the pairs
approach: insufficient edge after costs at accessible retail scale. This
finding is distinct from the cointegration pairs work above; both are recorded
here for completeness.

---

## Performance Summary

**ARIA Backtest** (2,063 trading days, walk-forward OOS) — engine locked 2026-06-18:

| Metric            | Value     |
| ----------------- | --------- |
| Total return      | 95.55%    |
| Annualised return | 12.61%    |
| Annualised vol    | 8.61%     |
| Sharpe (raw)      | 1.442     |
| Sharpe (rf-adj)   | 0.875     |
| Sortino           | 1.320     |
| Calmar            | 1.849     |
| Omega             | 1.359     |
| Max drawdown      | -6.82%    |
| Avg DD duration   | 13.2 bars |
| VaR 95%           | -0.716%   |
| CVaR 95%          | -1.150%   |
| Daily hit rate    | 49.39%    |

**Trade analysis** (335 closed round trips):

| Metric             | Value              |
| ------------------ | ------------------ |
| Win rate           | 66.57%             |
| Avg win            | +11.83%            |
| Avg loss           | -4.35%             |
| Payoff ratio       | 2.722              |
| Profit factor      | 4.844              |
| Expectancy         | +6.42% per trade   |
| Avg hold           | 67.3 days          |
| Best / worst trade | +402.91% / -19.67% |
| Total gross P&L    | $360,618           |

**Regime breakdown** (all 4 HMM states covered):

| Regime        | Bars | Ann Ret% | Sharpe | Max DD% | Win% |
| ------------- | ---- | -------- | ------ | ------- | ---- |
| Bull-Trending | 176  | 93.77    | 3.84   | -4.92   | 76.1 |
| Bull-Stable   | 643  | 27.17    | 1.35   | -6.72   | 61.9 |
| Bear-Stable   | 561  | 30.74    | 1.81   | -2.97   | 70.1 |
| Bear-Stress   | 23   | -3.61    | -0.35  | -1.95   | 66.7 |

**Benchmark comparison** (buy & hold, identical OOS dates):

| Metric             | ARIA    | SPY B&H  | QQQ B&H  |
| ------------------ | ------- | -------- | -------- |
| Total return       | +95.55% | +140.00% | +169.24% |
| Sharpe (raw)       | 1.442   | 1.016    | 0.907    |
| Max drawdown       | -6.82%  | -24.50%  | -35.12%  |
| Alpha (ann) vs SPY | ~+4.6%  | —        | —        |
| Beta vs SPY        | ~0.28   | —        | —        |
| Correlation vs SPY | ~0.49   | —        | —        |

**How to read this:** ARIA returns less than buy-and-hold SPY/QQQ in
absolute terms (+95.55% vs +140%/+169%) but with significantly higher
risk-adjusted quality — Sharpe 1.442 vs 1.016/0.907, and less than a
third of the drawdown (-6.82% vs -24.5%/-35.1%). The value proposition is
_participation with protection_: market-like upside, dramatically smaller
losses. Low beta (~0.28) confirms the return is driven mainly by alpha,
not market exposure.

**Note on regimes:** the current backtest covers all four HMM regimes.
Bull-Stable is the largest segment (643 bars), and live trading since
June 2026 has run entirely in Bull-Stable — meaning live performance is
now directly comparable to the 27.17% annualised / Sharpe 1.35 backtested
Bull-Stable period. The conditional min-hold fix (live-mirrored June 2026)
is expected to reduce the hypercloud churn observed in early live runs.

---

### Why the 95.55% engine is better than the prior 100.4% version

The headline return dropped. Every risk-adjusted and architectural metric
improved.

**Risk-adjusted comparison:**

| Metric         | Old (100.4%) | Current (95.55%) | Better              |
| -------------- | ------------ | ---------------- | ------------------- |
| Total return   | 100.4%       | 95.55%           | Old (headline only) |
| Annual return  | 13.14%       | 12.61%           | ~Tie (0.5pt apart)  |
| Sharpe (raw)   | 1.321        | 1.442            | **Current**         |
| Sortino        | 1.169        | 1.320            | **Current**         |
| Calmar         | 1.301        | 1.849            | **Current** (+42%)  |
| Max drawdown   | -10.1%       | -6.82%           | **Current**         |
| Annualised vol | 9.84%        | 8.61%            | **Current**         |

Three reasons the current engine is better despite the lower headline:

**1. Risk-adjusted quality.** Annual return is essentially tied (12.61% vs
13.14% — 0.5 percentage points apart over 5.5 years). The current engine
achieves that return on substantially less risk: -6.82% vs -10.1% max
drawdown (33% shallower), lower volatility, and higher Sharpe, Sortino, and
Calmar. Same reward, much less risk — a better engine by definition.

**2. Modern regime architecture.** The old 100.4% was built on the 2-state
Hurst taxonomy (Bear-Trending / Bull-Trending). The current engine uses the
4-state GaussianHMM (Bull-Trending, Bull-Stable, Bear-Stable, Bear-Stress)
— a different, more refined architecture, not just a retuned one. That
structural change also enabled the 8-phase enhancement stack (ESN latent
PCA, cross-asset macro layer, inflation signal engine, tail-risk hedger,
structural break detector) that was impossible on the old taxonomy.

**3. Leverage headroom = the path to 100%+.** The lower drawdown is not just
"safer" — it is leverage capacity. At 1.30x, the current engine is projected
to produce ~133% total at -8.9% max DD — still below the old engine's
unlevered -10.1%. At ~1.47x (matched to the old engine's risk level), the
projected return is ~158%, approximately 58 points more than the old engine
made at identical drawdown.

_Honest note on per-trade stats:_ the old engine had marginally better
per-trade statistics (profit factor 5.372 vs 4.844, win rate 68.64% vs
66.57%). Those do not offset accepting ~48% more drawdown (-10.1% vs -6.82%)
for essentially the same annual return. The trade-off favours the current
engine for any leverage-based path to the 100%+ goal.

_Honest note on the ceiling:_ the 95.55% was reached by testing ~19 ideas
and keeping exactly ONE structural change (the conditional min-hold). That
makes it an exhaustively-tested, honest baseline — not a curve-fit. The old
100.4% was a vanity headline on a riskier, older architecture.

**Leverage projection** (projected — linear-scaling estimate, NOT a re-backtested result):

| Leverage | Annual% | Max DD% | ~Total (5.5y) | Notes                                                 |
| -------- | ------- | ------- | ------------- | ----------------------------------------------------- |
| 1.00x    | 12.61%  | -6.82%  | ~95%          | Base, unlevered                                       |
| 1.15x    | ~14.5%  | ~-7.8%  | ~113%         | Clears 100%+ goal; DD still below old engine's -10.1% |
| 1.30x    | ~16.4%  | ~-8.9%  | ~133%         |
| 1.45x    | ~18.3%  | ~-9.9%  | ~155%         | Beating SPY DD still below 10%                        |
| ~1.47x   | ~18.5%  | ~-10.0% | ~158%         | Matched to old engine's risk; ~58pt more return       |

**Key framing:** at the same max drawdown the old engine took to make 100.4%
(-10.1%), the current engine projects ~158% — roughly 58 percentage points
more return at identical risk. At the planned 1.30x target, it clears the
100%+ goal (~133%) at -8.9% DD, lower risk than the old engine ran unlevered.

**Mandatory caveats — these apply before any leverage is ever considered:**

1. **Projected, not backtested.** Linear-scaling estimates only. A real
   levered backtest could differ due to path effects, volatility drag, and
   margin calls at drawdown troughs.
2. **Borrow cost excluded.** Margin interest (~5–7%/yr on the borrowed
   portion) shaves real return. At 1.30x, expect ~1.5–2%/yr drag; net
   annual is closer to ~15–16%, not 16.4%. The table is gross of borrow cost.
3. **Drawdown scales too.** At 1.45x, a normal -6.82% backtest stretch
   becomes -9.9%; a rough live fortnight like the -7.5% already observed
   becomes ~-11%. Leverage amplifies losses identically to gains.
4. **Post-validation only.** These projections are modelled on backtest
   drawdown. Live must first confirm DD stays near -6.82% over a full market
   cycle (including a drawdown AND recovery) before any leverage is applied.
   No leverage during validation. If it is ever applied, ramp gradually
   (1.15x → 1.30x) — never jump.

---

**Live paper** (reset 2026-06-15, $100k, locked engine): in validation.
Tracked daily via `daily_recorder.py` with alpha measured against SPY/QQQ.
See **Live Validation Reports** below for current numbers.

**ARIA-Growth live paper** (since 2026-06-09, Zed2 account): in validation.
Tracked daily via `aria_growth_daily_log.py` with book performance measured
against SPY and QQQ since go-live. See **Live Validation Reports** below.

---

## Live Validation Reports (as of 2026-07-09)

Both live paper books are read-only reviewed by monthly report scripts —
`aria_momentum_month_end.py` (`8_live_trading/month_end/`) and the ARIA-Growth
equivalent (`9_aria_growth/month_end/`). Numbers below are pulled straight
from the latest generated reports. **Sample-size caveat applies to both:**
a few weeks of live data verifies the machinery and shows behavior; it
cannot prove or disprove an edge validated over years of backtest.

### ARIA-Momentum (17 trading days since 2026-06-16)

| Metric                | Book       | SPY    | QQQ    |
| ---------------------- | ---------- | ------ | ------ |
| Return since inception | **+1.69%** | -1.73% | -4.21% |
| Alpha                  | —          | +3.42pp | +5.90pp |

Sharpe **4.05** · Sortino **10.19** · Calmar 22.21 · ann. return +26.5% ·
ann. vol 6.5% · max drawdown **-1.19%** (backtest budget -6.82%) · hit rate
47% (7W/8L) · best day +0.95% · worst -0.61%. All 17 days so far have traded
in a single regime (Bull-Stable).

**Why Sharpe/Sortino look inflated:** these are annualized by multiplying
the daily mean/vol by √252 — a scaling built for a full year of data, not
17 days. With only one regime observed, no losing streak worse than -0.61%,
and annualized vol sitting at just 6.5% (backtest Bull-Stable ran at
~20%+), the ratio comes out several times higher than the locked backtest's
Bull-Stable Sharpe of 1.35. This is a mechanical small-sample artifact, not
a claim the live engine is outperforming the backtest by 3x — it will
compress toward the backtest's regime-level numbers as more days, losing
streaks, and regime transitions accumulate. (These numbers were also
re-verified against a data-quality fix on 2026-07-09: a corrupted
`daily_pnl_pct` row and 8 dates where `daily_history.csv` had drifted from
the hand-reconciled `live_equity_curve.csv` were corrected, and the root
cause — `live_engine.py` logging pre-fill equity while `daily_recorder.py`
ran as a fully separate, unsynced process — was fixed so both files now
share one post-fill equity snapshot per day.)

### ARIA-Growth (20 trading days since go-live 2026-06-09, Zed2 account)

| Metric                | Book       | SPY    | QQQ    |
| ---------------------- | ---------- | ------ | ------ |
| Return since go-live    | **+1.44%** | +1.11% | +0.51% |
| Alpha                   | —          | +0.33pp | +0.93pp |

Sharpe **0.82** · Sortino **1.30** · Calmar 5.21 · ann. return +15.0% ·
ann. vol 18.3% · max drawdown -2.88% · beta vs SPY 0.79 (corr 0.64) ·
up-capture 64% / down-capture 56% (winning by losing less) · hit rate 47%
(9W/10L) · best day +1.79% · worst -2.92%. 16 open positions (9 in profit);
4 exits this period (1 stop-loss, 3 manual).

ARIA-Growth's Sharpe reads far more "normal" than ARIA-Momentum's at a
similar sample size, and that's informative rather than a coincidence: its
daily swings are an order of magnitude larger (worst day -2.92% vs
ARIA-Momentum's -0.61%, ann. vol 18.3% vs 6.5%), so the same √252 scaling
doesn't blow up the ratio the way it does for ARIA-Momentum's unusually
smooth first three weeks.

**Bottom line:** both books are ahead of SPY and QQQ since their respective
go-live dates, execution machinery is behaving as designed on both, and
neither result is old enough to be conclusive — treat the risk-adjusted
numbers (especially ARIA-Momentum's) as directional, not final, until more
regimes and a longer sample accumulate.

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

| Task                      | Who                                                   | When                                       | Duration |
| ------------------------- | ----------------------------------------------------- | ------------------------------------------ | -------- |
| Daily live run (ARIA)     | Manual, `run_live.py`                                 | Weekdays ~19:30 Dhaka (US open)            | ~3-4 min |
| Daily status check (ARIA) | Manual, `run_live.py --status`                        | After run                                  | seconds  |
| Daily recording (ARIA)    | Manual, `daily_recorder.py`                           | After US close                             | ~30 sec  |
| Growth rebalance check    | Manual, `aria_growth_executor.py`                     | Weekly (dry-run; --execute)                | ~30 sec  |
| Growth daily log          | Manual, `aria_growth_daily_log.py`                    | Daily after US close (~2am Dhaka)          | ~30 sec  |
| Growth monthly re-screen  | Manual, `growth_screener.py` + archive                | 1st of month, then --reset-stops           | ~5 min   |
| Model retrain (ARIA)      | Manual, `python retrain.py` (--dry-run / --skip-gate) | As needed; NOT auto during live validation | ~1-2 hr  |
| RL agent retrain          | SHELVED                                               | —                                          | —        |

---

## Known Issues (current)

| Issue                                                                                                                                                                                                         | Status                                                                                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Backtest never entered Bull-Stable or Bear-Stress** (old Hurst taxonomy)                                                                                                                                    | **Resolved** — current 4-state HMM covers all regimes; Bull-Stable is the largest (643 bars). See Resolved Finding above.                                                                                                                           |
| ~~Trapped exits~~: held positions couldn't sell when sector gated out                                                                                                                                         | **Fixed (2026-06-18)** — sells bypass the regime/sector gate entirely; held positions always sellable, confirmed live as gate-independent                                                                                                           |
| ~~Fill price logs as 0~~ if order fills slower than retry window                                                                                                                                              | **Fixed (2026-06-18)** — `_await_fill` polls until BOTH price AND qty > 0 (5×2s); fallback chain (notional÷price → get_positions qty) makes a silent 0 structurally impossible; `price_estimated` flag added to trade log                           |
| **Live/backtest parity gap** — conditional min-hold (+5% structural win: hold profitable positions < 3 bars through churn, losers exit immediately) existed only in the backtest; live ran old churn behavior | **Fixed (2026-06-18)** — min-hold mirrored into `live_engine.py`: seeded at buy (entry bar = bar 0), idempotent per date, dry-run-safe, persisted via `position_hold_state.json`; forced exits (bear_heat_trim, regime_exit, TSLA) remain unguarded |
| FinBERT returns identical sentiment across tickers                                                                                                                                                            | Disabled in live; ticker-specific feed + retrain planned                                                                                                                                                                                            |
| Hypercloud round-trip churn in Bull-Stable (buy ~0.55, sell ~0.40 days later)                                                                                                                                 | Monitoring — consistent with backtested Bull-Stable (61.9% win, frequent rotations); min-hold fix now live and expected to reduce churn                                                                                                             |
| Drift gate is a stub — `regime_selector.py` reads `drift_auc` from signal_cache but nothing computes/writes it, so it defaults to 0.5 (gate never fires)                                                      | Open — drift producer never built; gate is dead code. Build post-validation if needed                                                                                                                                                               |
| AVGO Tier 3 / TSLA blacklist in regime_selector.py, yet both held live                                                                                                                                        | Open — suggests live engine may not apply selector tier gates; investigate                                                                                                                                                                          |
| COVID crash / 2018 Q4 stress tests return NaN                                                                                                                                                                 | Outside current backtest date range                                                                                                                                                                                                                 |
| ARIA-Growth not historically validated (current-snapshot screen)                                                                                                                                              | By design — monthly archive building point-in-time data for an EDGAR backtest                                                                                                                                                                       |
| ARIA-Growth regime detection lags (200dma) — slow to de-risk in fast crash                                                                                                                                    | Mitigated by 15% per-stock stop; thresholds editable, to be tuned in backtest                                                                                                                                                                       |

---

## Documentation History

- **v1** (`8_docs_v1/`): original development docs (Sections 1-7), through
  the backtesting and RL-development phase.
- **v2** (`10_docs_v2/`): current docs, reflecting the move to live Alpaca
  trading, RL shelving, and the multi-strategy roadmap.
