# Daily run on GitHub Actions

`.github/workflows/daily-run.yml` runs, every US trading day, at **03:15 Asia/Dhaka**:

1. `python 9_aria_growth/aria_growth_daily_log.py`
2. `python run_live.py`

03:15 Dhaka is 21:15 UTC the day before, i.e. 16:15 / 17:15 in New York, right
after that day's close, so each run covers "today in New York". Weekends are
filtered by the cron, NYSE holidays by `scripts/market_day.py`. Your PC no longer
needs to be on.

GitHub doesn't guarantee scheduled runs, so there are backup slots at 03:50,
04:40 and 05:40 Dhaka. A backup only runs if no live run has happened yet for
that New York date (tracked in `last_run.json` on the `bot-state` branch), so
the books never trade twice in one day.

## One-time setup

### 1. Repository secrets

Settings → Secrets and variables → Actions → **New repository secret**:

| Secret | Used by | Value |
|---|---|---|
| `ALPACA_API_KEY` | `run_live.py` | ARIA paper account key (from `8_live_trading/.env` / root `.env`) |
| `ALPACA_SECRET_KEY` | `run_live.py` | ARIA paper account secret |
| `GROWTH_ALPACA_API_KEY` | `aria_growth_daily_log.py` | Zed2 growth account key (from `9_aria_growth/.env`) |
| `GROWTH_ALPACA_SECRET_KEY` | `aria_growth_daily_log.py` | Zed2 growth account secret |
| `STATE_KEY` | both | output of `python scripts/state_bundle.py keygen` (step 2) |
| `GROQ_API_KEY`, `NEWS_API_KEY` | optional | only if you later add `--explain` / news features |

### 2. Seed the state from your PC

Models, price caches, trade/equity logs and the growth CSVs are git-ignored, so
GitHub starts with none of them. Ship them once, encrypted:

```bash
pip install cryptography
python scripts/state_bundle.py keygen   # prints a key and saves it to .state_key (git-ignored)
                                        # -> paste the same key as the STATE_KEY secret
python scripts/state_bundle.py list     # check what will be uploaded (no .env files)
python scripts/state_bundle.py push     # encrypt + force-push to the bot-state branch
```

`push` also records your Python version and the exact versions of
pandas/xgboost/scikit-learn/etc. so the runner loads the pickled models with the
same libraries.

### 3. Test it

Actions → **Daily trading run** → **Run workflow** with *Dry run* ticked
(and *Run even if the NYSE was closed* ticked if it's a weekend/holiday).
Both scripts run with `--dry-run`; no orders are placed and nothing is saved
back. When that's green, the 03:15 schedule takes over.

## After it's live

* **Stop the local Task Scheduler job** (if any). Otherwise your PC and GitHub
  both trade and their logs diverge.
* To look at the latest logs on your PC: `python scripts/state_bundle.py pull`.
* If you retrain models or re-screen locally, `pull` first, do the work, then
  `push` so the bot picks it up.
* `bot-state` always holds a single commit (each run force-pushes), so the repo
  doesn't grow every day. Don't protect that branch.
* The repo is public, so **Actions logs are public**: anyone can read the
  printed equity, positions and orders (the state bundle itself is encrypted).
  Make the repo private if you'd rather keep those to yourself.
