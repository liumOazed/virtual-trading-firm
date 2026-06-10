"""
tail_risk_hedger.py
===================
VIX-triggered tail-risk hedging system.

Hedge triggers (any one fires hedge entry):
  1. VIX > 25 AND VIX 5-day momentum > +30%
  2. stocks_bonds_corr_30d >= 0.30 AND corr_60d >= 0.10 AND 5 recent bars all > 0.15
  3. credit_stress_score > +2.0 sigma
  4. HMM posterior P(Bear-Stress) > 0.40 OR total_bear > 0.65
  5. Avg cross-asset correlation > 0.70 AND sustained > 0.65 for 3 bars

Entry gates (all must pass before any trigger fires):
  - Bull gate:     P(Bull-Stable) + P(Bull-Trending) > 0.50  → suppress
  - Trending gate: P(Bull-Trending) > 0.40                    → suppress
  - Cooldown:      < hedge_cooldown_bars (20) since last entry → suppress
  - Annual cap:    >= max_hedges_per_year (20) this year       → suppress

Hedge sizing (% of portfolio):
  VIX > 40    → 1.5%
  VIX 30–40   → 1.0%
  VIX 25–30   → 0.8%
  VIX 20–25   → 0.5%
  VIX <= 20   → 0.5%

Hedge instruments (priority order):
  1. SH  (inverse SPY)
  2. PSQ (inverse QQQ)
  3. GLD (gold, flight-to-safety)
  4. TLT (long-duration bonds) — only if stocks-bonds corr < 0

Hedge exit conditions (minimum min_hold_bars = 10 enforced first):
  - VIX < 18 AND bars_held >= 15
  - P(Bull-Stable) + P(Bull-Trending) > 0.65 AND bars_held >= 15
  - Held for 30+ bars (forced time exit)

Caller MUST invoke on_hedge_executed(date) when a hedge BUY is filled.

Tunable class-level parameters (override on instance to change):
  ENTRY_COOLDOWN_BARS  = 20   hedge_cooldown_bars
  MIN_HOLD_BARS        = 10
  ANNUAL_HEDGE_CAP     = 20   max_hedges_per_year
  BULL_GATE_THRESHOLD  = 0.50 bull combined probability
  BULL_TRENDING_GATE   = 0.40 bull-trending individual probability
  CORR_FLIP_30D_MIN    = 0.30 30-day corr floor
  CORR_FLIP_60D_MIN    = 0.10 60-day corr floor (sustained)
  CORR_FLIP_5BAR_MIN   = 0.15 each of last 5 bars floor
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class TailRiskHedger:
    """
    Computes hedge entry/exit signals and selects hedge instruments.
    Depends on GaussianHMMRegimeDetector, CrossAssetSignalEngine, and
    RealizedCovarianceTracker — passed at construction time.
    All signal methods are causally correct (data up to and including date).
    """

    HEDGE_PRIORITY      = ['SH', 'PSQ', 'GLD', 'TLT']

    # ── tunable gate/trigger parameters ──────────────────────────────────────
    ENTRY_COOLDOWN_BARS = 20
    MIN_HOLD_BARS       = 10
    ANNUAL_HEDGE_CAP    = 20
    BULL_GATE_THRESHOLD  = 0.50
    BULL_TRENDING_GATE   = 0.40
    CORR_FLIP_30D_MIN   = 0.30
    CORR_FLIP_60D_MIN   = 0.10
    CORR_FLIP_5BAR_MIN  = 0.15

    def __init__(self, hmm, cross_asset, rcov):
        self.hmm         = hmm
        self.cross_asset = cross_asset
        self.rcov        = rcov

        # cooldown: date of last actual hedge entry (None = no prior entry)
        self.last_hedge_entry_date: Optional[str] = None
        self.hedge_cooldown_bars:   int           = self.ENTRY_COOLDOWN_BARS

        # annual cap
        self.hedge_entries_by_year: Dict[int, int] = {}
        self.max_hedges_per_year:   int            = self.ANNUAL_HEDGE_CAP

    # ── private helpers ───────────────────────────────────────────────────────

    def _get_vix(self, date: str) -> float:
        series = self.cross_asset._close_up_to('^VIX', date)
        if series is None or len(series) == 0:
            return 0.0
        ts    = pd.Timestamp(date)
        exact = series[series.index == ts]
        return float(exact.iloc[-1]) if len(exact) > 0 else float(series.iloc[-1])

    def _get_vix_5day_momentum(self, date: str) -> float:
        series = self.cross_asset._close_up_to('^VIX', date)
        if series is None or len(series) < 6:
            return 0.0
        return float(series.iloc[-1] / series.iloc[-6] - 1.0)

    def _get_bear_stress_prob(self, date: str) -> float:
        return float(self.hmm.get_posterior(date).get('Bear-Stress', 0.0))

    def _get_bull_prob(self, date: str) -> float:
        post = self.hmm.get_posterior(date)
        return float(post.get('Bull-Stable', 0.0) + post.get('Bull-Trending', 0.0))

    def _get_last_n_dates(self, date: str, n: int) -> List[str]:
        """Last n trading date strings up to and including date, oldest first."""
        series = self.cross_asset._close_up_to('^VIX', date)
        if series is None or len(series) < n:
            return []
        return [str(d.date()) for d in series.index[-n:]]

    def _bars_between(self, date1: Optional[str], date2: str) -> int:
        """Trading bars strictly after date1 and up to/including date2."""
        if date1 is None:
            return 9999
        series = self.cross_asset._close_up_to('^VIX', date2)
        if series is None:
            return 9999
        ts1 = pd.Timestamp(date1)
        return int((series.index > ts1).sum())

    def _check_corr_flip_trigger(self, date: str) -> bool:
        """Only fire on a SUSTAINED positive stocks-bonds correlation regime."""
        sbc_30d = self.cross_asset.get_stocks_bonds_corr(date, lookback=30)
        sbc_60d = self.cross_asset.get_stocks_bonds_corr(date, lookback=60)

        if sbc_30d < self.CORR_FLIP_30D_MIN:
            return False

        if sbc_60d < self.CORR_FLIP_60D_MIN:
            return False

        recent_dates = self._get_last_n_dates(date, n=5)
        if len(recent_dates) < 5:
            return False
        recent_sbc = [self.cross_asset.get_stocks_bonds_corr(d) for d in recent_dates]
        return all(s > self.CORR_FLIP_5BAR_MIN for s in recent_sbc)

    def _check_hmm_bear_trigger(self, date: str) -> bool:
        """Fire when Bear-Stress posterior dominates OR combined bear is very high."""
        if self.hmm is None:
            return False
        posterior     = self.hmm.get_posterior(date)
        p_bear_stress = posterior.get('Bear-Stress', 0.0)
        p_bear_stable = posterior.get('Bear-Stable', 0.0)
        p_bear_total  = p_bear_stress + p_bear_stable
        if p_bear_stress > 0.40:
            return True
        if p_bear_total > 0.65:
            return True
        return False

    def _check_crisis_corr_trigger(self, date: str) -> bool:
        """Fire only when avg cross-asset correlation is truly elevated AND sustained."""
        try:
            avg_corr = self.rcov.get_avg_correlation(date, lookback=30)
        except Exception:
            return False
        if avg_corr < 0.70:
            return False
        try:
            last_3 = self._get_last_n_dates(date, 3)
            recent_corrs = [self.rcov.get_avg_correlation(d, lookback=30) for d in last_3]
            if not all(c > 0.65 for c in recent_corrs):
                return False
        except Exception:
            return False
        return True

    # ── public interface ──────────────────────────────────────────────────────

    def should_hedge_enter(self, date: str) -> Tuple[bool, str]:
        """
        Evaluate all hedge entry gates and triggers for the given date.
        Returns (should_enter, trigger_or_gate_name).
        Caller must call on_hedge_executed(date) when a hedge BUY is filled.
        """
        # Gate 0: only allow hedges in actively crashing Bear-Stress, not Bottoming
        if self.hmm is not None:
            _posterior    = self.hmm.get_posterior(date)
            p_bear_stress = _posterior.get('Bear-Stress', 0.0)
            if hasattr(self, 'engine') and hasattr(self.engine, '_detect_bear_stress_substate'):
                if p_bear_stress > 0.40:
                    if self.engine._detect_bear_stress_substate(date) == 'Bear-Stress-Bottoming':
                        return False, 'bear_stress_bottoming_gate'
                else:
                    return False, 'not_bear_stress_active_gate'
            else:
                if p_bear_stress < 0.50:
                    return False, 'not_strong_bear_stress'

        # Gate 1: suppress in any meaningful bull market
        if self.hmm is not None:
            _posterior    = self.hmm.get_posterior(date)
            _p_bull_trend = _posterior.get('Bull-Trending', 0.0)
            _p_bull_stab  = _posterior.get('Bull-Stable',   0.0)
            _p_bull_total = _p_bull_trend + _p_bull_stab
            if _p_bull_total > self.BULL_GATE_THRESHOLD:
                return False, 'bull_regime_gate'
            if _p_bull_trend > self.BULL_TRENDING_GATE:
                return False, 'bull_trending_gate'
        elif self._get_bull_prob(date) > self.BULL_GATE_THRESHOLD:
            return False, 'bull_regime_gate'

        # Gate 2: cooldown between entries
        if self._bars_between(self.last_hedge_entry_date, date) < self.hedge_cooldown_bars:
            return False, 'cooldown_active'

        # Gate 3: annual cap
        year = pd.Timestamp(date).year
        if self.hedge_entries_by_year.get(year, 0) >= self.max_hedges_per_year:
            return False, 'annual_cap_reached'

        vix     = self._get_vix(date)
        vix_mom = self._get_vix_5day_momentum(date)
        css     = self.cross_asset.get_credit_stress_score(date)

        # Trigger 1: VIX spike with momentum
        if vix > 25.0 and vix_mom > 0.30:
            return True, 'vix_spike'

        # Trigger 2: sustained positive stocks-bonds correlation
        if self._check_corr_flip_trigger(date):
            return True, 'corr_flip'

        # Trigger 3: elevated credit stress
        if css > 2.0:
            return True, 'credit_stress'

        # Trigger 4: HMM bear-stress posterior (p_bear_stress>0.40 OR total_bear>0.65)
        if self._check_hmm_bear_trigger(date):
            return True, 'hmm_bear'

        # Trigger 5: cross-asset correlation crisis (avg_corr>0.70, sustained 3 bars)
        if self._check_crisis_corr_trigger(date):
            return True, 'crisis_corr'

        return False, ''

    def on_hedge_executed(self, date: str) -> None:
        """
        Call after a hedge BUY is actually filled.
        Resets cooldown and increments annual entry count.
        """
        year = pd.Timestamp(date).year
        self.hedge_entries_by_year[year] = self.hedge_entries_by_year.get(year, 0) + 1
        self.last_hedge_entry_date       = date

    def get_hedge_size(self, date: str) -> float:
        """
        Hedge allocation as a fraction of portfolio equity (0.005–0.015).
        Scaled by current VIX level.
        """
        vix = self._get_vix(date)
        if vix > 40.0:
            return 0.015
        if vix > 30.0:
            return 0.010
        if vix > 25.0:
            return 0.008
        if vix > 20.0:
            return 0.005
        return 0.005

    def get_hedge_instrument(self, date: str) -> str:
        """
        Highest-priority available instrument from HEDGE_PRIORITY.
        TLT excluded when stocks-bonds correlation >= 0 (no flight-to-quality).
        Caller must verify instrument price availability before executing.
        """
        sbc = self.cross_asset.get_stocks_bonds_corr(date)
        for ticker in self.HEDGE_PRIORITY:
            if ticker == 'TLT' and sbc >= 0:
                continue
            return ticker
        return 'GLD'

    def should_hedge_exit(self, date: str, hedge_position: Dict) -> Tuple[bool, str]:
        """
        Evaluate exit conditions for an active hedge.
        Must be called once per bar per hedge.
        Returns (should_exit, reason).
        """
        entry_date = hedge_position.get('entry_date', date)
        bars_held  = self._bars_between(entry_date, date)

        # Minimum hold — no exit before MIN_HOLD_BARS regardless of signals
        if bars_held < self.MIN_HOLD_BARS:
            return False, 'min_hold_active'

        # Maximum hold — forced time exit
        if bars_held >= 30:
            return True, 'time_exit'

        # VIX below 18 after minimum extended hold
        vix = self._get_vix(date)
        if vix < 18.0 and bars_held >= 15:
            return True, 'vix_normalized'

        # HMM signals bull recovery after extended hold
        if self._get_bull_prob(date) > 0.65 and bars_held >= 15:
            return True, 'bull_recovery'

        return False, ''

    def get_crash_beta(self, ticker: str, price_data: Dict,
                        date: str, lookback: int = 252) -> float:
        """
        Beta of `ticker` vs SPY measured only on days when SPY falls > 1%.
        Negative crash beta = ticker rises when SPY crashes (ideal hedge).
        """
        spy_df = price_data.get('SPY')
        tkr_df = price_data.get(ticker)
        if spy_df is None or tkr_df is None:
            return 1.0

        ts        = pd.Timestamp(date)
        spy_close = spy_df['close'][spy_df.index <= ts].iloc[-lookback:]
        tkr_close = tkr_df['close'][tkr_df.index <= ts].iloc[-lookback:]
        spy_rets  = spy_close.pct_change().dropna()
        tkr_rets  = tkr_close.pct_change().dropna()
        common    = spy_rets.index.intersection(tkr_rets.index)
        if len(common) < 20:
            return 1.0

        spy_r      = spy_rets.loc[common].values
        tkr_r      = tkr_rets.loc[common].values
        crash_mask = spy_r < -0.01
        if crash_mask.sum() < 5:
            return 1.0

        spy_crash = spy_r[crash_mask]
        tkr_crash = tkr_r[crash_mask]
        var_spy   = float(np.var(spy_crash))
        if var_spy < 1e-12:
            return 1.0
        cov_val = float(np.cov(tkr_crash, spy_crash)[0, 1])
        return cov_val / var_spy


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("TailRiskHedger: run via BacktestEngineV2 — requires hmm, cross_asset, rcov.")
