"""
structural_break.py
===================
Structural break and concept drift detection for automatic retrain triggers
and model staleness alerts.

Detection methods:
  cusum_test              — Two-sided CUSUM test for mean-shift in a series
  page_hinkley_test       — Page-Hinkley online test for monotonic error drift
  bai_perron_test         — Sequential RSS-minimisation multiple breakpoint test
  online_changepoint_bocpd — Bayesian online changepoint probability (BOCPD)
  model_drift_test        — Concept drift via prediction accuracy degradation

Applications in check_all_signals:
  1. SPY return distribution    (mean shift via CUSUM, threshold=4.0)
  2. SPY volatility regime      (variance shift via CUSUM on |returns|, threshold=3.5)
  3. XGBoost concept drift      (model_drift_test on predictions vs actuals)
  4. HMM regime structure drift (posterior entropy; high entropy = diffuse/drifting)
  5. Realized correlation break (Page-Hinkley on encoded regime history from rcov)

Severity thresholds:
  1 signal fires  → severity='warning'
  2+ signals fire → severity='critical'

Critical break actions (handled by BacktestEngineV2):
  - Pause new entries for 5 bars (entry_pause_until = bar_idx + 5)
  - Force model retrain at next window boundary
  - Tighten position sizes by 1.5x for 20 bars (gate_tightening_factor = 1.5)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class StructuralBreakDetector:
    """
    Detects structural breaks across multiple market and model signals.
    All methods are stateless — they operate on numpy arrays passed at call time.
    check_all_signals is the primary entry point from BacktestEngineV2.
    """

    # ── individual tests ──────────────────────────────────────────────────────

    def cusum_test(self, returns_series: np.ndarray,
                   threshold: float = 5.0) -> Tuple[bool, int]:
        """
        Two-sided CUSUM test for a mean shift in returns_series.
        S+ = cumulative upward deviation from baseline mean; S- = downward.
        Returns (break_detected, last_reset_idx).
        threshold: multiples of series std above which S+/S- signals a break.
        """
        n = len(returns_series)
        if n < 10:
            return False, -1

        mu  = float(np.mean(returns_series))
        std = float(np.std(returns_series, ddof=1))
        if std < 1e-12:
            return False, -1

        k          = 0.5 * std   # allowance (half-sigma slack)
        h          = threshold * std
        s_pos      = 0.0
        s_neg      = 0.0
        last_reset = -1

        for i, xi in enumerate(returns_series):
            s_pos = max(0.0, s_pos + (xi - mu) - k)
            s_neg = max(0.0, s_neg - (xi - mu) - k)
            if s_pos == 0.0 or s_neg == 0.0:
                last_reset = i

        return bool(s_pos > h or s_neg > h), last_reset

    def page_hinkley_test(self, error_series: np.ndarray,
                          alpha: float = 0.01) -> Tuple[bool, int]:
        """
        Page-Hinkley test for monotonic drift in an error (residual) series.
        Detects when the cumulative deviation from the initial mean
        exceeds a threshold = 1 / alpha.
        Returns (drift_detected, last_trigger_idx).
        alpha: sensitivity — smaller alpha → stricter threshold.
        """
        n = len(error_series)
        if n < 10:
            return False, -1

        threshold    = 1.0 / max(alpha, 1e-9)
        mu0          = float(np.mean(error_series[:max(n // 4, 5)]))
        m_t          = 0.0
        m_min        = 0.0
        last_trigger = -1

        for i, xi in enumerate(error_series):
            m_t   += xi - mu0
            m_min  = min(m_min, m_t)
            ph_stat = m_t - m_min
            if ph_stat > threshold:
                last_trigger = i
                m_t   = 0.0  # reset after detection
                m_min = 0.0

        return last_trigger >= 0, last_trigger

    def bai_perron_test(self, series: np.ndarray,
                        max_breaks: int = 5) -> List[int]:
        """
        Greedy sequential Bai-Perron breakpoint search.
        At each iteration, finds the split point that minimises total RSS
        across left and right sub-segments. Stops when the RSS improvement
        falls below 5% of the un-split segment's RSS.
        Returns sorted list of breakpoint indices.
        """
        n = len(series)
        if n < 20:
            return []

        breakpoints   = []
        segment_start = 0

        for _ in range(max_breaks):
            seg = series[segment_start:]
            m   = len(seg)
            if m < 10:
                break

            best_bp  = -1
            best_rss = np.inf

            for split in range(5, m - 5):
                left  = seg[:split]
                right = seg[split:]
                rss   = (float(np.var(left))  * len(left) +
                         float(np.var(right)) * len(right))
                if rss < best_rss:
                    best_rss = rss
                    best_bp  = split

            if best_bp < 0:
                break

            null_rss = float(np.var(seg)) * m
            if null_rss > 0 and (null_rss - best_rss) / null_rss < 0.05:
                break

            breakpoints.append(segment_start + best_bp)
            segment_start += best_bp

        return sorted(breakpoints)

    def online_changepoint_bocpd(self, series: np.ndarray,
                                 hazard: float = 0.01) -> float:
        """
        Bayesian Online Changepoint Detection (Adams & MacKay 2007).
        Maintains a run-length distribution log P(rl_t | x_{1:t}).
        Returns P(changepoint at last observation) in [0, 1].
        Uses an empirical Gaussian predictive (trailing 30-bar window).
        Complexity: O(n²) — intended for series up to ~500 points.
        """
        n = len(series)
        if n < 5:
            return 0.0

        log_H  = np.log(max(hazard, 1e-12))
        log_1H = np.log(max(1.0 - hazard, 1e-12))
        log_R  = np.full(n + 1, -np.inf)
        log_R[0] = 0.0  # at t=0, run length 0 has full probability

        for t in range(n):
            x = float(series[t])

            # Empirical Gaussian predictive from recent history
            if t >= 2:
                seg   = series[max(0, t - 30):t]
                mu    = float(np.mean(seg))
                sigma = max(float(np.std(seg, ddof=0)), 1e-6)
            else:
                mu    = float(series[0])
                sigma = 1.0

            # Gaussian log-likelihood
            log_lk = (-0.5 * ((x - mu) / sigma) ** 2
                      - np.log(sigma) - 0.9189385332)  # 0.5 * log(2π)

            new_log_R = np.full(n + 1, -np.inf)

            # Changepoint: all run lengths collapse to rl=0
            active = log_R[:t + 1]
            valid  = active[active > -np.inf]
            if len(valid) > 0:
                lse = float(valid.max() +
                            np.log(np.sum(np.exp(valid - valid.max()))))
                new_log_R[0] = log_H + lse + log_lk

            # Growth: run length rl → rl+1 for each existing run length
            new_log_R[1:t + 2] = np.logaddexp(
                new_log_R[1:t + 2],
                log_1H + log_R[:t + 1] + log_lk,
            )

            # Normalize the active prefix
            seg_lr = new_log_R[:t + 2]
            valid  = seg_lr[seg_lr > -np.inf]
            if len(valid) > 0:
                log_Z = float(valid.max() +
                              np.log(np.sum(np.exp(valid - valid.max()))))
                new_log_R[:t + 2] -= log_Z

            log_R = new_log_R

        cp0 = log_R[0]
        return float(np.clip(np.exp(cp0) if cp0 > -np.inf else 0.0, 0.0, 1.0))

    def model_drift_test(self, predictions: np.ndarray,
                          actuals: np.ndarray,
                          window: int = 60) -> Tuple[bool, float]:
        """
        Concept drift detection via direction-accuracy degradation.
        Compares accuracy of predictions vs actuals in the first half of the
        window against the second half.
        Returns (drift_detected, drift_score).
        drift_score > 0 means accuracy fell (positive = worse).
        Detection threshold: 15 percentage-point drop in accuracy.
        """
        if len(predictions) < 20 or len(actuals) < 20:
            return False, 0.0

        n     = min(len(predictions), len(actuals), window)
        preds = np.array(predictions[-n:], dtype=float)
        acts  = np.array(actuals[-n:],    dtype=float)

        pred_dir = (preds >= 0.5).astype(int)
        act_dir  = (acts  >  0.0).astype(int)

        half      = n // 2
        acc_early = float(np.mean(pred_dir[:half] == act_dir[:half]))
        acc_late  = float(np.mean(pred_dir[half:] == act_dir[half:]))

        drift_score    = acc_early - acc_late  # positive = accuracy fell
        drift_detected = drift_score > 0.15    # >15pp degradation threshold

        return drift_detected, round(drift_score, 4)

    # ── primary entry point ───────────────────────────────────────────────────

    def check_all_signals(
        self,
        spy_returns:   np.ndarray,
        model_preds:   np.ndarray,
        model_actuals: np.ndarray,
        rcov,
        hmm_posterior: Dict[str, float],
    ) -> Tuple[bool, Dict]:
        """
        Run all five structural break tests and aggregate.
        Returns (break_detected, break_info).
        break_info keys: type, severity, score, signals.
        Returns (False, {}) when no signals fire.

        rcov must expose a _sb_corr_history attribute (deque of floats
        0/1/2 encoding normal/elevated/crisis) populated by BacktestEngineV2.
        """
        signals_fired: List[str]        = []
        scores:        Dict[str, float] = {}

        # 1. SPY return distribution — mean shift
        if len(spy_returns) >= 20:
            mean_shift, _ = self.cusum_test(
                np.array(spy_returns, dtype=float), threshold=4.0)
            scores['spy_mean_shift'] = 1.0 if mean_shift else 0.0
            if mean_shift:
                signals_fired.append('spy_mean_shift')

        # 2. SPY volatility — variance shift on absolute returns
        if len(spy_returns) >= 20:
            abs_rets     = np.abs(np.array(spy_returns, dtype=float))
            vol_shift, _ = self.cusum_test(abs_rets, threshold=3.5)
            scores['spy_vol_shift'] = 1.0 if vol_shift else 0.0
            if vol_shift:
                signals_fired.append('spy_vol_shift')

        # 3. XGBoost concept drift (skipped if fewer than 20 observations)
        if len(model_preds) >= 20 and len(model_actuals) >= 20:
            drift_det, drift_sc = self.model_drift_test(
                np.array(model_preds,   dtype=float),
                np.array(model_actuals, dtype=float),
            )
            scores['model_drift'] = float(drift_sc)
            if drift_det:
                signals_fired.append('model_drift')

        # 4. HMM regime structure drift — normalized posterior entropy
        if hmm_posterior:
            probs       = np.array(list(hmm_posterior.values()), dtype=float)
            probs       = probs / max(float(probs.sum()), 1e-9)
            entropy     = float(-np.sum(probs * np.log(probs + 1e-9)))
            max_entropy = np.log(max(len(probs), 1))
            norm_ent    = float(entropy / max(float(max_entropy), 1e-9))
            scores['regime_structure_drift'] = norm_ent
            if norm_ent > 0.85:  # nearly-uniform posterior = HMM drifting
                signals_fired.append('regime_structure_drift')

        # 5. Realized correlation break — Page-Hinkley on encoded regime history
        if rcov is not None:
            try:
                corr_hist = getattr(rcov, '_sb_corr_history', None)
                if corr_hist is not None and len(corr_hist) >= 15:
                    corr_enc  = np.array(list(corr_hist), dtype=float)
                    ph_det, _ = self.page_hinkley_test(corr_enc, alpha=0.05)
                    scores['correlation_break'] = 1.0 if ph_det else 0.0
                    if ph_det:
                        signals_fired.append('correlation_break')
            except Exception:
                pass

        if not signals_fired:
            return False, {}

        n_sig      = len(signals_fired)
        severity   = 'critical' if n_sig >= 2 else 'warning'
        break_type = signals_fired[0] if n_sig == 1 else 'multi_signal'

        return True, {
            'type':     break_type,
            'severity': severity,
            'score':    round(sum(scores.values()), 4),
            'signals':  signals_fired,
        }


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    detector = StructuralBreakDetector()
    rng      = np.random.default_rng(42)

    # Stationary series
    s_stable = rng.normal(0.0, 0.01, 60)
    # Series with a mean shift at index 40
    s_break  = np.concatenate([
        rng.normal(0.0,  0.01, 40),
        rng.normal(0.05, 0.01, 20),
    ])

    print("CUSUM stationary  :", detector.cusum_test(s_stable))
    print("CUSUM break       :", detector.cusum_test(s_break, threshold=3.0))
    print("Page-Hinkley      :", detector.page_hinkley_test(s_break, alpha=0.02))
    print("Bai-Perron bkpts  :", detector.bai_perron_test(s_break))
    print("BOCPD P(cp_last)  :", round(detector.online_changepoint_bocpd(s_break), 4))

    preds   = rng.uniform(0.3, 0.7, 60)
    actuals = np.concatenate([
        rng.binomial(1, 0.6, 30),
        rng.binomial(1, 0.4, 30),
    ]).astype(float)
    print("Model drift test  :", detector.model_drift_test(preds, actuals))
