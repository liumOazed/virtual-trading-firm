import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


# ══════════════════════════════════════════════════════════════
# 1. RANDOM WALK INDEX (RWI)
# ══════════════════════════════════════════════════════════════

def random_walk_index(high: pd.Series, low: pd.Series,
                      close: pd.Series, window: int = 14) -> pd.DataFrame:
    """
    Random Walk Index — Michael Poulos (1992)
    
    Compares actual price range to expected range of a random walk.
    RWI_High > 1 → uptrend stronger than random
    RWI_Low  > 1 → downtrend stronger than random
    
    Formula:
        RWI_High = (High - Low[n]) / (ATR * sqrt(n))
        RWI_Low  = (High[n] - Low) / (ATR * sqrt(n))
    """
    atr = (high - low).rolling(window).mean()
    sqrt_n = np.sqrt(window)

    rwi_high_vals = []
    rwi_low_vals  = []

    for i in range(len(close)):
        if i < window:
            rwi_high_vals.append(np.nan)
            rwi_low_vals.append(np.nan)
            continue

        window_atr  = atr.iloc[i]
        if window_atr == 0 or np.isnan(window_atr):
            rwi_high_vals.append(np.nan)
            rwi_low_vals.append(np.nan)
            continue

        # Highest high and lowest low over window
        period_high = high.iloc[i - window + 1: i + 1].max()
        period_low  = low.iloc[i - window + 1:  i + 1].min()

        rwi_h = (period_high - low.iloc[i]) / (window_atr * sqrt_n)
        rwi_l = (high.iloc[i] - period_low) / (window_atr * sqrt_n)

        rwi_high_vals.append(rwi_h)
        rwi_low_vals.append(rwi_l)

    result = pd.DataFrame({
        "rwi_high":    rwi_high_vals,
        "rwi_low":     rwi_low_vals,
    }, index=close.index)

    # Composite: max of both (overall trend strength)
    result["rwi_max"]   = result[["rwi_high", "rwi_low"]].max(axis=1)

    # Signal: is there a genuine trend?
    result["rwi_trend"]  = (result["rwi_max"] > 1.0).astype(int)

    # Direction: +1 uptrend, -1 downtrend, 0 choppy
    result["rwi_direction"] = np.where(
        result["rwi_high"] > result["rwi_low"], 1,
        np.where(result["rwi_low"] > result["rwi_high"], -1, 0)
    )

    return result


# ══════════════════════════════════════════════════════════════
# 2. ORNSTEIN-UHLENBECK MEAN REVERSION FEATURES
# ══════════════════════════════════════════════════════════════

def fit_ou_process(prices: np.ndarray) -> dict:
    """
    Fit Ornstein-Uhlenbeck process to price series.
    
    dX = theta * (mu - X) * dt + sigma * dW
    
    Returns:
        theta: mean reversion speed (higher = faster reversion)
        mu:    long-run equilibrium price
        sigma: volatility of the process
        half_life: days to revert halfway to mean
        residual:  current deviation from equilibrium (z-score)
    """
    if len(prices) < 20:
        return {"ou_theta": np.nan, "ou_mu": np.nan,
                "ou_sigma": np.nan, "ou_half_life": np.nan,
                "ou_residual": np.nan, "ou_reversion_signal": 0}

    # Use OLS on lagged values: X(t) = a + b*X(t-1) + e
    x_lag  = prices[:-1]
    x_curr = prices[1:]

    # Linear regression: x_t = alpha + beta * x_{t-1}
    slope, intercept, r_val, p_val, _ = stats.linregress(x_lag, x_curr)

    # OU parameters
    # beta = exp(-theta * dt), dt = 1 day
    beta  = slope
    if beta <= 0 or beta >= 1:
        # Not mean-reverting
        return {"ou_theta": 0.0, "ou_mu": float(np.mean(prices)),
                "ou_sigma": float(np.std(np.diff(prices))),
                "ou_half_life": np.inf,
                "ou_residual": 0.0,
                "ou_reversion_signal": 0}

    theta     = -np.log(beta)           # mean reversion speed
    mu        = intercept / (1 - beta)  # long-run mean
    half_life = np.log(2) / theta       # days to revert 50%

    # Residuals
    residuals = x_curr - (intercept + beta * x_lag)
    sigma     = float(np.std(residuals))

    # Current deviation from equilibrium (z-score)
    current_price = prices[-1]
    recent_std    = np.std(prices[-20:]) if len(prices) >= 20 else sigma
    residual_z    = (current_price - mu) / (recent_std + 1e-9)

    # Reversion signal: -1 = overbought (expect down), +1 = oversold (expect up)
    if residual_z > 1.5:
        rev_signal = -1   # price too high, expect reversion down
    elif residual_z < -1.5:
        rev_signal = 1    # price too low, expect reversion up
    else:
        rev_signal = 0    # within normal range

    return {
        "ou_theta":            round(float(theta), 6),
        "ou_mu":               round(float(mu), 4),
        "ou_sigma":            round(float(sigma), 6),
        "ou_half_life":        round(float(half_life), 2),
        "ou_residual":         round(float(residual_z), 4),
        "ou_reversion_signal": int(rev_signal),
    }


def rolling_ou_features(prices: pd.Series, window: int = 60) -> pd.DataFrame:
    """
    Compute rolling OU features over a sliding window.
    """
    results = []
    for i in range(len(prices)):
        if i < window:
            results.append({
                "ou_theta":            np.nan,
                "ou_mu":               np.nan,
                "ou_sigma":            np.nan,
                "ou_half_life":        np.nan,
                "ou_residual":         np.nan,
                "ou_reversion_signal": 0,
            })
        else:
            window_prices = prices.iloc[i - window: i + 1].values
            ou = fit_ou_process(window_prices)
            results.append(ou)

    return pd.DataFrame(results, index=prices.index)


# ══════════════════════════════════════════════════════════════
# 3. QUADRATIC VARIATION (VOLATILITY SIGNATURE)
# ══════════════════════════════════════════════════════════════

def quadratic_variation(prices: pd.Series,
                        windows: list = [1, 5, 10, 20]) -> pd.DataFrame:
    """
    Realized quadratic variation at multiple timescales.
    
    QV(n) = sum of squared returns over n-period intervals
    
    Volatility signature plot:
    - If QV increases with sampling frequency → microstructure noise
    - If QV is flat → true price signal found
    - Ratio QV_fast/QV_slow > 1 → noise dominated
    - Ratio QV_fast/QV_slow ≈ 1 → clean signal
    """
    log_returns = np.log(prices / prices.shift(1))
    result      = pd.DataFrame(index=prices.index)

    for w in windows:
        # Realized variance over window w
        rv = (log_returns ** 2).rolling(w * 5).sum()
        result[f"qv_{w}d"] = rv

    # Volatility signature ratio (noise detector)
    if len(windows) >= 2:
        fast = f"qv_{windows[0]}d"
        slow = f"qv_{windows[-1]}d"
        result["qv_signature_ratio"] = result[fast] / (result[slow] + 1e-9)

        # Noise flag: ratio >> 1 means high-frequency noise
        result["qv_noise_flag"] = (result["qv_signature_ratio"] > 2.0).astype(int)

    # Annualized realized volatility (clean measure)
    result["realized_vol_20d"] = (
        np.sqrt((log_returns ** 2).rolling(20).sum() * 252)
    )

    # Vol of vol (second-order uncertainty)
    result["vol_of_vol"] = result["realized_vol_20d"].rolling(20).std()

    return result


# ══════════════════════════════════════════════════════════════
# 4. DRIFT vs DIFFUSION RATIO
# ══════════════════════════════════════════════════════════════

def drift_diffusion_ratio(prices: pd.Series,
                          window: int = 20) -> pd.DataFrame:
    """
    Separates directional price movement (drift) from random noise (diffusion).
    
    For a price path:
        Total variation  = sum(|returns|)       ← drift + diffusion
        Quadratic var    = sum(returns^2)^0.5   ← diffusion only
        
    Drift component  = Total - Diffusion
    D/D Ratio        = |Drift| / Diffusion
    
    High ratio (>1)  → price has clear direction → trend trade
    Low ratio (<0.5) → price is mostly noise → avoid
    
    This is essentially the signal-to-noise ratio of price.
    """
    log_returns = np.log(prices / prices.shift(1))
    result      = pd.DataFrame(index=prices.index)

    # Total variation (sum of absolute returns)
    total_var = log_returns.abs().rolling(window).sum()

    # Quadratic variation (realized vol proxy)
    quad_var  = np.sqrt((log_returns ** 2).rolling(window).sum())

    # Drift component
    net_drift = log_returns.rolling(window).sum().abs()

    # D/D Ratio
    result["dd_ratio"]      = net_drift / (quad_var + 1e-9)

    # Normalized drift (signed direction of drift)
    result["drift_signed"]  = log_returns.rolling(window).sum()

    # Diffusion (noise level)
    result["diffusion"]     = quad_var

    # Signal quality score (0 to 1)
    # 1 = pure trend, 0 = pure noise
    result["signal_quality"] = np.clip(result["dd_ratio"], 0, 1)

    # Regime classification
    result["dd_regime"] = np.where(
        result["dd_ratio"] > 0.7, 1,    # trending
        np.where(result["dd_ratio"] < 0.3, -1, 0)  # noisy / neutral
    )

    return result


# ══════════════════════════════════════════════════════════════
# 5. OPTIMAL ENTRY (HJB APPROXIMATION)
# ══════════════════════════════════════════════════════════════

def hjb_optimal_entry(prices: pd.Series,
                      ou_features: pd.DataFrame,
                      dd_features: pd.DataFrame,
                      window: int = 20) -> pd.DataFrame:
    """
    Practical approximation of HJB optimal entry score.
    
    Full HJB requires solving a PDE — not practical for daily trading.
    This approximation captures the core insight:
    
    "Enter when: (1) mean reversion is strong, (2) you are far from
     equilibrium, (3) the signal-to-noise is high, (4) momentum
     supports the direction."
    
    HJB Entry Score = f(reversion_speed, deviation, signal_quality, momentum)
    
    Score > 0.7 → strong entry signal
    Score < 0.3 → avoid entry
    """
    result = pd.DataFrame(index=prices.index)

    log_returns = np.log(prices / prices.shift(1))
    momentum    = log_returns.rolling(window).sum()

    # Normalize OU theta to 0-1 (higher theta = faster reversion = better)
    theta_raw  = ou_features["ou_theta"].clip(0, 2)
    theta_norm = theta_raw / 2.0

    # Normalize residual deviation (abs value, further = more opportunity)
    resid_norm = ou_features["ou_residual"].abs().clip(0, 3) / 3.0

    # Signal quality from D/D ratio
    sig_qual   = dd_features["signal_quality"]

    # Momentum alignment with reversion signal
    # If OU says oversold (+1) and momentum is positive → aligned
    ou_signal   = ou_features["ou_reversion_signal"]
    mom_sign    = np.sign(momentum)
    alignment   = ((ou_signal == mom_sign) & (ou_signal != 0)).astype(float)

    # HJB Score: weighted combination
    result["hjb_entry_score"] = (
        0.30 * theta_norm +     # reversion speed
        0.25 * resid_norm +     # deviation from equilibrium
        0.25 * sig_qual   +     # signal quality
        0.20 * alignment        # momentum alignment
    ).clip(0, 1)

    # Entry signal: 1 if score high enough
    result["hjb_entry_signal"] = (result["hjb_entry_score"] > 0.60).astype(int)

    # Direction: follow OU reversion signal
    result["hjb_direction"] = ou_features["ou_reversion_signal"]

    return result


# ══════════════════════════════════════════════════════════════
# MASTER FUNCTION — adds all features to a dataframe
# ══════════════════════════════════════════════════════════════

def add_advanced_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all 5 advanced price features to an OHLCV dataframe.
    
    Required columns: open, high, low, close, volume
    All feature names prefixed to avoid conflicts.
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]

    print("      → Computing RWI...")
    rwi = random_walk_index(high, low, close, window=14)
    for col in rwi.columns:
        df[col] = rwi[col].values

    print("      → Computing OU mean reversion...")
    ou = rolling_ou_features(close, window=60)
    for col in ou.columns:
        df[col] = ou[col].values

    print("      → Computing Quadratic Variation...")
    qv = quadratic_variation(close, windows=[1, 5, 10, 20])
    for col in qv.columns:
        df[col] = qv[col].values

    print("      → Computing Drift/Diffusion ratio...")
    dd = drift_diffusion_ratio(close, window=20)
    for col in dd.columns:
        df[col] = dd[col].values

    print("      → Computing HJB optimal entry...")
    hjb = hjb_optimal_entry(close, ou, dd, window=20)
    for col in hjb.columns:
        df[col] = hjb[col].values

    return df


# ══════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import yfinance as yf
    from datetime import date

    TODAY = date.today().strftime("%Y-%m-%d")

    print("=" * 60)
    print("  ADVANCED PRICE FEATURES TEST")
    print("=" * 60)

    print("\n📥 Fetching AAPL...")
    raw = yf.download("AAPL", period="2y", progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    df = raw.dropna().copy()

    print(f"✅ {len(df)} rows loaded\n")

    df = add_advanced_price_features(df)

    last = df.iloc[-1]

    print(f"\n📊 Latest values ({TODAY}):")
    print(f"\n  — RWI —")
    print(f"   rwi_high      : {last.get('rwi_high', 'N/A'):.4f}")
    print(f"   rwi_low       : {last.get('rwi_low', 'N/A'):.4f}")
    print(f"   rwi_max       : {last.get('rwi_max', 'N/A'):.4f}")
    print(f"   rwi_trend     : {'✅ Trending' if last.get('rwi_trend') else '❌ Choppy'}")
    print(f"   rwi_direction : {'+1 UP' if last.get('rwi_direction') == 1 else '-1 DOWN' if last.get('rwi_direction') == -1 else '0 NEUTRAL'}")

    print(f"\n  — OU Mean Reversion —")
    print(f"   ou_theta      : {last.get('ou_theta', 'N/A'):.6f}  (speed)")
    print(f"   ou_mu         : ${last.get('ou_mu', 'N/A'):.2f}  (equilibrium)")
    print(f"   ou_half_life  : {last.get('ou_half_life', 'N/A'):.1f} days")
    print(f"   ou_residual   : {last.get('ou_residual', 'N/A'):.4f}  (z-score from mean)")
    rev = last.get('ou_reversion_signal', 0)
    print(f"   ou_reversion  : {'+1 OVERSOLD' if rev == 1 else '-1 OVERBOUGHT' if rev == -1 else '0 NEUTRAL'}")

    print(f"\n  — Quadratic Variation —")
    print(f"   realized_vol  : {last.get('realized_vol_20d', 'N/A'):.4f}  (annualized)")
    print(f"   vol_of_vol    : {last.get('vol_of_vol', 'N/A'):.4f}")
    print(f"   qv_sig_ratio  : {last.get('qv_signature_ratio', 'N/A'):.4f}")
    print(f"   noise_flag    : {'⚠️ Noisy' if last.get('qv_noise_flag') else '✅ Clean'}")

    print(f"\n  — Drift / Diffusion —")
    print(f"   dd_ratio      : {last.get('dd_ratio', 'N/A'):.4f}")
    print(f"   signal_quality: {last.get('signal_quality', 'N/A'):.4f}")
    reg = last.get('dd_regime', 0)
    print(f"   dd_regime     : {'📈 Trending' if reg == 1 else '📉 Noisy' if reg == -1 else '➡️ Neutral'}")

    print(f"\n  — HJB Optimal Entry —")
    print(f"   hjb_score     : {last.get('hjb_entry_score', 'N/A'):.4f}")
    print(f"   hjb_entry     : {'✅ ENTER' if last.get('hjb_entry_signal') else '⏳ WAIT'}")
    d = last.get('hjb_direction', 0)
    print(f"   hjb_direction : {'+1 LONG' if d == 1 else '-1 SHORT' if d == -1 else '0 NEUTRAL'}")

    print(f"\n✅ All advanced price features ready")
    print(f"   Total new features added: {len([c for c in df.columns if any(p in c for p in ['rwi','ou_','qv_','dd_','hjb_','realized','vol_of','signal_q','diffusion','drift'])])}")