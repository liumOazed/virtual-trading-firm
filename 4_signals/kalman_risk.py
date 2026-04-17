import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter


def build_kalman_filter(observation_noise: float = 1.0,
                         process_noise: float = 0.1) -> KalmanFilter:
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1, 1],
                     [0, 1]])        # state transition: position + velocity
    kf.H = np.array([[1, 0]])        # observation: we only see price
    kf.R = np.array([[observation_noise]])   # observation noise
    kf.Q = np.array([[process_noise, 0],
                     [0, process_noise]])    # process noise
    kf.P = np.eye(2) * 100           # initial uncertainty
    return kf


def run_kalman(prices: pd.Series,
               observation_noise: float = 1.0,
               process_noise: float = 0.1) -> pd.DataFrame:
    kf = build_kalman_filter(observation_noise, process_noise)

    # Initialize with first price
    kf.x = np.array([[prices.iloc[0]], [0.0]])

    smoothed     = []
    upper_bands  = []
    lower_bands  = []
    innovations  = []   # residual: actual - predicted

    for price in prices:
        kf.predict()
        predicted = kf.x[0, 0]
        kf.update(np.array([[price]]))

        smoothed_price = kf.x[0, 0]
        uncertainty    = np.sqrt(kf.P[0, 0])

        smoothed.append(smoothed_price)
        upper_bands.append(smoothed_price + 2 * uncertainty)
        lower_bands.append(smoothed_price - 2 * uncertainty)
        innovations.append(price - predicted)

    return pd.DataFrame({
        "price":       prices.values,
        "smoothed":    smoothed,
        "upper_band":  upper_bands,
        "lower_band":  lower_bands,
        "innovation":  innovations,
        "uncertainty": [np.sqrt(kf.P[0, 0])] * len(prices),
    }, index=prices.index)


def compute_risk_signals(kf_df: pd.DataFrame) -> pd.DataFrame:
    df = kf_df.copy()

    # Distance from Kalman smoothed price (normalized)
    df["kalman_deviation"] = (
        (df["price"] - df["smoothed"]) /
        (df["upper_band"] - df["smoothed"]).clip(lower=1e-6)
    )

    # Is price outside Kalman bands? → anomaly
    df["above_upper"]  = (df["price"] > df["upper_band"]).astype(int)
    df["below_lower"]  = (df["price"] < df["lower_band"]).astype(int)
    df["in_band"]      = (
        (df["price"] <= df["upper_band"]) &
        (df["price"] >= df["lower_band"])
    ).astype(int)

    # Innovation z-score (how surprising was the latest price move)
    inn_std = df["innovation"].rolling(20).std().clip(lower=1e-6)
    df["innovation_zscore"] = df["innovation"] / inn_std

    # Dynamic stop loss: smoothed price - 2 * uncertainty
    df["dynamic_stop_loss"] = df["lower_band"]

    # Risk level
    def risk_level(row):
        if abs(row["innovation_zscore"]) > 2.5:
            return "HIGH"
        elif abs(row["innovation_zscore"]) > 1.5:
            return "MEDIUM"
        else:
            return "LOW"

    df["risk_level"] = df.apply(risk_level, axis=1)

    return df


def get_kalman_features(prices: pd.Series) -> dict:
    """
    Returns latest Kalman features as a dict for use in feature_builder.
    """
    kf_df = run_kalman(prices)
    kf_df = compute_risk_signals(kf_df)
    last  = kf_df.iloc[-1]

    return {
        "kalman_smoothed":      last["smoothed"],
        "kalman_deviation":     last["kalman_deviation"],
        "kalman_upper":         last["upper_band"],
        "kalman_lower":         last["lower_band"],
        "kalman_innovation_z":  last["innovation_zscore"],
        "kalman_risk_level":    last["risk_level"],
        "dynamic_stop_loss":    last["dynamic_stop_loss"],
        "above_kalman_upper":   int(last["above_upper"]),
        "below_kalman_lower":   int(last["below_lower"]),
    }


def evaluate_trade_risk(
    entry_price: float,
    current_price: float,
    prices: pd.Series,
) -> dict:
    """
    Given an entry price and current price,
    evaluate whether to exit based on Kalman dynamic stop loss.
    """
    features    = get_kalman_features(prices)
    stop_loss   = features["dynamic_stop_loss"]
    risk_level  = features["kalman_risk_level"]
    deviation   = features["kalman_deviation"]

    pnl_pct     = (current_price - entry_price) / entry_price * 100
    should_exit = current_price < stop_loss

    return {
        "entry_price":    entry_price,
        "current_price":  current_price,
        "stop_loss":      round(stop_loss, 2),
        "pnl_pct":        round(pnl_pct, 2),
        "should_exit":    should_exit,
        "risk_level":     risk_level,
        "deviation":      round(deviation, 4),
        "verdict":        "EXIT ⚠️" if should_exit else "HOLD ✅",
    }


if __name__ == "__main__":
    import yfinance as yf
    from datetime import date

    TODAY = date.today().strftime("%Y-%m-%d")

    print("=" * 55)
    print("  KALMAN RISK FILTER TEST")
    print("=" * 55)

    ticker = "AAPL"
    print(f"\n📥 Fetching {ticker} prices...")
    df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    prices = df["Close"].dropna()

    print(f"✅ {len(prices)} price points loaded")

    # Run Kalman
    kf_df = run_kalman(prices)
    kf_df = compute_risk_signals(kf_df)

    last = kf_df.iloc[-1]
    print(f"\n📊 Latest Kalman Analysis ({TODAY}):")
    print(f"   Price          : ${last['price']:.2f}")
    print(f"   Kalman Smooth  : ${last['smoothed']:.2f}")
    print(f"   Upper Band     : ${last['upper_band']:.2f}")
    print(f"   Lower Band     : ${last['lower_band']:.2f}")
    print(f"   Dynamic Stop   : ${last['dynamic_stop_loss']:.2f}")
    print(f"   Innovation Z   : {last['innovation_zscore']:.3f}")
    print(f"   Risk Level     : {last['risk_level']}")
    print(f"   In Band        : {'✅ Yes' if last['in_band'] else '⚠️ No'}")

    # Test risk evaluation
    entry = float(prices.iloc[-20])
    curr  = float(prices.iloc[-1])
    print(f"\n⚖️  Trade Risk Evaluation:")
    print(f"   Entry (20d ago): ${entry:.2f}")
    result = evaluate_trade_risk(entry, curr, prices)
    for k, v in result.items():
        print(f"   {k:<18}: {v}")

    # Kalman features for ML
    print(f"\n🔧 Kalman Features for XGBoost:")
    features = get_kalman_features(prices)
    for k, v in features.items():
        if k != "kalman_risk_level":
            print(f"   {k:<25}: {v:.4f}" if isinstance(v, float) else f"   {k:<25}: {v}")
        else:
            print(f"   {k:<25}: {v}")

    print("\n✅ Kalman risk filter ready")