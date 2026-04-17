import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os


class EchoStateNetwork:
    """
    Reservoir Computing / Echo State Network for temporal pattern detection.
    Only the output layer is trained (ridge regression) — reservoir is fixed random.
    Trains in seconds on CPU. Great for financial time series.
    """

    def __init__(
        self,
        reservoir_size: int = 200,
        spectral_radius: float = 0.9,
        sparsity: float = 0.1,
        input_scaling: float = 0.5,
        leak_rate: float = 0.3,
        random_state: int = 42,
    ):
        self.reservoir_size  = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity        = sparsity
        self.input_scaling   = input_scaling
        self.leak_rate       = leak_rate
        self.random_state    = random_state

        self.W_in    = None   # input weights
        self.W_res   = None   # reservoir weights
        self.scaler  = StandardScaler()
        self.readout = RidgeClassifier(alpha=1.0)
        self.is_fitted = False

        self._init_reservoir()

    def _init_reservoir(self):
        rng = np.random.RandomState(self.random_state)

        # Input weights
        self.W_in = rng.uniform(
            -self.input_scaling,
             self.input_scaling,
            (self.reservoir_size, 1)
        )

        # Sparse random reservoir
        W = rng.randn(self.reservoir_size, self.reservoir_size)
        mask = rng.rand(*W.shape) > self.sparsity
        W[mask] = 0

        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(W)
        sr = np.max(np.abs(eigenvalues))
        if sr > 0:
            W = W * (self.spectral_radius / sr)
        self.W_res = W

    def _run_reservoir(self, X: np.ndarray) -> np.ndarray:
        """
        Drive reservoir with input sequence X.
        Returns reservoir states.
        """
        n_steps  = len(X)
        states   = np.zeros((n_steps, self.reservoir_size))
        state    = np.zeros(self.reservoir_size)

        for t in range(n_steps):
            u = np.array([[X[t]]])
            pre_activation = (
                self.W_res @ state +
                (self.W_in @ u).flatten()
            )
            state = (
                (1 - self.leak_rate) * state +
                self.leak_rate * np.tanh(pre_activation)
            )
            states[t] = state

        return states

    def fit(self, prices: np.ndarray, labels: np.ndarray):
        """
        Train on price series and binary labels (1=up, 0=down).
        """
        # Normalize prices
        prices_norm = self.scaler.fit_transform(
            prices.reshape(-1, 1)
        ).flatten()

        # Run reservoir
        states = self._run_reservoir(prices_norm)

        # Drop warmup period (first 50 steps)
        warmup   = 50
        states   = states[warmup:]
        labels   = labels[warmup:]

        # Train readout layer
        self.readout.fit(states, labels)
        self.is_fitted = True

        # Training accuracy
        preds = self.readout.predict(states)
        acc   = accuracy_score(labels, preds)
        return acc

    def predict(self, prices: np.ndarray) -> dict:
        """
        Predict signal from price series.
        Returns dict with signal, confidence, and reservoir state features.
        """
        if not self.is_fitted:
            raise RuntimeError("ESN not fitted yet. Call fit() first.")

        prices_norm = self.scaler.transform(
            prices.reshape(-1, 1)
        ).flatten()

        states = self._run_reservoir(prices_norm)
        last_state = states[-1].reshape(1, -1)

        signal     = self.readout.predict(last_state)[0]
        # Confidence via decision function distance from boundary
        decision   = self.readout.decision_function(last_state)[0]
        confidence = float(np.tanh(abs(decision)))  # 0 to 1

        return {
            "signal":     int(signal),        # 1=BUY 0=SELL
            "confidence": round(confidence, 4),
            "decision":   round(float(decision), 4),
            "label":      "BUY" if signal == 1 else "SELL",
        }

    def save(self, path: str):
        joblib.dump({
            "W_in":          self.W_in,
            "W_res":         self.W_res,
            "scaler":        self.scaler,
            "readout":       self.readout,
            "reservoir_size": self.reservoir_size,
            "leak_rate":     self.leak_rate,
            "input_scaling": self.input_scaling,
            "spectral_radius": self.spectral_radius,
            "sparsity":      self.sparsity,
            "is_fitted":     self.is_fitted,
        }, path)
        print(f"   ✅ ESN saved to {path}")

    def load(self, path: str):
        data = joblib.load(path)
        self.W_in            = data["W_in"]
        self.W_res           = data["W_res"]
        self.scaler          = data["scaler"]
        self.readout         = data["readout"]
        self.reservoir_size  = data["reservoir_size"]
        self.leak_rate       = data["leak_rate"]
        self.input_scaling   = data["input_scaling"]
        self.spectral_radius = data["spectral_radius"]
        self.sparsity        = data["sparsity"]
        self.is_fitted       = data["is_fitted"]
        print(f"   ✅ ESN loaded from {path}")


def train_esn(
    prices: pd.Series,
    labels: pd.Series,
    save_path: str = None,
) -> tuple:
    """
    Train ESN on price series and labels.
    Returns (esn, accuracy).
    """
    esn = EchoStateNetwork(
        reservoir_size  = 200,
        spectral_radius = 0.95,
        sparsity        = 0.1,
        input_scaling   = 0.5,
        leak_rate       = 0.3,
    )

    print(f"   → Training ESN on {len(prices)} price points...")
    acc = esn.fit(prices.values, labels.values)
    print(f"   ✅ ESN trained — accuracy: {acc:.3f}")

    if save_path:
        esn.save(save_path)

    return esn, acc


def get_esn_signal(
    prices: pd.Series,
    labels: pd.Series = None,
    model_path: str = None,
) -> dict:
    """
    Get ESN temporal signal. Trains if no model exists.
    """
    esn = EchoStateNetwork()

    if model_path and os.path.exists(model_path):
        esn.load(model_path)
    elif labels is not None:
        esn, acc = train_esn(prices, labels, save_path=model_path)
    else:
        raise ValueError("Either model_path or labels required.")

    result = esn.predict(prices.values)
    return result


if __name__ == "__main__":
    import yfinance as yf
    from datetime import date, timedelta

    print("=" * 55)
    print("  ECHO STATE NETWORK (RC) TEST")
    print("=" * 55)

    # Fetch data
    print("\n📥 Fetching AAPL prices...")
    df = yf.download("AAPL", period="2y", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    prices = df["Close"].dropna()

    # Labels: 1 if price goes up in next 5 days
    future_return = prices.pct_change(5).shift(-5)
    labels        = (future_return > 0).astype(int).dropna()
    prices_clean  = prices.loc[labels.index]

    print(f"✅ {len(prices_clean)} samples")
    print(f"   Label distribution: {labels.value_counts().to_dict()}")

    # Train
    print(f"\n🧠 Training Echo State Network...")
    esn, acc = train_esn(
        prices_clean,
        labels,
        save_path="4_signals/esn_model.pkl",
    )

    # Predict on latest prices
    print(f"\n🔮 Predicting on latest data...")
    result = esn.predict(prices_clean.values)

    print(f"\n📊 ESN Signal:")
    print(f"   Signal     : {result['label']}")
    print(f"   Confidence : {result['confidence']:.4f}")
    print(f"   Decision   : {result['decision']:.4f}")

    # Show last 5 predictions vs actual
    print(f"\n🔍 Last 10 predictions vs actual:")
    prices_arr = prices_clean.values
    actual      = labels.values
    correct     = 0
    for i in range(-10, 0):
        snippet = prices_arr[:len(prices_arr)+i] if i < 0 else prices_arr
        pred    = esn.predict(snippet)
        actual_label = "BUY" if actual[i] == 1 else "SELL"
        match   = "✅" if pred["label"] == actual_label else "❌"
        correct += (pred["label"] == actual_label)
        print(f"   [{i:3d}] Pred: {pred['label']:<4} | Actual: {actual_label:<4} | Conf: {pred['confidence']:.3f} {match}")

    print(f"\n   Last 10 accuracy: {correct}/10")
    print(f"\n✅ Echo State Network ready")