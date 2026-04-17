import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import pytz

# Import your specialists
from finbert_sentiment import get_sentiment
from xgboost_model import add_kalman_features, add_esn_features, add_advanced_features, check_adversarial_drift

class SignalEngine:
    def __init__(self, model_path="4_signals/xgboost_global_model.pkl"):
        """Initialize the engine by loading the 'Global Brain'."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Global model not found at {model_path}. Train it first!")
            
        print(f"🧠 Signal Engine: Loading Global Brain from {model_path}...")
        self.data = joblib.load(model_path)        
        # Ensure the model file has the required keys
        required_keys = ["model", "scaler", "feature_cols"]
        for key in required_keys:
            if key not in self.data:
                raise KeyError(f"❌ Model file missing required key: '{key}'. Retrain the model!")
        
        self.model = self.data["model"]
        self.scaler = self.data["scaler"]
        self.feature_cols = self.data["feature_cols"]
        self.X_train_sample = self.data.get("X_train_sample") # For drift detection
        self.threshold = 0.55  # Default threshold for BUY signal
        
    
    def get_full_signals(self, df, ticker):
        """
        Generate signals for all rows in the dataframe.
        This method uses the same approach as get_state but for historical data.
        """
        import pandas_ta as ta
        from xgboost_model import add_kalman_features, add_esn_features, add_advanced_features, check_adversarial_drift
        
        # Load model data once
        data = joblib.load("4_signals/xgboost_global_model.pkl")
        model = data["model"]
        scaler = data["scaler"]
        feature_cols = data["feature_cols"]
        global_esn = data.get("global_esn")
        X_train_sample = data.get("X_train_sample")
        
        # Add features to the entire dataframe at once (more efficient)
        df_work = df.copy()
        
        # Add basic technical features (similar to feature_builder.py)
        close = df_work["close"]
        high = df_work["high"]
        low = df_work["low"]
        vol = df_work["volume"]
        
        # Trend
        df_work["sma_20"] = ta.sma(close, length=20)
        df_work["sma_50"] = ta.sma(close, length=50)
        df_work["ema_10"] = ta.ema(close, length=10)
        df_work["ema_20"] = ta.ema(close, length=20)
        
        # Momentum
        df_work["rsi_14"] = ta.rsi(close, length=14)
        df_work["rsi_7"] = ta.rsi(close, length=7)
        df_work["mom_10"] = ta.mom(close, length=10)
        
        # MACD
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        if macd is not None:
            df_work["macd"] = macd.iloc[:, 0]
            df_work["macd_signal"] = macd.iloc[:, 1]
            df_work["macd_hist"] = macd.iloc[:, 2]
        
        # Volatility
        df_work["atr_14"] = ta.atr(high, low, close, length=14)
        df_work["atr_7"] = ta.atr(high, low, close, length=7)
        
        # Bollinger Bands
        bb = ta.bbands(close, length=20, std=2)
        if bb is not None:
            df_work["bb_lower"] = bb.iloc[:, 0]
            df_work["bb_mid"] = bb.iloc[:, 1]
            df_work["bb_upper"] = bb.iloc[:, 2]
            df_work["bb_width"] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / bb.iloc[:, 1]
            df_work["bb_pct"] = (close - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0])
        
        # Volume
        df_work["vwma_20"] = ta.vwma(close, vol, length=20)
        df_work["vol_ratio"] = vol / vol.rolling(20).mean()
        
        # Price derived
        df_work["return_1d"] = close.pct_change(1)
        df_work["return_5d"] = close.pct_change(5)
        df_work["return_10d"] = close.pct_change(10)
        df_work["return_20d"] = close.pct_change(20)
        df_work["hl_ratio"] = (high - low) / close
        df_work["gap"] = (close - close.shift(1)) / close.shift(1)
        
        # Trend strength
        df_work["above_sma20"] = (close > df_work["sma_20"]).astype(int)
        df_work["above_sma50"] = (close > df_work["sma_50"]).astype(int)
        df_work["golden_cross"] = (df_work["sma_20"] > df_work["sma_50"]).astype(int)
        
        # Add placeholder columns needed by feature engineering
        if 'label' not in df_work.columns:
            df_work['label'] = 0
        if 'sentiment' not in df_work.columns:
            df_work['sentiment'] = 0.0
        
        # Add Kalman features
        df_work = add_kalman_features(df_work)
        
        # Add ESN features
        df_work = add_esn_features(df_work, global_esn)
        
        # Add advanced features
        df_work = add_advanced_features(df_work)
        
        # Drop rows with NaN in feature columns
        df_work = df_work.dropna()
        
        if df_work.empty:
            print(f"   ⚠️ {ticker}: No valid signals generated (empty after feature engineering)")
            return pd.DataFrame(columns=['date', 'signal', 'confidence', 'proba_buy'])
        
        # Fill missing feature columns with 0
        for col in feature_cols:
            if col not in df_work.columns:
                df_work[col] = 0.0
        
        X = df_work[feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = scaler.transform(X)
        
        # Get predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        results = []
        for idx in range(len(df_work)):
            try:
                row = df_work.iloc[idx]
                proba = probabilities[idx]
                proba_buy = float(proba[1])
                confidence = float(max(proba))
                signal_idx = int(predictions[idx])
                
                # Check drift
                drift_score = 0.5
                if X_train_sample is not None and len(X_train_sample) >= 6:
                    drift_score = check_adversarial_drift(X_train_sample[-200:], X_scaled[idx:idx+1])
                
                signal = "BUY" if (signal_idx == 1 and drift_score < 0.70) else "WAIT/SELL"
                
                results.append({
                    "date": row['date'],
                    "signal": signal,
                    "confidence": round(confidence, 4),
                    "proba_buy": round(proba_buy, 4)
                })
            except Exception as e:
                # Skip this row on error
                continue
        
        if not results:
            print(f"   ⚠️ {ticker}: No valid signals generated")
            return pd.DataFrame(columns=['date', 'signal', 'confidence', 'proba_buy'])
        
        return pd.DataFrame(results)

    def get_state(self, ticker: str, target_date: str) -> dict:
        """
        Inference Pipeline: 
        Raw Data -> Features -> Specialist Signals -> Scaling -> Prediction
        """
        try:
            # 1. Access Feature Builder (Section 3 legacy)
            import importlib.util
            spec = importlib.util.spec_from_file_location("fb", "4_signals/feature_builder.py")
            fb = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fb)

            # 2. Build Base Features (We fetch extra lookback to warm up Hurst/ESN)
            # 252 (Hurst) + 100 (ESN Z-Score) = ~350 days needed
            df = fb.build_features(
                ticker=ticker, 
                end_date=target_date, 
                lookback_days=400, 
                forward_days=5,
                sentiment_score=0.0 # Placeholder
            )

            if df is None or df.empty:
                return {"status": "error", "message": f"No data for {ticker}"}

            # 3. Add Specialist Layers (Variance-Reduced Versions)
            df = add_kalman_features(df)
            df = add_esn_features(df)      # This now includes our Step 5 Z-Scoring
            df = add_advanced_features(df) # This now includes our Step 2 Hurst-252
            
            # ----------------------------------
            # MODEL PREDICTION (ADD THIS)
            # ----------------------------------

            X = df[self.feature_cols].values

            # Predict probabilities
            proba = self.model.predict_proba(X)[:, 1]

            df['proba_buy'] = proba

            # Convert to signal (optional but useful)
            df['signal'] = np.where(proba > self.threshold, 'BUY', 'WAIT/SELL')
            df['confidence'] = proba

            # 4. Inject Live Sentiment (Step 1: Anti-Leak Logic)
            # Only use real sentiment if target_date is TODAY or YESTERDAY
            target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
            today_dt = date.today()
            
            if abs((today_dt - target_dt).days) <= 1:
                sent_res = get_sentiment(ticker, target_date, use_local=True)
                df.loc[df.index[-1], "sentiment"] = sent_res["score"]
            else:
                df.loc[df.index[-1], "sentiment"] = 0.0 # Historical baseline

            # 5. Scaling
            available_cols = [c for c in self.feature_cols if c in df.columns]
            X_raw = df[available_cols].iloc[[-1]].values
            X_scaled = self.scaler.transform(X_raw)

            # 6. Prediction & Confidence
            signal_idx = self.model.predict(X_scaled)[0]
            probs = self.model.predict_proba(X_scaled)[0]
            confidence = float(np.max(probs))

            # 7. Drift Detection (Step 2: Adversarial AUC)
            drift_auc = 0.5
            if self.X_train_sample is not None:
                # Compare latest scaled row against training memory
                drift_auc = check_adversarial_drift(self.X_train_sample[-200:], X_scaled)

            # 8. Create the "Signal Packet"
            # This packet is exactly what the RL Agent in Section 5 will consume as its "STATE"
            return {
                "status": "success",
                "ticker": ticker,
                "date": target_date,
                "signal": "BUY" if (signal_idx == 1 and drift_auc < 0.70) else "WAIT/SELL",
                "raw_signal": int(signal_idx),
                "confidence": confidence,
                "drift_auc": drift_auc,
                "regime_warning": drift_auc > 0.70,
                # Metadata for the RL Agent's state space
                "state_vector": {
                    "confidence": confidence,
                    "proba_buy": float(probs[1]),
                    "sentiment": float(df.iloc[-1]["sentiment"]),
                    
                    # Top SHAP drivers
                    "esn_signal": float(df.iloc[-1]["esn_signal"]),
                    "return_5d": float(df.iloc[-1]["return_5d"]),
                    "bb_width": float(df.iloc[-1]["bb_width"]),
                    "vol_regime": float(df.iloc[-1]["vol_regime"]),
                    
                    # Regime / structure
                    "hurst": float(df.iloc[-1]["hurst"]),
                    
                    # Trend & Momentum Indicators
                    "rwi": float(df.iloc[-1]["rwi_low"]),
                    "rsi": float(df.iloc[-1]["rsi_14"]),
                    
                    # Volatility & Risk Metrics
                    "realized_vol": float(df.iloc[-1]["realized_vol_20d"]), # Historical price variance
                    "atr": float(df.iloc[-1]["atr_14"]),                   # Absolute price movement range
                    "return_skew": float(df.iloc[-1]["return_skew_20"]),   # Tail risk / distribution asymmetry

                    
                    # Risk / mean reversion
                    "kalman_dist": float(df.iloc[-1]["kalman_deviation"]),
                    "ou_mu": float(df.iloc[-1]["ou_mu"]),
                    "ou_half_life": float(df.iloc[-1]["ou_half_life"])
                }
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Test Run
    engine = SignalEngine()
    test_date = date.today().strftime("%Y-%m-%d")
    print(f"\n🚀 Testing Signal Engine for NVDA on {test_date}...")
    
    packet = engine.get_state("NVDA", test_date)
    
    if packet["status"] == "success":
        print(f"   Signal: {packet['signal']} ({packet['confidence']:.2%})")
        print(f"   Drift AUC: {packet['drift_auc']:.3f}")
        print(f"   State Vector Summary: {list(packet['state_vector'].keys())}")
    else:
        print(f"   ❌ Error: {packet['message']}")