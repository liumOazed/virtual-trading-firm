import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import pytz

# Import your specialists
from finbert_sentiment import get_sentiment
from xgboost_model import (
    add_esn_features,
    add_advanced_features,
    SECTOR_META_FEATURES,
    META_FEATURE_NAMES,
)

class SignalEngine:
    def __init__(self, model_path="4_signals/xgboost_global_model.pkl"):
        """Initialize the engine by loading the 'Global Brain'."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Global model not found at {model_path}. Train it first!")
            
        print(f"🧠 Signal Engine: Loading Global Brain from {model_path}...")
        self.data      = joblib.load(model_path)
        self._pkl_path = model_path   # stored for sector detection at inference        
        # Ensure the model file has the required keys
        required_keys = ["scaler", "feature_cols"]
        for key in required_keys:
            if key not in self.data:
                raise KeyError(f"❌ Model file missing required key: '{key}'. Retrain the model!")
        
        # support both old pkl (model) and new regime-conditional pkl (global_model)
        self.model = self.data.get("model") or self.data.get("global_model")
        if self.model is None:
            raise KeyError("❌ Model file has neither 'model' nor 'global_model' key. Retrain.")
        self.scaler = self.data["scaler"]
        self.feature_cols = self.data["feature_cols"]
        self.threshold = self.data.get("optimal_threshold", 0.55)  # Default threshold for BUY signal
        
    
    def get_full_signals(self, df, ticker):
        """
        Generate signals for all rows in the dataframe.
        This method uses the same approach as get_state but for historical data.
        """
        import pandas_ta as ta
        from xgboost_model import (
            add_esn_features, add_advanced_features,
            _specialist_probas_batch,
        )
        esn_pca = self.data.get("esn_pca")
        
        # Use already-loaded model data from __init__ — no re-read needed
        scaler          = self.scaler
        feature_cols    = self.feature_cols
        global_esn      = self.data.get("global_esn")
        regime_models   = self.data.get("regime_models", {})
        global_model    = self.model
        global_threshold= self.threshold
        
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
        
        # v6: global_esn is a dict {ticker: ESN} — extract correct ticker's ESN
        if isinstance(global_esn, dict):
            ticker_esn = global_esn.get(ticker)
        else:
            ticker_esn = global_esn   # v5 fallback — single ESN object

        # Add ESN features; collect reservoir states when esn_pca is available
        if esn_pca is not None:
            df_work, _esn_states = add_esn_features(
                df_work, ticker_esn, collect_states=True)
            df_work['_esn_state'] = list(_esn_states)  # survives dropna
        else:
            df_work = add_esn_features(df_work, ticker_esn)

        # Add advanced features
        df_work = add_advanced_features(df_work)

        # Drop rows with NaN in feature columns
        df_work = df_work.dropna()

        # Recover ESN reservoir states and apply PCA (aligned after dropna)
        if esn_pca is not None and '_esn_state' in df_work.columns:
            _states_survived = np.vstack(df_work['_esn_state'].values)
            df_work = df_work.drop(columns=['_esn_state'])
            esn_latent = esn_pca.transform(_states_survived).astype(np.float32)
        else:
            esn_latent = None
        
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
        
        # Detect architecture and route accordingly
        architecture = self.data.get("architecture", "unknown")
        predictions  = np.zeros(len(X_scaled), dtype=int)

        if global_model is None:
            raise ValueError("No global model loaded")

        if architecture == "stacked_ensemble_v6":
            # v6: run batch specialist layer first, then meta-learner
            specialists = self.data.get("specialists", {})
            meta_scaler = self.data.get("meta_scaler")

            if not specialists or meta_scaler is None:
                raise ValueError("v6 pkl missing specialists or meta_scaler")

            # ── base 20 meta-features ──────────────────────────────
            meta_matrix = _specialist_probas_batch(
                X_raw             = X,
                specialists       = specialists,
                scaler            = scaler,
                feature_cols      = feature_cols,
                esn_latent_matrix = esn_latent,
            )

            # ── sector extra meta-features ─────────────────────────
            # Use saved sector_key; fall back to pkl path inference
            _sector_key = self.data.get("sector_key", "global")
            if _sector_key == "global":
                _pkl_path = getattr(self, '_pkl_path', '')
                for _sk in ["hardware", "hypercloud", "software",
                             "autos", "defensive"]:
                    if _sk in _pkl_path.lower():
                        _sector_key = _sk
                        break

            # prefer saved sector_meta_names; fall back to constant
            _sector_meta = (
                self.data.get("sector_meta_names")
                or SECTOR_META_FEATURES.get(_sector_key, META_FEATURE_NAMES)
            )
            _extra_names = _sector_meta[len(META_FEATURE_NAMES):]

            if _extra_names:
                try:
                    from cross_asset_signals import CrossAssetSignalEngine
                    from inflation_signals import InflationSignalEngine

                    # get date range from df_work
                    if "date" in df_work.columns:
                        _dates = pd.to_datetime(df_work["date"])
                    else:
                        _dates = pd.Series(df_work.index)

                    _start = _dates.min().strftime("%Y-%m-%d")
                    _end   = _dates.max().strftime("%Y-%m-%d")

                    _ca   = CrossAssetSignalEngine()
                    _infl = InflationSignalEngine()
                    _ca.load_data(_start, _end)
                    _infl.load_data(_start, _end)

                    _extra_cols = []
                    for _ename in _extra_names:
                        if _ename == "dollar_strength":
                            _arr = np.array([
                                _ca.get_dollar_strength_score(
                                    d.strftime('%Y-%m-%d'))
                                for d in _dates
                            ], dtype=np.float32)
                        elif _ename == "stocks_dollar_corr":
                            _arr = np.array([
                                _ca.get_stocks_dollar_corr(
                                    d.strftime('%Y-%m-%d'))
                                for d in _dates
                            ], dtype=np.float32)
                        elif _ename == "yield_curve_slope":
                            _arr = np.array([
                                _ca.get_yield_curve_slope(
                                    d.strftime('%Y-%m-%d'))
                                for d in _dates
                            ], dtype=np.float32)
                        elif _ename == "breakeven_inflation":
                            _arr = np.array([
                                _infl.get_breakeven_inflation(
                                    d.strftime('%Y-%m-%d'))
                                for d in _dates
                            ], dtype=np.float32)
                        elif _ename == "commodity_signal":
                            _arr = np.array([
                                _infl.get_commodity_inflation_signal(
                                    d.strftime('%Y-%m-%d'))
                                for d in _dates
                            ], dtype=np.float32)
                        elif _ename == "gold_ret":
                            _arr = np.array([
                                _ca._recent_return(
                                    'GLD', d.strftime('%Y-%m-%d'),
                                    lookback=30)
                                for d in _dates
                            ], dtype=np.float32)
                        else:
                            _arr = np.zeros(len(_dates), dtype=np.float32)

                        _extra_cols.append(_arr.reshape(-1, 1))

                    if _extra_cols:
                        _extra_block = np.hstack(_extra_cols)
                        meta_matrix  = np.hstack(
                            [meta_matrix, _extra_block]
                        ).astype(np.float32)
                        print(f"  ✅ {_sector_key} extra meta "
                              f"({len(_extra_names)}): "
                              f"{_extra_names}")

                except Exception as _e:
                    print(f"  ⚠️  Extra meta-features failed "
                          f"for {_sector_key}: {_e} — "
                          f"padding with zeros")
                    _pad = np.zeros(
                        (len(meta_matrix), len(_extra_names)),
                        dtype=np.float32
                    )
                    meta_matrix = np.hstack(
                        [meta_matrix, _pad]
                    ).astype(np.float32)

            # ── final scaler transform ─────────────────────────────
            _n_meta_expected = getattr(
                meta_scaler, "n_features_in_",
                meta_matrix.shape[1]
            )
            if meta_matrix.shape[1] > _n_meta_expected:
                meta_matrix = meta_matrix[:, :_n_meta_expected]
            elif meta_matrix.shape[1] < _n_meta_expected:
                # pad with zeros if still short
                _pad = np.zeros(
                    (len(meta_matrix),
                     _n_meta_expected - meta_matrix.shape[1]),
                    dtype=np.float32
                )
                meta_matrix = np.hstack(
                    [meta_matrix, _pad]
                ).astype(np.float32)

            meta_matrix_sc = meta_scaler.transform(meta_matrix)
            probabilities  = global_model.predict_proba(meta_matrix_sc)
            predictions    = (
                probabilities[:, 1] >= global_threshold
            ).astype(int)

        else:
            # v5 fallback: direct prediction on raw scaled features
            probabilities = global_model.predict_proba(X_scaled)
            predictions   = (probabilities[:, 1] >= global_threshold).astype(int)

        # temporary placeholder regime labels
        regime_labels = np.zeros(len(X_scaled), dtype=int)

        results = []
        for idx in range(len(df_work)):
            try:
                row = df_work.iloc[idx]
                proba = probabilities[idx]
                proba_buy = float(proba[1])
                confidence = float(max(proba))
                signal_idx = int(predictions[idx])
                
                # signal_idx already accounts for regime-specific threshold (set above)
                results.append({
                    "date":           row['date'] if 'date' in df_work.columns else str(df_work.index[idx]),
                    "signal":         "BUY" if signal_idx == 1 else "WAIT/SELL",
                    "confidence":     round(confidence, 4),
                    "proba_buy":      round(proba_buy, 4),
                    "regime":         (int(regime_labels[idx])
                                      if idx < len(regime_labels) and pd.notna(regime_labels[idx])
                                      else 0),
                    "threshold_used": round(global_threshold, 3),
                    "ou_half_life":   float(row.get("ou_half_life", 0.0)),
                })
            except Exception as e:
                # Skip this row on error
                continue
        
        if not results:
            print(f"   ⚠️ {ticker}: No valid signals generated")
            return pd.DataFrame(columns=['date', 'signal', 'confidence', 'proba_buy', 'regime'])

        result_df = pd.DataFrame(results)
        result_df["regime"] = result_df["regime"].fillna(0).astype(int)
        return result_df

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

            # 3. Add Specialist Layers
            architecture = self.data.get("architecture", "unknown")
            esn_pca      = self.data.get("esn_pca")
            global_esn   = self.data.get("global_esn")

            ticker_esn = (global_esn.get(ticker)
                          if isinstance(global_esn, dict) else global_esn)

            if esn_pca is not None:
                df, esn_states = add_esn_features(df, ticker_esn, collect_states=True)
                df['_esn_state'] = list(esn_states)
            else:
                df = add_esn_features(df, ticker_esn)

            df = add_advanced_features(df)
            df = df.dropna()

            if df.empty:
                return {"status": "error",
                        "message": f"No data after feature engineering for {ticker}"}

            if esn_pca is not None and '_esn_state' in df.columns:
                _last_state    = df['_esn_state'].iloc[-1]
                esn_latent_row = esn_pca.transform(
                    _last_state.reshape(1, -1)).flatten()
                df = df.drop(columns=['_esn_state'])
            else:
                esn_latent_row = None

            # 4. Sentiment
            # Match backtest: model was trained with sentiment=0.0 always.
            # Real sentiment is out-of-distribution until model is retrained with it.
            df.loc[df.index[-1], "sentiment"] = 0.0

            # 5. Build raw feature vector for last row
            for col in self.feature_cols:
                if col not in df.columns:
                    df[col] = 0.0

            available_cols = [c for c in self.feature_cols if c in df.columns]
            X_raw = np.nan_to_num(
                df[available_cols].iloc[[-1]].values,
                nan=0.0, posinf=0.0, neginf=0.0)

            # 6. Prediction & Confidence
            if architecture == "stacked_ensemble_v6":
                from xgboost_model import _specialist_proba_single_row
                from cross_asset_signals import CrossAssetSignalEngine
                from inflation_signals import InflationSignalEngine

                specialists = self.data.get("specialists", {})
                meta_scaler = self.data.get("meta_scaler")
                start_date  = (pd.Timestamp(target_date) -
                               pd.DateOffset(days=450)).strftime("%Y-%m-%d")
                try:
                    ca   = CrossAssetSignalEngine()
                    infl = InflationSignalEngine()
                    ca.load_data(start_date, target_date)
                    infl.load_data(start_date, target_date)
                    sbc = ca.get_stocks_bonds_corr(target_date)
                    css = ca.get_credit_stress_score(target_date)
                    rs  = ca.get_risk_on_off_signal(target_date)
                    im  = infl.get_inflation_momentum(target_date)
                    rr  = infl.get_real_rate_proxy(target_date)
                except Exception:
                    sbc, css, rs, im, rr = 0.0, 0.0, 0.0, 0.0, 0.0

                meta_row = _specialist_proba_single_row(
                    X_raw=X_raw, specialists=specialists,
                    scaler=self.scaler, feature_cols=available_cols,
                    esn_latent_row=esn_latent_row,
                    stocks_bonds_corr=sbc, credit_stress_score=css,
                    risk_score=rs, inflation_momentum=im, real_rate=rr,
                )
                _n_expected = getattr(meta_scaler, "n_features_in_", meta_row.shape[1])
                if meta_row.shape[1] < _n_expected:
                    meta_row = np.hstack([meta_row,
                        np.zeros((1, _n_expected - meta_row.shape[1]),
                                 dtype=np.float32)])
                elif meta_row.shape[1] > _n_expected:
                    meta_row = meta_row[:, :_n_expected]

                probs = self.model.predict_proba(
                    meta_scaler.transform(meta_row))[0]
            else:
                probs = self.model.predict_proba(
                    self.scaler.transform(X_raw))[0]

            signal_idx = int(probs[1] >= self.threshold)
            confidence = float(np.max(probs))
            
            # For debugging: print shapes and check for NaNs or extreme values before drift detection
            # print(X_scaled.shape)
            # print(np.isnan(X_scaled).sum())
            # print(np.max(np.abs(X_scaled)))

            # 8. Create the "Signal Packet"
            # This packet is exactly what the RL Agent in Section 5 will consume as its "STATE"
            return {
                "status": "success",
                "ticker": ticker,
                "date": target_date,
                "signal": "BUY" if signal_idx == 1 else "WAIT/SELL",
                "raw_signal": int(signal_idx),
                "confidence": confidence,
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
                    "macd": float(df.iloc[-1]["macd"]),
                    
                    # Regime / structure
                    "hurst": float(df.iloc[-1]["hurst"]),

                    # Trend & Momentum Indicators
                    "rwi": float(df.iloc[-1]["rwi_low"]),
                    "rsi": float(df.iloc[-1]["rsi_14"]),

                    # Volatility & Risk Metrics
                    "realized_vol": float(df.iloc[-1]["realized_vol_20d"]),
                    "atr": float(df.iloc[-1]["atr_14"]),
                    "return_skew": float(df.iloc[-1]["return_skew_20"]),

                    # Mean reversion
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
        print(f"   State Vector Summary: {list(packet['state_vector'].keys())}")
    else:
        print(f"   ❌ Error: {packet['message']}")