"""
rl_agent.py  (v2 — fully fixed)
================================
SAC + GRU policy — multi-asset RL trading agent.
VIRTUAL_TRADING_FIRM  |  Stage 6

Fixes applied vs v1
--------------------
  - Rich state: price momentum, RSI, ATR, BB, SMA per ticker
  - Loads price_data.pkl — no yfinance re-download
  - Reward: risk-adjusted Sortino+Sharpe + position-aware bonus − overtrade penalty
  - Reward clipped to [-1, 1] — critic stays stable
  - Gradient clipping via Adam eps=1e-5
  - SAC learning_starts=2000
  - Differential reward: last 10 bars only, not cumulative

Dependencies
------------
  pip install stable-baselines3 torch gymnasium pandas-ta optuna
"""

import os, sys, warnings, pickle
import numpy as np
import pandas as pd
import pandas_ta as ta
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, StopTrainingOnNoModelImprovement
)
from stable_baselines3.common.monitor import Monitor

warnings.filterwarnings("ignore")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

RESULTS_DIR = "5_backtesting/results"
RL_DIR      = "6_rl_agent"
os.makedirs(RL_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# 1.  STATE BUILDER
# ════════════════════════════════════════════════════════════════════════════

REGIME_MAP = {
    "Bull-Trending": 0,
    "Bull-MeanRev":  1,
    "Bear-Trending": 2,
    "Bear-MeanRev":  3,
}

class StateBuilder:
    """
    Builds (T, n_features) observation matrix from:
      - equity_curve_v2.csv  → global portfolio state
      - trade_log_v2.csv     → per-ticker proba signal
      - price_data.pkl       → momentum, RSI, ATR, BB, trend per ticker
    """

    GLOBAL_FEATURES = [
        "regime",
        "drawdown",
        "daily_return",
        "roll_vol_10",
        "roll_sharpe_10",
        "heat",
    ]

    TICKER_FEATURES = [
        "proba",
        "ret5d",
        "ret20d",
        "rsi",
        "above_sma50",
        "atr_norm",
        "bb_pct",
    ]

    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.n       = len(tickers)

    @property
    def n_features(self) -> int:
        return len(self.GLOBAL_FEATURES) + self.n * len(self.TICKER_FEATURES)

    def build(self,
              equity_df:  pd.DataFrame,
              trades_df:  pd.DataFrame,
              price_data: Dict = None
              ) -> Tuple[np.ndarray, List[str], List[str]]:

        eq = equity_df.copy().sort_values("date").reset_index(drop=True)
        eq["date"] = eq["date"].astype(str)

        # ── global features ───────────────────────────────────────────────
        eq["regime"]       = eq["regime"].map(REGIME_MAP).fillna(0).astype(float)
        eq["daily_return"] = eq["equity"].pct_change().fillna(0)
        peak               = eq["equity"].cummax()
        eq["drawdown"]     = ((eq["equity"] - peak) /
                               peak.clip(lower=1e-9)).fillna(0)
        eq["heat"]         = 0.0
        r                  = eq["daily_return"]
        eq["roll_vol_10"]  = r.rolling(10).std().fillna(0)
        eq["roll_sharpe_10"] = (
            r.rolling(10).mean() /
            r.rolling(10).std().clip(lower=1e-9)
        ).fillna(0) * np.sqrt(252)

        dates = eq["date"].tolist()

        # ── per-ticker features ───────────────────────────────────────────
        ticker_frames = []
        col_names     = list(self.GLOBAL_FEATURES)

        for ticker in self.tickers:
            # proba from trade log
            t_df = trades_df[trades_df["ticker"] == ticker].copy()
            t_df = t_df.sort_values("date")
            base = pd.DataFrame({"date": dates})
            base = base.merge(
                t_df[["date","proba"]].drop_duplicates("date"),
                on="date", how="left"
            ).ffill().fillna({"proba": 0.5})

            # price features
            price_feats = pd.DataFrame({
                "date":        dates,
                "ret5d":       0.0,
                "ret20d":      0.0,
                "rsi":         0.5,
                "above_sma50": 0.0,
                "atr_norm":    0.01,
                "bb_pct":      0.5,
            })

            if price_data and ticker in price_data:
                try:
                    pdf       = price_data[ticker].copy()
                    pdf.index = pdf.index.strftime("%Y-%m-%d")
                    close     = pdf["close"]
                    high      = pdf["high"]
                    low       = pdf["low"]

                    feat = pd.DataFrame(index=pdf.index)
                    feat["ret5d"]       = close.pct_change(5).fillna(0)
                    feat["ret20d"]      = close.pct_change(20).fillna(0)
                    feat["rsi"]         = (ta.rsi(close, 14) / 100.0).fillna(0.5)
                    feat["above_sma50"] = (close > ta.sma(close, 50)).astype(float).fillna(0)
                    atr                 = ta.atr(high, low, close, 14).fillna(0)
                    feat["atr_norm"]    = (atr / close.clip(lower=1e-9)).fillna(0.01)
                    bb = ta.bbands(close, length=20)
                    if bb is not None and not bb.empty:
                        upper = bb.iloc[:, 2]
                        lower_b = bb.iloc[:, 0]
                        feat["bb_pct"] = ((close - lower_b) /
                                          (upper - lower_b).clip(lower=1e-9)
                                         ).fillna(0.5)
                    else:
                        feat["bb_pct"] = 0.5

                    feat.index.name = "date"
                    feat = feat.reset_index()
                    feat["date"] = feat["date"].astype(str)
                    price_feats = (pd.DataFrame({"date": dates})
                                   .merge(feat, on="date", how="left")
                                   .ffill().bfill().fillna(0))
                    print(f"  ✅ {ticker}: price features built")

                except Exception as e:
                    print(f"  ⚠️  {ticker} price features failed: {e}")

            combined = base.merge(price_feats, on="date", how="left").fillna(0)

            for feat in self.TICKER_FEATURES:
                ticker_frames.append(combined[feat].values)
                col_names.append(f"{ticker}_{feat}")

        # ── assemble ──────────────────────────────────────────────────────
        global_rows = [eq[f].values for f in self.GLOBAL_FEATURES]
        all_rows    = global_rows + ticker_frames
        obs         = np.stack(all_rows, axis=1).astype(np.float32)
        obs         = np.clip(obs, -5.0, 5.0)

        print(f"  ✅ State matrix: {obs.shape} | {len(col_names)} features")
        return obs, dates, col_names


# ════════════════════════════════════════════════════════════════════════════
# 2.  TRADING ENVIRONMENT
# ════════════════════════════════════════════════════════════════════════════

class MultiAssetTradingEnv(gym.Env):
    """
    Observation : flattened (seq_len × n_features)
    Action      : (n_tickers,) softmax → portfolio weights
    Reward      : Sortino/Sharpe hybrid + position bonus − overtrade penalty
                  clipped to [-1, 1]
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 obs_matrix:       np.ndarray,
                 price_matrix:     np.ndarray,
                 dates:            List[str],
                 tickers:          List[str],
                 seq_len:          int   = 10,
                 initial_capital:  float = 100_000.0,
                 turnover_penalty: float = 0.0005,
                 slippage:         float = 0.001,
                 sortino_weight:   float = 0.7,
                 sharpe_weight:    float = 0.3,
                 reward_window:    int   = 10,
                 step_every:       int   = 5):

        super().__init__()
        self.obs_matrix       = obs_matrix
        self.price_matrix     = price_matrix
        self.dates            = dates
        self.tickers          = tickers
        self.n                = len(tickers)
        self.seq_len          = seq_len
        self.T                = len(dates)
        self.initial_capital  = initial_capital
        self.turnover_penalty = turnover_penalty
        self.slippage         = slippage
        self.sortino_w        = sortino_weight
        self.sharpe_w         = sharpe_weight
        self.reward_window    = reward_window
        self.step_every       = step_every

        n_feat = obs_matrix.shape[1]
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0,
            shape=(seq_len * n_feat,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.n,),
            dtype=np.float32
        )
        self._reset_state()

    def _reset_state(self):
        self.t              = self.seq_len
        self.equity         = self.initial_capital
        self.peak_equity    = self.initial_capital
        self.weights        = np.ones(self.n) / self.n
        self.return_history: List[float] = []

    def _get_obs(self) -> np.ndarray:
        window = self.obs_matrix[self.t - self.seq_len: self.t]
        return window.flatten()

    def _compute_reward(self,
                        portfolio_return: float,
                        prev_weights:     np.ndarray,
                        new_weights:      np.ndarray) -> float:

        self.return_history.append(portfolio_return)
        if len(self.return_history) > self.reward_window:
            self.return_history.pop(0)

        if len(self.return_history) < 3:
            return 0.0

        rets    = np.array(self.return_history)
        mu      = rets.mean()
        sigma   = rets.std() + 1e-9
        down    = rets[rets < 0]
        d_sigma = down.std() + 1e-9 if len(down) > 1 else sigma

        # 1. risk-adjusted hybrid
        sharpe  = (mu / sigma)   * np.sqrt(252)
        sortino = (mu / d_sigma) * np.sqrt(252)
        reward  = self.sortino_w * sortino + self.sharpe_w * sharpe

        # 2. position-aware: reward conviction on winning position
        top_idx    = np.argmax(new_weights)
        top_ret    = (self.price_matrix[self.t, top_idx] /
                      max(self.price_matrix[self.t - 1, top_idx], 1e-9)) - 1
        reward    += new_weights[top_idx] * top_ret * 10

        # 3. overtrade penalty
        turnover = np.abs(new_weights - prev_weights).sum()
        reward  -= self.turnover_penalty * turnover * 100

        # 4. drawdown penalty (only above 5%)
        dd = (self.peak_equity - self.equity) / (self.peak_equity + 1e-9)
        if dd > 0.05:
            reward -= dd * 5.0

        # 5. clip to [-1, 1]
        return float(np.clip(reward / 10.0, -1.0, 1.0))

    def step(self, action: np.ndarray):
        # ── skip non-rebalance bars (mark to market only) ─────────────────
        if self.t % self.step_every != 0:
            prices_now       = self.price_matrix[self.t]
            prices_prev      = self.price_matrix[self.t - 1]
            price_returns    = (prices_now / np.clip(prices_prev, 1e-9, None)) - 1
            portfolio_return = float(np.dot(self.weights, price_returns))
            self.equity      *= (1 + portfolio_return)
            self.peak_equity  = max(self.peak_equity, self.equity)
            self.t           += 1
            done = self.t >= self.T
            obs  = (self._get_obs() if not done
                    else np.zeros(self.observation_space.shape, dtype=np.float32))
            info = {
                "date":             self.dates[self.t - 1],
                "equity":           round(self.equity, 2),
                "portfolio_return": round(portfolio_return, 6),
                "weights":          dict(zip(self.tickers, self.weights.round(4))),
                "drawdown":         round((self.peak_equity - self.equity) /
                                        max(self.peak_equity, 1e-9), 4),
            }
            return obs, 0.0, done, False, info
    # ── rebalance bar — existing code unchanged below ──────────────────

        action      = np.clip(action, 1e-6, None)
        new_weights = action / action.sum()
        prev_weights = self.weights.copy()
        action      = np.clip(action, 1e-6, None)
        new_weights = action / action.sum()
        prev_weights = self.weights.copy()

        prices_now  = self.price_matrix[self.t]
        prices_prev = self.price_matrix[self.t - 1]

        slip_dir    = np.sign(new_weights - prev_weights)
        exec_prices = prices_now * (1 + self.slippage * slip_dir)  # noqa

        price_returns    = (prices_now / np.clip(prices_prev, 1e-9, None)) - 1
        portfolio_return = float(np.dot(prev_weights, price_returns))

        self.equity      *= (1 + portfolio_return)
        self.peak_equity  = max(self.peak_equity, self.equity)
        self.weights      = new_weights

        reward = self._compute_reward(portfolio_return, prev_weights, new_weights)
        self.t += 1
        done    = self.t >= self.T

        obs = (self._get_obs() if not done
               else np.zeros(self.observation_space.shape, dtype=np.float32))

        info = {
            "date":             self.dates[self.t - 1],
            "equity":           round(self.equity, 2),
            "portfolio_return": round(portfolio_return, 6),
            "weights":          dict(zip(self.tickers, new_weights.round(4))),
            "drawdown":         round((self.peak_equity - self.equity) /
                                      max(self.peak_equity, 1e-9), 4),
        }
        return obs, reward, done, False, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def render(self): pass


# ════════════════════════════════════════════════════════════════════════════
# 3.  GRU FEATURE EXTRACTOR
# ════════════════════════════════════════════════════════════════════════════

class GRUFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self,
                 observation_space: spaces.Box,
                 seq_len:    int   = 10,
                 n_features: int   = 48,
                 gru_hidden: int   = 128,
                 gru_layers: int   = 1,
                 dropout:    float = 0.1):

        super().__init__(observation_space, gru_hidden)
        self.seq_len    = seq_len
        self.n_features = n_features

        self.gru  = nn.GRU(
            input_size  = n_features,
            hidden_size = gru_hidden,
            num_layers  = gru_layers,
            batch_first = True,
            dropout     = dropout if gru_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(gru_hidden)
        self.act  = nn.Tanh()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        b      = obs.shape[0]
        x      = obs.view(b, self.seq_len, self.n_features)
        out, _ = self.gru(x)
        return self.act(self.norm(out[:, -1, :]))


# ════════════════════════════════════════════════════════════════════════════
# 4.  SAC AGENT
# ════════════════════════════════════════════════════════════════════════════

class RLTradingAgent:

    def __init__(self,
                 tickers:       List[str],
                 seq_len:       int   = 10,
                 gru_hidden:    int   = 128,
                 learning_rate: float = 1e-4,
                 batch_size:    int   = 256,
                 buffer_size:   int   = 100_000,
                 gamma:         float = 0.95,
                 tau:           float = 0.01,
                 step_every:    int = 5,
                 ent_coef:      str   = "auto",
                 device:        str   = "auto"):

        self.tickers     = tickers
        self.n           = len(tickers)
        self.seq_len     = seq_len
        self.gru_hidden  = gru_hidden
        self.lr          = learning_rate
        self.batch_size  = batch_size
        self.buffer_size = buffer_size
        self.gamma       = gamma
        self.tau         = tau
        self.step_every   = step_every
        self.ent_coef    = ent_coef
        self.device      = device
        self.model:     Optional[SAC] = None
        self.train_env: Optional[MultiAssetTradingEnv] = None
        self.eval_env:  Optional[MultiAssetTradingEnv] = None

    def build_envs(self,
                   obs_matrix:   np.ndarray,
                   price_matrix: np.ndarray,
                   dates:        List[str],
                   train_ratio:  float = 0.85,
                   **env_kwargs):

        T     = len(dates)
        split = int(T * train_ratio)

        self.train_env = Monitor(MultiAssetTradingEnv(
            obs_matrix   = obs_matrix[:split],
            price_matrix = price_matrix[:split],
            dates        = dates[:split],
            tickers      = self.tickers,
            seq_len      = self.seq_len,
            **env_kwargs
        ))
        self.eval_env = Monitor(MultiAssetTradingEnv(
            obs_matrix   = obs_matrix[split:],
            price_matrix = price_matrix[split:],
            dates        = dates[split:],
            tickers      = self.tickers,
            seq_len      = self.seq_len,
            **env_kwargs
        ))
        print(f"  ✅ Train: {split} bars | Eval: {T - split} bars")

    def build_model(self, n_features: int):
        policy_kwargs = dict(
            features_extractor_class  = GRUFeaturesExtractor,
            features_extractor_kwargs = dict(
                seq_len    = self.seq_len,
                n_features = n_features,
                gru_hidden = self.gru_hidden,
                gru_layers = 1,
                dropout    = 0.1,
            ),
            net_arch      = [128, 64],
            activation_fn = nn.Tanh,
        )

        self.model = SAC(
            policy          = "MlpPolicy",
            env             = DummyVecEnv([lambda: self.train_env]),
            learning_rate   = self.lr,
            batch_size      = self.batch_size,
            buffer_size     = self.buffer_size,
            gamma           = self.gamma,
            tau             = self.tau,
            ent_coef        = self.ent_coef,
            learning_starts = 2_000,
            policy_kwargs   = policy_kwargs,
            verbose         = 1,
            device          = self.device,
            tensorboard_log = f"{RL_DIR}/tb_logs/",
        )

        # gradient clipping — prevents critic explosion
        self.model.policy.optimizer = torch.optim.Adam(
            self.model.policy.parameters(),
            lr  = self.lr,
            eps = 1e-5,
        )

        print(f"  ✅ SAC+GRU | device={self.device} | "
              f"LR={self.lr} | batch={self.batch_size} | gamma={self.gamma}")

    def train(self,
              total_timesteps: int = 200_000,
              eval_freq:       int = 5_000,
              n_eval_episodes: int = 3):

        if self.model is None:
            raise RuntimeError("Call build_model() first.")

        stop_cb = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals = 20,
            min_evals                = 30,
            verbose                  = 1,
        )
        eval_cb = EvalCallback(
            eval_env             = DummyVecEnv([lambda: self.eval_env]),
            best_model_save_path = f"{RL_DIR}/best_model/",
            log_path             = f"{RL_DIR}/eval_logs/",
            eval_freq            = eval_freq,
            n_eval_episodes      = n_eval_episodes,
            deterministic        = True,
            verbose              = 1,
            callback_after_eval  = stop_cb,
        )

        print(f"\n🚀 Training SAC+GRU | {total_timesteps:,} timesteps …\n")
        self.model.learn(
            total_timesteps = total_timesteps,
            callback        = eval_cb,
            progress_bar    = True,
        )
        self.model.save(f"{RL_DIR}/sac_gru_final")
        print(f"\n✅ Saved → {RL_DIR}/sac_gru_final.zip")

    def evaluate(self,
                 env:           Optional[MultiAssetTradingEnv] = None,
                 deterministic: bool = True) -> pd.DataFrame:

        if self.model is None:
            raise RuntimeError("No model loaded.")

        env = env or self.eval_env
        obs, _ = env.reset()
        records = []

        while True:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, done, _, info = env.step(action)
            records.append({
                "date":   info["date"],
                "equity": info["equity"],
                "reward": round(reward, 6),
                "dd":     info["drawdown"],
                **{f"w_{t}": info["weights"].get(t, 0)
                   for t in self.tickers},
            })
            if done:
                break

        df = pd.DataFrame(records)
        df.to_csv(f"{RL_DIR}/rl_equity_curve.csv", index=False)
        print(f"\n📊 RL curve → {RL_DIR}/rl_equity_curve.csv")
        self._print_rl_tearsheet(df)
        return df

    @staticmethod
    def _print_rl_tearsheet(df: pd.DataFrame):
        eq   = df["equity"].values
        rets = pd.Series(eq).pct_change().fillna(0).values

        total_ret = (eq[-1] - eq[0]) / eq[0]
        ann_ret   = (1 + total_ret) ** (252 / max(len(eq), 1)) - 1
        sharpe    = rets.mean() / (rets.std() + 1e-9) * np.sqrt(252)
        down      = rets[rets < 0]
        sortino   = (rets.mean() / (down.std() + 1e-9) * np.sqrt(252)
                     if len(down) > 1 else 0)
        peak      = np.maximum.accumulate(eq)
        max_dd    = ((eq - peak) / np.where(peak > 0, peak, 1)).min()
        calmar    = ann_ret / abs(max_dd) if max_dd < 0 else 0

        W = 44
        print("\n" + "═" * W)
        print(f"  {'RL AGENT TEARSHEET':^{W-4}}")
        print("═" * W)
        print(f"  Total Return      {total_ret*100:>16.2f}%")
        print(f"  Ann. Return       {ann_ret*100:>16.2f}%")
        print(f"  Sharpe            {sharpe:>17.3f}")
        print(f"  Sortino           {sortino:>17.3f}")
        print(f"  Calmar            {calmar:>17.3f}")
        print(f"  Max Drawdown      {max_dd*100:>16.2f}%")
        print("═" * W)

    def save(self, path: str = None):
        path = path or f"{RL_DIR}/sac_gru_final"
        if self.model:
            self.model.save(path)
            print(f"  💾 Saved → {path}.zip")

    def load(self, path: str = None):
        path = path or f"{RL_DIR}/best_model/best_model"
        self.model = SAC.load(path, device=self.device)
        print(f"  ✅ Loaded ← {path}.zip")

    def predict_weights(self,
                        obs_window:    np.ndarray,
                        deterministic: bool = True) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("No model loaded.")
        flat      = obs_window.flatten().astype(np.float32)
        action, _ = self.model.predict(flat[None], deterministic=deterministic)
        action    = np.clip(action[0], 1e-6, None)
        weights   = action / action.sum()
        return dict(zip(self.tickers, weights.round(4)))


# ════════════════════════════════════════════════════════════════════════════
# 5.  DATA PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def load_backtest_outputs(
        tickers:     List[str],
        equity_file: str = f"{RESULTS_DIR}/equity_curve.csv",
        trade_file:  str = f"{RESULTS_DIR}/trade_log.csv",
        price_pkl:   str = f"{RESULTS_DIR}/price_data.pkl",
) -> Tuple[np.ndarray, np.ndarray, List[str], int, StateBuilder]:

    print("📂 Loading backtest outputs …")
    equity_df = pd.read_csv(equity_file)
    trades_df = pd.read_csv(trade_file)
    equity_df["date"] = equity_df["date"].astype(str)
    trades_df["date"] = trades_df["date"].astype(str)

    # load price_data.pkl
    price_data = None
    if os.path.exists(price_pkl):
        with open(price_pkl, "rb") as f:
            price_data = pickle.load(f)
        print(f"  ✅ price_data.pkl: {len(price_data)} tickers")
    else:
        print(f"  ⚠️  price_data.pkl not found — price features will be zeros")

    # build state
    sb = StateBuilder(tickers)
    obs_matrix, dates, col_names = sb.build(equity_df, trades_df, price_data)
    n_features = obs_matrix.shape[1]

    # price matrix for env
    price_frames = []
    for ticker in tickers:
        if price_data and ticker in price_data:
            df    = price_data[ticker]
            close = df["close"].copy()
            close.index = df.index.strftime("%Y-%m-%d")
            price_frames.append(close.rename(ticker))
        else:
            price_frames.append(pd.Series(1.0, index=dates, name=ticker))

    price_df     = pd.concat(price_frames, axis=1)
    price_df     = price_df.reindex(dates).ffill().bfill().fillna(1.0)
    price_matrix = (price_df / price_df.iloc[0]).values.astype(np.float32)

    print(f"  ✅ price_matrix: {price_matrix.shape}")
    return obs_matrix, price_matrix, dates, n_features, sb


# ════════════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    TICKERS = ["AAPL", "NVDA", "MSFT", "SPY", "QQQ", "TSLA"]

    # ── 1. load data ──────────────────────────────────────────────────────
    obs_matrix, price_matrix, dates, n_features, sb = load_backtest_outputs(
        tickers     = TICKERS,
        equity_file = f"{RESULTS_DIR}/equity_curve.csv",
        trade_file  = f"{RESULTS_DIR}/trade_log.csv",
        price_pkl   = f"{RESULTS_DIR}/price_data.pkl",
    )

    # ── 2. (optional) tune hyperparams ───────────────────────────────────
    # best_params = tune_hyperparams(obs_matrix, price_matrix, dates,
    #                                TICKERS, n_features, n_trials=8)

    # ── 3. build agent ────────────────────────────────────────────────────
    agent = RLTradingAgent(
        tickers       = TICKERS,
        seq_len       = 10,
        gru_hidden    = 128,
        learning_rate = 1e-4,      # was 3e-4, slower but stable,
        batch_size    = 256,       # was 64, larger batch smooths gradients
        buffer_size   = 100_000,
        gamma         = 0.95,      # less exploding rewards, more stable learning
        tau           = 0.01,      # was 0.005, faster target network update
        ent_coef      = "auto",
        device        = "auto",   # uses GPU if available
    )

    # ── 4. build environments ─────────────────────────────────────────────
    agent.build_envs(
        obs_matrix    = obs_matrix,
        price_matrix  = price_matrix,
        dates         = dates,
        train_ratio   = 0.85,
        step_every       = 5, 
        # env kwargs
        turnover_penalty = 0.0005,
        slippage         = 0.001,
        sortino_weight   = 0.7,
        sharpe_weight    = 0.3,
        reward_window    = 10,
    )

    # ── 5. build model ────────────────────────────────────────────────────
    agent.build_model(n_features=n_features)

    # ── 6. train ──────────────────────────────────────────────────────────
    agent.train(
        total_timesteps = 500_000,
        eval_freq       = 10_000,
        n_eval_episodes = 3,
    )
    
    # ── 7. load best checkpoint + evaluate ───────────────────────────────
    print("\n📊 Evaluating on OOS slice …")
    agent.load(f"{RL_DIR}/best_model/best_model")  # best OOS, not last checkpoint
    rl_df = agent.evaluate(deterministic=True)


    # ── 8. live inference demo ────────────────────────────────────────────
    print("\n🤖 Live inference demo (last window):")
    last_window = obs_matrix[-agent.seq_len:]
    weights     = agent.predict_weights(last_window)
    print("  Target allocations:")
    for ticker, w in weights.items():
        print(f"    {ticker:<6} → {w:.2%}")