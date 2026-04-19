"""
benchmark.py
============
Algorithm benchmarking for VIRTUAL_TRADING_FIRM — Stage 6b.
Trains SAC, PPO, A2C, TD3 on identical environment + data split.
Produces side-by-side tearsheet + saves best model.

Run:
    python 6_rl_agent/benchmark.py
Or in Colab — import and call run_benchmark() directly.
"""

import os, sys, warnings, time
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

warnings.filterwarnings("ignore")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "5_backtesting"))
sys.path.insert(0, os.path.join(project_root, "6_rl_agent"))

from stable_baselines3 import SAC, PPO, A2C, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from rl_agent import (
    MultiAssetTradingEnv,
    GRUFeaturesExtractor,
    load_backtest_outputs,
    StateBuilder,
)

import torch.nn as nn

RESULTS_DIR = "5_backtesting/results"
BENCH_DIR   = "6_rl_agent/benchmark"
os.makedirs(BENCH_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# 1.  ALGORITHM CONFIGS
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class AlgoConfig:
    name:            str
    cls:             type
    total_timesteps: int
    extra_kwargs:    Dict  # algo-specific SB3 kwargs


def get_algo_configs(n_actions: int,
                     n_features: int,
                     seq_len: int,
                     gru_hidden: int = 128) -> List[AlgoConfig]:
    """
    Returns list of algorithm configs.
    All share the same GRU feature extractor where supported.
    TD3 uses a separate action noise config.
    A2C/PPO use on-policy settings.
    """

    gru_kwargs = dict(
        features_extractor_class  = GRUFeaturesExtractor,
        features_extractor_kwargs = dict(
            seq_len    = seq_len,
            n_features = n_features,
            gru_hidden = gru_hidden,
            gru_layers = 1,
            dropout    = 0.1,
        ),
        net_arch      = [128, 64],
        activation_fn = nn.Tanh,
    )

    # TD3 needs action noise for exploration
    action_noise = NormalActionNoise(
        mean  = np.zeros(n_actions),
        sigma = 0.1 * np.ones(n_actions),
    )

    return [
        AlgoConfig(
            name = "SAC",
            cls  = SAC,
            total_timesteps = 200_000,
            extra_kwargs = dict(
                learning_rate   = 1e-4,
                batch_size      = 256,
                buffer_size     = 100_000,
                gamma           = 0.95,
                tau             = 0.01,
                ent_coef        = "auto",
                learning_starts = 2_000,
                policy_kwargs   = gru_kwargs,
            ),
        ),
        AlgoConfig(
            name = "TD3",
            cls  = TD3,
            total_timesteps = 200_000,
            extra_kwargs = dict(
                learning_rate   = 1e-4,
                batch_size      = 256,
                buffer_size     = 100_000,
                gamma           = 0.95,
                tau             = 0.01,
                action_noise    = action_noise,
                learning_starts = 2_000,
                policy_kwargs   = gru_kwargs,
            ),
        ),
        AlgoConfig(
            name = "PPO",
            cls  = PPO,
            total_timesteps = 200_000,
            extra_kwargs = dict(
                learning_rate = 3e-4,
                n_steps       = 64,
                batch_size    = 32,
                n_epochs      = 5,
                gamma         = 0.95,
                gae_lambda    = 0.95,
                clip_range    = 0.2,
                ent_coef      = 0.01,
                policy_kwargs = gru_kwargs,
            ),
        ),
        AlgoConfig(
            name = "A2C",
            cls  = A2C,
            total_timesteps = 200_000,
            extra_kwargs = dict(
                learning_rate = 7e-4,
                n_steps       = 32,
                gamma         = 0.95,
                gae_lambda    = 0.9,
                ent_coef      = 0.01,
                rms_prop_eps  = 1e-5,
                policy_kwargs = gru_kwargs,
            ),
        ),
    ]


# ════════════════════════════════════════════════════════════════════════════
# 2.  SINGLE ALGORITHM TRAINER
# ════════════════════════════════════════════════════════════════════════════

def train_algo(
    cfg:          AlgoConfig,
    train_env:    MultiAssetTradingEnv,
    eval_env:     MultiAssetTradingEnv,
    device:       str = "auto",
) -> Tuple[object, float]:
    """
    Trains one algorithm. Returns (model, training_time_seconds).
    """
    print(f"\n{'='*55}")
    print(f"  🚀 Training {cfg.name} | {cfg.total_timesteps:,} timesteps")
    print(f"{'='*55}")

    vec_train = DummyVecEnv([lambda: train_env])

    model = cfg.cls(
        policy  = "MlpPolicy",
        env     = vec_train,
        verbose = 0,           # silent — benchmark handles printing
        device  = device,
        tensorboard_log = f"{BENCH_DIR}/tb_{cfg.name}/",
        **cfg.extra_kwargs,
    )

    # gradient clipping for off-policy algos
    if cfg.name in ("SAC", "TD3"):
        model.policy.optimizer = torch.optim.Adam(
            model.policy.parameters(),
            lr  = cfg.extra_kwargs.get("learning_rate", 1e-4),
            eps = 1e-5,
        )

    t0 = time.time()
    model.learn(
        total_timesteps = cfg.total_timesteps,
        progress_bar    = True,
    )
    elapsed = time.time() - t0

    save_path = f"{BENCH_DIR}/{cfg.name}_model"
    model.save(save_path)
    print(f"  💾 {cfg.name} saved → {save_path}.zip")
    print(f"  ⏱️  Training time: {elapsed/60:.1f} min")

    return model, elapsed


# ════════════════════════════════════════════════════════════════════════════
# 3.  EVALUATOR
# ════════════════════════════════════════════════════════════════════════════

def evaluate_algo(
    model,
    algo_name:  str,
    eval_env:   MultiAssetTradingEnv,
    tickers:    List[str],
) -> Dict:
    """
    Runs one full episode on eval_env.
    Returns metrics dict.
    """
    obs, _ = eval_env.reset()
    records = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = eval_env.step(action)
        records.append({
            "date":   info["date"],
            "equity": info["equity"],
            "dd":     info["drawdown"],
            **{f"w_{t}": info["weights"].get(t, 0) for t in tickers},
        })
        if done:
            break

    df = pd.DataFrame(records)
    df.to_csv(f"{BENCH_DIR}/{algo_name}_equity.csv", index=False)

    eq   = df["equity"].values
    rets = pd.Series(eq).pct_change().fillna(0).values

    total_ret  = (eq[-1] - eq[0]) / eq[0]
    ann_ret    = (1 + total_ret) ** (252 / max(len(eq), 1)) - 1
    sharpe     = rets.mean() / (rets.std()  + 1e-9) * np.sqrt(252)
    down       = rets[rets < 0]
    sortino    = (rets.mean() / (down.std() + 1e-9) * np.sqrt(252)
                  if len(down) > 1 else 0)
    peak       = np.maximum.accumulate(eq)
    max_dd     = ((eq - peak) / np.where(peak > 0, peak, 1)).min()
    calmar     = ann_ret / abs(max_dd) if max_dd < 0 else 0
    hit_rate   = float((rets > 0).mean())

    # avg allocation per ticker
    weight_cols = [f"w_{t}" for t in tickers]
    avg_weights = {t: round(float(df[f"w_{t}"].mean()), 4) for t in tickers}

    return {
        "algo":         algo_name,
        "total_ret":    round(total_ret * 100, 2),
        "ann_ret":      round(ann_ret   * 100, 2),
        "sharpe":       round(sharpe,    3),
        "sortino":      round(sortino,   3),
        "calmar":       round(calmar,    3),
        "max_dd":       round(max_dd   * 100, 2),
        "hit_rate":     round(hit_rate * 100, 2),
        "avg_weights":  avg_weights,
        "equity_df":    df,
    }


# ════════════════════════════════════════════════════════════════════════════
# 4.  COMPARISON TEARSHEET
# ════════════════════════════════════════════════════════════════════════════

def print_comparison(results: List[Dict], train_times: Dict[str, float]):
    W = 72
    print("\n" + "═" * W)
    print(f"  {'ALGORITHM BENCHMARK — COMPARISON TEARSHEET':^{W-4}}")
    print("═" * W)
    print(f"  {'Metric':<18}", end="")
    for r in results:
        print(f"  {r['algo']:>10}", end="")
    print()
    print("  " + "─" * (W - 2))

    metrics = [
        ("Total Return %",  "total_ret"),
        ("Ann. Return %",   "ann_ret"),
        ("Sharpe",          "sharpe"),
        ("Sortino",         "sortino"),
        ("Calmar",          "calmar"),
        ("Max DD %",        "max_dd"),
        ("Hit Rate %",      "hit_rate"),
    ]

    for label, key in metrics:
        print(f"  {label:<18}", end="")
        vals = [r[key] for r in results]
        best = max(vals) if key != "max_dd" else max(vals)  # max_dd least negative = best
        for r in results:
            v    = r[key]
            star = " ★" if v == best else "  "
            print(f"  {v:>9.2f}{star}", end="")
        print()

    print("  " + "─" * (W - 2))
    print(f"  {'Train time (min)':<18}", end="")
    for r in results:
        t = train_times.get(r["algo"], 0) / 60
        print(f"  {t:>10.1f}", end="")
    print()

    print("═" * W)

    # winner
    best = max(results, key=lambda x: x["sharpe"])
    print(f"\n  🏆 WINNER: {best['algo']} "
          f"(Sharpe {best['sharpe']:.3f} | "
          f"Return {best['total_ret']:.2f}%)")

    print("\n  📊 AVERAGE ALLOCATIONS PER ALGO:")
    print(f"  {'Ticker':<8}", end="")
    for r in results:
        print(f"  {r['algo']:>10}", end="")
    print()
    tickers = list(results[0]["avg_weights"].keys())
    for ticker in tickers:
        print(f"  {ticker:<8}", end="")
        for r in results:
            w = r["avg_weights"].get(ticker, 0)
            print(f"  {w*100:>9.1f}%", end="")
        print()
    print("═" * W + "\n")

    return best["algo"]


# ════════════════════════════════════════════════════════════════════════════
# 5.  MAIN BENCHMARK RUNNER
# ════════════════════════════════════════════════════════════════════════════

def run_benchmark(
    tickers:      List[str]  = None,
    equity_file:  str        = f"{RESULTS_DIR}/equity_curve.csv",
    trade_file:   str        = f"{RESULTS_DIR}/trade_log.csv",
    price_pkl:    str        = f"{RESULTS_DIR}/price_data.pkl",
    train_ratio:  float      = 0.85,
    seq_len:      int        = 10,
    gru_hidden:   int        = 128,
    device:       str        = "auto",
    algos:        List[str]  = None,   # None = run all 4
    env_kwargs:   Dict       = None,
) -> Dict:
    """
    Full benchmark pipeline.
    Returns dict with all results + winner name.
    """
    tickers   = tickers or ["AAPL","NVDA","MSFT","SPY","QQQ","TSLA"]
    algos     = algos   or ["SAC", "TD3", "PPO", "A2C"]
    env_kwargs = env_kwargs or dict(
        turnover_penalty = 0.0005,
        slippage         = 0.001,
        sortino_weight   = 0.7,
        sharpe_weight    = 0.3,
        reward_window    = 10,
    )

    # ── load data ─────────────────────────────────────────────────────────
    obs_matrix, price_matrix, dates, n_features, sb = load_backtest_outputs(
        tickers     = tickers,
        equity_file = equity_file,
        trade_file  = trade_file,
        price_pkl   = price_pkl,
    )

    T     = len(dates)
    split = int(T * train_ratio)

    print(f"\n📊 Data: {T} bars | Train: {split} | Eval: {T-split}")
    print(f"   Algos: {algos}")
    print(f"   Device: {device}\n")

    # ── algo configs ──────────────────────────────────────────────────────
    all_configs = {c.name: c for c in get_algo_configs(
        n_actions  = len(tickers),
        n_features = n_features,
        seq_len    = seq_len,
        gru_hidden = gru_hidden,
    )}

    # ── train + evaluate each algo ────────────────────────────────────────
    results     = []
    train_times = {}

    for algo_name in algos:
        if algo_name not in all_configs:
            print(f"  ⚠️  Unknown algo: {algo_name} — skipping")
            continue

        cfg = all_configs[algo_name]

        # fresh envs per algo (avoid state leakage)
        train_env = Monitor(MultiAssetTradingEnv(
            obs_matrix   = obs_matrix[:split],
            price_matrix = price_matrix[:split],
            dates        = dates[:split],
            tickers      = tickers,
            seq_len      = seq_len,
            **env_kwargs,
        ))
        eval_env = MultiAssetTradingEnv(
            obs_matrix   = obs_matrix[split:],
            price_matrix = price_matrix[split:],
            dates        = dates[split:],
            tickers      = tickers,
            seq_len      = seq_len,
            **env_kwargs,
        )

        # train
        model, elapsed = train_algo(cfg, train_env, eval_env, device)
        train_times[algo_name] = elapsed

        # evaluate
        metrics = evaluate_algo(model, algo_name, eval_env, tickers)
        results.append(metrics)

        # quick inline result
        print(f"  📈 {algo_name}: "
              f"Return={metrics['total_ret']:.2f}% | "
              f"Sharpe={metrics['sharpe']:.3f} | "
              f"DD={metrics['max_dd']:.2f}%")

    # ── comparison tearsheet ──────────────────────────────────────────────
    winner = print_comparison(results, train_times)

    # ── save summary CSV ──────────────────────────────────────────────────
    summary = [{k: v for k, v in r.items() if k not in ("avg_weights","equity_df")}
               for r in results]
    pd.DataFrame(summary).to_csv(f"{BENCH_DIR}/benchmark_summary.csv", index=False)
    print(f"  💾 Summary → {BENCH_DIR}/benchmark_summary.csv")

    return {
        "results":      results,
        "winner":       winner,
        "train_times":  train_times,
        "obs_matrix":   obs_matrix,
        "price_matrix": price_matrix,
        "dates":        dates,
        "n_features":   n_features,
    }


# ════════════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    out = run_benchmark(
        tickers     = ["AAPL", "NVDA", "MSFT", "SPY", "QQQ", "TSLA"],
        equity_file = f"{RESULTS_DIR}/equity_curve.csv",
        trade_file  = f"{RESULTS_DIR}/trade_log.csv",
        price_pkl   = f"{RESULTS_DIR}/price_data.pkl",
        train_ratio = 0.85,
        seq_len     = 10,
        gru_hidden  = 128,
        device      = "auto",
        algos       = ["SAC", "TD3", "PPO", "A2C"],
    )

    print(f"\n🏆 Best algorithm: {out['winner']}")
    print(f"   Load it from: 6_rl_agent/benchmark/{out['winner']}_model.zip")