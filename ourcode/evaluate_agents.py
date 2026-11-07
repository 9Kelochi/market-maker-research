"""
Evaluate 3 PPO market makers in ONE shared ABIDES-NG market, with a sell shock.

2x2 design, identical agent count in every cell:
    policy : ppo | passive   (passive = same agents, always KEEP_CURRENT_ORDERS)
    shock  : on  | off       (off = same ImpactOrderAgent scheduled after the sim ends)
A non-trading OBSERVER wakes every 5 s and is the single data source for the book.

Run from ourcode/:
    uv run --with stable-baselines3 python evaluate_agents.py                      # seeds 1-5, 1000-share shock
    uv run --with stable-baselines3 python evaluate_agents.py --seeds 1 2 --shock-size 5000
"""
import argparse
from copy import deepcopy

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from abides_core import Kernel
from abides_core.messaging.generators import ConstantTimeGenerator
from abides_core.utils import str_to_ns
from abides_gym.experimental_agents.financial_gym_agent import FinancialGymAgent
from abides_markets.agents.impact_order_agent import ImpactOrderAgent
from abides_markets.configs import rmsc04
from abides_markets.orders import Side
from abides_markets.utils import config_add_agents

import config as C
import metrics
from train_market_maker_ppo import MarketMakingEnv

KEEP = 5
_book_parse_failures = 0


class EvaluatedGymAgent(FinancialGymAgent):
    def act_on_wakeup(self):
        raw_state = super().act_on_wakeup()
        return {"agent_id": self.id, "raw_state": raw_state}


def load_models():
    return [PPO.load(str(C.MODEL_DIR / f"ppo_agent_{i + 1}.zip"), device="cpu")
            for i in range(C.N_PPO)]


def resolve_mkt_open_ns(config):
    for key in ("mkt_open", "market_open"):
        if key in config:
            return int(config[key]), f"config['{key}']"
    return pd.Timestamp(C.MKT_OPEN_FALLBACK).value, "fallback constant"



def book_depth(raw):
    """Depth from the RAW book (not the gym's possibly filled-in spread). NaN on any format surprise."""
    out = {"bid_depth": float("nan"), "ask_depth": float("nan"),
           "n_bid_levels": float("nan"), "n_ask_levels": float("nan"), "book_empty": float("nan")}
    try:
        snap = list(raw[-1]["parsed_mkt_data"])[-1]
        bids, asks = snap["bids"], snap["asks"]
        out["n_bid_levels"], out["n_ask_levels"] = len(bids), len(asks)
        out["bid_depth"] = float(sum(float(l[1]) for l in bids))
        out["ask_depth"] = float(sum(float(l[1]) for l in asks))
        out["book_empty"] = int(len(bids) == 0 or len(asks) == 0)
    except Exception:
        pass
    return out

def make_gym_agent(agent_id, name, first, step, seed):
    return EvaluatedGymAgent(
        id=agent_id, symbol="ABM", starting_cash=10_000_000, name=name,
        type="EvaluatedPPO", first_interval=str_to_ns(first),
        wakeup_interval_generator=ConstantTimeGenerator(step_duration=str_to_ns(step)),
        state_buffer_length=4, market_data_buffer_length=5,
        random_state=np.random.RandomState(seed), log_orders=False,
    )


def book_stats(raw):
    """Spread/depth straight from book levels.
    Spread is NaN and one_sided=1 when either = empty."""
    global _book_parse_failures
    try:
        snap = raw[-1]["parsed_mkt_data"][-1]
        bids, asks = snap["bids"], snap["asks"]
        one_sided = int(len(bids) == 0 or len(asks) == 0)
        return {
            "spread_raw_cents": np.nan if one_sided else float(asks[0][0] - bids[0][0]),
            "bid_depth": int(sum(q for _, q in bids)),
            "ask_depth": int(sum(q for _, q in asks)),
            "one_sided": one_sided,
        }
    except Exception:
        _book_parse_failures += 1
        return {"spread_raw_cents": np.nan, "bid_depth": np.nan,
                "ask_depth": np.nan, "one_sided": np.nan}


def run_one(models, use_ppo, shock_on, seed):
    config = rmsc04.build_config(
        seed=seed, end_time=C.END_TIME, log_orders=False, book_logging=False,
    )
    mkt_open, src = resolve_mkt_open_ns(config)
    base_id = len(config["agents"])

    gym_agents = [make_gym_agent(base_id + i, f"EVAL_PPO_{i + 1}",
                                 C.PPO_FIRST_INTERVAL, C.PPO_STEP, 900 + i)
                  for i in range(C.N_PPO)]
    gym_agents.append(make_gym_agent(base_id + C.N_PPO, "OBSERVER",
                                     C.OBSERVER_FIRST_INTERVAL, C.OBSERVER_STEP, 950))

    shock_time = mkt_open + str_to_ns(C.SHOCK_OFFSET)
    dormant_time = config["stop_time"] + str_to_ns("01:00:00")
    shock_agent = ImpactOrderAgent(
        id=base_id + C.N_PPO + 1,
        order_time=shock_time if shock_on else dormant_time,
        quantity=C.SHOCK_SIZE, name="SHOCK", type="ImpactOrderAgent",
        random_state=np.random.RandomState(777), symbol="ABM",
        starting_cash=100_000_000,
        side=Side.ASK if C.SHOCK_SIDE == "ASK" else Side.BID,
        order_type="MARKET",
    )

    config = config_add_agents(config, gym_agents + [shock_agent], np.random.RandomState(1000))
    kernel = Kernel(
        agents=config["agents"], random_state=config["random_state_kernel"],
        start_time=config["start_time"], stop_time=config["stop_time"],
        agent_latency_model=config["agent_latency_model"],
        default_computation_delay=config["default_computation_delay"],
        oracle=config["oracle"], runner_hook=gym_agents[0],
    )

    adapters = [MarketMakingEnv() for _ in gym_agents]
    by_id = {a.id: i for i, a in enumerate(gym_agents)}
    names = [f"PPO_{i + 1}" for i in range(C.N_PPO)] + ["OBSERVER"]
    counters = [0] * len(gym_agents)
    rows, checked = [], False

    kernel.initialize()
    try:
        event = kernel.runner()
        while not event["done"]:
            payload = event["result"]
            i = by_id[payload["agent_id"]]
            agent, adapter, raw = gym_agents[i], adapters[i], payload["raw_state"]
            is_observer = names[i] == "OBSERVER"

            if not checked:
                raw_open = raw[-1]["internal_data"]["mkt_open"]
                if raw_open != mkt_open:
                    raise RuntimeError(f"mkt_open mismatch: used {mkt_open} ({src}), "
                                       f"simulation says {raw_open}. Shock would be mistimed.")
                checked = True

            info = adapter.raw_state_to_info(deepcopy(raw))
            if is_observer or not use_ppo:
                action = KEEP
            else:
                obs = adapter.raw_state_to_state(deepcopy(raw))
                action, _ = models[i].predict(obs, deterministic=True)
                action = int(np.asarray(action).item())
            commands = adapter._map_action_space_to_ABIDES_SIMULATOR_SPACE(action)

            counters[i] += 1
            row = {
                "seed": seed, "policy": "ppo" if use_ppo else "passive",
                "shock": int(shock_on), "who": names[i], "decision": counters[i],
                "sim_time_s": (agent.current_time - mkt_open) / 1e9,
                "action": adapter.ACTION_NAMES[action],
                "inventory": int(info["holdings"]),
                "n_open_orders": len(getattr(agent, "orders", {}) or {}),
                "best_bid": info["best_bid"], "best_ask": info["best_ask"],
                "spread_cents": info["spread"],   
            }
            row.update(book_stats(raw))
            rows.append(row)
            event = kernel.runner((agent, commands))
    finally:
        kernel.terminate()
        for a in adapters:
            try:
                a.close()
            except Exception:
                pass

    filled = -shock_agent.holdings.get("ABM", 0)
    for r in rows:
        r["shock_filled"] = filled
    print(f"  shock filled {filled} shares (expect {C.SHOCK_SIZE if shock_on else 0})", flush=True)
    if shock_on and filled < 0.95 * C.SHOCK_SIZE:
        print(f"  WARNING: only {filled}/{C.SHOCK_SIZE} filled; shock size differs across runs.", flush=True)
    if not shock_on and filled != 0:
        print("  WARNING: shock agent traded in a no-shock run.", flush=True)
    return rows


def save_merged(new_rows, seeds):
    path = C.results_file("multi_agent_ppo_evaluation")
    new = pd.DataFrame(new_rows)
    if path.exists():
        old = pd.read_csv(path)
        if {"spread_raw_cents", "shock_filled"} <= set(old.columns):
            new = pd.concat([old[~old.seed.isin(seeds)], new], ignore_index=True)
    new.to_csv(path, index=False)
    return new, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=C.SEEDS)
    ap.add_argument("--shock-size", type=int, default=C.SHOCK_SIZE)
    args = ap.parse_args()
    C.SHOCK_SIZE = args.shock_size

    models = load_models()
    new_rows, done = [], []
    for seed in args.seeds:
        for use_ppo in (False, True):
            for shock_on in (False, True):
                print(f"seed={seed} policy={'ppo' if use_ppo else 'passive'} "
                      f"shock={shock_on} size={C.SHOCK_SIZE}", flush=True)
                new_rows += run_one(models, use_ppo, shock_on, seed)
        done.append(seed)
        save_merged(new_rows, done)           # checkpoint after every seed

    if _book_parse_failures:
        print(f"\nWARNING: {_book_parse_failures} book snapshots could not be parsed; "
              f"spread_raw/depth are NaN there. Check book_stats().")

    df, path = save_merged(new_rows, done)
    summary = metrics.summarize(df)
    summary.to_csv(C.results_file("summary_metrics"), index=False)
    paired = metrics.paired_effects(summary)
    paired.to_csv(C.results_file("paired_effects"), index=False)
    pd.set_option("display.width", 200)
    print("\n", summary.to_string(index=False))
    print("\nPaired shock effect (shock - no shock), mean over seeds:")
    print(paired.groupby("policy").mean(numeric_only=True).drop(columns="seed").to_string())
    print(f"\n[SUCCESS] wrote {path.name}, summary and paired_effects in {C.RESULTS_DIR}")


if __name__ == "__main__":
    main()