"""Stage 1 of independent multi-agent market-making research.

Train 3 distinct single-agent PPO policies, then evaluate all 3 inside ONE
shared ABIDES-NG limit-order-book simulation.

This is a shared-market evaluation smoke test, NOT joint-market IPPO training.
Run from the ABIDES-NG repo root:
    uv run --with stable-baselines3 python ourcode/three_ppo_agents.py

Requires ourcode/train_market_maker_ppo.py from the previous working stage.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from abides_core import Kernel
from abides_core.messaging.generators import ConstantTimeGenerator
from abides_core.utils import str_to_ns
from abides_gym.experimental_agents.financial_gym_agent import FinancialGymAgent
from abides_markets.configs import rmsc04
from abides_markets.utils import config_add_agents

from train_market_maker_ppo import MarketMakingEnv


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "independent_ppo"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_AGENTS = 3
TRAIN_TIMESTEPS = 256  # A functional test, not enough for research conclusions
BASE_SEED = 42


class IdentifiedGymAgent(FinancialGymAgent):
    """Keep track of which of the three market makers caused a kernel pause."""

    def act_on_wakeup(self):
        raw_state = super().act_on_wakeup()
        return {"agent_id": self.id, "raw_state": raw_state}


def train_or_load_independent_models():
    """Each policy is trained from scratch with its own random seed."""
    models = []
    for index in range(N_AGENTS):
        path = MODEL_DIR / f"ppo_agent_{index + 1}.zip"
        if path.exists():
            print(f"Loading existing independent model {index + 1}: {path}", flush=True)
            model = PPO.load(str(path), device="cpu")
        else:
            print(f"Training independent PPO model {index + 1}...", flush=True)
            env = MarketMakingEnv()
            try:
                model = PPO(
                    "MlpPolicy",
                    env,
                    verbose=0,
                    n_steps=32,
                    batch_size=32,
                    n_epochs=2,
                    learning_rate=0.0003,
                    seed=BASE_SEED + index,
                    device="cpu",
                )
                model.learn(total_timesteps=TRAIN_TIMESTEPS)
                model.save(str(path))
                print(f"Saved model: {path}", flush=True)
            finally:
                env.close()
        models.append(model)
    return models


def run_shared_market(models):
    print("\nStarting ONE ABIDES market with THREE PPO traders...", flush=True)

    config = rmsc04.build_config(
        seed=2026,
        end_time="09:45:00",
        num_noise_agents=100,
        num_value_agents=20,
        num_momentum_agents=4,
        log_orders=False,
        book_logging=False,
    )

    # Non-learning background market makers remain in rmsc04 at this stage.
    # Each new experimental trader has a different account and agent ID.
    agents = []
    for index in range(N_AGENTS):
        agents.append(
            IdentifiedGymAgent(
                id=len(config["agents"]) + index,
                symbol="ABM",
                starting_cash=10_000_000,
                name=f"INDEPENDENT_PPO_{index + 1}",
                type="IndependentPPO",
                first_interval=str_to_ns("00:05:00"),
                wakeup_interval_generator=ConstantTimeGenerator(
                    step_duration=str_to_ns("60s")
                ),
                state_buffer_length=4,
                market_data_buffer_length=5,
                random_state=np.random.RandomState(900 + index),
                log_orders=False,
            )
        )

    config = config_add_agents(
        config, agents, np.random.RandomState(1000)
    )

    kernel = Kernel(
        agents=config["agents"],
        random_state=config["random_state_kernel"],
        start_time=config["start_time"],
        stop_time=config["stop_time"],
        agent_latency_model=config["agent_latency_model"],
        default_computation_delay=config["default_computation_delay"],
        oracle=config["oracle"],
        runner_hook=agents[0],
    )

    # Reuse the confirmed observation and action mapping from the
    # single-agent MarketMakingEnv. No additional simulations are started.
    adapters = [MarketMakingEnv() for _ in range(N_AGENTS)]
    by_id = {agent.id: index for index, agent in enumerate(agents)}
    actions_taken = [Counter() for _ in agents]
    cumulative_rewards = [0.0 for _ in agents]
    decision_counts = [0 for _ in agents]
    last_holdings = [0 for _ in agents]

    kernel.initialize()
    try:
        event = kernel.runner()
        while not event["done"]:
            payload = event["result"]
            agent_id = payload["agent_id"]
            index = by_id[agent_id]
            raw_state = payload["raw_state"]
            adapter = adapters[index]

            # Each model sees only its own inventory and the public book.
            obs = adapter.raw_state_to_state(deepcopy(raw_state))
            info = adapter.raw_state_to_info(deepcopy(raw_state))
            reward = adapter.raw_state_to_reward(deepcopy(raw_state))

            action, _ = models[index].predict(obs, deterministic=True)
            action = int(np.asarray(action).item())
            action_commands = adapter._map_action_space_to_ABIDES_SIMULATOR_SPACE(
                action
            )

            actions_taken[index][adapter.ACTION_NAMES[action]] += 1
            decision_counts[index] += 1
            cumulative_rewards[index] += float(reward)
            last_holdings[index] = int(info["holdings"])

            print(
                f"PPO {index + 1} | "
                f"decision {decision_counts[index]:02d} | "
                f"{adapter.ACTION_NAMES[action]:20s} | "
                f"inventory={int(info['holdings']):+d} | "
                f"spread={info['spread']}",
                flush=True,
            )

            # ABIDES executes the submitted action and advances ALL agents
            # until one of the three market makers next requests a decision.
            event = kernel.runner((agents[index], action_commands))
    finally:
        kernel.terminate()

    print("\n=== THREE-AGENT SHARED-MARKET SUMMARY ===")
    for index, agent in enumerate(agents):
        print(
            f"Agent {index + 1}: "
            f"decisions={decision_counts[index]}, "
            f"last_seen_inventory={last_holdings[index]}, "
            f"total_reward={cumulative_rewards[index]:.6f}, "
            f"filled_order_events={len(agent.episode_executed_orders)}"
        )
        print(f"    Actions: {dict(actions_taken[index])}")

    if all(count > 0 for count in decision_counts):
        print("\nPASS: all three independent policies acted in ONE shared ABIDES market.")
    else:
        raise RuntimeError("At least one policy did not act in the shared market")


if __name__ == "__main__":
    policies = train_or_load_independent_models()
    run_shared_market(policies)
