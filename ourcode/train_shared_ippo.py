"""
With this script, all three agents participate in the same market during
training, but only one agent learns at a time. For example, Agent 1 trains
while Agents 2 and 3 continue trading using their existing models. Once
Agent 1 finishes, Agent 2 begins training against the other two agents.
The same process is repeated for Agent 3, completing one training round.

From the repository root:
    uv run --with stable-baselines3 python ourcode/train_shared_ippo.py --timesteps 128 --rounds 1

"""

import argparse
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
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
START_MODELS = ROOT / "models" / "independent_ppo"
TRAINED_MODELS = ROOT / "models" / "shared_ippo"
N_AGENTS = 3


class NamedGymAgent(FinancialGymAgent):
    """Tell the outer trainer which agent is requesting a decision."""

    def act_on_wakeup(self):
        return {"agent_id": self.id, "raw_state": super().act_on_wakeup()}


def current_model_path(index):
    name = f"ppo_agent_{index + 1}.zip"
    trained = TRAINED_MODELS / name
    base = START_MODELS / name
    if trained.exists():
        return trained
    if base.exists():
        return base
    raise FileNotFoundError(
        f"Missing {base}. Run ourcode/three_ppo_agents.py first."
    )


class SharedMarketTrainingEnv(gym.Env):
    """One learner + two frozen PPO peers, all trading in one ABIDES kernel.

    The selected learner receives only its own Gym observation and reward.
    Other agents act autonomously in the exact same limit order book.
    """

    metadata = {"render_modes": []}

    def __init__(self, learner_index, peer_model_paths):
        super().__init__()
        if learner_index not in range(N_AGENTS):
            raise ValueError("learner_index must be 0, 1, or 2")

        self.learner_index = learner_index
        self.adapters = [MarketMakingEnv() for _ in range(N_AGENTS)]
        self.action_space = self.adapters[0].action_space
        self.observation_space = self.adapters[0].observation_space
        self.peers = {}
        for index, path in peer_model_paths.items():
            if index != learner_index:
                self.peers[index] = PPO.load(str(path), device="cpu")

        self.kernel = None
        self.kernel_running = False
        self.agents = []
        self.by_id = {}
        self.last_observation = None
        self.episode_seed = None
        self.episode_actions = 0

    def _close_kernel(self):
        if self.kernel is not None and self.kernel_running:
            self.kernel.terminate()
            self.kernel_running = False
        self.kernel = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._close_kernel()
        self.episode_seed = int(self.np_random.integers(1, 2**31 - 1))
        self.episode_actions = 0

        for adapter in self.adapters:
            adapter._best_bid = None
            adapter._best_ask = None
            adapter._holdings = 0
            adapter.previous_marked_to_market = adapter.starting_cash

        config = rmsc04.build_config(
            seed=self.episode_seed,
            end_time="09:45:00",
            num_noise_agents=100,
            num_value_agents=20,
            num_momentum_agents=4,
            book_logging=False,
            log_orders=False,
        )
        base_id = len(config["agents"])
        self.agents = [
            NamedGymAgent(
                id=base_id + i,
                name=f"SHARED_TRAIN_PPO_{i + 1}",
                type="IndependentPPO",
                symbol="ABM",
                starting_cash=10_000_000,
                first_interval=str_to_ns("00:05:00"),
                wakeup_interval_generator=ConstantTimeGenerator(
                    step_duration=str_to_ns("60s")
                ),
                state_buffer_length=4,
                market_data_buffer_length=5,
                random_state=np.random.RandomState(900 + i),
                log_orders=False,
            )
            for i in range(N_AGENTS)
        ]
        config = config_add_agents(config, self.agents, np.random.RandomState(1000))
        self.by_id = {a.id: i for i, a in enumerate(self.agents)}
        self.kernel = Kernel(
            agents=config["agents"],
            random_state=config["random_state_kernel"],
            start_time=config["start_time"],
            stop_time=config["stop_time"],
            agent_latency_model=config["agent_latency_model"],
            default_computation_delay=config["default_computation_delay"],
            oracle=config["oracle"],
            runner_hook=self.agents[0],
        )
        self.kernel.initialize()
        self.kernel_running = True
        event = self.kernel.runner()
        event = self._advance_past_peers(event)

        if event["done"]:
            self._close_kernel()
            raise RuntimeError("ABIDES ended before the learner's first wakeup")

        raw = event["result"]["raw_state"]
        adapter = self.adapters[self.learner_index]
        info = adapter.raw_state_to_info(deepcopy(raw))
        adapter.previous_marked_to_market = float(info["marked_to_market"])
        obs = adapter.raw_state_to_state(deepcopy(raw))
        self.last_observation = obs
        return obs, {"seed": self.episode_seed, "holdings": info["holdings"]}

    def _advance_past_peers(self, event):
        """Advance one kernel and auto-act for peers until learner wakes."""
        while not event["done"]:
            payload = event["result"]
            index = self.by_id[payload["agent_id"]]
            if index == self.learner_index:
                break

            raw = payload["raw_state"]
            adapter = self.adapters[index]
            adapter.raw_state_to_info(deepcopy(raw))
            obs = adapter.raw_state_to_state(deepcopy(raw))
            action, _ = self.peers[index].predict(obs, deterministic=True)
            commands = adapter._map_action_space_to_ABIDES_SIMULATOR_SPACE(
                int(np.asarray(action).item())
            )
            event = self.kernel.runner((self.agents[index], commands))
        return event

    def step(self, action):
        if self.kernel is None or not self.kernel_running:
            raise RuntimeError("Call reset() before step()")
        adapter = self.adapters[self.learner_index]
        commands = adapter._map_action_space_to_ABIDES_SIMULATOR_SPACE(int(action))
        self.episode_actions += 1
        event = self.kernel.runner((self.agents[self.learner_index], commands))
        event = self._advance_past_peers(event)

        if event["done"]:
            # At the close, ABIDES does not issue another scheduled wakeup.
            # Calculate the remaining change in marked-to-market wealth
            # from this agent's final account and latest observed trade.
            agent = self.agents[self.learner_index]
            account = agent.get_internal_data()
            last_trade = agent.parsed_mkt_data.get("last_transaction")
            if last_trade is None:
                # Rare case: no valid transaction price was received.
                reward = 0.0
                has_final_price = False
            else:
                wealth = (
                    account["cash"] + account["holdings"] * last_trade
                )
                # Match the normalization in MarketMakingEnv's parent.
                steps_per_day = str_to_ns("06:30:00") / adapter.timestep_duration
                reward = (
                    (wealth - adapter.previous_marked_to_market)
                    / adapter.order_fixed_size
                    / steps_per_day
                )
                has_final_price = True
            self._close_kernel()
            return self.last_observation, float(reward), False, True, {
                "episode_actions": self.episode_actions,
                "seed": self.episode_seed,
                "has_final_price": has_final_price,
            }

        raw = event["result"]["raw_state"]
        obs = adapter.raw_state_to_state(deepcopy(raw))
        reward = float(adapter.raw_state_to_reward(deepcopy(raw)))
        info = adapter.raw_state_to_info(deepcopy(raw))
        self.last_observation = obs
        return obs, reward, False, False, {
            "seed": self.episode_seed,
            "holdings": int(info["holdings"]),
            "spread_cents": int(info["spread"]),
        }

    def close(self):
        self._close_kernel()
        for adapter in self.adapters:
            adapter.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    if args.timesteps <= 0 or args.rounds <= 0:
        parser.error("--timesteps and --rounds must be positive")

    TRAINED_MODELS.mkdir(parents=True, exist_ok=True)
    for round_number in range(1, args.rounds + 1):
        print(f"\n=== Shared-market training round {round_number} ===")
        for learner_index in range(N_AGENTS):
            paths = {
                i: current_model_path(i)
                for i in range(N_AGENTS)
                if i != learner_index
            }
            env = SharedMarketTrainingEnv(learner_index, paths)
            source_model = current_model_path(learner_index)
            print(
                f"Training Agent {learner_index + 1} in a shared market; "
                f"frozen peers: {list(paths.values())}",
                flush=True,
            )
            try:
                learner = PPO.load(str(source_model), env=env, device="cpu")
                learner.learn(
                    total_timesteps=args.timesteps,
                    reset_num_timesteps=False,
                )
                target = TRAINED_MODELS / f"ppo_agent_{learner_index + 1}"
                learner.save(str(target))
                print(f"Saved {target}.zip", flush=True)
            finally:
                env.close()

    print("\nShared-market alternating independent PPO training completed.")
    print("These are NEW training models, not yet cooperative MAPPO agents.")


if __name__ == "__main__":
    main()
