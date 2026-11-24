"""Cooperative MAPPO-style prototype for the existing ABIDES-NG market-maker project.

Three market makers interact with ONE ABIDES kernel. Each actor decision sees
only that agent's 7-feature market/inventory observation. A centralized critic
receives the latest observed state of all three agents (plus availability and
active-agent indicators). Agents share policy parameters and a TEAM P&L reward.

This is an event-driven, parameter-sharing CTDE PPO prototype: wakeups are
asynchronous and the critic's other-agent observations can be stale. It is NOT
an implementation of the complete standard synchronous MAPPO algorithm.

From the ABIDES-NG repository root:
    uv run --with stable-baselines3 python ourcode/train_cooperative_mappo.py --episodes 3

Outputs:
    models/cooperative_mappo/team_actor_critic.pt
"""

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from abides_core import Kernel
from abides_core.messaging.generators import ConstantTimeGenerator
from abides_core.utils import str_to_ns
from abides_gym.experimental_agents.financial_gym_agent import FinancialGymAgent
from abides_markets.configs import rmsc04
from abides_markets.utils import config_add_agents

from train_market_maker_ppo import MarketMakingEnv


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "cooperative_mappo" / "team_actor_critic.pt"
N_AGENTS = 3
N_ACTIONS = 6
OBS_DIM = 7
CENTRAL_DIM = N_AGENTS * OBS_DIM + N_AGENTS + N_AGENTS
STARTING_CASH_CENTS = 10_000_000
SHARES_PER_ORDER = 10
STEP_SECONDS = 60
# Match the per-agent reward scale used by the earlier PPO experiments.
REWARD_SCALE = N_AGENTS * SHARES_PER_ORDER * (6.5 * 3600 / STEP_SECONDS)
OBS_SCALE = np.array([50, 1, 100, 100, 100, 100, 100], dtype=np.float32)


class IdentifiedGymAgent(FinancialGymAgent):
    """Identify which agent caused the ABIDES runner to pause."""

    def act_on_wakeup(self):
        return {"agent_id": self.id, "raw_state": super().act_on_wakeup()}


class SharedActorCentralCritic(nn.Module):
    """Decentralized shared actor, centralized critic (CTDE)."""

    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(OBS_DIM, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, N_ACTIONS),
        )
        self.critic = nn.Sequential(
            nn.Linear(CENTRAL_DIM, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    @torch.no_grad()
    def choose_action(self, local_observation, central_state):
        local = torch.as_tensor(local_observation, dtype=torch.float32)
        global_state = torch.as_tensor(central_state, dtype=torch.float32)
        distribution = Categorical(logits=self.actor(local))
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        value = self.critic(global_state).squeeze(-1)
        return int(action.item()), float(log_prob.item()), float(value.item())


def normalize_observation(raw_observation):
    values = np.asarray(raw_observation, dtype=np.float32).reshape(-1)
    if values.size != OBS_DIM:
        raise ValueError(f"Expected {OBS_DIM} observation features; got {values.size}")
    values = np.nan_to_num(values / OBS_SCALE, nan=0.0, posinf=10.0, neginf=-10.0)
    return np.clip(values, -10.0, 10.0)


def central_observation(latest_observations, seen, active_agent):
    active = np.zeros(N_AGENTS, dtype=np.float32)
    active[active_agent] = 1.0
    return np.concatenate([latest_observations.flatten(), seen, active]).astype(np.float32)


def last_trade_price(raw_state):
    """Use the most recent transaction price (ABIDES prices are in cents)."""
    for state in reversed(raw_state):
        market_data = state.get("parsed_mkt_data", [])
        if market_data:
            price = market_data[-1].get("last_transaction")
            if price is not None and float(price) > 0:
                return float(price)
    return None


def terminal_trade_price(agents, previous_price):
    """Use the newest observed market-data price at simulation close."""
    latest_time, latest_price = -1, previous_price
    for agent in agents:
        data = agent.parsed_mkt_data
        price = data.get("last_transaction")
        timestamp = data.get("exchange_ts")
        if price is not None and float(price) > 0 and timestamp is not None:
            if int(timestamp) > latest_time:
                latest_time, latest_price = int(timestamp), float(price)
    return latest_price


def team_wealth(agents, last_price_cents):
    return sum(
        float(agent.get_holdings("CASH"))
        + float(agent.get_holdings("ABM")) * last_price_cents
        for agent in agents
    )


def build_simulation(seed):
    """Reuse the same short-market configuration as the shared-IPPO trainer."""
    config = rmsc04.build_config(
        seed=seed,
        end_time="09:45:00",
        num_noise_agents=100,
        num_value_agents=20,
        num_momentum_agents=4,
        book_logging=False,
        log_orders=False,
    )
    initial_id = len(config["agents"])
    agents = [
        IdentifiedGymAgent(
            id=initial_id + i,
            symbol="ABM",
            name=f"COOP_MAPPO_{i + 1}",
            type="CooperativeMAPPOPrototype",
            starting_cash=STARTING_CASH_CENTS,
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
    config = config_add_agents(config, agents, np.random.RandomState(1000))
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
    return kernel, agents


def play_episode(model, seed):
    """Sample on-policy transitions from three agents sharing one order book."""
    kernel, agents = build_simulation(seed)
    adapters = [MarketMakingEnv() for _ in agents]
    by_id = {agent.id: i for i, agent in enumerate(agents)}

    latest = np.zeros((N_AGENTS, OBS_DIM), dtype=np.float32)
    seen = np.zeros(N_AGENTS, dtype=np.float32)
    actions_taken = Counter()
    samples = []
    final_price = None

    kernel.initialize()
    try:
        event = kernel.runner()
        while not event["done"]:
            payload = event["result"]
            index = by_id[payload["agent_id"]]
            raw_state = payload["raw_state"]
            adapter = adapters[index]

            # raw_state_to_info updates the quote prices used by the action mapper.
            adapter.raw_state_to_info(deepcopy(raw_state))
            observation = normalize_observation(
                adapter.raw_state_to_state(deepcopy(raw_state))
            )
            latest[index] = observation
            seen[index] = 1.0
            global_state = central_observation(latest, seen, index)

            price_now = last_trade_price(raw_state)
            if price_now is None:
                raise RuntimeError("No recent trade price in ABIDES market data")
            wealth_before = team_wealth(agents, price_now)

            action, log_prob, value = model.choose_action(observation, global_state)
            commands = adapter._map_action_space_to_ABIDES_SIMULATOR_SPACE(action)
            actions_taken[adapter.ACTION_NAMES[action]] += 1

            next_event = kernel.runner((agents[index], commands))
            done = bool(next_event["done"])
            if done:
                price_after = terminal_trade_price(agents, price_now)
            else:
                price_after = last_trade_price(next_event["result"]["raw_state"])
                if price_after is None:
                    price_after = price_now
            wealth_after = team_wealth(agents, price_after)
            final_price = price_after

            # All three agents receive the SAME economic objective.
            reward = (wealth_after - wealth_before) / REWARD_SCALE
            samples.append({
                "local": observation.copy(),
                "central": global_state.copy(),
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "reward": float(reward),
                "done": done,
            })
            event = next_event
    finally:
        kernel.terminate()
        for adapter in adapters:
            adapter.close()

    if final_price is None:
        raise RuntimeError("No market-maker decisions occurred in this episode")
    pnl_dollars = (
        team_wealth(agents, final_price) - N_AGENTS * STARTING_CASH_CENTS
    ) / 100.0
    return samples, actions_taken, pnl_dollars


def generalized_advantages(samples, gamma=0.99, gae_lambda=0.95):
    """GAE over the event stream, with zero bootstrap at episode end."""
    advantage = 0.0
    for t in range(len(samples) - 1, -1, -1):
        current = samples[t]
        next_value = samples[t + 1]["value"] if t + 1 < len(samples) else 0.0
        live = 0.0 if current["done"] else 1.0
        delta = current["reward"] + gamma * live * next_value - current["value"]
        advantage = delta + gamma * gae_lambda * live * advantage
        current["advantage"] = advantage
        current["return"] = advantage + current["value"]


def ppo_update(model, optimizer, samples, epochs=3, batch_size=64, clip=0.2):
    """Clipped PPO policy loss plus centralized critic value regression."""
    local = torch.as_tensor(np.stack([s["local"] for s in samples]))
    central = torch.as_tensor(np.stack([s["central"] for s in samples]))
    actions = torch.as_tensor([s["action"] for s in samples], dtype=torch.long)
    old_logp = torch.as_tensor([s["log_prob"] for s in samples], dtype=torch.float32)
    returns = torch.as_tensor([s["return"] for s in samples], dtype=torch.float32)
    advantages = torch.as_tensor([s["advantage"] for s in samples], dtype=torch.float32)
    if len(samples) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    last_loss = 0.0
    for _ in range(epochs):
        order = torch.randperm(len(samples))
        for inds in order.split(batch_size):
            dist = Categorical(logits=model.actor(local[inds]))
            new_logp = dist.log_prob(actions[inds])
            ratio = torch.exp(new_logp - old_logp[inds])
            p1 = ratio * advantages[inds]
            p2 = ratio.clamp(1.0 - clip, 1.0 + clip) * advantages[inds]
            actor_loss = -torch.minimum(p1, p2).mean()
            predicted_value = model.critic(central[inds]).squeeze(-1)
            critic_loss = torch.nn.functional.mse_loss(predicted_value, returns[inds])
            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            last_loss = float(loss.detach().item())
    return last_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    if args.episodes < 1 or args.epochs < 1:
        parser.error("episodes and epochs must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = SharedActorCentralCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Training 3 cooperative PPO actors with one centralized critic")
    for episode in range(args.episodes):
        # Keep policy fixed while collecting each episode (on-policy PPO).
        samples, actions, pnl = play_episode(model, args.seed + episode)
        generalized_advantages(samples)
        loss = ppo_update(model, optimizer, samples, epochs=args.epochs)
        print(
            f"Episode {episode + 1}/{args.episodes} | "
            f"decisions={len(samples)} | team_PnL=${pnl:.2f} | "
            f"loss={loss:.5f} | actions={dict(actions)}",
            flush=True,
        )

    torch.save({
        "actor": model.actor.state_dict(),
        "critic": model.critic.state_dict(),
        "n_agents": N_AGENTS,
        "obs_dim": OBS_DIM,
        "central_dim": CENTRAL_DIM,
        "action_names": MarketMakingEnv.ACTION_NAMES,
        "prototype": True,
        "training_episodes": args.episodes,
        "seed": args.seed,
    }, MODEL_PATH)
    print(f"\nSaved cooperative MAPPO-style prototype: {MODEL_PATH}")
    print("This is a training smoke test, not evidence of strategy quality.")


if __name__ == "__main__":
    main()
