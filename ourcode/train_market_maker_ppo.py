
from pathlib import Path

from gymnasium import spaces
from stable_baselines3 import PPO

from abides_gym.envs.markets_daily_investor_environment_v0 import (
    SubGymMarketsDailyInvestorEnv_v0,
)


class MarketMakingEnv(SubGymMarketsDailyInvestorEnv_v0):
    """
    Single-agent PPO market-making prototype.

    The agent can place bid/ask limit orders,
    cancel existing orders, or keep its quotes.
    """

    ACTION_NAMES = (
        "WITHDRAW_BOTH",
        "QUOTE_BOTH_TIGHT",
        "QUOTE_BOTH_WIDE",
        "QUOTE_BID_ONLY",
        "QUOTE_ASK_ONLY",
        "KEEP_CURRENT_ORDERS",
    )

    def __init__(self):
        super().__init__(
            background_config="rmsc04",
            mkt_close="09:45:00",
            first_interval="00:05:00",
            timestep_duration="60s",
            order_fixed_size=10,
            starting_cash=10_000_000,
            debug_mode=True,
        )

        self.action_space = spaces.Discrete(6)

        self._best_bid = None
        self._best_ask = None
        self._holdings = 0

        self.max_abs_inventory = 50

    def reset(self, *, seed=None, options=None):
        self._best_bid = None
        self._best_ask = None
        self._holdings = 0

        self.previous_marked_to_market = self.starting_cash

        return super().reset(seed=seed, options=options)

    def raw_state_to_info(self, raw_state):
        info = super().raw_state_to_info(raw_state)

        self._best_bid = int(info["best_bid"])
        self._best_ask = int(info["best_ask"])
        self._holdings = int(info["holdings"])

        return info

    def _map_action_space_to_ABIDES_SIMULATOR_SPACE(self, action):
        action = int(action)

        # Keep existing orders unchanged.
        if action == 5:
            return []

        # Cancel existing quotes.
        orders = [{"type": "CCL_ALL"}]

        if action == 0:
            return orders

        if action not in (1, 2, 3, 4):
            raise ValueError(f"Invalid action: {action}")

        bid = self._best_bid
        ask = self._best_ask

        if bid is None or ask is None:
            return orders

        if bid <= 0 or ask <= bid:
            return orders

        # Prices are represented in cents.
        offset = 2 if action == 2 else 0

        bid_price = bid - offset
        ask_price = ask + offset

        qty = self.order_fixed_size

        # Place a buy limit order.
        if action in (1, 2, 3):
            if (
                bid_price > 0
                and self._holdings + qty <= self.max_abs_inventory
            ):
                orders.append({
                    "type": "LMT",
                    "direction": "BUY",
                    "size": qty,
                    "limit_price": bid_price,
                })

        # Place a sell limit order.
        if action in (1, 2, 4):
            if self._holdings - qty >= -self.max_abs_inventory:
                orders.append({
                    "type": "LMT",
                    "direction": "SELL",
                    "size": qty,
                    "limit_price": ask_price,
                })

        return orders


def main():
    print("Creating PPO market-making environment...", flush=True)

    env = MarketMakingEnv()

    output = Path(__file__).resolve().parent.parent / "models"
    output.mkdir(parents=True, exist_ok=True)

    try:
        print("Action space:", env.action_space)
        print("Observation space:", env.observation_space)

        print("\nTesting limit-order quoting...", flush=True)

        observation, info = env.reset(seed=42)

        observation, reward, terminated, truncated, info = (
            env.step(1)
        )

        print("Limit-order action executed.")
        print("Market spread:", info["spread"])
        print("Inventory:", info["holdings"])
        print("Reward:", reward)

        print("\nStarting PPO training...", flush=True)

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=32,
            batch_size=32,
            n_epochs=2,
            learning_rate=0.0003,
            seed=42,
            device="cpu",
        )

        model.learn(total_timesteps=128)

        model_path = output / "market_maker_ppo_smoke_test"
        model.save(str(model_path))

        print("\nPPO training completed!")
        print("Model saved to:", model_path)

        print("\nTesting trained agent...", flush=True)

        observation, info = env.reset(seed=123)

        for step in range(10):
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            action = int(action)

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            print(
                f"Step {step + 1}: "
                f"{env.ACTION_NAMES[action]} | "
                f"Inventory: {info['holdings']} | "
                f"Spread: {info['spread']} | "
                f"Reward: {reward:.6f}"
            )

            if terminated or truncated:
                break

        print("\nMarket-making PPO prototype completed!")

    finally:
        env.close()


if __name__ == "__main__":
    main()
