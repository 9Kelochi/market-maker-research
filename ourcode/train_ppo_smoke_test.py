import gymnasium as gym
import abides_gym
from stable_baselines3 import PPO

env = gym.make(
    "markets-daily_investor-v0",
    background_config="rmsc04",
    mkt_close="09:45:00",
    timestep_duration="60s",
)

try:
    print("Starting PPO training...", flush=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=16,
        batch_size=16,
        n_epochs=2,
        learning_rate=0.0003,
        seed=42,
        device="cpu",
    )

    model.learn(total_timesteps=64)

    model.save("ppo_daily_investor_smoke_test")
    print("PPO training completed and model saved!")

finally:
    env.close()
