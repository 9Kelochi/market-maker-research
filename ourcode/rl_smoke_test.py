
import gymnasium as gym
import abides_gym

# Create an RL environment connected to ABIDES
print("Creating environment...", flush=True)

env = gym.make(
    "markets-daily_investor-v0",
    background_config="rmsc04",
)

try:
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    # Start a new simulation
    observation, info = env.reset(seed=42)

    print("\nEnvironment initialized!")
    print("Initial observation:", observation)

    # Test three actions (not yet trained)
    action_names = {
        0: "BUY",
        1: "HOLD",
        2: "SELL",
    }

    for step in range(3):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        print(f"\nStep {step + 1}")
        print("Action:", action_names[int(action)])
        print("Reward:", reward)
        print("Observation:", observation)
        print("Terminated:", terminated)
        print("Truncated:", truncated)

        if terminated or truncated:
            break

finally:
    env.close()

print("\nRL smoke test completed!")
