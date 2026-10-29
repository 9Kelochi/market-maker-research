
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from abides_markets.config_system import SimulationBuilder
from abides_markets.simulation import (
    run_simulation,
    ResultProfile,
)

# Create an output folder
output = Path("results_volatile")
output.mkdir(exist_ok=True)

# Configure the market
config = (
    SimulationBuilder()
    .apply_template("volatile_day")
    .seed(42)
    .build()
)

# Request detailed order-book history
print("Starting simulation...", flush=True)

result = run_simulation(
    config,
    profile=ResultProfile.QUANT
)

print("\n=== MARKET SUMMARY ===")
print(result.summary())

# Extract Level 1 market data
market = result.markets["ABM"]

if market.l1_series is None:
    raise RuntimeError("No L1 market data returned")

df = market.l1_series.as_dataframe()

# Calculate elapsed time in minutes
df["minutes"] = (
    df["time_ns"] - df["time_ns"].iloc[0]
) / 60_000_000_000

# Spread = Best Ask - Best Bid
df["spread_usd"] = (
    df["ask_price_cents"] - df["bid_price_cents"]
) / 100.0

# Shares available at the best bid and ask
df["best_level_depth"] = (
    df["bid_qty"] + df["ask_qty"]
)

# Export L1 measurements
df.to_csv(output / "market_data.csv", index=False)

# Export agent results
agents = pd.DataFrame([
    {
        "agent_id": a.agent_id,
        "agent_type": a.agent_type,
        "pnl_usd": a.pnl_cents / 100,
    }
    for a in result.agents
])

agents.to_csv(output / "agent_results.csv", index=False)

# Print important metrics
print("\n=== CALCULATED METRICS ===")
print(f"Average spread: ${df['spread_usd'].mean():.4f}")
print(f"Average best-level depth: {df['best_level_depth'].mean():.2f}")
print(f"L1 observations: {len(df):,}")

# Graph 1: Bid-ask spread
spread = df.dropna(subset=["spread_usd"])

plt.figure(figsize=(10, 5))
plt.plot(
    spread["minutes"].to_numpy(dtype=float),
    spread["spread_usd"].to_numpy(dtype=float),
    linewidth=0.8
)
plt.xlabel("Elapsed time (minutes)")
plt.ylabel("Bid-ask spread ($)")
plt.title("Market Bid-Ask Spread Over Time")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output / "spread.png", dpi=150)
plt.close()

# Graph 2: Best-level liquidity
depth = df.dropna(subset=["best_level_depth"])

plt.figure(figsize=(10, 5))
plt.plot(
    depth["minutes"].to_numpy(dtype=float),
    depth["best_level_depth"].to_numpy(dtype=float),
    linewidth=0.8
)
plt.xlabel("Elapsed time (minutes)")
plt.ylabel("Available shares")
plt.title("Best-Level Order Book Liquidity")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output / "liquidity.png", dpi=150)
plt.close()

print("\nResults saved in the results folder!")
