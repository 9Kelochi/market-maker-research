
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from abides_markets.config_system import SimulationBuilder
from abides_markets.config_system.agent_configs import ImpactOrderAgentConfig
from abides_markets.simulation import run_simulation, ResultProfile

OUTPUT = Path("results_shock")
OUTPUT.mkdir(exist_ok=True)

SHOCK_MINUTE = 15
SHOCK_SIZE = 5000
SEED = 42


def run_experiment(name, with_shock):
    builder = (
        SimulationBuilder()
        .apply_template("rmsc04")
        .seed(SEED)
    )

    # Add a large sell order only in the shock experiment
    if with_shock:
        builder.enable_agent(
            ImpactOrderAgentConfig(
                order_time_offset="00:15:00",
                side="ASK",
                quantity=SHOCK_SIZE,
                order_type="MARKET",
            ),
            count=1,
        )

    config = builder.build()
    print(f"\nRunning {name}...", flush=True)

    result = run_simulation(
        config,
        profile=ResultProfile.QUANT
    )

    print(result.summary())

    market = result.markets["ABM"]
    df = market.l1_series.as_dataframe()

    # Align timestamps with the actual market opening
    opening = datetime.strptime(
        f"{config.market.date} {config.market.start_time}",
        "%Y%m%d %H:%M:%S"
    )
    open_ns = pd.Timestamp(opening).value

    df["minutes"] = (
        df["time_ns"].astype("int64") - open_ns
    ) / 60_000_000_000

    df["spread_usd"] = (
        df["ask_price_cents"] - df["bid_price_cents"]
    ) / 100.0

    df.to_csv(OUTPUT / f"{name}.csv", index=False)

    # Check actual execution of the shock order
    if with_shock:
        impact_ids = {
            a.agent_id for a in result.agents
            if "ImpactOrder" in a.agent_type
        }
        filled = sum(
            t.quantity for t in (market.trades or [])
            if t.aggressive_agent_id in impact_ids
        )
        print(f"Shock order requested: {SHOCK_SIZE} shares")
        print(f"Shock shares executed: {filled}")

    # Measurements during first minute after shock
    window = df[
        (df["minutes"] >= 15) &
        (df["minutes"] < 16)
    ]

    print(f"\n{name} - Minute 15 to 16")
    print(f"Maximum spread: ${window['spread_usd'].max():.4f}")
    print(f"Minimum bid quantity: {window['bid_qty'].min()}")

    return df


# Run both scenarios
normal = run_experiment("normal", False)
shock = run_experiment("shock", True)


# Plot comparisons close to the shock
for column, ylabel, filename in [
    ("spread_usd", "Bid-ask spread ($)", "spread_comparison.png"),
    ("bid_qty", "Shares at best bid", "bid_comparison.png"),
]:
    plt.figure(figsize=(11, 5))

    for name, df in [("Normal", normal), ("Shock", shock)]:
        section = df[
            (df["minutes"] >= 12) &
            (df["minutes"] <= 20)
        ].copy()

        # One-second median for a clearer chart
        section["second"] = (section["minutes"] * 60).astype(int)
        grouped = section.groupby("second")[column].median()

        plt.plot(
            grouped.index / 60,
            grouped.astype(float).to_numpy(),
            label=name,
            linewidth=1.2,
        )

    plt.axvline(
        SHOCK_MINUTE,
        color="red",
        linestyle="--",
        label="Shock at 15 min"
    )

    plt.xlabel("Minutes since market open")
    plt.ylabel(ylabel)
    plt.title(f"Normal vs Sell Shock: {ylabel}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / filename, dpi=150)
    plt.close()

print("\nExperiment complete! Results saved to results_shock/")
