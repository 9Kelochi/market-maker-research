
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

output = Path("results_volatile")

# Load existing simulation data
df = pd.read_csv(output / "market_data.csv")

# Divide the trading day into five-minute intervals
df["interval"] = (df["minutes"] // 5) * 5

# Calculate median quantities per interval
summary = df.groupby("interval", as_index=False)[
    ["bid_qty", "ask_qty"]
].median()

# Save summarized data
summary.to_csv(
    output / "liquidity_by_side.csv",
    index=False
)

# Plot buying and selling liquidity separately
plt.figure(figsize=(12, 5))

plt.plot(
    summary["interval"],
    summary["bid_qty"],
    label="Bid-side liquidity"
)

plt.plot(
    summary["interval"],
    summary["ask_qty"],
    label="Ask-side liquidity"
)

plt.xlabel("Elapsed time (minutes)")
plt.ylabel("Shares at best price")
plt.title("Bid vs Ask Liquidity")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    output / "liquidity_by_side.png",
    dpi=150
)

plt.show()
