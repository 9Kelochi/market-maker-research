from pathlib import Path
import pandas as pd

OUTPUT = Path("results_shock")

normal = pd.read_csv(OUTPUT / "normal.csv")
shock = pd.read_csv(OUTPUT / "shock.csv")

# Recompute spread just in case
for df in [normal, shock]:
    df["spread_usd"] = (df["ask_price_cents"] - df["bid_price_cents"]) / 100.0

# Windows
before_normal = normal[(normal["minutes"] >= 12) & (normal["minutes"] < 15)]
after_normal  = normal[(normal["minutes"] >= 15) & (normal["minutes"] < 18)]

before_shock = shock[(shock["minutes"] >= 12) & (shock["minutes"] < 15)]
after_shock  = shock[(shock["minutes"] >= 15) & (shock["minutes"] < 18)]

print("=== NORMAL ===")
print(f"Avg spread before shock window: ${before_normal['spread_usd'].mean():.4f}")
print(f"Avg spread after  shock window: ${after_normal['spread_usd'].mean():.4f}")
print(f"Peak spread after shock time:   ${after_normal['spread_usd'].max():.4f}")
print(f"Min bid qty after shock time:   {after_normal['bid_qty'].min()}")

print("\n=== SHOCK ===")
print(f"Avg spread before shock window: ${before_shock['spread_usd'].mean():.4f}")
print(f"Avg spread after  shock window: ${after_shock['spread_usd'].mean():.4f}")
print(f"Peak spread after shock time:   ${after_shock['spread_usd'].max():.4f}")
print(f"Min bid qty after shock time:   {after_shock['bid_qty'].min()}")

# Simple recovery rule:
# recovery happens when spread returns to <= pre-shock average * 1.2
baseline = before_shock["spread_usd"].mean()
threshold = baseline * 1.2

post = shock[shock["minutes"] >= 15].copy()
recovery = post[post["spread_usd"] <= threshold]

print("\n=== RECOVERY ===")
print(f"Shock pre-event average spread: ${baseline:.4f}")
print(f"Recovery threshold:             ${threshold:.4f}")

if len(recovery) > 0:
    recovery_time = recovery["minutes"].iloc[0] - 15
    print(f"Estimated recovery time:        {recovery_time:.4f} minutes")
else:
    print("Estimated recovery time:        Not recovered in observed window")