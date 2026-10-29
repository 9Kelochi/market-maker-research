
from pathlib import Path
import numpy as np
import pandas as pd

#df = pd.read_csv(Path("results_shock") / "shock.csv")
df = pd.read_csv(Path("results_shock") / "normal.csv")
df = df.sort_values("minutes").copy()

df["spread_usd"] = (
    df["ask_price_cents"] - df["bid_price_cents"]
) / 100

baseline = df.loc[
    (df["minutes"] >= 12) & (df["minutes"] < 15),
    "spread_usd"
].mean()

#threshold = baseline * 1.2
threshold = 0.0295
duration = 0.5  # 30 seconds in minutes

post = df[df["minutes"] >= 15].copy()

# Treat unavailable quotes or excessive spread as unstable
unstable = (
    post["spread_usd"].isna()
    | (post["spread_usd"] > threshold)
)

bad_times = post.loc[unstable, "minutes"].to_numpy()

print(f"Pre-shock spread: ${baseline:.4f}")
print(f"Recovery threshold: ${threshold:.4f}")

if len(bad_times) == 0:
    print("No threshold breach observed after shock.")
else:
    first_bad = bad_times[0]

    good_times = post.loc[
        (~unstable) & (post["minutes"] > first_bad),
        "minutes"
    ].to_numpy()

    recovery = None
    end_time = post["minutes"].max()

    for t in good_times:
        if t + duration > end_time:
            break

        next_bad_idx = np.searchsorted(
            bad_times, t, side="left"
        )

        next_bad = (
            bad_times[next_bad_idx]
            if next_bad_idx < len(bad_times)
            else float("inf")
        )

        if next_bad >= t + duration:
            recovery = t
            break

    if recovery is None:
        print("No sustained 30-second recovery observed.")
    else:
        print(
            f"Recovery began: "
            f"{(recovery - 15) * 60:.2f} seconds "
            f"after shock"
        )
        print(
            f"Recovery confirmed: "
            f"{(recovery + duration - 15) * 60:.2f} "
            f"seconds after shock"
        )
