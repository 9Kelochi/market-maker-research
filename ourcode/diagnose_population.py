"""Paired shock effects in short windows right after the shock + noise-floor check.
Run from ourcode/:  uv run python analyze_effects.py [path/to/evaluation.csv]
Default path: ../results/shared_ippo_shock_5000/evaluation.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

path = Path(sys.argv[1]) if len(sys.argv) > 1 else C.RESULTS_DIR / "shared_ippo_shock_5000" / "evaluation.csv"
df = pd.read_csv(path)
obs = df[df.who == "OBSERVER"].copy()
T = C.SHOCK_SECONDS
WINDOWS = {"first 2 min": (T, T + 120), "first 5 min": (T, T + 300), "min 5-20": (T + 300, T + 1200)}


def window_stats(g, lo, hi):
    w = g[(g.sim_time_s >= lo) & (g.sim_time_s < hi)]
    pre = g[(g.sim_time_s >= T - 600) & (g.sim_time_s < T)]
    return {
        "mean_spread": pd.to_numeric(w.spread_cents, errors="coerce").mean(),
        "excess_vs_pre": pd.to_numeric(w.spread_cents, errors="coerce").mean()
        - pd.to_numeric(pre.spread_cents, errors="coerce").median(),
        "min_bid_depth": w.bid_depth.min(),
        "one_sided": w.one_sided.mean() if "one_sided" in w else np.nan,
    }


def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return np.nan, np.nan, n
    se = x.std(ddof=1) / np.sqrt(n)
    t = {2: 12.7, 3: 4.30, 4: 3.18, 5: 2.78}.get(n, 2.0 if n > 30 else 2.1)
    return x.mean(), t * se, n


rows = []
for (seed, pol, sh), g in obs.groupby(["seed", "policy", "shock"]):
    for name, (lo, hi) in WINDOWS.items():
        rows.append({"seed": seed, "policy": pol, "shock": sh, "window": name, **window_stats(g, lo, hi)})
r = pd.DataFrame(rows)

print("=== Paired effect = shock minus no-shock (same seed, same policy); mean +/- 95% CI half-width ===")
for pol in r.policy.unique():
    for name in WINDOWS:
        a = r[(r.policy == pol) & (r.window == name)]
        s = a[a.shock == 1].set_index("seed")
        n = a[a.shock == 0].set_index("seed")
        for metric in ["mean_spread", "one_sided", "min_bid_depth"]:
            d = (s[metric] - n[metric]).dropna()
            m, h, k = ci95(d)
            pos = int((d > 0).sum())
            print(f"{pol:8s} {name:12s} {metric:14s} mean={m:9.3f} +/- {h:8.3f}  (n={k}, positive in {pos}/{k})")
    print()

print("=== Noise floor: spread of the NO-SHOCK 20-min excess-spread across seeds ===")
nf = r[(r.shock == 0) & (r.window == "min 5-20")].groupby("policy").excess_vs_pre.agg(["mean", "std", "min", "max"])
print(nf.round(2).to_string())

print("\n=== Shock fill vs. pre-shock visible bid depth (is the shock capacity-limited?) ===")
fill = df[df.shock == 1].groupby(["seed", "policy"]).shock_filled.first()
pre = obs[(obs.shock == 1) & (obs.sim_time_s >= T - 60) & (obs.sim_time_s < T)].groupby(["seed", "policy"]).bid_depth.mean()
t = pd.concat([fill, pre.rename("bid_depth_pre")], axis=1)
t["fill_over_depth"] = t.shock_filled / t.bid_depth_pre
print(t.round(2).to_string())