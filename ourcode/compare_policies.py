"""
Compare policy families (e.g. shared IPPO vs cooperative MAPPO) from their evaluation.csv files.

    uv run python ourcode/compare_policies.py results/shared_ippo_shock_20000 results/mappo_shock_20000

For each folder prints, for the PPO-condition runs only:
  1. paired shock effects (shock minus no-shock, same seed) with 95% CI
  2. action synchrony: how often all 3 agents pick the same action in the same minute,
     compared with the rate expected by chance if the agents acted independently
     (given each agent's own action frequencies). Excess synchrony is the thing to look for.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

T = C.SHOCK_SECONDS
WIN = {"first 2 min": (T, T + 120), "first 5 min": (T, T + 300), "min 5-20": (T + 300, T + 1200)}
TCRIT = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26,
         15: 2.14, 20: 2.09, 30: 2.05}


def tcrit(n):
    for k in sorted(TCRIT):
        if n <= k:
            return TCRIT[k]
    return 1.98


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan, len(x)
    return x.mean(), tcrit(len(x)) * x.std(ddof=1) / np.sqrt(len(x)), len(x)


def window_metrics(g, lo, hi):
    w = g[(g.sim_time_s >= lo) & (g.sim_time_s < hi)]
    return {
        "mean_spread": pd.to_numeric(w.spread_cents, errors="coerce").mean(),
        "min_bid_depth": w.bid_depth.min() if "bid_depth" in w else np.nan,
    }


def paired_effects(df):
    obs = df[(df.who == "OBSERVER") & (df.policy == "ppo")]
    rows = []
    for (seed, shock), g in obs.groupby(["seed", "shock"]):
        for name, (lo, hi) in WIN.items():
            rows.append({"seed": seed, "shock": shock, "window": name, **window_metrics(g, lo, hi)})
    r = pd.DataFrame(rows)
    print("  Paired shock effect (shock - no-shock), mean +/- 95% CI")
    for name in WIN:
        a = r[r.window == name]
        s, n = a[a.shock == 1].set_index("seed"), a[a.shock == 0].set_index("seed")
        for metric in ("mean_spread", "min_bid_depth"):
            d = (s[metric] - n[metric]).dropna()
            m, h, k = ci(d)
            print(f"    {name:12s} {metric:14s} {m:9.3f} +/- {h:7.3f}   (n={k}, positive in {int((d > 0).sum())}/{k})")


def synchrony(df, bin_s=60):
    p = df[(df.policy == "ppo") & df.who.str.startswith("PPO")].copy()
    p["bin"] = (p.sim_time_s // bin_s).astype(int)
    wide = p.pivot_table(index=["seed", "shock", "bin"], columns="who", values="action", aggfunc="first").dropna()
    if wide.empty:
        print("  (no complete 3-agent minutes)"); return
    acts = wide.to_numpy()
    all_same = (acts == acts[:, [0]]).all(axis=1)
    all_withdraw = (acts == "WITHDRAW_BOTH").all(axis=1)
    # chance rate if agents acted independently with their own action frequencies
    cats = np.unique(acts)
    probs = np.array([[np.mean(acts[:, i] == c) for c in cats] for i in range(acts.shape[1])])
    chance_same = float(np.sum(np.prod(probs, axis=0)))
    chance_wd = float(np.prod([np.mean(acts[:, i] == "WITHDRAW_BOTH") for i in range(acts.shape[1])]))
    shock = wide.index.get_level_values("shock").to_numpy()
    post = wide.index.get_level_values("bin").to_numpy() * bin_s >= T
    print(f"  Action synchrony over {len(wide)} agent-minutes")
    print(f"    all 3 agents same action : {all_same.mean():.3f}   expected by chance: {chance_same:.3f}   excess: {all_same.mean() - chance_same:+.3f}")
    print(f"    all 3 WITHDRAW_BOTH      : {all_withdraw.mean():.3f}   expected by chance: {chance_wd:.3f}   excess: {all_withdraw.mean() - chance_wd:+.3f}")
    for s_val, lab in ((0, "no shock"), (1, "shock")):
        m = post & (shock == s_val)
        if m.any():
            print(f"    after t={T}s, {lab:8s}: all-same {all_same[m].mean():.3f}, all-withdraw {all_withdraw[m].mean():.3f}  (n={int(m.sum())})")


def main():
    folders = [Path(a) for a in sys.argv[1:]] or [C.RESULTS_DIR / "shared_ippo_shock_20000"]
    for f in folders:
        path = f / "evaluation.csv"
        print("\n" + "=" * 78 + f"\n{f.name}\n" + "=" * 78)
        if not path.exists():
            print(f"  missing {path}"); continue
        df = pd.read_csv(path)
        fills = df[df.shock == 1].groupby(["seed", "policy"]).shock_filled.first() if "shock_filled" in df else None
        if fills is not None:
            print(f"  seeds: {df.seed.nunique()}   median shock fill: {fills.median():.0f}")
        paired_effects(df)
        synchrony(df)


if __name__ == "__main__":
    main()
