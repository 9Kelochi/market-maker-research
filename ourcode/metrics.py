"""Shared metrics. Spreads in CENTS """
import numpy as np
import pandas as pd

import config as C


def _obs_all(df):
    o = df[df["who"] == "OBSERVER"].copy()
    col = "spread_raw_cents" if "spread_raw_cents" in o else "spread_cents"
    o["spread_cents"] = pd.to_numeric(o[col], errors="coerce")
    return o.sort_values("sim_time_s")


def _obs(df):
    return _obs_all(df).dropna(subset=["spread_cents"])


def baseline_spread(obs, t_shock=C.SHOCK_SECONDS, window=C.BASELINE_WINDOW_S):
    pre = obs[(obs.sim_time_s >= t_shock - window) & (obs.sim_time_s < t_shock)]
    return float(pre.spread_cents.median()) if len(pre) else np.nan


def excess_spread_area(obs, baseline, t_shock=C.SHOCK_SECONDS, horizon=C.POST_HORIZON_S):
    """Integral of max(spread - baseline, 0) dt after the shock (cent-seconds).
    One-sided (NaN) samples are skipped"""
    post = obs[(obs.sim_time_s >= t_shock) & (obs.sim_time_s <= t_shock + horizon)]
    if len(post) < 2 or np.isnan(baseline):
        return np.nan
    excess = np.clip(post.spread_cents.to_numpy() - baseline, 0, None)
    t = post.sim_time_s.to_numpy()
    return float(np.sum(0.5 * (excess[1:] + excess[:-1]) * np.diff(t)))


def recovery_time(obs, baseline, tol_cents=2.0, hold=60, roll=30,
                  t_shock=C.SHOCK_SECONDS, horizon=C.POST_HORIZON_S):
    """Seconds after shock until rolling-median spread <= baseline + tol_cents for `hold` s."""
    post = obs[(obs.sim_time_s >= t_shock) & (obs.sim_time_s <= t_shock + horizon)]
    if post.empty or np.isnan(baseline):
        return np.nan
    s = post.set_index("sim_time_s").spread_cents
    step = max(float(np.median(np.diff(s.index))), 1.0)
    smooth = s.rolling(max(int(roll / step), 1), min_periods=1).median()
    ok = (smooth <= baseline + tol_cents).to_numpy()
    need = max(int(hold / step), 1)
    idx, run = smooth.index.to_numpy(), 0
    for k, flag in enumerate(ok):
        run = run + 1 if flag else 0
        if run >= need:
            return float(idx[k - need + 1] - t_shock)
    return np.nan


def withdrawn_fraction(df, bin_s=60):
    """fraction of PPO agents that chose WITHDRAW_BOTH or hold no open orders.
    NOTE: high even without a shock (agents often hold no quotes); compare shock vs no-shock."""
    p = df[df["who"].str.startswith("PPO")].copy()
    p["withdrawn"] = (p.action == "WITHDRAW_BOTH") | (p.n_open_orders == 0)
    p["bin"] = (p.sim_time_s // bin_s) * bin_s
    return p.groupby("bin").withdrawn.mean()


def summarize(df):
    out = []
    for (seed, policy, shock), g in df.groupby(["seed", "policy", "shock"]):
        allobs, obs = _obs_all(g), _obs(g)
        base = baseline_spread(obs)
        pre = allobs[(allobs.sim_time_s >= C.SHOCK_SECONDS - 60) & (allobs.sim_time_s < C.SHOCK_SECONDS)]
        post = allobs[(allobs.sim_time_s >= C.SHOCK_SECONDS)
                      & (allobs.sim_time_s <= C.SHOCK_SECONDS + C.POST_HORIZON_S)]
        wf = withdrawn_fraction(g)
        post_wf = wf[wf.index >= C.SHOCK_SECONDS]
        has_depth = "bid_depth" in allobs
        out.append({
            "seed": seed, "policy": policy, "shock": shock,
            "shock_filled": g["shock_filled"].iloc[0] if "shock_filled" in g else np.nan,
            "baseline_spread_cents": base,
            "peak_spread_cents": post.spread_cents.max(),
            "excess_spread_area_cent_s": excess_spread_area(obs, base),
            "recovery_time_s": recovery_time(obs, base),
            "pre_bid_depth": pre.bid_depth.mean() if has_depth else np.nan,
            "post_bid_depth": post.bid_depth.mean() if has_depth else np.nan,
            "post_one_sided_frac": post.one_sided.mean() if "one_sided" in post else np.nan,
            "min_bid_depth_post": (obs[obs.sim_time_s >= C.SHOCK_SECONDS].bid_depth.min()
                                   if "bid_depth" in obs else np.nan),
            "post_shock_withdrawn_frac": post_wf.mean() if policy == "ppo" else np.nan,
        })
    return pd.DataFrame(out)


def paired_effects(summary):
    """Shock minus no-shock, same seed and policy. This is the quantity to report."""
    cols = ["excess_spread_area_cent_s", "peak_spread_cents", "post_one_sided_frac",
            "post_bid_depth", "post_shock_withdrawn_frac"]
    a = summary[summary.shock == 1].set_index(["seed", "policy"])
    b = summary[summary.shock == 0].set_index(["seed", "policy"])
    d = (a[cols] - b[cols]).add_prefix("d_")
    d["shock_filled"] = a["shock_filled"]
    return d.reset_index()