"""
Plot shock reaction from results/multi_agent_ppo_evaluation.csv (4 conditions x seeds).
Run: uv run python ourcode/plot_shock_reaction.py
"""
import matplotlib.pyplot as plt
import pandas as pd

import config as C
import metrics

CSV = C.RESULTS_DIR / "multi_agent_ppo_evaluation.csv"
OUT = C.RESULTS_DIR / "ppo_shock_reaction.png"
BIN = 30  # seconds


def main():
    if not CSV.exists():
        raise FileNotFoundError(f"{CSV} missing. Run evaluate_agents.py first.")
    df = pd.read_csv(CSV)
    t_shock_min = C.SHOCK_SECONDS / 60
    n_seeds = df.seed.nunique()

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(10, 13), sharex=True, dpi=200)
    fig.suptitle(f"PPO market makers vs. {C.SHOCK_SIZE}-share sell shock "
                 f"(mean ± std over {n_seeds} seeds)", fontweight="bold")

    #Spread (observer) per condition
    obs = df[df.who == "OBSERVER"].copy()
    obs["spread_cents"] = pd.to_numeric(obs.spread_cents, errors="coerce")
    obs["bin"] = (obs.sim_time_s // BIN) * BIN / 60
    styles = {("passive", 0): ("tab:gray", "-"), ("passive", 1): ("tab:gray", "--"),
              ("ppo", 0): ("tab:blue", "-"), ("ppo", 1): ("tab:red", "--")}
    for (pol, sh), g in obs.groupby(["policy", "shock"]):
        per_seed = g.groupby(["seed", "bin"]).spread_cents.mean().unstack(0)
        m, s = per_seed.mean(axis=1), per_seed.std(axis=1).fillna(0)
        col, ls = styles[(pol, sh)]
        a1.plot(m.index, m, color=col, ls=ls, label=f"{pol}, {'shock' if sh else 'no shock'}")
        a1.fill_between(m.index, m - s, m + s, color=col, alpha=0.12)
    a1.set_ylabel("Bid-ask spread (cents)")
    a1.set_title("Market spread (observer, 5 s sampling)")

    # Withdrawn fraction (PPO runs only)
    for sh, col in [(0, "tab:blue"), (1, "tab:red")]:
        g = df[(df.policy == "ppo") & (df.shock == sh)]
        wf = pd.concat([metrics.withdrawn_fraction(x) for _, x in g.groupby("seed")], axis=1)
        m = wf.mean(axis=1)
        a2.plot(m.index / 60, m, color=col, marker="o", ms=3,
                label="shock" if sh else "no shock")
    a2.set_ylabel("Fraction of PPO agents\nwithdrawn / no open orders")
    a2.set_ylim(-0.05, 1.05)
    a2.set_title("Simultaneous withdrawal indicator")

    # Inventory, first seed, shocked PPO run
    first = df.seed.min()
    g = df[(df.seed == first) & (df.policy == "ppo") & (df.shock == 1) & df.who.str.startswith("PPO")]
    for who, x in g.groupby("who"):
        a3.plot(x.sim_time_s / 60, x.inventory, marker="s", ms=3, label=who)
    a3.axhline(0, color="gray", lw=0.8)
    a3.set_ylabel("Inventory (shares)")
    a3.set_xlabel("Minutes since market open")
    a3.set_title(f"PPO inventory, shocked run (seed {first})")

    for ax in (a1, a2, a3):
        ax.axvline(t_shock_min, color="black", ls="--", lw=1.2, label="Shock")
        ax.grid(True, ls=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, bbox_inches="tight")
    print(f"[SUCCESS] saved {OUT}")


if __name__ == "__main__":
    main()