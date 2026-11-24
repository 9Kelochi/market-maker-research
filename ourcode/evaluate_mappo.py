"""
Evaluate the cooperative MAPPO prototype in the SAME shock harness as independent/shared PPO.

Reuses evaluate_agents.run_one, so the market, observer, shock agent, seeds and metrics are
identical. The shared actor is wrapped so run_one can call it like an SB3 model.
No centralized critic is needed at execution time (decentralized execution).

From repo root:
    uv run --with stable-baselines3 python ourcode/evaluate_mappo.py --seeds 1 2 3 --shock-size 20000
Options:
    --sample           sample actions from the policy instead of argmax (matches training)
    --with-passive     also run passive baselines (otherwise reuse them from the IPPO folder)
    --checkpoint PATH  default models/cooperative_mappo/team_actor_critic.pt
    --tag NAME         output folder: results/NAME_shock_SIZE  (default: mappo)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config as C
import evaluate_agents as evaluation
import evaluate_shared_ippo as shared
from train_cooperative_mappo import SharedActorCentralCritic, normalize_observation


class MappoPolicy:
    """Gives the shared MAPPO actor an SB3-style predict() interface."""

    def __init__(self, actor, deterministic=True):
        self.actor = actor
        self.default_deterministic = deterministic

    def predict(self, obs, deterministic=None):
        # evaluate_agents.run_one always passes deterministic=True, which would silently
        # override the --sample flag. The policy's own setting wins; the argument is ignored.
        det = self.default_deterministic
        x = torch.as_tensor(normalize_observation(obs))
        with torch.no_grad():
            logits = self.actor(x)
            if det:
                action = int(torch.argmax(logits).item())
            else:
                action = int(torch.distributions.Categorical(logits=logits).sample().item())
        return np.array(action), None


def load_actor(path):
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SharedActorCentralCritic()
    model.actor.load_state_dict(ckpt["actor"])
    model.actor.eval()
    info = {k: ckpt.get(k) for k in ("training_episodes", "seed", "reward_mode", "prototype")}
    print(f"Loaded actor from {path}  (checkpoint info: {info})", flush=True)
    return model.actor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--shock-size", type=int, default=C.SHOCK_SIZE)
    ap.add_argument("--checkpoint", type=Path,
                    default=C.ROOT / "models" / "cooperative_mappo" / "team_actor_critic.pt")
    ap.add_argument("--tag", default="mappo")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--with-passive", action="store_true")
    args = ap.parse_args()
    C.SHOCK_SIZE = args.shock_size

    actor = load_actor(args.checkpoint)
    policy = MappoPolicy(actor, deterministic=not args.sample)
    models = [policy] * C.N_PPO          # parameter sharing: same actor for all 3 agents

    modes = [True] + ([False] if args.with_passive else [])
    rows = []
    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===", flush=True)
        for use_ppo in modes:
            for shock_on in (False, True):
                print(f"  policy={'MAPPO' if use_ppo else 'passive'} shock={'ON' if shock_on else 'OFF'}",
                      flush=True)
                # Re-seed the sampling RNG so the shock-off and shock-on runs of a seed draw the SAME
                # actions until the shock; otherwise the pair differs before t=shock and the paired
                # comparison is contaminated by sampling noise.
                torch.manual_seed(10_000 + seed)
                rows.extend(evaluation.run_one(models=models, use_ppo=use_ppo,
                                               shock_on=shock_on, seed=seed))
    df = pd.DataFrame(rows)

    # sanity check: a collapsed policy (one action almost always) makes the comparison meaningless
    acts = df[(df.policy == "ppo") & df.who.str.startswith("PPO")].action.value_counts(normalize=True)
    print("\nAction shares used in evaluation:\n" + acts.round(3).to_string())
    if len(acts) and acts.iloc[0] > 0.9:
        print(f"WARNING: {acts.index[0]} is {acts.iloc[0]:.0%} of all decisions; this policy has collapsed to "
              "(nearly) one action. Results describe that action, not a learned strategy.")

    out = C.RESULTS_DIR / f"{args.tag}_shock_{args.shock_size}"
    path = shared.save_results(df, out)
    try:
        shared.generate_graph(path, out)
    except Exception as e:                # the plot is optional; never lose the CSVs
        print(f"(plot skipped: {e})")
    print("\nMAPPO evaluation completed. NOTE: policy label stays 'ppo' in the CSV; "
          f"the folder name ({out.name}) identifies it.")


if __name__ == "__main__":
    main()