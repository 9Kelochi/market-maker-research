
"""
Evaluate three independently trained PPO agents in one shared ABIDES-NG market.

This script uses the models trained by train_shared_ippo.py and evaluates
their behavior during normal market conditions and sudden sell shocks.

The experiment includes four scenarios:
1. Passive agents without a shock
2. Passive agents with a shock
3. PPO agents without a shock
4. PPO agents with a shock

We measure market spreads, order-book depth, liquidity withdrawal, and
recovery time using our existing evaluation and metrics scripts.

The script does not retrain the PPO agents or implement cooperative MAPPO.

Run from the repository root:
    uv run --with stable-baselines3 python ourcode/evaluate_shared_ippo.py --seeds 1

For all five seeds:
    uv run --with stable-baselines3 python ourcode/evaluate_shared_ippo.py --seeds 1 2 3 4 5

The results are saved separately in results/shared_ippo_shock_5000/.
"""

import argparse
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

import config as C
import evaluate_agents as evaluation
import metrics


def load_shared_models(model_dir: Path):

    models = []

    for i in range(C.N_PPO):
        path = model_dir / f"ppo_agent_{i + 1}.zip"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing PPO model: {path}\n"
                "Run train_shared_ippo.py first."
            )

        print(f"Loading PPO Agent {i + 1}: {path}", flush=True)

        model = PPO.load(str(path), device="cpu")
        models.append(model)

    return models


def run_evaluation(models, seeds):
    """
    Run four experimental conditions for each seed.

    Each condition uses three trading agents in the same ABIDES market.
    """

    rows = []

    for seed in seeds:
        print(f"\n=== Evaluating Seed {seed} ===", flush=True)

        for use_ppo in (False, True):
            for shock_on in (False, True):
                policy = "PPO" if use_ppo else "Passive"
                shock = "ON" if shock_on else "OFF"

                print(
                    f"Seed: {seed} | "
                    f"Policy: {policy} | "
                    f"Shock: {shock}",
                    flush=True,
                )

                result = evaluation.run_one(
                    models=models,
                    use_ppo=use_ppo,
                    shock_on=shock_on,
                    seed=seed,
                )

                rows.extend(result)

    return pd.DataFrame(rows)


def save_results(df, output_dir):
    """Calculate performance metrics and save the results."""

    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_path = output_dir / "evaluation.csv"
    summary_path = output_dir / "summary_metrics.csv"
    paired_path = output_dir / "paired_effects.csv"

    df.to_csv(evaluation_path, index=False)

    summary = metrics.summarize(df)
    summary.to_csv(summary_path, index=False)

    paired = metrics.paired_effects(summary)
    paired.to_csv(paired_path, index=False)

    print("\n=== Evaluation Summary ===")
    print(summary.to_string(index=False))

    print("\n=== Shock Effects ===")
    print(paired.to_string(index=False))

    incomplete = summary[
        (summary["shock"] == 1)
        & (summary["shock_filled"] < C.SHOCK_SIZE)
    ]

    if not incomplete.empty:
        print(
            "\nWARNING: Some sell shocks were only partially filled."
        )

    if "spread_raw_cents" in df.columns:
        spreads = pd.to_numeric(
            df["spread_raw_cents"],
            errors="coerce",
        )

        negative_count = int((spreads < 0).sum())

        if negative_count > 0:
            print(
                f"\nWARNING: Found {negative_count} "
                "negative raw bid-ask spreads."
            )

    print(f"\nResults saved to: {output_dir}")

    return evaluation_path


def generate_graph(evaluation_path, output_dir):
    """Generate the shock reaction graph using our partner's plot script."""

    import plot_shock_reaction as plots

    plots.CSV = evaluation_path
    plots.OUT = output_dir / "shock_reaction.png"

    plots.main()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1],
    )

    parser.add_argument(
        "--shock-size",
        type=int,
        default=C.SHOCK_SIZE,
    )

    args = parser.parse_args()

    if args.shock_size <= 0:
        parser.error("Shock size must be positive.")

    C.SHOCK_SIZE = args.shock_size

    model_dir = C.ROOT / "models" / "shared_ippo"

    output_dir = (
        C.RESULTS_DIR
        / f"shared_ippo_shock_{args.shock_size}"
    )

    print("Loading shared-market independent PPO models...")

    models = load_shared_models(model_dir)

    print("\nStarting evaluation...")

    df = run_evaluation(models, args.seeds)

    evaluation_path = save_results(df, output_dir)

    print("\nGenerating shock reaction graph...")

    generate_graph(evaluation_path, output_dir)

    print("\nShared-market PPO evaluation completed!")


if __name__ == "__main__":
    main()
