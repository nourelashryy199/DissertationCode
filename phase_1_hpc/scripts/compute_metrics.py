# ============================================================
# scripts/compute_metrics.py — Phase 1 (HPC)
# Computes per-(category, strategy, rephrasing) accuracy from the
# consolidated dataset produced by parse_predictions.py. This is
# the summary table that analysis.py will use to compute the
# Category Fit Score, Prompt Transfer Penalty, and the
# Wording-Sensitivity vs. Strategy-Sensitivity decomposition.
#
# Safe to run against partial (mid-generation) data — just re-run
# once more data exists for a fuller picture.
# ============================================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import pandas as pd


def main():
    model_name = config.get_model_name_from_args().model
    safe_model_name = model_name.replace("/", "_")

    parsed_path = os.path.join(config.PARSED_DIR, f"all_generations_parsed_rescored__{safe_model_name}.csv")
    if not os.path.exists(parsed_path):
        print(f"ERROR: {parsed_path} not found. Run parse_predictions.py then rescore.py first.")
        return

    df = pd.read_csv(parsed_path)
    print(f"Loaded {len(df)} generations for {model_name}")

    df["is_correct"] = df["is_correct"].fillna(False)

    per_strategy_rephrasing = (
        df.groupby(["category", "strategy", "rephrasing_id"])
        .agg(
            n_generations=("is_correct", "size"),
            accuracy=("is_correct", "mean"),
        )
        .reset_index()
    )
    print("\n=== Per (category, strategy, rephrasing) accuracy ===")
    print(per_strategy_rephrasing.to_string(index=False))

    per_strategy = (
        df.groupby(["category", "strategy"])
        .agg(
            n_generations=("is_correct", "size"),
            accuracy_mean=("is_correct", "mean"),
            accuracy_std=("is_correct", "std"),
        )
        .reset_index()
    )
    print("\n=== Per (category, strategy) accuracy (mean, std) ===")
    print(per_strategy.to_string(index=False))

    zero_shot_baseline = (
        per_strategy[per_strategy["strategy"] == "zero_shot"]
        .set_index("category")["accuracy_mean"]
    )
    print("\n=== Zero-shot baseline per category ===")
    print(zero_shot_baseline)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    out1 = os.path.join(config.RESULTS_DIR, f"metrics_per_strategy_rephrasing__{safe_model_name}.csv")
    per_strategy_rephrasing.to_csv(out1, index=False)

    out2 = os.path.join(config.RESULTS_DIR, f"metrics_per_strategy__{safe_model_name}.csv")
    per_strategy.to_csv(out2, index=False)

    print(f"\nSaved: {out1}")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()