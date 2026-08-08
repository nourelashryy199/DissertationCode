# ============================================================
# scripts/analysis.py — Phase 1 (HPC)
# Computes the Category Fit Score (genuine Sharpe-style: run-level
# accuracy distribution, not pooled), identifies each category's
# champion strategy, and produces the Prompt Transfer Penalty
# matrix: Penalty(i -> j) = FitScore(j, champion_j) - FitScore(j, champion_i).
#
# Since Stage A already ran every strategy against every category's
# tasks (a full exhaustive sweep), no new generations are needed —
# this re-slices the existing parsed dataset.
# ============================================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import pandas as pd
import numpy as np


def compute_run_level_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (category, strategy, run_id), compute accuracy across
    all instances and rephrasings within that run. This gives 5
    accuracy values per (category, strategy) — the genuine
    "repeated trial" observations a Sharpe-style ratio needs,
    rather than pooling all generations into one flat group.
    """
    return (
        df.groupby(["category", "strategy", "run_id"])["is_correct"]
        .mean()
        .reset_index()
        .rename(columns={"is_correct": "run_accuracy"})
    )


def compute_fit_scores(run_level: pd.DataFrame) -> pd.DataFrame:
    """
    Fit Score(category, strategy) = mean(run_accuracy gain over
    zero-shot) / std(run_accuracy), computed across the 5 run-level
    accuracy observations — genuine Sharpe-style construction.
    """
    zero_shot = (
        run_level[run_level["strategy"] == "zero_shot"]
        .groupby("category")["run_accuracy"]
        .mean()
        .rename("zero_shot_baseline")
    )

    summary = (
        run_level.groupby(["category", "strategy"])["run_accuracy"]
        .agg(mean_accuracy="mean", std_accuracy="std", n_runs="count")
        .reset_index()
        .merge(zero_shot, on="category", how="left")
    )

    summary["accuracy_gain"] = summary["mean_accuracy"] - summary["zero_shot_baseline"]

    # Guard against divide-by-zero when a strategy has zero variance
    # across its 5 runs (e.g., perfect or identical accuracy every
    # run) — treat as effectively infinite stability, capped at a
    # large finite value so it doesn't break downstream sorting/plots.
    summary["fit_score"] = np.where(
        summary["std_accuracy"] > 1e-9,
        summary["accuracy_gain"] / summary["std_accuracy"],
        np.where(summary["accuracy_gain"] > 0, 100.0, np.where(summary["accuracy_gain"] < 0, -100.0, 0.0)),
    )

    return summary


def identify_champions(fit_scores: pd.DataFrame) -> pd.Series:
    """For each category, the strategy with the highest Fit Score."""
    idx = fit_scores.groupby("category")["fit_score"].idxmax()
    champions = fit_scores.loc[idx].set_index("category")["strategy"]
    return champions


def compute_transfer_penalty(fit_scores: pd.DataFrame, champions: pd.Series) -> pd.DataFrame:
    """
    Penalty(i -> j) = FitScore(j, champion_j) - FitScore(j, champion_i)
    Diagonal (i == j) is 0 by construction (a category's own champion
    applied to itself has zero penalty). Off-diagonal cells show the
    cost of using category i's champion strategy on category j's tasks.
    """
    categories = list(champions.index)
    matrix = pd.DataFrame(index=categories, columns=categories, dtype=float)

    fit_lookup = fit_scores.set_index(["category", "strategy"])["fit_score"]

    for j in categories:
        champion_j_score = fit_lookup.get((j, champions[j]), np.nan)
        for i in categories:
            if i == j:
                matrix.loc[i, j] = 0.0
                continue
            champion_i_on_j_score = fit_lookup.get((j, champions[i]), np.nan)
            if pd.isna(champion_i_on_j_score) or pd.isna(champion_j_score):
                matrix.loc[i, j] = np.nan
            else:
                matrix.loc[i, j] = champion_j_score - champion_i_on_j_score

    return matrix


def main():
    model_name = config.get_model_name_from_args().model
    safe_model_name = model_name.replace("/", "_")

    parsed_path = os.path.join(config.PARSED_DIR, f"all_generations_parsed__{safe_model_name}.csv")
    if not os.path.exists(parsed_path):
        print(f"ERROR: {parsed_path} not found. Run parse_predictions.py first.")
        return

    df = pd.read_csv(parsed_path)
    df["is_correct"] = df["is_correct"].fillna(False)
    print(f"Loaded {len(df)} generations for {model_name}")

    run_level = compute_run_level_accuracy(df)
    fit_scores = compute_fit_scores(run_level)
    champions = identify_champions(fit_scores)

    print("\n=== Category Champions (highest Fit Score) ===")
    print(champions)

    print("\n=== Fit Scores (all category x strategy pairs) ===")
    print(fit_scores.sort_values(["category", "fit_score"], ascending=[True, False]).to_string(index=False))

    penalty_matrix = compute_transfer_penalty(fit_scores, champions)
    print("\n=== Prompt Transfer Penalty Matrix: Penalty(i -> j) ===")
    print("Rows = origin strategy's category (i), Columns = applied-to category (j)")
    print(penalty_matrix.round(3))

    # Aggregate: mean and std of penalty per target category j
    # (averaging across all i != j) — magnitude and unpredictability
    # of precaution needed for that category.
    off_diagonal = penalty_matrix.copy()
    np.fill_diagonal(off_diagonal.values, np.nan)
    mean_penalty = off_diagonal.mean(axis=0)
    std_penalty = off_diagonal.std(axis=0)

    risk_summary = pd.DataFrame({
        "mean_penalty": mean_penalty,
        "std_penalty": std_penalty,
    })
    print("\n=== Per-Category Risk Summary (mean/std of incoming penalty) ===")
    print(risk_summary.round(3))

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    fit_scores.to_csv(os.path.join(config.RESULTS_DIR, f"fit_scores__{safe_model_name}.csv"), index=False)
    champions.to_csv(os.path.join(config.RESULTS_DIR, f"champions__{safe_model_name}.csv"))
    penalty_matrix.to_csv(os.path.join(config.RESULTS_DIR, f"transfer_penalty_matrix__{safe_model_name}.csv"))
    risk_summary.to_csv(os.path.join(config.RESULTS_DIR, f"risk_summary__{safe_model_name}.csv"))

    print(f"\nSaved all results to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
