# ============================================================
# scripts/rescore.py — Phase 1 (HPC)
# Retroactively recomputes is_correct using normalized string
# matching (case-insensitive, whitespace/trailing-period stripped)
# against the already-generated parsed_answer values. Does NOT
# touch generation — reads existing parsed CSV, writes a corrected
# version. Existing raw .jsonl files and running jobs are untouched.
# ============================================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import pandas as pd
import json


def normalize_answer(s) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip().lower().rstrip(".")


def main():
    model_name = config.get_model_name_from_args().model
    safe_model_name = model_name.replace("/", "_")

    manifest_df = pd.read_csv(config.MANIFEST_PATH)

    # Need the true expected_output per task_id, which isn't stored
    # in the parsed CSV — pull it back from the raw eval pool files.
    task_field_map_path = os.path.join(config.HPC_ROOT, "data", "task_field_map.json")
    with open(task_field_map_path) as f:
        task_field_map = json.load(f)

    expected_lookup = {}  # instance task_id (e.g. "abercrombie_17") -> true label
    import random
    SAMPLE_SIZE = 45  # must match the --sample_size used in run_stage_a.py

    for _, row in manifest_df.iterrows():
        base_task_id = row["task_id"]
        eval_path = os.path.join(config.EVAL_POOLS_DIR, f"{base_task_id}_eval.json")
        with open(eval_path) as f:
            pool = json.load(f)
        pool = pool.copy()
        random.Random(config.CLUSTERING_RANDOM_STATE).shuffle(pool)
        pool = pool[:SAMPLE_SIZE]
        for idx, r in enumerate(pool):
            expected_lookup[f"{base_task_id}_{idx}"] = r.get("answer", "")

    parsed_path = os.path.join(config.PARSED_DIR, f"all_generations_parsed__{safe_model_name}.csv")
    df = pd.read_csv(parsed_path)
    print(f"Loaded {len(df)} generations")

    df["expected_output"] = df["task_id"].map(expected_lookup)
    missing = df["expected_output"].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} rows had no matching expected_output lookup — check task_id alignment.")

    old_correct_rate = df["is_correct"].mean()

    df["is_correct_normalized"] = (
        df["parsed_answer"].apply(normalize_answer) == df["expected_output"].apply(normalize_answer)
    ) & df["parsed_answer"].notna()

    new_correct_rate = df["is_correct_normalized"].mean()

    print(f"\nOld (strict-match) accuracy:      {old_correct_rate:.4f}")
    print(f"New (normalized-match) accuracy:  {new_correct_rate:.4f}")
    print(f"Difference: {new_correct_rate - old_correct_rate:+.4f}")

    n_changed = (df["is_correct"] != df["is_correct_normalized"]).sum()
    print(f"\nRows where correctness changed: {n_changed} / {len(df)} ({n_changed/len(df)*100:.2f}%)")

    # Replace is_correct with the corrected version, keep the old
    # one for reference/audit.
    df["is_correct_original"] = df["is_correct"]
    df["is_correct"] = df["is_correct_normalized"]
    df = df.drop(columns=["is_correct_normalized"])

    output_path = os.path.join(config.PARSED_DIR, f"all_generations_parsed_rescored__{safe_model_name}.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved rescored dataset to {output_path}")


if __name__ == "__main__":
    main()
