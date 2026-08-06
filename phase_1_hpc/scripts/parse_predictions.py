# ============================================================
# scripts/parse_predictions.py — Phase 1 (HPC)
# Loads all raw generation .jsonl files, deduplicates, flags
# parsing failures, and saves one consolidated dataset. Safe to
# run against partial data (mid-generation) or the final complete
# run — just re-run once generation finishes for the full dataset.
# ============================================================

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import pandas as pd


def run_task_id_key(strategy, rephrasing_id, run_id, instance_task_id):
    return f"{strategy}|{rephrasing_id}|{run_id}|{instance_task_id}"


def load_and_dedupe(filepath: str) -> list:
    if not os.path.exists(filepath):
        print(f"WARNING: file not found: {filepath}")
        return []

    with open(filepath) as f:
        raw_records = [json.loads(line) for line in f if line.strip()]

    deduped = {}
    for r in raw_records:
        key = run_task_id_key(r["strategy"], r["rephrasing_id"], r["run_id"], r["task_id"])
        deduped[key] = r

    if len(deduped) < len(raw_records):
        print(f"  {os.path.basename(filepath)}: {len(raw_records) - len(deduped)} duplicate(s) found, deduplicated to {len(deduped)}.")

    return list(deduped.values())


def main():
    manifest_df = pd.read_csv(config.MANIFEST_PATH)
    model_name = config.get_model_name_from_args().model
    safe_model_name = model_name.replace("/", "_")

    all_records = []
    for task_id in manifest_df["task_id"]:
        filename = f"{task_id}__{safe_model_name}_generations.jsonl"
        filepath = os.path.join(config.RAW_GEN_DIR, filename)
        records = load_and_dedupe(filepath)
        all_records.extend(records)
        print(f"{task_id}: {len(records)} generations loaded")

    print(f"\nTotal generations loaded: {len(all_records)}")

    df = pd.DataFrame(all_records)

    if len(df) == 0:
        print("No records found — nothing to save.")
        return

    parsing_failures = df[df["parsed_answer"].isna()]
    n_failures = len(parsing_failures)
    n_total = len(df)
    failure_rate = (n_failures / n_total * 100) if n_total > 0 else 0
    print(f"\nParsing failures: {n_failures} / {n_total} ({failure_rate:.2f}%)")

    if n_failures > 0:
        print("\nBreakdown by (task, strategy):")
        print(parsing_failures.groupby(["task_id", "strategy"]).size().sort_values(ascending=False).head(20))

    os.makedirs(config.PARSED_DIR, exist_ok=True)

    output_path = os.path.join(config.PARSED_DIR, f"all_generations_parsed__{safe_model_name}.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved consolidated dataset to {output_path}")
    print(f"Shape: {df.shape}")

    if n_failures > 0:
        failures_path = os.path.join(config.PARSED_DIR, f"parsing_failures__{safe_model_name}.csv")
        parsing_failures.to_csv(failures_path, index=False)
        print(f"Saved {n_failures} parsing failures to {failures_path}")


if __name__ == "__main__":
    main()
