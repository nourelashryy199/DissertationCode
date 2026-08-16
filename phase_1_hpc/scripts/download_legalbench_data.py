# ============================================================
# scripts/download_legalbench_data.py — Phase 1 (HPC)
# Ported from 01_Download_LegalBench.ipynb (Colab). Run this ONCE
# on the Stanage LOGIN node (needs internet access, which compute
# nodes typically don't have) — same pattern as
# download_data_qwen*.py.
#
# Does four things:
#   1. Enumerates all LegalBench task configs on Hugging Face.
#   2. Filters candidate tasks per category down to
#      classification-style tasks only (small, fixed label space).
#   3. Selects the task with the smallest test set per category.
#   4. Downloads/caches the selected tasks' train+test data
#      locally, and writes manifest.csv, task_field_map.json,
#      and question_templates.json.
#
# SAFETY: refuses to overwrite an existing manifest.csv unless
# --force is passed. Re-running this for real re-queries HF for
# LegalBench's CURRENT state — if the dataset has changed at all
# since your original run, task selection could differ and would
# silently desynchronize your already-completed demonstrations,
# eval pools, and generation data. Do not pass --force unless you
# specifically intend to re-select tasks and are prepared to
# re-run build_demonstrations.py and generation from scratch.
# ============================================================

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import pandas as pd
import schema
from datasets import get_dataset_config_names, load_dataset

LEGALBENCH_DATASET = "nguha/legalbench"

# --- Candidate tasks per category ---
# Only tasks with a small, fixed, discrete label space are eligible.
# Open-ended / generative tasks (e.g., rule_qa, definition_extraction,
# summarization-style tasks) are deliberately excluded, since this
# dissertation's accuracy metric requires a finite label set.
CATEGORY_TASK_MAP = {
    "issue-spotting": [
        "learned_hands_torts", "learned_hands_housing", "learned_hands_family",
    ],
    "rule-recall": [
        "citation_prediction_classification",
        "international_citizenship_questions",
        "telemarketing_sales_rule",
        # rule_qa deliberately excluded: open-ended free-text answer, not classification
    ],
    "rule-application": [
        "abercrombie", "diversity_1", "diversity_2",
    ],
    "rule-conclusion": [
        "contract_qa", "consumer_contracts_qa",
    ],
    "interpretation": [
        "cuad_governing_law", "cuad_termination_for_convenience",
    ],
    "rhetorical-understanding": [
        "overruling", "legal_reasoning_causality",
    ],
}

# --- Classification-style screening thresholds ---
MAX_UNIQUE_LABEL_RATIO = 0.1     # unique answers should be <=10% of examples
MAX_UNIQUE_LABELS_ABSOLUTE = 20  # hard cap regardless of dataset size

# --- Manual field mapping and question templates for the six ---
# --- tasks this dissertation ultimately selected. Hand-authored, ---
# --- not derived — kept here so the full pipeline is reproducible ---
# --- from this one script, matching what's already in your repo. ---
QUESTION_TEMPLATES = {
    "learned_hands_torts": "Does this post describe a potential tort law issue?",
    "telemarketing_sales_rule": "Does this scenario describe a violation of the Telemarketing Sales Rule?",
    "abercrombie": "How should this trademark be classified on the Abercrombie distinctiveness spectrum?",
    "contract_qa": None,  # this task already has its own per-instance "question" column
    "cuad_termination_for_convenience": "Does this contract clause permit termination for convenience?",
    "legal_reasoning_causality": "Does this passage describe legal reasoning that relies on statistical or causal inference?",
}

TASK_FIELD_MAP = {
    "learned_hands_torts": {"context": "text", "question": None},
    "telemarketing_sales_rule": {"context": "text", "question": None},
    "abercrombie": {"context": "text", "question": None},
    "contract_qa": {"context": "text", "question": "question"},
    "cuad_termination_for_convenience": {"context": "text", "question": None},
    "legal_reasoning_causality": {"context": "text", "question": None},
}


def screen_and_select():
    print("Fetching available LegalBench task configs from Hugging Face...")
    all_task_configs = get_dataset_config_names(LEGALBENCH_DATASET)
    print(f"Total LegalBench task configs available: {len(all_task_configs)}")

    candidate_records = []
    for category, task_names in CATEGORY_TASK_MAP.items():
        for task_name in task_names:
            if task_name not in all_task_configs:
                print(f"WARNING: '{task_name}' not found in current LegalBench configs — skipping.")
                continue
            try:
                ds = load_dataset(LEGALBENCH_DATASET, task_name)
                test_split = ds["test"] if "test" in ds else ds[list(ds.keys())[0]]
                candidate_records.append({
                    "category": category,
                    "task_name": task_name,
                    "test_size": len(test_split),
                    "columns": test_split.column_names,
                })
                print(f"[{category}] {task_name}: {len(test_split)} test examples")
            except Exception as e:
                print(f"FAILED to load {task_name} ({category}): {e}")

    print("\nScreening candidates for classification-style label spaces...")
    screened_records = []
    for record in candidate_records:
        task_name = record["task_name"]
        ds = load_dataset(LEGALBENCH_DATASET, task_name)
        test_split = ds["test"] if "test" in ds else ds[list(ds.keys())[0]]

        answer_col = None
        for candidate_col in ["answer", "label"]:
            if candidate_col in test_split.column_names:
                answer_col = candidate_col
                break

        if answer_col is None:
            print(f"SKIP {task_name}: no recognizable answer/label column — likely open-ended.")
            continue

        unique_answers = set(test_split[answer_col])
        n_unique = len(unique_answers)
        n_total = len(test_split)
        ratio = n_unique / n_total

        is_classification = (n_unique <= MAX_UNIQUE_LABELS_ABSOLUTE) and (ratio <= MAX_UNIQUE_LABEL_RATIO)

        record["n_unique_labels"] = n_unique
        record["unique_label_ratio"] = round(ratio, 4)
        record["is_classification"] = is_classification

        if not is_classification:
            print(f"EXCLUDED {task_name}: {n_unique} unique answers across {n_total} examples "
                  f"(ratio {ratio:.3f}) — looks open-ended, not classification.")
        screened_records.append(record)

    candidates_df = pd.DataFrame(screened_records)
    candidates_df = candidates_df[candidates_df["is_classification"]]
    print(f"\n{len(candidates_df)} tasks passed the classification screen.")

    os.makedirs(os.path.dirname(config.MANIFEST_PATH), exist_ok=True)
    candidates_df.sort_values(["category", "test_size"]).to_csv(
        os.path.join(os.path.dirname(config.MANIFEST_PATH), "candidate_tasks.csv"), index=False)

    pilot_selection = (
        candidates_df
        .sort_values("test_size")
        .groupby("category", as_index=False)
        .first()
    )
    print("\nSelected task per category (smallest test set):")
    print(pilot_selection[["category", "task_name", "test_size"]].to_string(index=False))

    return pilot_selection


def download_selected(pilot_selection: pd.DataFrame):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    selected_datasets = {}

    for _, row in pilot_selection.iterrows():
        task_name = row["task_name"]
        ds = load_dataset(LEGALBENCH_DATASET, task_name)
        selected_datasets[task_name] = ds

        task_dir = os.path.join(config.DATA_DIR, task_name)
        os.makedirs(task_dir, exist_ok=True)
        for split_name, split_data in ds.items():
            split_data.to_csv(os.path.join(task_dir, f"{split_name}.csv"))

        print(f"Saved {task_name} to {task_dir}/")

    return selected_datasets


def build_manifest_and_field_maps(pilot_selection: pd.DataFrame, selected_datasets: dict):
    manifest_rows = []
    for _, row in pilot_selection.iterrows():
        task_name = row["task_name"]
        test_data = selected_datasets[task_name]["test"]
        unique_labels = sorted(set(str(r.get("answer", "")) for r in test_data))

        manifest_rows.append({
            "task_id": task_name,
            "category": row["category"],
            "test_size": row["test_size"],
            "n_labels": len(unique_labels),
            "source_dataset": "LegalBench",
            # Note: no "phase" column here — this dissertation presents one
            # unified experimental protocol, not a phased pilot/final split.
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(config.MANIFEST_PATH, index=False)
    print(f"\nSaved manifest to {config.MANIFEST_PATH}")
    print(manifest_df.to_string(index=False))

    field_map_path = os.path.join(config.HPC_ROOT, "data", "task_field_map.json")
    with open(field_map_path, "w") as f:
        json.dump(TASK_FIELD_MAP, f, indent=2)
    print(f"Saved field map to {field_map_path}")

    question_templates_path = os.path.join(config.HPC_ROOT, "data", "question_templates.json")
    with open(question_templates_path, "w") as f:
        json.dump(QUESTION_TEMPLATES, f, indent=2)
    print(f"Saved question templates to {question_templates_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                         help="Overwrite an existing manifest.csv. DANGEROUS if generation "
                              "has already run — see the warning in this file's header.")
    args, _ = parser.parse_known_args()

    if os.path.exists(config.MANIFEST_PATH) and not args.force:
        print(f"manifest.csv already exists at {config.MANIFEST_PATH}.")
        print("Refusing to overwrite without --force.")
        print("\nWARNING: if you've already run build_demonstrations.py or run_stage_a.py "
              "against the current manifest.csv, re-running this script with --force could "
              "select DIFFERENT tasks (LegalBench's hosted data may have changed since your "
              "original run) and silently desynchronize your existing demonstrations, eval "
              "pools, and generation data from the new manifest. Only use --force if you "
              "specifically intend to re-select tasks and re-run the full pipeline from "
              "scratch.")
        sys.exit(1)

    pilot_selection = screen_and_select()
    selected_datasets = download_selected(pilot_selection)
    build_manifest_and_field_maps(pilot_selection, selected_datasets)

    print("\nDone. Run build_demonstrations.py next.")


if __name__ == "__main__":
    main()