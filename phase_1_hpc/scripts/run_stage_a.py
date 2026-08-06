# ============================================================
# scripts/run_stage_a.py — Phase 1 (HPC)
# Full Stage A generation loop: all strategies x all tasks x
# all rephrasings x all runs, for a single specified model.
# Interleaved by instance index across tasks, so a job that
# times out mid-run leaves partial coverage across every
# category rather than concentrating progress in one task.
#
# Run via SLURM as:
#   python run_stage_a.py --model <model_name> [--sample_size N]
# Omit --sample_size for the full Phase 1 eval pools.
# ============================================================

import os
import sys
import json
import random
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import schema
import pandas as pd
from model import LegalPromptModel
from strategy_functions import build_prompt


def load_task_field_map():
    with open(os.path.join(config.HPC_ROOT, "data", "task_field_map.json")) as f:
        return json.load(f)


def load_question_templates():
    with open(os.path.join(config.HPC_ROOT, "data", "question_templates.json")) as f:
        return json.load(f)


def run_task_id_key(strategy, rephrasing_id, run_id, instance_task_id):
    return f"{strategy}|{rephrasing_id}|{run_id}|{instance_task_id}"


def output_file_key(task_id, model_name):
    safe_model_name = model_name.replace("/", "_")
    return f"{task_id}__{safe_model_name}"


def load_existing_records(file_key: str) -> list:
    path = os.path.join(config.RAW_GEN_DIR, f"{file_key}_generations.jsonl")
    if not os.path.exists(path):
        return []

    with open(path) as f:
        raw_records = [json.loads(line) for line in f if line.strip()]

    deduped = {}
    for r in raw_records:
        key = run_task_id_key(r["strategy"], r["rephrasing_id"], r["run_id"], r["task_id"])
        deduped[key] = r

    if len(deduped) < len(raw_records):
        print(f"  NOTE: {file_key} had {len(raw_records) - len(deduped)} duplicate record(s) — deduplicated.")
        with open(path, "w") as f:
            for r in deduped.values():
                f.write(json.dumps(r) + "\n")

    return list(deduped.values())


def append_record(file_key: str, record: dict):
    path = os.path.join(config.RAW_GEN_DIR, f"{file_key}_generations.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def build_legal_task(row: dict, task_id: str, category: str, idx: int,
                      task_field_map: dict, question_templates: dict) -> schema.LegalTask:
    field_map = task_field_map[task_id]
    context = str(row.get(field_map["context"], ""))

    if field_map.get("question"):
        question = str(row.get(field_map["question"], ""))
    else:
        question = question_templates.get(task_id)

    return schema.LegalTask(
        task_id=f"{task_id}_{idx}",
        task_type=category,
        context=context,
        question=question,
        label_options=[],
        expected_output=str(row.get("answer", "")),
        jurisdiction="US General",
        source_dataset="LegalBench",
    )


def main():
    args = config.get_model_name_from_args()
    model_name = args.model
    sample_size = args.sample_size

    os.makedirs(config.RAW_GEN_DIR, exist_ok=True)
    print(f"=== Stage A run starting: model={model_name}, sample_size={sample_size or 'FULL'} ===")

    task_field_map = load_task_field_map()
    question_templates = load_question_templates()

    manifest_df = pd.read_csv(config.MANIFEST_PATH)

    eval_pools, demonstration_sets = {}, {}
    for task_id in manifest_df["task_id"]:
        with open(os.path.join(config.EVAL_POOLS_DIR, f"{task_id}_eval.json")) as f:
            eval_pools[task_id] = json.load(f)
        with open(os.path.join(config.DEMO_DIR, f"{task_id}_demos.json")) as f:
            demo_raw = json.load(f)
        demonstration_sets[task_id] = [
            schema.Demonstration(context=d["context"], question=d["question"], label=d["label"])
            for d in demo_raw
        ]

    legal_tasks = {}
    for _, row in manifest_df.iterrows():
        task_id, category = row["task_id"], row["category"]
        pool = eval_pools[task_id]
        if sample_size:
            pool = pool.copy()
            random.Random(config.CLUSTERING_RANDOM_STATE).shuffle(pool)
            pool = pool[:sample_size]

        instances = [
            build_legal_task(r, task_id, category, i, task_field_map, question_templates)
            for i, r in enumerate(pool)
        ]
        label_options = sorted(set(t.expected_output for t in instances))
        for t in instances:
            t.label_options = label_options
        legal_tasks[task_id] = instances
        print(f"{task_id}: {len(instances)} instances loaded (sample_size={sample_size or 'full'})")

    total_planned = sum(len(v) for v in legal_tasks.values()) * len(config.ALL_STRATEGIES) * config.N_REPHRASINGS * config.N_RUNS

    lpm = LegalPromptModel(model_name)
    lpm.load()

    task_existing_keys = {}
    completed_count = 0
    for task_id in manifest_df["task_id"]:
        file_key = output_file_key(task_id, model_name)
        existing = load_existing_records(file_key)
        task_existing_keys[task_id] = {
            run_task_id_key(r["strategy"], r["rephrasing_id"], r["run_id"], r["task_id"])
            for r in existing
        }
        completed_count += len(existing)
        print(f"{task_id}: {len(existing)} generations already saved for {model_name}")

    print(f"\nStarting from {completed_count}/{total_planned} already complete.\n")

    start_time = time.time()
    max_instances = max(len(v) for v in legal_tasks.values())

    for instance_idx in range(max_instances):
        print(f"\n########## Instance index {instance_idx + 1}/{max_instances} (across all tasks) ##########")

        for _, row in manifest_df.iterrows():
            task_id, category = row["task_id"], row["category"]
            if instance_idx >= len(legal_tasks[task_id]):
                continue

            task_instance = legal_tasks[task_id][instance_idx]
            existing_keys = task_existing_keys[task_id]
            file_key = output_file_key(task_id, model_name)

            for strategy in config.ALL_STRATEGIES:
                for rephrasing_id in range(config.N_REPHRASINGS):
                    for run_id in range(config.N_RUNS):
                        key = run_task_id_key(strategy, rephrasing_id, run_id, task_instance.task_id)
                        if key in existing_keys:
                            continue

                        prompt_text = build_prompt(task_instance, strategy, rephrasing_id, task_id, demonstration_sets)
                        raw_output, parsed_answer = lpm.generate_and_parse(prompt_text)
                        is_correct = (parsed_answer == task_instance.expected_output) if parsed_answer else False

                        record = schema.GenerationRecord(
                            task_id=task_instance.task_id,
                            category=category,
                            strategy=strategy,
                            rephrasing_id=rephrasing_id,
                            run_id=run_id,
                            model_name=model_name,
                            prompt_text=prompt_text,
                            raw_output=raw_output,
                            parsed_answer=parsed_answer,
                            is_correct=is_correct,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                        )
                        append_record(file_key, record.__dict__)
                        existing_keys.add(key)
                        completed_count += 1

                        if completed_count % 100 == 0:
                            elapsed = time.time() - start_time
                            rate = completed_count / elapsed if elapsed > 0 else 0
                            remaining = total_planned - completed_count
                            eta_hours = (remaining / rate / 3600) if rate > 0 else float("inf")
                            print(f"  Progress: {completed_count}/{total_planned} "
                                  f"({rate:.2f} gen/sec, ETA: {eta_hours:.1f} hrs) "
                                  f"[currently: {task_id}, instance {instance_idx}]")

            print(f"  Finished instance {instance_idx} for {task_id} ({category})")

    lpm.unload()
    print(f"\n=== Stage A run COMPLETE for {model_name}. Total generations: {completed_count} ===")


if __name__ == "__main__":
    main()