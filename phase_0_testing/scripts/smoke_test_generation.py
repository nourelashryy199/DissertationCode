# ============================================================
# phase_0/scripts/smoke_test_generation.py
# Tiny-scale, real end-to-end generation test using the CURRENT
# production code from phase_1_hpc (not a copy, not an old
# version) — validates that model loading, real demonstration
# sets, prompt construction, generation, parsing, and scoring
# all work together correctly before committing to the full
# 6-task x 13-strategy x 3-rephrasing x 5-run sweep.
#
# Scope: 1 task, 1 instance, 3 representative strategies
# (zero_shot, cot, irac), 1 rephrasing, 1 run = 3 generations.
#
# Needs a GPU — run via SLURM (see ../slurm/smoke_test.sbatch)
# or an interactive GPU session. NOT laptop-runnable.
#
# Output is written to phase_0/outputs/ — separate from
# phase_1_hpc/outputs/raw_generations/, so this never touches
# or pollutes real experiment data.
#
# Run via:
#   python smoke_test_generation.py --model Qwen/Qwen2.5-7B-Instruct
# ============================================================

import os
import sys
import json
from datetime import datetime, timezone

PHASE0_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE0_DIR = os.path.dirname(PHASE0_SCRIPTS_DIR)
REPO_ROOT = os.path.dirname(PHASE0_DIR)
PHASE1_DIR = os.path.join(REPO_ROOT, "phase_1_hpc")

if not os.path.isdir(PHASE1_DIR):
    print(f"ERROR: expected to find phase_1_hpc at {PHASE1_DIR} — adjust PHASE1_DIR if your repo layout differs.")
    sys.exit(1)

sys.path.insert(0, PHASE1_DIR)

import config
import schema
import pandas as pd
from model import LegalPromptModel
from strategy_functions import build_prompt

SMOKE_STRATEGIES = ["zero_shot", "cot", "irac"]  # one generic, one CoT, one legal framework
PHASE0_OUTPUT_DIR = os.path.join(PHASE0_DIR, "outputs")


def main():
    model_name = config.get_model_name_from_args().model
    os.makedirs(PHASE0_OUTPUT_DIR, exist_ok=True)

    print(f"=== Smoke test starting: model={model_name} ===\n")

    manifest_df = pd.read_csv(config.MANIFEST_PATH)
    smoke_task_id = manifest_df["task_id"].iloc[0]
    smoke_category = manifest_df[manifest_df["task_id"] == smoke_task_id]["category"].iloc[0]

    with open(os.path.join(config.EVAL_POOLS_DIR, f"{smoke_task_id}_eval.json")) as f:
        eval_pool = json.load(f)
    with open(os.path.join(config.DEMO_DIR, f"{smoke_task_id}_demos.json")) as f:
        demo_raw = json.load(f)
    demonstration_sets = {
        smoke_task_id: [schema.Demonstration(context=d["context"], question=d["question"], label=d["label"])
                         for d in demo_raw]
    }
    print(f"Loaded {len(demonstration_sets[smoke_task_id])} real demonstrations for {smoke_task_id} "
          f"(confirms build_demonstrations.py output is being read correctly)")

    with open(os.path.join(PHASE1_DIR, "data", "task_field_map.json")) as f:
        task_field_map = json.load(f)
    with open(os.path.join(PHASE1_DIR, "data", "question_templates.json")) as f:
        question_templates = json.load(f)

    field_map = task_field_map[smoke_task_id]
    row = eval_pool[0]
    context = str(row.get(field_map["context"], ""))
    question = str(row.get(field_map["question"], "")) if field_map.get("question") else question_templates.get(smoke_task_id)

    smoke_instance = schema.LegalTask(
        task_id=f"{smoke_task_id}_0",
        task_type=smoke_category,
        context=context,
        question=question,
        label_options=sorted(set(str(r.get("answer", "")) for r in eval_pool[:20])),  # small sample for label set
        expected_output=str(row.get("answer", "")),
    )
    print(f"Smoke test instance: {smoke_instance.task_id} (category: {smoke_category})\n")

    lpm = LegalPromptModel(model_name)
    lpm.load()

    output_path = os.path.join(PHASE0_OUTPUT_DIR, f"smoke_test__{model_name.replace('/', '_')}.jsonl")
    results = []

    for strategy in SMOKE_STRATEGIES:
        prompt_text = build_prompt(smoke_instance, strategy, rephrasing_id=0,
                                    task_id=smoke_task_id, demonstration_sets=demonstration_sets)
        raw_output, parsed_answer = lpm.generate_and_parse(prompt_text)
        is_correct = (parsed_answer == smoke_instance.expected_output) if parsed_answer else False

        record = {
            "task_id": smoke_instance.task_id,
            "category": smoke_category,
            "strategy": strategy,
            "model_name": model_name,
            "prompt_text": prompt_text,
            "raw_output": raw_output,
            "parsed_answer": parsed_answer,
            "expected_output": smoke_instance.expected_output,
            "is_correct": is_correct,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        results.append(record)

        print(f"--- {strategy} ---")
        print(f"Prompt (first 200 chars): {prompt_text[:200]}...")
        print(f"Expected: {smoke_instance.expected_output}")
        print(f"Parsed:   {parsed_answer}")
        print(f"Correct:  {is_correct}\n")

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    lpm.unload()

    n_correct = sum(r["is_correct"] for r in results)
    n_parsed = sum(r["parsed_answer"] is not None for r in results)
    print(f"=== Smoke test complete: {n_parsed}/{len(results)} parsed successfully, "
          f"{n_correct}/{len(results)} correct ===")
    print(f"Full output saved to: {output_path}")
    print("\nThis does NOT validate the model's accuracy (n=3, not a real result) — "
          "it validates that model loading, real demo sets, prompt construction, "
          "generation, and parsing all work together end-to-end with the current code.")


if __name__ == "__main__":
    main()