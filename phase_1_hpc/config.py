# ============================================================
# config.py — Phase 1 (HPC)
# Central configuration for the full experiment: model, sampling,
# category list, and the shared answer-extraction convention.
# ============================================================

import re
import os
import argparse

# --- Model settings ---
# Overridden via --model CLI arg in each SLURM job, so one config
# file works across the whole Qwen2.5 -> Llama-3.x progression.
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 512

# --- Sampling / rephrasing design ---
# Restored to the full Phase 1 spec (Phase 0's Colab constraints
# forced N_RUNS down to 3 and PILOT_SAMPLE_SIZE down to 5 per task —
# neither restriction applies on HPC).
TEMPERATURE = 0.7
TOP_P = 0.95
N_REPHRASINGS = 3
N_RUNS = 5

# --- LegalBench reasoning-type categories ---
CATEGORIES = [
    "issue-spotting",
    "rule-recall",
    "rule-application",
    "rule-conclusion",
    "interpretation",
    "rhetorical-understanding",
]

# --- Prompting strategies ---
GENERIC_STRATEGIES = [
    "zero_shot",
    "one_shot",
    "few_shot_2",
    "few_shot_3",
    "role_based",
    "structured",
    "cot",
]

LEGAL_FRAMEWORK_STRATEGIES = [
    "irac",
    "crac",
    "creac",
    "cleo",
    "treacc",
    "ireac",
]

ALL_STRATEGIES = GENERIC_STRATEGIES + LEGAL_FRAMEWORK_STRATEGIES

DEMO_REQUIRED_STRATEGIES = {
    "one_shot": 1,
    "few_shot_2": 2,
    "few_shot_3": 3,
}

# --- Canonical legal reasoning framework step sequences ---
# Sourced from Burton (2017) / Turner (2012); see methodology
# for full citation and discussion of acronym instability
# (e.g. "IRREAC" in the AI-prompting literature vs. the
# pedagogically-standard IREAC used here).
FRAMEWORK_STEPS = {
    "irac":   ["Issue", "Rule", "Application", "Conclusion"],
    "crac":   ["Conclusion", "Rule", "Application", "Conclusion"],
    "creac":  ["Conclusion", "Rule", "Explanation", "Application", "Conclusion"],
    "cleo":   ["Claim", "Law", "Evaluation", "Outcome"],
    "treacc": ["Topic", "Rule", "Explanation", "Analysis", "Counterarguments", "Conclusion"],
    "ireac":  ["Issue", "Rule", "Explanation", "Application", "Conclusion"],
}

# --- Parsing convention ---
FINAL_ANSWER_PREFIX = "Final Answer:"
FINAL_ANSWER_INSTRUCTION = (
    f"End your response with a line in exactly this format: {FINAL_ANSWER_PREFIX} <your answer>"
)


def extract_final_answer(raw_output: str) -> str | None:
    """
    Extracts the model's final answer, taking the LAST occurrence
    of the Final Answer prefix — important because some legal
    reasoning frameworks (e.g., CRAC, CREAC) instruct the model to
    restate a conclusion mid-reasoning, and some models echo the
    instruction text before the true final line.
    """
    matches = re.findall(r"Final Answer:\s*(.+)", raw_output)
    return matches[-1].strip() if matches else None


# --- Clustering / demonstration selection ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLUSTERING_RANDOM_STATE = 42

# --- Paths ---
# HPC_ROOT should be set to wherever this repo is cloned on Stanage
# (e.g. /users/msp25noe/dissertation/phase1_hpc), passed via the
# DISSERTATION_ROOT environment variable set in each .sbatch script,
# so paths aren't hardcoded to any one user's home directory layout.
HPC_ROOT = os.environ.get("DISSERTATION_ROOT", os.getcwd())

DATA_DIR = os.path.join(HPC_ROOT, "data", "legalbench_csv")
EVAL_POOLS_DIR = os.path.join(HPC_ROOT, "data", "eval_pools")
MANIFEST_PATH = os.path.join(HPC_ROOT, "data", "manifest.csv")
DEMO_DIR = os.path.join(HPC_ROOT, "demonstrations")
RAW_GEN_DIR = os.path.join(HPC_ROOT, "outputs", "raw_generations")
PARSED_DIR = os.path.join(HPC_ROOT, "outputs", "parsed_predictions")
LOG_DIR = os.path.join(HPC_ROOT, "outputs", "logs")
RESULTS_DIR = os.path.join(HPC_ROOT, "results")
FIGURES_DIR = os.path.join(HPC_ROOT, "figures")


def get_model_name_from_args() -> str:
    """
    Allows each SLURM job to specify which model to run via
    `python run_stage_a.py --model meta-llama/Llama-3.3-70B-Instruct`,
    so one script serves the entire Qwen -> Llama progression without
    editing this file per job.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--sample_size", type=int, default=None,
                         help="Optional: limit instances per task (for testing). "
                              "Omit for full Phase 1 eval pools.")
    args, _ = parser.parse_known_args()
    return args
