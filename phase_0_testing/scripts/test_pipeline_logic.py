# ============================================================
# phase_0/scripts/test_pipeline_logic.py
# Validates the deterministic LOGIC of the pipeline — prompt
# construction, answer parsing — using small synthetic inputs.
# Imports the REAL phase_1_hpc modules (not copies), so this
# always tests the actual current production code.
#
# No model loading, no GPU, no internet. Safe to run on a laptop.
# Run via: python test_pipeline_logic.py
# ============================================================

import os
import sys

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
import strategy_functions as sf

FAILURES = []


def check(condition, description):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {description}")
    if not condition:
        FAILURES.append(description)


# ------------------------------------------------------------
# 1. Schema construction
# ------------------------------------------------------------
print("\n=== 1. Schema objects ===")

dummy_task = schema.LegalTask(
    task_id="dummy_0",
    task_type="rule-application",
    context="The mark 'Ice' for an ice cream shop.",
    question="How should this trademark be classified?",
    label_options=["generic", "descriptive", "suggestive", "arbitrary", "fanciful"],
    expected_output="descriptive",
)
check(dummy_task.task_id == "dummy_0", "LegalTask constructs with expected fields")

demos = [
    schema.Demonstration(context="The mark 'Pictures' for a photography service.", question="", label="generic"),
    schema.Demonstration(context="The mark 'Shark' for a custom t-shirt maker.", question="", label="arbitrary"),
]
check(len(demos) == 2, "Demonstration list has expected length (mirrors real UNIFORM_K=2 case)")
check("Final Answer: generic" in demos[0].render(), "Demonstration.render() includes Final Answer line")


# ------------------------------------------------------------
# 2. Prompt construction — all 13 strategies must build without error
# ------------------------------------------------------------
print("\n=== 2. Strategy prompt construction ===")

demonstration_sets = {"dummy_0": demos}

built_prompts = {}
for strategy in config.ALL_STRATEGIES:
    try:
        prompt = sf.build_prompt(dummy_task, strategy, rephrasing_id=0, task_id="dummy_0",
                                  demonstration_sets=demonstration_sets)
        built_prompts[strategy] = prompt
        ok = isinstance(prompt, str) and len(prompt) > 0 and config.FINAL_ANSWER_PREFIX in prompt
        check(ok, f"'{strategy}' builds a non-empty prompt containing the Final Answer instruction")
    except Exception as e:
        check(False, f"'{strategy}' builds without raising an exception (raised: {e})")

if "few_shot_2" in built_prompts and "few_shot_3" in built_prompts:
    check(
        built_prompts["few_shot_2"] == built_prompts["few_shot_3"],
        "few_shot_2 and few_shot_3 produce IDENTICAL prompts when only 2 demos are available "
        "(expected/known limitation — see Methodology; should stay TRUE until demo count changes)"
    )

zero_shot_variants = {
    sf.build_prompt(dummy_task, "zero_shot", r, "dummy_0", demonstration_sets)
    for r in range(config.N_REPHRASINGS)
}
check(len(zero_shot_variants) == config.N_REPHRASINGS,
      f"zero_shot produces {config.N_REPHRASINGS} distinct rephrasing variants")

for framework, steps in config.FRAMEWORK_STEPS.items():
    prompt = built_prompts.get(framework, "")
    all_steps_present = all(step in prompt for step in steps)
    check(all_steps_present, f"'{framework}' prompt mentions all its canonical steps {steps}")


# ------------------------------------------------------------
# 3. Answer parsing (config.extract_final_answer)
# ------------------------------------------------------------
print("\n=== 3. Answer parsing edge cases ===")

check(config.extract_final_answer("blah blah\nFinal Answer: Yes") == "Yes",
      "extracts a simple, well-formed final answer")

check(config.extract_final_answer("no final answer line here") is None,
      "returns None when no Final Answer line is present")

multi = "Final Answer: Draft\nsome more reasoning\nFinal Answer: Confirmed"
check(config.extract_final_answer(multi) == "Confirmed",
      "takes the LAST occurrence when Final Answer appears multiple times (framework restatement case)")

check(config.extract_final_answer("Final Answer:   Yes   ") == "Yes",
      "strips surrounding whitespace from the extracted answer")

check(config.extract_final_answer("Final Answer: Yes.") == "Yes.",
      "extraction is exact-match (trailing period NOT stripped — normalization happens in rescore.py)")


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All checks passed.")
    sys.exit(0)