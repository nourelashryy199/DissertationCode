# ============================================================
# phase_0/scripts/validate_metrics.py
# Validates that Category Fit Score and Prompt Transfer Penalty
# behave as DESIGNED, using synthetic scenarios with a known
# correct answer. Imports the REAL phase_1_hpc/analysis.py.
#
# No model, no GPU, no real data required.
# Run via: python validate_metrics.py
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
sys.path.insert(0, os.path.join(PHASE1_DIR, "scripts"))

import pandas as pd
import numpy as np
import analysis

FAILURES = []


def check(condition, description):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {description}")
    if not condition:
        FAILURES.append(description)


def make_runs(category, strategy, values):
    return [{"category": category, "strategy": strategy, "run_id": i, "run_accuracy": v}
            for i, v in enumerate(values)]


# ============================================================
# SCENARIO 1 — Consistency vs. raw magnitude
# ============================================================
print("=== Scenario 1: Does Fit Score reward CONSISTENCY, not just raw accuracy? ===\n")

rows = []
rows += make_runs("Consistency", "zero_shot", [0.40, 0.40, 0.40, 0.40, 0.40])
rows += make_runs("Consistency", "volatile",  [0.20, 0.90, 0.70, 0.30, 1.00])
rows += make_runs("Consistency", "stable",    [0.53, 0.54, 0.56, 0.57, 0.55])

df = pd.DataFrame(rows)
fit = analysis.compute_fit_scores(df)

volatile = fit[(fit.category == "Consistency") & (fit.strategy == "volatile")].iloc[0]
stable   = fit[(fit.category == "Consistency") & (fit.strategy == "stable")].iloc[0]

print(f"  volatile: mean_accuracy={volatile.mean_accuracy:.3f}, std={volatile.std_accuracy:.3f}, fit_score={volatile.fit_score:.3f}")
print(f"  stable:   mean_accuracy={stable.mean_accuracy:.3f}, std={stable.std_accuracy:.3f}, fit_score={stable.fit_score:.3f}\n")

check(volatile.mean_accuracy > stable.mean_accuracy,
      "Sanity: 'volatile' genuinely has higher RAW accuracy than 'stable' (by construction)")
check(stable.fit_score > volatile.fit_score,
      "Fit Score REVERSES the raw-accuracy ranking: 'stable' scores higher despite lower mean accuracy")


# ============================================================
# SCENARIO 2 — Negative gain must produce a negative score
# ============================================================
print("\n=== Scenario 2: Does a genuinely WORSE strategy get a negative Fit Score? ===\n")

rows = []
rows += make_runs("NegControl", "zero_shot", [0.50, 0.50, 0.50, 0.50, 0.50])
rows += make_runs("NegControl", "bad_strategy", [0.30, 0.32, 0.28, 0.31, 0.29])

df2 = pd.DataFrame(rows)
fit2 = analysis.compute_fit_scores(df2)
bad = fit2[(fit2.category == "NegControl") & (fit2.strategy == "bad_strategy")].iloc[0]

print(f"  bad_strategy: mean_accuracy={bad.mean_accuracy:.3f}, accuracy_gain={bad.accuracy_gain:.3f}, fit_score={bad.fit_score:.3f}\n")
check(bad.accuracy_gain < 0, "bad_strategy genuinely underperforms zero_shot (negative gain, by construction)")
check(bad.fit_score < 0, "Fit Score is correctly negative for a genuinely worse strategy")


# ============================================================
# SCENARIO 3 — Tie-breaking behavior is deterministic
# ============================================================
print("\n=== Scenario 3: Deterministic tie-breaking when two strategies are equal ===\n")

rows = []
rows += make_runs("TieCase", "zero_shot", [0.40, 0.40, 0.40, 0.40, 0.40])
rows += make_runs("TieCase", "strategy_a", [0.60, 0.61, 0.59, 0.60, 0.60])
rows += make_runs("TieCase", "strategy_b", [0.60, 0.61, 0.59, 0.60, 0.60])

df3 = pd.DataFrame(rows)
fit3 = analysis.compute_fit_scores(df3)
champ3 = analysis.identify_champions(fit3)

a_score = fit3[(fit3.category == "TieCase") & (fit3.strategy == "strategy_a")].iloc[0].fit_score
b_score = fit3[(fit3.category == "TieCase") & (fit3.strategy == "strategy_b")].iloc[0].fit_score
print(f"  strategy_a fit_score={a_score:.4f}, strategy_b fit_score={b_score:.4f}")
print(f"  champion selected: {champ3['TieCase']}\n")

check(abs(a_score - b_score) < 1e-9, "Tied strategies genuinely produce equal Fit Scores")
check(champ3["TieCase"] in ("strategy_a", "strategy_b"),
      "Champion selection resolves ties deterministically (no crash/ambiguity)")


# ============================================================
# SCENARIO 4 — Transfer Penalty: hand-verify one matrix value
# ============================================================
print("\n=== Scenario 4: Hand-verified Transfer Penalty value across 3 categories ===\n")

rows = []
rows += make_runs("Cat1", "zero_shot", [0.30]*5)
rows += make_runs("Cat1", "strat_X",   [0.70, 0.71, 0.69, 0.70, 0.70])
rows += make_runs("Cat1", "strat_Y",   [0.50, 0.55, 0.45, 0.52, 0.48])

rows += make_runs("Cat2", "zero_shot", [0.40]*5)
rows += make_runs("Cat2", "strat_X",   [0.45, 0.60, 0.30, 0.55, 0.35])
rows += make_runs("Cat2", "strat_Y",   [0.65, 0.66, 0.64, 0.65, 0.65])

df4 = pd.DataFrame(rows)
fit4 = analysis.compute_fit_scores(df4)
champ4 = analysis.identify_champions(fit4)
print(f"  Champions: {dict(champ4)}\n")

penalty4 = analysis.compute_transfer_penalty(fit4, champ4, score_col="fit_score")
print(penalty4.round(4))

fit_lookup = fit4.set_index(["category", "strategy"])["fit_score"]
expected_penalty_1_to_2 = fit_lookup[("Cat2", champ4["Cat2"])] - fit_lookup[("Cat2", champ4["Cat1"])]
actual_penalty_1_to_2 = penalty4.loc["Cat1", "Cat2"]

print(f"\n  Hand-calculated Penalty(Cat1 -> Cat2) = {expected_penalty_1_to_2:.4f}")
print(f"  Matrix's Penalty(Cat1 -> Cat2)        = {actual_penalty_1_to_2:.4f}\n")

check(champ4["Cat1"] == "strat_X", "strat_X correctly identified as Cat1's champion (by construction)")
check(champ4["Cat2"] == "strat_Y", "strat_Y correctly identified as Cat2's champion (by construction)")
check(abs(expected_penalty_1_to_2 - actual_penalty_1_to_2) < 1e-9,
      "Transfer Penalty matrix value matches the hand-calculated formula exactly")
check(actual_penalty_1_to_2 > 0,
      "Penalty is positive: applying Cat1's champion to Cat2 is genuinely worse than Cat2's own champion")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All metric validation checks passed.")
    sys.exit(0)