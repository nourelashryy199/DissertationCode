# ============================================================
# scripts/diagnose_parsing_failures.py — Phase 1 (HPC)
# Diagnostic: checks whether parsing-failure rate (no "Final
# Answer:" line found) is elevated for legal-framework strategies
# vs. generic ones, and whether that correlates with step count
# (a proxy for how likely a strategy is to be truncated by the
# shared MAX_NEW_TOKENS cap). Read-only — does not touch
# generation, parsing, or scoring outputs.
#
# Run via:
#   python diagnose_parsing_failures.py --model <model_name>
# ============================================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import pandas as pd


def main():
    model_name = config.get_model_name_from_args().model
    safe_model_name = model_name.replace("/", "_")

    parsed_path = os.path.join(config.PARSED_DIR, f"all_generations_parsed__{safe_model_name}.csv")
    if not os.path.exists(parsed_path):
        print(f"ERROR: {parsed_path} not found. Run parse_predictions.py first.")
        return

    df = pd.read_csv(parsed_path)
    print(f"Loaded {len(df)} generations for {model_name}\n")

    df["parsed_fail"] = df["parsed_answer"].isna()
    df["strategy_type"] = df["strategy"].apply(
        lambda s: "legal_framework" if s in config.LEGAL_FRAMEWORK_STRATEGIES else "generic"
    )
    df["step_count"] = df["strategy"].apply(
        lambda s: len(config.FRAMEWORK_STEPS[s]) if s in config.FRAMEWORK_STEPS else 0
    )

    print("=== Parsing failure rate per strategy (highest first) ===")
    print(df.groupby("strategy")["parsed_fail"].mean().sort_values(ascending=False).to_string())

    print("\n=== Parsing failure rate: legal_framework vs generic ===")
    print(df.groupby("strategy_type")["parsed_fail"].mean().to_string())

    print("\n=== Parsing failure rate vs step count (frameworks only) ===")
    framework_df = df[df["strategy_type"] == "legal_framework"]
    print(
        framework_df.groupby(["strategy", "step_count"])["parsed_fail"]
        .mean()
        .sort_values(ascending=False)
        .to_string()
    )

    # If raw_output is available, check completion length as a more
    # direct truncation signal (word count as a rough token proxy).
    if "raw_output" in df.columns:
        df["output_word_count"] = df["raw_output"].astype(str).str.split().str.len()
        print("\n=== Mean output word count per strategy (proxy for token usage) ===")
        print(df.groupby("strategy")["output_word_count"].mean().sort_values(ascending=False).to_string())

        print("\n=== Correlation: output length vs parsing failure (within legal frameworks) ===")
        for strategy in config.LEGAL_FRAMEWORK_STRATEGIES:
            sub = df[df["strategy"] == strategy]
            if len(sub) == 0:
                continue
            failed_len = sub.loc[sub["parsed_fail"], "output_word_count"].mean()
            ok_len = sub.loc[~sub["parsed_fail"], "output_word_count"].mean()
            print(f"  {strategy}: mean length when FAILED={failed_len:.1f} words, "
                  f"when PARSED OK={ok_len:.1f} words")


if __name__ == "__main__":
    main()
