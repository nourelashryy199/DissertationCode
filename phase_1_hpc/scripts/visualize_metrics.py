# ============================================================
# scripts/visualize_metrics.py — Phase 1 (HPC)
# Produces every reasonable visualization from the currently
# available data: parsed generations (parse_predictions.py) and
# per-category/strategy/rephrasing accuracy (compute_metrics.py).
# All figures saved to config.FIGURES_DIR as PNG (non-interactive,
# matplotlib "Agg" backend — no display on Stanage).
# ============================================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import config

sns.set_theme(style="whitegrid")


def savefig(fig, name, safe_model_name):
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    path = os.path.join(config.FIGURES_DIR, f"{name}__{safe_model_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    model_name = config.get_model_name_from_args().model
    safe_model_name = model_name.replace("/", "_")

    parsed_path = os.path.join(config.PARSED_DIR, f"all_generations_parsed__{safe_model_name}.csv")
    per_strategy_path = os.path.join(config.RESULTS_DIR, f"metrics_per_strategy__{safe_model_name}.csv")
    per_strategy_rephrasing_path = os.path.join(config.RESULTS_DIR, f"metrics_per_strategy_rephrasing__{safe_model_name}.csv")

    for p in [parsed_path, per_strategy_path, per_strategy_rephrasing_path]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found. Run parse_predictions.py and compute_metrics.py first.")
            return

    df_raw = pd.read_csv(parsed_path)
    df_raw["is_correct"] = df_raw["is_correct"].fillna(False)
    df_strategy = pd.read_csv(per_strategy_path)
    df_rephrasing = pd.read_csv(per_strategy_rephrasing_path)

    strategy_order = config.ALL_STRATEGIES
    category_order = config.CATEGORIES

    pivot = df_strategy.pivot(index="category", columns="strategy", values="accuracy_mean")
    pivot = pivot.reindex(index=category_order, columns=strategy_order)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                cbar_kws={"label": "Accuracy"}, ax=ax)
    ax.set_title(f"Accuracy Heatmap: Category x Strategy ({model_name})")
    plt.xticks(rotation=45, ha="right")
    savefig(fig, "heatmap_category_strategy", safe_model_name)

    fig, ax = plt.subplots(figsize=(16, 7))
    sns.barplot(data=df_strategy, x="category", y="accuracy_mean", hue="strategy",
                order=category_order, hue_order=strategy_order, ax=ax)
    ax.set_title(f"Accuracy per Strategy, Grouped by Category ({model_name})")
    ax.set_ylabel("Mean Accuracy")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    savefig(fig, "grouped_bar_category_strategy", safe_model_name)

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, strategy in enumerate(strategy_order):
        sub = df_strategy[df_strategy["strategy"] == strategy].set_index("category").reindex(category_order)
        x = np.arange(len(category_order)) + i * 0.06
        ax.errorbar(x, sub["accuracy_mean"], yerr=sub["accuracy_std"], fmt="o", capsize=3, label=strategy, markersize=4)
    ax.set_xticks(np.arange(len(category_order)) + 0.06 * len(strategy_order) / 2)
    ax.set_xticklabels(category_order, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (mean +/- std)")
    ax.set_title(f"Accuracy with Variability, per Category and Strategy ({model_name})")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    savefig(fig, "errorbar_category_strategy", safe_model_name)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df_strategy, x="strategy", y="accuracy_mean", order=strategy_order, ax=ax)
    ax.set_title(f"Accuracy Distribution per Strategy, Across Categories ({model_name})")
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    savefig(fig, "boxplot_strategy_spread", safe_model_name)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_strategy, x="category", y="accuracy_mean", order=category_order, ax=ax)
    ax.set_title(f"Accuracy Distribution per Category, Across Strategies ({model_name})")
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=30, ha="right")
    savefig(fig, "boxplot_category_spread", safe_model_name)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
    for ax, category in zip(axes.flat, category_order):
        sub = df_rephrasing[df_rephrasing["category"] == category]
        for strategy in strategy_order:
            s = sub[sub["strategy"] == strategy].sort_values("rephrasing_id")
            ax.plot(s["rephrasing_id"], s["accuracy"], marker="o", label=strategy, linewidth=1)
        ax.set_title(category)
        ax.set_xlabel("Rephrasing ID")
        ax.set_xticks([0, 1, 2])
    axes.flat[0].set_ylabel("Accuracy")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=7, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"Accuracy Across Rephrasings, per Category and Strategy ({model_name})")
    savefig(fig, "lineplot_rephrasing_sensitivity", safe_model_name)

    overall = df_strategy.groupby("strategy")["accuracy_mean"].mean().reindex(strategy_order).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=overall.values, y=overall.index, ax=ax, palette="viridis")
    ax.set_title(f"Overall Strategy Ranking (Mean Accuracy Across Categories) ({model_name})")
    ax.set_xlabel("Mean Accuracy")
    savefig(fig, "ranked_bar_overall_strategy", safe_model_name)

    top5 = overall.head(5).index.tolist()
    angles = np.linspace(0, 2 * np.pi, len(category_order), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for strategy in top5:
        sub = df_strategy[df_strategy["strategy"] == strategy].set_index("category").reindex(category_order)
        values = sub["accuracy_mean"].tolist()
        values += values[:1]
        ax.plot(angles, values, marker="o", label=strategy)
        ax.fill(angles, values, alpha=0.05)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(category_order, fontsize=8)
    ax.set_title(f"Top 5 Strategies: Accuracy Profile Across Categories ({model_name})")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    savefig(fig, "radar_top5_strategy_profile", safe_model_name)

    df_raw["parsed_fail"] = df_raw["parsed_answer"].isna()
    fail_rate = (
        df_raw.groupby(["category", "strategy"])["parsed_fail"]
        .mean()
        .reset_index()
        .pivot(index="category", columns="strategy", values="parsed_fail")
        .reindex(index=category_order, columns=strategy_order)
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(fail_rate, annot=True, fmt=".1%", cmap="Reds", ax=ax, cbar_kws={"label": "Parsing Failure Rate"})
    ax.set_title(f"Parsing Failure Rate: Category x Strategy ({model_name})")
    plt.xticks(rotation=45, ha="right")
    savefig(fig, "heatmap_parsing_failures", safe_model_name)

    counts = df_raw.groupby("category").size().reindex(category_order)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="Blues_d")
    ax.set_title(f"Total Generations per Category ({model_name})")
    ax.set_ylabel("Generation Count")
    plt.xticks(rotation=30, ha="right")
    savefig(fig, "bar_generation_counts", safe_model_name)
# ============================================================
    # 11. TRANSFER PENALTY MATRIX — heatmap, entries are Penalty(i -> j)
    # ============================================================
    penalty_path = os.path.join(config.RESULTS_DIR, f"transfer_penalty_matrix__{safe_model_name}.csv")
    if os.path.exists(penalty_path):
        penalty_matrix = pd.read_csv(penalty_path, index_col=0)
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(penalty_matrix, annot=True, fmt=".3f", cmap="RdYlGn_r", center=0,
                    cbar_kws={"label": "Penalty(i -> j)"}, ax=ax)
        ax.set_xlabel("Category j (strategy applied TO)")
        ax.set_ylabel("Category i (champion strategy FROM)")
        ax.set_title(f"Prompt Transfer Penalty Matrix ({model_name})")
        savefig(fig, "matrix_transfer_penalty", safe_model_name)
    else:
        print(f"WARNING: {penalty_path} not found — run analysis.py first for the Transfer Penalty matrix.")
    # ============================================================
    # 12. TRANSFER PENALTY MATRIX (NORMALIZED, 0-1 scale) — heatmap
    # ============================================================
    penalty_norm_path = os.path.join(config.RESULTS_DIR, f"transfer_penalty_matrix_normalized__{safe_model_name}.csv")
    if os.path.exists(penalty_norm_path):
        penalty_matrix_norm = pd.read_csv(penalty_norm_path, index_col=0)
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(penalty_matrix_norm, annot=True, fmt=".3f", cmap="RdYlGn_r", vmin=0, vmax=1,
                    cbar_kws={"label": "Normalized Penalty(i -> j)"}, ax=ax)
        ax.set_xlabel("Category j (strategy applied TO)")
        ax.set_ylabel("Category i (champion strategy FROM)")
        ax.set_title(f"Prompt Transfer Penalty Matrix — Normalized 0-1 Scale ({model_name})")
        savefig(fig, "matrix_transfer_penalty_normalized", safe_model_name)
    else:
        print(f"WARNING: {penalty_norm_path} not found — run analysis.py first for the normalized Transfer Penalty matrix.")
    print("\nAll visualizations generated.")


if __name__ == "__main__":
    main()
