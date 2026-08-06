# ============================================================
# scripts/build_demonstrations.py — Phase 1 (HPC)
# Reads LegalBench data from LOCAL CSVs (already downloaded from
# Colab), instead of calling load_dataset() directly — since
# Stanage's compute nodes typically have no internet access, and
# this script needs to be safely runnable inside an sbatch job.
# ============================================================

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np


def load_task_field_map():
    with open(os.path.join(config.HPC_ROOT, "data", "task_field_map.json")) as f:
        return json.load(f)


def load_local_task_data(task_id: str) -> list:
    """Reads train.csv + test.csv from the pre-downloaded local folder, combines into one pool."""
    task_dir = os.path.join(config.DATA_DIR, task_id)
    train_path = os.path.join(task_dir, "train.csv")
    test_path = os.path.join(task_dir, "test.csv")

    pool = []
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        pool.extend(train_df.to_dict(orient="records"))
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        pool.extend(test_df.to_dict(orient="records"))

    return pool


def main():
    manifest_df = pd.read_csv(config.MANIFEST_PATH)
    task_field_map = load_task_field_map()

    embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    MAX_SHOTS = max(config.DEMO_REQUIRED_STRATEGIES.values())

    combined_pools, task_embeddings, task_best_k = {}, {}, {}

    for _, row in manifest_df.iterrows():
        task_id = row["task_id"]
        pool = load_local_task_data(task_id)
        combined_pools[task_id] = pool

        field_map = task_field_map[task_id]
        texts = [str(r.get(field_map["context"], "")) for r in pool]
        embeddings = embedder.encode(texts, show_progress_bar=False)
        task_embeddings[task_id] = embeddings

        max_k = min(MAX_SHOTS, len(pool) - 1, 8)
        best_k, best_score = 2, -1
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=config.CLUSTERING_RANDOM_STATE, n_init=10)
            labels = km.fit_predict(embeddings)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_k, best_score = k, score

        task_best_k[task_id] = best_k
        print(f"{task_id}: pool={len(pool)}, own silhouette-optimal k={best_k} (score={best_score:.3f})")

    UNIFORM_K = min(task_best_k.values())
    print(f"\nUniform demonstration count applied to ALL tasks: k={UNIFORM_K}\n")

    os.makedirs(config.DEMO_DIR, exist_ok=True)
    os.makedirs(config.EVAL_POOLS_DIR, exist_ok=True)

    for _, row in manifest_df.iterrows():
        task_id = row["task_id"]
        pool = combined_pools[task_id]
        field_map = task_field_map[task_id]
        embeddings = task_embeddings[task_id]

        kmeans = KMeans(n_clusters=UNIFORM_K, random_state=config.CLUSTERING_RANDOM_STATE, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        demo_indices = []
        for cluster_id in range(UNIFORM_K):
            member_idxs = np.where(cluster_labels == cluster_id)[0]
            cluster_embeddings = embeddings[member_idxs]
            centroid = kmeans.cluster_centers_[cluster_id]
            dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            demo_indices.append(int(member_idxs[np.argmin(dists)]))

        demos, demo_context_texts = [], set()
        for idx in demo_indices:
            r = pool[idx]
            context = str(r.get(field_map["context"], ""))
            question = str(r.get(field_map["question"], "")) if field_map.get("question") else ""
            label = str(r.get("answer", ""))
            demos.append({"context": context, "question": question, "label": label})
            demo_context_texts.add(context)

        with open(os.path.join(config.DEMO_DIR, f"{task_id}_demos.json"), "w") as f:
            json.dump(demos, f, indent=2)

        eval_pool = [r for r in pool if str(r.get(field_map["context"], "")) not in demo_context_texts]
        with open(os.path.join(config.EVAL_POOLS_DIR, f"{task_id}_eval.json"), "w") as f:
            json.dump(eval_pool, f, indent=2, default=str)

        n_excluded = len(pool) - len(eval_pool)
        print(f"{task_id}: demos={len(demos)}, rows_excluded_from_eval={n_excluded}, eval_pool_size={len(eval_pool)}")

    print("\nDemonstration building complete.")


if __name__ == "__main__":
    main()