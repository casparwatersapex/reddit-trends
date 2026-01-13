from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def compute_topic_embeddings(
    embeddings: np.ndarray, topics: np.ndarray, topic_ids: list[int]
) -> np.ndarray:
    topic_vectors = []
    for topic_id in topic_ids:
        topic_mask = topics == topic_id
        topic_vectors.append(embeddings[topic_mask].mean(axis=0))
    return np.vstack(topic_vectors)


def compute_top_n_assignments(
    embeddings: np.ndarray, topic_embeddings: np.ndarray, topic_ids: list[int], top_n: int
) -> tuple[list[list[int]], list[list[float]]]:
    sims = cosine_similarity(embeddings, topic_embeddings)
    order = np.argsort(-sims, axis=1)[:, :top_n]
    top_topics: list[list[int]] = []
    top_scores: list[list[float]] = []
    for row_idx, indices in enumerate(order):
        top_topics.append([topic_ids[i] for i in indices])
        top_scores.append([float(sims[row_idx, i]) for i in indices])
    return top_topics, top_scores


def cluster_once(
    embeddings: np.ndarray,
    texts: list[str],
    min_cluster_size: int,
    min_samples: int,
    n_neighbors: int,
    n_components: int,
    seed: int,
) -> np.ndarray:
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        random_state=seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(texts, embeddings)
    return np.array(topics)


def map_clusters_to_base(
    base_labels: np.ndarray,
    run_labels: np.ndarray,
    jaccard_threshold: float,
) -> dict[int, int]:
    base_sizes = Counter(base_labels[base_labels != -1])
    mapping: dict[int, int] = {}
    for cluster_id in sorted(set(run_labels)):
        if cluster_id == -1:
            mapping[cluster_id] = -1
            continue
        members = np.where(run_labels == cluster_id)[0]
        if len(members) == 0:
            mapping[cluster_id] = -1
            continue
        base_counts = Counter(base_labels[members])
        base_counts.pop(-1, None)
        if not base_counts:
            mapping[cluster_id] = -1
            continue
        best_base, overlap = base_counts.most_common(1)[0]
        union = len(members) + base_sizes.get(best_base, 0) - overlap
        jaccard = overlap / union if union else 0.0
        mapping[cluster_id] = best_base if jaccard >= jaccard_threshold else -1
    return mapping


def build_consensus(
    base_labels: np.ndarray,
    mapped_runs: list[np.ndarray],
    consensus_k: int,
) -> np.ndarray:
    consensus = np.full_like(base_labels, fill_value=-1)
    for idx in range(len(base_labels)):
        votes = []
        if base_labels[idx] != -1:
            votes.append(int(base_labels[idx]))
        for run in mapped_runs:
            if run[idx] != -1:
                votes.append(int(run[idx]))
        if not votes:
            continue
        counts = Counter(votes)
        label, count = counts.most_common(1)[0]
        if count >= consensus_k:
            consensus[idx] = label
    return consensus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stability/consensus clustering.")
    parser.add_argument("embeddings_path", help="Parquet with post_id/text/embedding.")
    parser.add_argument("--output-path", required=True, help="Parquet output path.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--consensus-k", type=int, default=2)
    parser.add_argument("--jaccard-threshold", type=float, default=0.6)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--min-cluster-size", type=int, default=75)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--jitter", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.embeddings_path)
    embeddings = np.vstack(df["embedding"].values)
    texts = df["text"].tolist()

    rng = np.random.RandomState(args.seed)
    runs: list[np.ndarray] = []
    for run_idx in range(args.runs):
        jitter = int(rng.randint(-args.jitter, args.jitter + 1))
        run_labels = cluster_once(
            embeddings,
            texts,
            args.min_cluster_size,
            max(2, args.min_samples + jitter),
            max(5, args.n_neighbors + jitter),
            args.n_components,
            args.seed + run_idx,
        )
        runs.append(run_labels)

    base_labels = runs[0]
    mapped_runs = []
    for run_labels in runs[1:]:
        mapping = map_clusters_to_base(base_labels, run_labels, args.jaccard_threshold)
        mapped = np.array([mapping[int(label)] for label in run_labels])
        mapped_runs.append(mapped)

    consensus = build_consensus(base_labels, mapped_runs, args.consensus_k)
    topic_ids = sorted([t for t in set(consensus) if t != -1])
    topic_embeddings = compute_topic_embeddings(embeddings, consensus, topic_ids)
    top_topics, top_scores = compute_top_n_assignments(
        embeddings, topic_embeddings, topic_ids, args.top_n
    )

    out_df = pd.DataFrame(
        {
            "post_id": df["post_id"],
            "topic": consensus,
            "top_n_topics": top_topics,
            "top_n_scores": top_scores,
        }
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    print(f"Wrote consensus clustering output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
