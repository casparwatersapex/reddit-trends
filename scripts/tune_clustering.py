from __future__ import annotations

import argparse
from dataclasses import dataclass

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


def reassign_noise_points(
    embeddings: np.ndarray,
    topics: np.ndarray,
    topic_embeddings: np.ndarray,
    topic_ids: list[int],
    threshold: float,
) -> np.ndarray:
    reassigned = topics.copy()
    noise_mask = topics == -1
    if not noise_mask.any():
        return reassigned
    sims = cosine_similarity(embeddings[noise_mask], topic_embeddings)
    best_idx = np.argmax(sims, axis=1)
    best_scores = sims[np.arange(len(best_idx)), best_idx]
    for i, score in enumerate(best_scores):
        if score >= threshold:
            reassigned[np.where(noise_mask)[0][i]] = topic_ids[best_idx[i]]
    return reassigned


@dataclass
class TuningResult:
    min_cluster_size: int
    min_samples: int
    n_neighbors: int
    topics: int
    noise_pct: float
    avg_cluster_size: float
    median_cluster_size: float


def compute_metrics(topics: np.ndarray) -> tuple[int, float, float, float]:
    unique = set(topics)
    topic_count = len(unique) - (1 if -1 in unique else 0)
    noise_pct = float((topics == -1).mean()) * 100
    counts = pd.Series(topics[topics != -1]).value_counts()
    avg_size = float(counts.mean()) if not counts.empty else 0.0
    median_size = float(counts.median()) if not counts.empty else 0.0
    return topic_count, round(noise_pct, 2), round(avg_size, 2), round(median_size, 2)


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search clustering parameters.")
    parser.add_argument("embeddings_path", help="Parquet with post_id/text/embedding.")
    parser.add_argument("--output-path", default="data/clustering_tuning.csv")
    parser.add_argument("--min-cluster-sizes", default="25,50")
    parser.add_argument("--min-samples", default="5,10")
    parser.add_argument("--n-neighbors", default="15,30")
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reassign-noise-threshold", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.embeddings_path)
    embeddings = np.vstack(df["embedding"].values)
    texts = df["text"].tolist()

    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    results: list[TuningResult] = []
    for min_cluster_size in parse_int_list(args.min_cluster_sizes):
        for min_samples in parse_int_list(args.min_samples):
            for n_neighbors in parse_int_list(args.n_neighbors):
                umap_model = UMAP(
                    n_neighbors=n_neighbors,
                    n_components=args.n_components,
                    metric="cosine",
                    random_state=args.seed,
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
                    verbose=False,
                )
                topics, _ = topic_model.fit_transform(texts, embeddings)
                topics = np.array(topics)

                topic_ids = sorted([t for t in set(topics) if t != -1])
                if topic_model.topic_embeddings_ is not None:
                    topic_embeddings = topic_model.topic_embeddings_[topic_ids]
                else:
                    topic_embeddings = compute_topic_embeddings(embeddings, topics, topic_ids)

                if args.reassign_noise_threshold is not None:
                    topics = reassign_noise_points(
                        embeddings,
                        topics,
                        topic_embeddings,
                        topic_ids,
                        args.reassign_noise_threshold,
                    )

                topic_count, noise_pct, avg_size, median_size = compute_metrics(topics)
                results.append(
                    TuningResult(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        n_neighbors=n_neighbors,
                        topics=topic_count,
                        noise_pct=noise_pct,
                        avg_cluster_size=avg_size,
                        median_cluster_size=median_size,
                    )
                )

    out_df = pd.DataFrame([r.__dict__ for r in results]).sort_values(["noise_pct", "topics"])
    out_df.to_csv(args.output_path, index=False)
    print(out_df.to_string(index=False))
    print(f"Wrote tuning results: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
