from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BERTopic clustering.")
    parser.add_argument("embeddings_path", help="Parquet with post_id/text/embedding.")
    parser.add_argument("--output-path", required=True, help="Parquet output path.")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--min-cluster-size", type=int, default=25)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--reassign-noise-threshold",
        type=float,
        default=None,
        help="Reassign -1 points to nearest topic if similarity exceeds threshold.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.embeddings_path)
    embeddings = np.vstack(df["embedding"].values)
    texts = df["text"].tolist()

    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    umap_model = UMAP(
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        metric="cosine",
        random_state=args.seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        verbose=True,
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

    top_topics, top_scores = compute_top_n_assignments(
        embeddings, topic_embeddings, topic_ids, args.top_n
    )

    out_df = pd.DataFrame(
        {
            "post_id": df["post_id"],
            "topic": topics,
            "top_n_topics": top_topics,
            "top_n_scores": top_scores,
        }
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    if args.model_path:
        topic_model.save(args.model_path)
    print(f"Wrote clustering output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
