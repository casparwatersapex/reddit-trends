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


def cluster_once(
    embeddings: np.ndarray,
    texts: list[str],
    min_cluster_size: int,
    min_samples: int,
    n_neighbors: int,
    n_components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
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
    topics = np.array(topics)
    topic_ids = sorted([t for t in set(topics) if t != -1])
    if topic_model.topic_embeddings_ is not None:
        topic_embeddings = topic_model.topic_embeddings_[topic_ids]
    else:
        topic_embeddings = compute_topic_embeddings(embeddings, topics, topic_ids)
    return topics, topic_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage BERTopic clustering.")
    parser.add_argument("embeddings_path", help="Parquet with post_id/text/embedding.")
    parser.add_argument("--output-path", required=True, help="Parquet output path.")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--min-cluster-size", type=int, default=75)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split-min-size", type=int, default=1200)
    parser.add_argument("--stage2-min-cluster-size", type=int, default=25)
    parser.add_argument("--stage2-min-samples", type=int, default=5)
    parser.add_argument("--stage2-n-neighbors", type=int, default=15)
    parser.add_argument("--stage2-n-components", type=int, default=5)
    parser.add_argument("--stage2-seed", type=int, default=13)
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

    topics, topic_embeddings = cluster_once(
        embeddings,
        texts,
        args.min_cluster_size,
        args.min_samples,
        args.n_neighbors,
        args.n_components,
        args.seed,
    )

    topic_counts = pd.Series(topics).value_counts()
    large_topics = [
        int(topic_id)
        for topic_id, count in topic_counts.items()
        if topic_id != -1 and int(count) >= args.split_min_size
    ]

    next_topic_id = max([t for t in set(topics) if t != -1], default=0) + 1
    final_topics = topics.copy()
    for topic_id in large_topics:
        topic_mask = topics == topic_id
        if topic_mask.sum() < args.split_min_size:
            continue
        sub_embeddings = embeddings[topic_mask]
        sub_texts = [texts[i] for i in np.where(topic_mask)[0]]
        sub_topics, _ = cluster_once(
            sub_embeddings,
            sub_texts,
            args.stage2_min_cluster_size,
            args.stage2_min_samples,
            args.stage2_n_neighbors,
            args.stage2_n_components,
            args.stage2_seed,
        )
        unique_sub = sorted([t for t in set(sub_topics) if t != -1])
        if len(unique_sub) <= 1:
            continue
        for sub_id in unique_sub:
            sub_mask = sub_topics == sub_id
            final_topics[np.where(topic_mask)[0][sub_mask]] = next_topic_id
            next_topic_id += 1
        final_topics[np.where(topic_mask)[0][sub_topics == -1]] = -1

    final_topic_ids = sorted([t for t in set(final_topics) if t != -1])
    if not final_topic_ids:
        top_topics = [[] for _ in range(len(df))]
        top_scores = [[] for _ in range(len(df))]
        out_df = pd.DataFrame(
            {
                "post_id": df["post_id"],
                "topic": final_topics,
                "top_n_topics": top_topics,
                "top_n_scores": top_scores,
            }
        )
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(output_path, index=False)
        print(f"Wrote two-stage clustering output: {output_path}")
        return 0

    final_embeddings = compute_topic_embeddings(embeddings, final_topics, final_topic_ids)
    if args.reassign_noise_threshold is not None:
        final_topics = reassign_noise_points(
            embeddings,
            final_topics,
            final_embeddings,
            final_topic_ids,
            args.reassign_noise_threshold,
        )

    final_topic_ids = sorted([t for t in set(final_topics) if t != -1])
    final_embeddings = compute_topic_embeddings(embeddings, final_topics, final_topic_ids)
    top_topics, top_scores = compute_top_n_assignments(
        embeddings, final_embeddings, final_topic_ids, args.top_n
    )

    out_df = pd.DataFrame(
        {
            "post_id": df["post_id"],
            "topic": final_topics,
            "top_n_topics": top_topics,
            "top_n_scores": top_scores,
        }
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    print(f"Wrote two-stage clustering output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
