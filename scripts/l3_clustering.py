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


def cluster_l3(
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build L3 clusters within L2 topics.")
    parser.add_argument("embeddings_path", help="Parquet with post_id/text/embedding.")
    parser.add_argument("clusters_path", help="Parquet with L2 post_id/topic assignments.")
    parser.add_argument("--output-path", required=True, help="Parquet output path.")
    parser.add_argument("--l2-threshold", type=int, default=500)
    parser.add_argument("--min-cluster-size", type=int, default=25)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-n", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    embeddings_df = pd.read_parquet(args.embeddings_path)
    clusters_df = pd.read_parquet(args.clusters_path)
    merged = clusters_df.merge(embeddings_df, on="post_id", how="inner")
    embeddings = np.vstack(merged["embedding"].values)
    texts = merged["text"].tolist()

    l2_topics = merged["topic"].to_numpy()
    l3_topics = np.full_like(l2_topics, fill_value=-1)
    next_l3_id = 0

    l2_counts = merged.loc[merged["topic"] != -1, "topic"].value_counts()
    for l2_topic, count in l2_counts.items():
        if int(count) < args.l2_threshold:
            continue
        l2_mask = l2_topics == l2_topic
        sub_embeddings = embeddings[l2_mask]
        sub_texts = [texts[i] for i in np.where(l2_mask)[0]]
        sub_topics = cluster_l3(
            sub_embeddings,
            sub_texts,
            args.min_cluster_size,
            args.min_samples,
            args.n_neighbors,
            args.n_components,
            args.seed,
        )
        sub_unique = sorted([t for t in set(sub_topics) if t != -1])
        if len(sub_unique) <= 1:
            continue
        for sub_id in sub_unique:
            sub_mask = sub_topics == sub_id
            l3_topics[np.where(l2_mask)[0][sub_mask]] = next_l3_id
            next_l3_id += 1

    topic_ids = sorted([t for t in set(l3_topics) if t != -1])
    if topic_ids:
        topic_embeddings = compute_topic_embeddings(embeddings, l3_topics, topic_ids)
        top_topics, top_scores = compute_top_n_assignments(
            embeddings, topic_embeddings, topic_ids, args.top_n
        )
    else:
        top_topics = [[] for _ in range(len(merged))]
        top_scores = [[] for _ in range(len(merged))]

    out_df = pd.DataFrame(
        {
            "post_id": merged["post_id"],
            "topic": l3_topics,
            "parent_topic": l2_topics,
            "top_n_topics": top_topics,
            "top_n_scores": top_scores,
        }
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    print(f"Wrote L3 clusters: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
