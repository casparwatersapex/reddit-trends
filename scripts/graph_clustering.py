from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(embeddings: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    return distances, indices


def run_leiden(indices: np.ndarray, distances: np.ndarray) -> list[int]:
    try:
        import igraph as ig  # type: ignore
        import leidenalg  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Leiden requires igraph + leidenalg.") from exc

    n_nodes = indices.shape[0]
    edges = []
    weights = []
    for i in range(n_nodes):
        for j, dist in zip(indices[i], distances[i], strict=False):
            if i == j:
                continue
            weight = 1.0 - float(dist)
            edges.append((i, int(j)))
            weights.append(weight)
    graph = ig.Graph(n=n_nodes, edges=edges, directed=False)
    graph.es["weight"] = weights
    partition = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition)
    return list(partition.membership)


def run_louvain(indices: np.ndarray, distances: np.ndarray) -> list[int]:
    try:
        import networkx as nx  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Louvain requires networkx.") from exc

    n_nodes = indices.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        for j, dist in zip(indices[i], distances[i], strict=False):
            if i == j:
                continue
            weight = 1.0 - float(dist)
            graph.add_edge(i, int(j), weight=weight)
    communities = nx.algorithms.community.louvain_communities(graph, weight="weight")
    memberships = [-1] * n_nodes
    for idx, community in enumerate(communities):
        for node in community:
            memberships[int(node)] = idx
    return memberships


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph-based topic clustering.")
    parser.add_argument("embeddings_path", help="Parquet with post_id/text/embedding.")
    parser.add_argument("--output-path", required=True, help="Parquet output path.")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--method", choices=["leiden", "louvain"], default="leiden")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.embeddings_path)
    embeddings = np.vstack(df["embedding"].values)

    distances, indices = build_knn_graph(embeddings, args.n_neighbors)
    if args.method == "leiden":
        try:
            topics = run_leiden(indices, distances)
        except RuntimeError:
            topics = run_louvain(indices, distances)
    else:
        topics = run_louvain(indices, distances)

    topics = np.array(topics)
    topic_ids = sorted([t for t in set(topics) if t != -1])
    if topic_ids:
        topic_embeddings = compute_topic_embeddings(embeddings, topics, topic_ids)
        top_topics, top_scores = compute_top_n_assignments(
            embeddings, topic_embeddings, topic_ids, args.top_n
        )
    else:
        top_topics = [[] for _ in range(len(df))]
        top_scores = [[] for _ in range(len(df))]

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
    print(f"Wrote graph clustering output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
