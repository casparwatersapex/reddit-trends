from __future__ import annotations

import numpy as np

from scripts.compare_clusterings import jaccard
from scripts.run_clustering import compute_top_n_assignments, reassign_noise_points


def test_compute_top_n_assignments_orders_by_similarity() -> None:
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    topic_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    topic_ids = [10, 20]

    top_topics, top_scores = compute_top_n_assignments(
        embeddings, topic_embeddings, topic_ids, top_n=1
    )

    assert top_topics == [[10], [20]]
    assert top_scores[0][0] > 0.9
    assert top_scores[1][0] > 0.9


def test_jaccard_handles_empty_sets() -> None:
    assert jaccard(set(), set()) == 1.0
    assert jaccard({1, 2}, {2, 3}) == 1 / 3


def test_reassign_noise_points_replaces_noise_when_threshold_met() -> None:
    embeddings = np.array([[0.0, 1.0], [0.0, 1.0]])
    topics = np.array([-1, 0])
    topic_embeddings = np.array([[0.0, 1.0]])
    topic_ids = [0]

    reassigned = reassign_noise_points(
        embeddings,
        topics,
        topic_embeddings,
        topic_ids,
        threshold=0.5,
    )

    assert reassigned[0] == 0
