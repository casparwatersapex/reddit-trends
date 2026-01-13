import numpy as np

from scripts.stability_consensus import build_consensus, map_clusters_to_base


def test_map_clusters_to_base_basic() -> None:
    base = np.array([1, 1, 2, 2, -1])
    run = np.array([10, 10, 20, 20, -1])
    mapping = map_clusters_to_base(base, run, jaccard_threshold=0.5)
    assert mapping[10] == 1
    assert mapping[20] == 2
    assert mapping[-1] == -1


def test_map_clusters_to_base_threshold() -> None:
    base = np.array([1, 1, 2, 2, -1, -1])
    run = np.array([10, 10, 10, 20, 20, 20])
    mapping = map_clusters_to_base(base, run, jaccard_threshold=0.8)
    assert mapping[10] == -1
    assert mapping[20] == -1


def test_build_consensus_majority_vote() -> None:
    base = np.array([1, 1, 2, 2, -1])
    run_a = np.array([1, 1, 2, -1, -1])
    run_b = np.array([1, 1, -1, 2, -1])
    consensus = build_consensus(base, [run_a, run_b], consensus_k=2)
    assert np.array_equal(consensus, np.array([1, 1, 2, 2, -1]))
