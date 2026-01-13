# Analytics agent notes

## What analysis means here
- Deterministic, repeatable metrics + charts.
- Prefer config-driven analysis rather than client-specific code forks.

## Where to implement
- `src/client_portal/analysis/metrics.py` for computed tables
- `src/client_portal/analysis/charts.py` for Plotly figure builders
- `scripts/embedding_benchmark.py` for stakeholder-friendly embedding benchmarks (supports live OpenAI sampling)
- `scripts/run_clustering.py` for BERTopic + top-N topic assignment outputs
- `scripts/compare_clusterings.py` for comparing clustering outputs across embeddings
- `scripts/name_clusters.py` for topic labeling experiments across sample sizes/backends
- `scripts/tune_clustering.py` for quick tuning runs to shrink -1 share
- `scripts/label_hierarchy.py` for L1/L2 hierarchy labeling
- `scripts/two_stage_clustering.py` for coarse-to-fine clustering on large topics
- `scripts/graph_clustering.py` for k-NN graph community detection
- `scripts/stability_consensus.py` for consensus clustering across runs
- `scripts/topic_refinement.py` for LLM-guided split/merge refinements
- `scripts/l3_clustering.py` for L3 clustering within large L2 topics
- `scripts/keyword_extraction.py` for keyword idea extraction from L2/L3 topics
- `scripts/keyword_growth.py` for growth summaries and timeseries by L3 topic
- `scripts/anchor_l3_grouping.py` for LLM-guided L3 anchors and keyword-based grouping
- `src/client_portal/analysis/label_review.py` for cleaning label text in review flows
- `src/client_portal/app/label_review_app.py` for interactive label review
