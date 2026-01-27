# Runbook

## End-to-end (recommended)
Use the Project Manager UI:
- `streamlit run src/client_portal/app/project_manager_app.py`
- Create a project, upload JSONL files, run pipeline.
- Enable "Resume pipeline" to skip completed steps.

## Manual pipeline (CLI)
Replace `<slug>` with your project slug.

1) Export canonical
- `python scripts/export_canonical.py projects/<slug>/raw/combined.jsonl clients/reddit/config.yml --output-path projects/<slug>/outputs/canonical.parquet`

2) Embeddings (OpenAI)
- `python scripts/compute_embeddings.py projects/<slug>/outputs/canonical.parquet --output-path projects/<slug>/outputs/embeddings_openai.parquet --backend openai --model text-embedding-3-small --batch-size 128 --stream-write`

3) Graph clustering (Leiden)
- `python scripts/graph_clustering.py projects/<slug>/outputs/embeddings_openai.parquet --output-path projects/<slug>/outputs/clusters_openai_graph.parquet --n-neighbors 30 --top-n 3 --method leiden`

4) Refinement (OpenAI)
- `python scripts/topic_refinement.py projects/<slug>/outputs/embeddings_openai.parquet projects/<slug>/outputs/clusters_openai_graph.parquet --output-path projects/<slug>/outputs/clusters_openai_refined.parquet --report-path projects/<slug>/outputs/topic_refinement.csv --backend openai --openai-model gpt-4o-mini --sample-size 20 --top-n 3`

5) L2 labels (OpenAI)
- `python scripts/name_clusters.py projects/<slug>/outputs/embeddings_openai.parquet projects/<slug>/outputs/clusters_openai_refined.parquet --output-path projects/<slug>/outputs/cluster_labels_openai_refined.csv --sample-sizes 20 --backends openai --openai-model gpt-4o-mini --resume`

6) L1/L2 hierarchy (OpenAI)
- `python scripts/label_hierarchy.py projects/<slug>/outputs/embeddings_openai.parquet projects/<slug>/outputs/clusters_openai_refined.parquet --output-path projects/<slug>/outputs/hierarchy_openai_refined_openai.csv --backend openai --openai-model gpt-4o-mini`

7) L3 anchors + labels (OpenAI)
- `python scripts/anchor_l3_grouping.py projects/<slug>/outputs/canonical.parquet projects/<slug>/outputs/embeddings_openai.parquet projects/<slug>/outputs/clusters_openai_refined.parquet --output-clusters projects/<slug>/outputs/clusters_openai_l3_anchor.parquet --output-anchors projects/<slug>/outputs/l3_anchors.csv --output-keywords projects/<slug>/outputs/keywords_l3_anchor.csv --output-labels projects/<slug>/outputs/cluster_labels_openai_l3_anchor.csv --l2-threshold 500 --sample-size 20 --min-similarity 0.2 --backend openai --openai-model gpt-4o-mini --embedding-model text-embedding-3-small --l2-labels-path projects/<slug>/outputs/cluster_labels_openai_refined.csv`

## Troubleshooting
- OpenAI failures: ensure `.env` has `OPENAI_API_KEY` and retry with `--resume` where supported.
- Unicode errors: re-run `scripts/name_clusters.py` after upgrading (safe output handling).
- Large uploads: Streamlit upload limit is set in `.streamlit/config.toml`.
