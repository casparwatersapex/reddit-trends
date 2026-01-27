# Project summary

## What this is
A repeatable analytics portal focused on Reddit topic discovery from user-supplied JSONL posts.

Non-technical stakeholders should be able to:
- upload subreddit post JSONL with metadata preserved
- run embeddings and topic clustering without preset category counts
- review human-readable topic labels and a two-level hierarchy
- view dashboards including growth trends
- export clustered JSONL and summary tables

## How to run locally
- Create venv, `pip install -e ".[dev]"`
- `streamlit run src/client_portal/app/streamlit_app.py`
- Label review UI: `streamlit run src/client_portal/app/label_review_app.py` (defaults to refined OpenAI L2/L3 outputs; includes L3 growth tab)
- Project manager UI: `streamlit run src/client_portal/app/project_manager_app.py` (creates named projects, runs OpenAI pipeline, and reuses review + growth views)
- Reddit JSONL config: `clients/reddit/config.yml`
- Streamlit upload limit is set to 5 GB in `.streamlit/config.toml`.
- Project manager supports resume mode to skip steps whose outputs already exist.

## How to stay up to date
- `docs/STATE.md` for current defaults and active status.
- `docs/RUNBOOK.md` for step-by-step pipeline execution.
- `docs/DATASETS.md` for canonical input inventory.
- `docs/CONFIGS.md` for client config conventions.
- `docs/CHANGES.md` for recent updates.

## Architecture (high level)
- `pipeline/` handles ingestion (CSV/Parquet/JSONL), validation, and canonical schema
- `scripts/export_canonical.py` exports canonical Parquet with basic validation metrics (JSONL uses chunked reads)
- `scripts/embedding_benchmark.py` produces a simple time/cost benchmark table for embeddings (optional live OpenAI sampling)
- `scripts/compute_embeddings.py` writes full-post embeddings to Parquet (local or OpenAI)
- `scripts/run_clustering.py` runs BERTopic and stores top-N topic assignments per post
- `scripts/two_stage_clustering.py` runs a coarse-to-fine BERTopic pass for large clusters
- `scripts/graph_clustering.py` runs k-NN graph clustering (Leiden/Louvain)
- `scripts/stability_consensus.py` builds a consensus clustering across multiple runs
- `scripts/topic_refinement.py` applies LLM-guided split/merge refinements
- `scripts/l3_clustering.py` builds L3 clusters within large L2 topics (thresholded)
- `scripts/keyword_extraction.py` derives keyword ideas from L2/L3 topics for niche discovery
- `scripts/keyword_growth.py` computes growth summaries and timeseries for L3 keywords
- `scripts/anchor_l3_grouping.py` builds L3 anchors with LLM guidance and assigns posts by keyword similarity
- `scripts/compare_clusterings.py` compares clustering outputs across embedding backends
- `scripts/name_clusters.py` labels topics using sample sizes (10/30/50) via local or OpenAI models
- `scripts/tune_clustering.py` runs a small parameter grid to reduce -1 and topic count
- `scripts/label_hierarchy.py` builds and labels a simple L1/L2 hierarchy
- `analysis/` computes topic metrics and chart specs
- `app/` is the UI (Streamlit), including label review and L3 growth tabs
- `app/project_manager_app.py` manages repeatable projects with JSONL uploads and OpenAI-first clustering defaults
- `projects.py` stores project metadata and standard output paths under `projects/`
- `reporting/` generates PPTX from charts

## Current constraints / known gotchas
- v1 focuses on JSONL posts with a canonical Reddit column set (post_id, date, subreddit, title, body, url, author, score, num_comments, is_self, permalink).
- Topic quality depends on embedding choice and BERTopic parameters.
- No fixed category count; tuning is required to manage the -1 cluster.

## Roadmap (next up)
- Embedding benchmark harness (local GPU vs OpenAI)
- BERTopic clustering with LLM labels and 2-level hierarchy
- Streamlit sunburst UI + growth windows (90/180/360 days)
- Future: NER and sentiment by topic and entity

## Benchmarks (locked for testing)
- GardeningUK canonical data: local `sentence-transformers/all-MiniLM-L6-v2` ~0.13 hours, $0.00; OpenAI `text-embedding-3-small` ~0.36 hours, ~$0.11 (sampled ~4.1k tokens/sec).
