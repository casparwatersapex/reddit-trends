# Project brief: Reddit topic clustering

## Problem
Teams need repeatable topic discovery from subreddit posts with metadata preserved for downstream export. Current pain: large -1 cluster, too-specific or unrelated topics, and unstable category counts.

## Goal
Build a pipeline and UI to ingest user-supplied Reddit JSONL, compute embeddings (local GPU or OpenAI), cluster with BERTopic without preset category counts, name clusters with an LLM, and present a 2-level hierarchy with growth signals.

## Users
- Analysts exploring subreddit themes
- Clients reviewing summarized trends

## Scope (v1)
- Input: JSONL posts from multiple subreddits (start with r/gardeninguk), preserving full metadata.
- Validation + canonical schema for posts.
- Embedding benchmarks: speed, cost, quality for local vs OpenAI models.
- BERTopic clustering with iterative parameter tuning; no fixed k.
- LLM cluster naming (human readable summaries) and L1/L2 hierarchy builder.
- Streamlit UI with sunburst + growth windows (90/180/360 days).
- Export-ready outputs (clustered JSONL + summary tables).

## Non-goals (v1)
- Fine-tuned classifiers or supervised taxonomy.
- Advanced NER or sentiment (planned later).
- Production multi-tenant deployment.

## Success metrics
- <25% in -1 cluster on test data or clear rationale for remaining outliers.
- Coherent cluster labels judged by analyst review.
- Embedding benchmark report with latency/cost/quality notes.
- UI supports fast browsing and growth detection.

## Risks and mitigations
- Quality variance across models -> benchmark harness + human review loop.
- Large -1 cluster -> tune UMAP/HDBSCAN and consider re-clustering outliers.
- Over-fragmentation -> merge step to create 2-level hierarchy.

## Open questions
- Target metadata fields for exports.
- GPU availability assumptions for local models.
- Acceptance criteria for cluster quality.
