# Project plan

## Milestone 1: Data contract and ingestion
Checklist:
- Define canonical Reddit post schema for JSONL uploads.
- Validate required metadata fields and timestamps.
- Preserve raw metadata for export.
- Create test fixture for r/gardeninguk.

## Milestone 2: Embedding benchmark harness
Checklist:
- Implement local embedding option (GPU when available).
- Implement OpenAI embedding option.
- Collect latency, throughput, and estimated cost.
- Capture quality notes from quick manual review.

## Milestone 3: BERTopic clustering pipeline
Checklist:
- Configure UMAP + HDBSCAN without fixed topic count.
- Add parameter iteration support for relevance/accuracy.
- Track -1 cluster share and re-cluster strategy.
- Persist cluster assignments and metadata.

## Milestone 4: LLM labeling and hierarchy
Checklist:
- Generate human-readable cluster names and summaries.
- Build L1/L2 hierarchy with at most 2 layers.
- Add merge logic to avoid over-specific topics.
- Store prompts and outputs for audit.

## Milestone 5: Streamlit UI + trend analytics
Checklist:
- Sunburst diagram for L1/L2 topics.
- Growth metrics for 90/180/360 day windows.
- Filters by subreddit, time range, and topic.
- Export clustered JSONL and summary tables.

## Milestone 6: Quality review and tuning
Checklist:
- Review topic coherence and label quality.
- Document best-performing model and parameters.
- Update docs and client-facing guidance.

## Future milestones
Checklist:
- Add NER for brands, ingredients, conditions.
- Add sentiment by topic and entity.
- Evaluate supervised classifiers if taxonomy stabilizes.
