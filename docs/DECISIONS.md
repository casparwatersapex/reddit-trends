# Decision log

## Data format
Choice: User-provided JSONL posts with full metadata preserved and a canonical schema for pipeline use.
Tradeoffs: More upfront validation work, but stable downstream analysis and exports.

## Embeddings
Choice: Support both local GPU models and OpenAI embeddings with a benchmark harness.
Tradeoffs: Local models reduce marginal cost and latency on GPU but may vary in quality; OpenAI is stable but adds per-call cost and network dependency.
Assumptions (locked for testing): For GardeningUK canonical data (~68k rows), local `sentence-transformers/all-MiniLM-L6-v2` estimated ~0.13 hours and $0.00; OpenAI `text-embedding-3-small` sampled ~4.1k tokens/sec, estimated ~0.36 hours and ~$0.11.

## Clustering
Choice: BERTopic with UMAP + HDBSCAN and no fixed topic count.
Tradeoffs: Requires parameter tuning and monitoring of the -1 cluster; avoids arbitrary category counts.

## Cluster labeling
Choice: LLM-generated human-readable labels and summaries, stored alongside prompts.
Tradeoffs: Extra runtime and cost but improves interpretability beyond keyword lists.

## Hierarchy
Choice: Build a two-level topic hierarchy via an LLM merge step.
Tradeoffs: Adds complexity but prevents overly fragmented topics and keeps UI usable.

## UI
Choice: Streamlit with Plotly sunburst and growth metrics (90/180/360 days).
Tradeoffs: Fast to iterate, less flexible than custom web apps.

## Fine-tuning guidance
Choice: Do not fine-tune initially; evaluate prompt-based labeling and clustering quality first.
Tradeoffs: Fine-tuning can be faster and cheaper at scale only if the taxonomy is stable and training data is large and consistent; otherwise it risks locking in weak categories.
