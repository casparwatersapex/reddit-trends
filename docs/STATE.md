# Project state

## Current defaults
- Embeddings: OpenAI `text-embedding-3-small` (full post text: title + body).
- L2 clustering: k-NN graph clustering (Leiden, Louvain fallback).
- Refinement: LLM-guided split/merge (OpenAI `gpt-4o-mini`).
- L2 labels: OpenAI `gpt-4o-mini` with 20 examples per topic.
- L1/L2 hierarchy: OpenAI `gpt-4o-mini`.
- L3: anchor-based grouping + labels (OpenAI `gpt-4o-mini`).

## Data and outputs
- Canonical inputs live under `projects/<slug>/raw/`.
- Outputs live under `projects/<slug>/outputs/` (canonical, embeddings, clusters, labels).
- Canonical schema is defined in `clients/reddit/config.yml`.

## UI entry points
- Project manager: `src/client_portal/app/project_manager_app.py`
- Label review: `src/client_portal/app/label_review_app.py`

## Status
- Stable: canonical export, embeddings, clustering, labeling, review apps.
- Experimental: alternative clustering methods (BERTopic/HDBSCAN).
