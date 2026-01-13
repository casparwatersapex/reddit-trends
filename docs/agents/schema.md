# Schema / data contracts agent notes

## Canonical dataset contract (v1)
The pipeline should produce a canonical dataframe with:
- typed columns (dates parsed, numerics numeric)
- stable column names used by the analysis layer

For Reddit JSONL ingestion, the canonical columns are:
- post_id
- date
- subreddit
- title
- body
- url
- author
- score
- num_comments
- is_self
- permalink

Validation metrics should track null counts for required columns plus invalid dates.

## Where to define
- `clients/<client>/config.yml` describes column mapping and required fields.
- `src/client_portal/pipeline/validate.py` enforces required columns.
