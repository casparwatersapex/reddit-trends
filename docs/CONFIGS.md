# Configs

## Reddit JSONL
- File: `clients/reddit/config.yml`
- Purpose: canonical schema for Reddit posts (post_id, date, subreddit, title, body, score, etc.).
- Notes: uses `created_utc` (seconds) for date parsing and `selftext` for body.

## Example client
- File: `clients/example_client/config.yml`
- Purpose: demo template for the generic portal (date/value).
