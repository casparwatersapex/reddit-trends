# Project summary

## What this is
A repeatable analytics portal for client datasets.

Non-technical stakeholders should be able to:
- upload a dataset
- run standard transforms + analysis
- view dashboards
- export a PowerPoint report

## How to run locally
- Create venv, `pip install -e ".[dev]"`
- `streamlit run src/client_portal/app/streamlit_app.py`

## Architecture (high level)
- `pipeline/` handles ingestion, validation, transform, and produces a canonical dataframe
- `analysis/` computes metrics and chart specs
- `app/` is the UI (Streamlit)
- `reporting/` generates PPTX from charts

## Current constraints / known gotchas
- v1 expects CSV/Parquet-like tabular data.
- Each client is configured via `clients/<client>/config.yml`.

## Roadmap (next up)
- Add stronger data validation + nicer error messages
- Add multi-file uploads + joins
- Add authentication (if deployed externally)
