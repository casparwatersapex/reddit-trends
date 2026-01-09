# Analytics agent notes

## What analysis means here
- Deterministic, repeatable metrics + charts.
- Prefer config-driven analysis rather than client-specific code forks.

## Where to implement
- `src/client_portal/analysis/metrics.py` for computed tables
- `src/client_portal/analysis/charts.py` for Plotly figure builders
