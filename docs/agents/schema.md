# Schema / data contracts agent notes

## Canonical dataset contract (v1)
The pipeline should produce a canonical dataframe with:
- typed columns (dates parsed, numerics numeric)
- stable column names used by the analysis layer

## Where to define
- `clients/<client>/config.yml` describes column mapping and required fields.
- `src/client_portal/pipeline/validate.py` enforces required columns.
