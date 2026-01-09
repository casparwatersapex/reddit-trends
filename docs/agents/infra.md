# Infra agent notes

## Deployment options (choose one)
- Local desktop app (single-user)
- Internal web app (company-only)
- Client-hosted deployment (their environment)

## Minimal v1 infra assumptions
- Runs as a Streamlit app.
- Stores uploaded files on disk for the current session only (configurable).
- For multi-user: add a database + object storage.
