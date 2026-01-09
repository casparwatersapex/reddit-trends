# Client Data Product Template (Python)

A starter template for repeatable client analytics projects:

- Upload data → validate → transform → run analysis → show dashboards → export a PowerPoint report.
- AI guardrails: `AGENTS.md` + `docs/agents/*` (“specialist memory”).
- Quality gates: Ruff + Pytest + pre-commit + GitHub Actions CI.

## Quickstart (Windows 11, venv + pip)

### 0) Prereqs
- Git
- Python 3.10+ (3.11 recommended)
- VS Code

### 1) Bootstrap the repo (recommended)
In PowerShell from the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
```

This will:
- create `.venv/`
- install dependencies (including dev tools)
- install pre-commit hooks

### 2) Run the app (Streamlit)
```powershell
.\.venv\Scripts\python.exe -m streamlit run src\client_portal\app\streamlit_app.py
```

### 3) Run checks
```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest -q
```

## Quickstart (macOS/Linux)
```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
streamlit run src/client_portal/app/streamlit_app.py
```

## How to use for a new client
1. Copy `clients/example_client/config.yml` to `clients/<client_name>/config.yml`
2. Upload data in the app or point the pipeline at a file path you control
3. Implement client-specific mapping/transforms in `src/client_portal/pipeline/transform.py`
4. Keep the **canonical output stable** so dashboards + PPTX stay reusable

## Repo docs
- `docs/PROJECT_SUMMARY.md` – living overview (update this as “human memory”)
- `docs/SECURITY_OVERVIEW.md` – client-friendly security posture
- `docs/agents/*` – domain notes (schema/infra/security/analytics)
