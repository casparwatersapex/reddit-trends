# AGENTS.md — instructions for coding agents (Codex, etc.)

## Goal
This repo is a repeatable client analytics product: ingest → validate → transform → analyze → visualize → export.

## Non-negotiables
- Do NOT rename existing public functions/classes/modules or widely-used variables unless explicitly asked.
- Avoid drive-by refactors. Keep diffs small and reviewable.
- Prefer extending existing patterns over creating new ones.

## How to work here
1. **Start with a plan**: list files you will touch and why.
2. **Search the repo** before adding new utilities.
3. **Add/adjust tests** for any behavior change.
4. **Run checks** (below) before finishing.

## Commands (expected)
- Lint: `ruff check .`
- Format: `ruff format .`
- Tests: `pytest -q`

## Documentation updates (required)
After meaningful changes:
- Update `docs/PROJECT_SUMMARY.md` (keep it short and human-readable).
- If the change affects clients, update `docs/SECURITY_OVERVIEW.md` as needed.
- If the change affects one “domain”, update the matching file in `docs/agents/`:
  - infra: `docs/agents/infra.md`
  - schema/data contracts: `docs/agents/schema.md`
  - security: `docs/agents/security.md`
  - analytics: `docs/agents/analysis.md`

## Definition of Done
- Tests pass locally.
- Ruff passes (check + format).
- Docs updated where relevant.
