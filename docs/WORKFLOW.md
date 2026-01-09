# Workflow (human + agent)

This repo is designed so you **don’t have to remember rules**:
- Coding rules live in `AGENTS.md` (read automatically by Codex).
- Domain “specialist memory” lives in `docs/agents/*`.
- Formatting/lint/testing is enforced by pre-commit + CI.

## Daily loop (recommended)
1. Create a branch.
2. Use Codex in VS Code to propose a plan and implement.
3. Run the VS Code task **Check (ruff + tests)** or run:
   - `ruff check .`
   - `ruff format --check .`
   - `pytest -q`
4. Update docs:
   - `docs/PROJECT_SUMMARY.md`
   - and any relevant `docs/agents/*.md`
5. Commit, push, open PR. CI repeats the checks.

## Orchestrator prompt (copy/paste)
Ask Codex:

- Read `AGENTS.md` and relevant `docs/agents/*.md`.
- Propose a short plan and list the files you will touch.
- Implement with minimal diffs.
- Run ruff + tests.
- Update `docs/PROJECT_SUMMARY.md` and the relevant `docs/agents/*.md`.
- Summarize changes and any risks.
