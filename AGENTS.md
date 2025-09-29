# Repository Guidelines

## Project Structure & Module Organization

- `session_mgmt_mcp/` contains the runtime server (`server.py`), CLI entry point (`cli.py`), and integration layers such as `crackerjack_integration.py`.
- Feature domains live under `session_mgmt_mcp/core/`, `session_mgmt_mcp/tools/`, and `session_mgmt_mcp/utils/`; keep new modules small and protocol-driven to match existing patterns.
- Tests reside in `tests/` with fixtures in `tests/fixtures/` and strategy notes in `tests/README.md`; mirror package paths when adding cases.
- Documentation and reference material live in `docs/`; build outputs (`dist/`, `htmlcov/`, coverage artifacts) are generated—avoid committing manual edits there.

## Build, Test, and Development Commands

- `uv sync --group dev` installs runtime and development dependencies in a reproducible environment.
- `uv run session-mgmt-mcp --start-mcp-server --verbose` launches the MCP server locally; pair with `--status` to verify ports 8677/8678.
- `uv run pre-commit run --all-files` executes Ruff, Pyright, Bandit, Complexipy, and additional quality gates; fixes formatting as needed.
- `uv run pytest --cov=session_mgmt_mcp --cov-report=term-missing` runs the full suite with coverage; use `--maxfail=1` during rapid iteration.

## Coding Style & Naming Conventions

- Target Python 3.13 syntax, require explicit type hints, and import typing as `import typing as t` with `t.` prefixes.
- Prefer `pathlib.Path`, f-strings, dataclasses, and protocol-based abstractions; avoid docstrings and excess inline comments.
- Keep functions small (cognitive complexity ≤13) and use descriptive snake_case for modules, functions, and variables.
- Consolidate repeated logic into utilities to uphold DRY and KISS principles outlined in `RULES.md`.

## Testing Guidelines

- Write pytest cases alongside code under matching paths (`session_mgmt_mcp/foo.py` → `tests/unit/test_foo.py`); lean on async fixtures already provided.
- Preserve or raise the project’s 30.8% coverage toward the 42% minimum and long-term 85% goal noted in `docs/TEST_PROGRESS_REPORT.md`.
- Regenerate reports with `uv run pytest --cov ... --cov-report=html` and review `htmlcov/index.html` before submitting.
- Document flaky behavior in test notes and mark with `@pytest.mark.xfail` plus an issue link when a temporary skip is unavoidable.

## Commit & Pull Request Guidelines

- Follow conventional-style messages (`fix(core): tighten session cleanup`) even though automated `checkpoint:` commits exist; include scope when practical.
- Ensure PRs summarize intent, list executed commands, attach coverage deltas, and link related issues or MCP transcripts.
- Screenshots or logs are expected when touching CLI output or websocket telemetry; redact secrets and tokens.
- Request review only after pre-commit, pytest, and server smoke tests complete cleanly in your branch.

## Security & Configuration Tips

- Store MCP client configs in `example.mcp.json`-style files and keep credentials out of version control.
- Prefer `uv run session-mgmt-mcp --config` to audit runtime settings; never hardcode temp paths—use `tempfile` utilities as enforced by Bandit.
- Review permission changes via the `permissions` MCP tool before merging to maintain least-privilege defaults.
