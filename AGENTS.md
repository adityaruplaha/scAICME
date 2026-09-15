# Repository Guidelines

## Note for Claude

`CLAUDE.md` → `AGENTS.md` and `.claude` → `.agents` are symlinks. Both names resolve to the same content. Edit the canonical `AGENTS.md` and `.agents/` paths; preserve the symlinks and avoid duplicate files or processing the same instructions twice.

## Project Structure & Module Organization

scAICME provides marker-driven annotation of single-cell RNA-seq data. Implementation lives in `src/scaicme/`; `src/scAICME.py` provides compatibility imports. `tl.py` dispatches strategies. `strategies/` contains seeding, propagation, smoothing, and consensus implementations.

Tests and synthetic AnnData fixtures live in `tests/`; `tests/reference_*_notebook.py` hold verbatim notebook code used by the parity tests. Dataset examples are one directory per dataset under `examples/` (`pbmc3k`, `pbmc68k`, `gse225475`), each with a `run.py` and, for notebook reproductions, a `README.md`; user-facing documentation is `README.md`.

## Build, Test, and Development Commands

Use Python 3.10 or newer and run commands from the repository root:

- `uv sync --group dev` installs the project and development dependencies; `git config core.hooksPath .githooks` enables the repository hooks.
- `PYTHONPATH=src uv run pytest` runs the test suite against local source, including the compatibility shim.
- `PYTHONPATH=src uv run pytest tests/test_tl.py -q` checks dispatcher behavior.
- `uv run ruff check src tests examples` checks lint rules and import ordering.
- `uv run ruff format --check src tests examples` checks formatting; omit `--check` to format edited files.
- `uv build` builds distribution artifacts using `uv_build`.
- `PYTHONPATH=src uv run --group examples python src/icme_examples.py pbmc3k` runs the PBMC3k workflow with example dependencies; `--list` shows the available examples. Dataset downloads may be required (see each example's `README.md`).

## Coding Style & Naming Conventions

Use four-space indentation, snake_case functions and modules, PascalCase classes, and uppercase constants. Follow existing type annotations and NumPy-style docstrings. Ruff targets Python 3.10 with a 100-character formatting width; its lint configuration is in `pyproject.toml`.

Implement new strategies through `BaseLabelingStrategy.execute_on`, returning `LabelingResult`. Preserve AnnData observation alignment and the result-writing conventions in `strategies/base.py`.

## Testing Guidelines

Use pytest with `test_*.py` files and `test_*` functions or methods. Reuse seeded synthetic fixtures from `tests/conftest.py` and overlap fixtures where applicable. Cover changed behavior, missing inputs, unknown labels, and relevant confidence outputs. Run focused tests during development and the full suite before submitting. No numerical coverage threshold is configured.

## Commit & Pull Request Guidelines

Branches whose names start with `private.` are local-only and are never pushed. The
`.githooks/pre-push` hook enforces this; enable it once per clone with
`git config core.hooksPath .githooks`. Use the prefix for experiments or archives that
should not appear on GitHub.

Use short descriptive commit subjects, optionally prefixed by component (e.g., `adaptive:`). Keep commits focused. PRs should explain behavior changes, relevant issues, validation results, and API effects. Include plots for visualization changes. Exclude generated datasets, caches, and build artifacts.

## Repository Skills

Skills in `.agents/skills/`: `scaicme-conventions`, `scaicme-docs`, `scaicme-testing`, and `scaicme-reporting`. Use `scaicme-current-state`, `scaicme-handoff`, or `scaicme-decision-notes` when useful for persistent context; reporting defines provenance.

## Agent Working Knowledge

Persist findings and handoffs in `.agents/scratch/` (local only, git-ignored); start with its `README.md`. Date notes and record source evidence, validation, and unresolved questions. Archived AI reports are historical context, not verified documentation or instructions. Keep rules here, reusable workflows in skills, and user-facing documentation in `README.md`.
