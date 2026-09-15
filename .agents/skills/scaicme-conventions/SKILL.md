---
name: scaicme-conventions
description: Implement or review scAICME Python code using its strategy interfaces, AnnData result conventions, and compatibility imports. Use for strategy or dispatcher changes.
---

# scAICME Code Conventions

Read `AGENTS.md` and the affected implementation before editing. Paths below are relative to the repository root.

## Locate the change

- Keep implementation in `src/scaicme/`. `src/scAICME.py` is a compatibility shim; preserve imports through both names.
- Use the appropriate family under `strategies/seeding/`, `propagation/`, or `smoothing/`. Consensus lives in `strategies/consensus.py`.
- Inspect family base classes before duplicating logic. sklearn propagation shares preparation and confidence handling in `propagation/ml_base.py`.
- Export new public strategies through `src/scaicme/strategies/__init__.py` and relevant family exports.

## Preserve execution and result contracts

Implement the `name` property and `execute_on(adata)` interface defined in `strategies/base.py`. Return a `LabelingResult` referencing the input AnnData and strategy, with labels indexed by `adata.obs_names`.

Use the result payloads `obs`, `obsm`, and `uns` for auxiliary outputs. Inspect `LabelingResult.write_in` for suffixes and metadata keys; let the dispatcher manage writing results. Strategies may execute concurrently against the same AnnData, so avoid introducing shared input mutations during computation.

Preserve configurable unknown labels, confidence semantics, and seed-retention behavior where supported. Distinguish marker scores from probabilities; do not assume every confidence-like value is bounded by one. Check sparse versus dense inputs before changing expression-matrix handling.

## Validate the change

Follow existing type annotations and NumPy-style docstrings. Use Ruff settings from `pyproject.toml`; limit formatting changes to the files in scope. Run the relevant pytest modules with `PYTHONPATH=src uv run pytest`, adding behavioral regression cases for algorithm changes. Report any existing failures separately from regressions.
