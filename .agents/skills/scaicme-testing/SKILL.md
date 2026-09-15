---
name: scaicme-testing
description: Add or diagnose scAICME pytest regressions for annotation strategies, confidence handling, consensus, and AnnData dispatch. Use for behavioral validation of package changes.
---

# scAICME Testing

Run commands from the repository root. Consult `AGENTS.md` for environment setup; tests use pytest and synthetic AnnData rather than requiring PBMC downloads.

## Select relevant tests

- `tests/test_phase1.py`: seeding and related first-phase behavior.
- `tests/test_phase2.py`: propagation behavior.
- `tests/test_dpgmm.py`: mixture-model strategy behavior.
- `tests/test_consensus.py`: voting and agreement metadata.
- `tests/test_tl.py`: dispatcher input forms and result writing.
- `tests/test_complex_overlaps.py`: overlapping marker signals.

Inspect the current tests before assuming their expectations reflect the intended change. Reuse fixtures in `tests/conftest.py` and `tests/fixtures_complex_overlap.py` when suitable; use smaller targeted fixtures for cases that do not need their full preprocessing.

## Design meaningful regressions

Exercise the public API and assert observable label, confidence, or error behavior. For changes affecting result storage, check observation-index alignment and expected AnnData keys. Use explicit random seeds for stochastic data or models; avoid exact floating-point or label-count expectations unless the case makes them deterministic.

Select edge cases relevant to the change: missing markers or feature keys, absent or insufficient seeds, unknown labels, confidence thresholds, consensus ties, or overlapping markers. For propagation, distinguish seed retention from predictions on unlabeled cells. Do not loosen assertions just to make a changed algorithm pass; establish the intended behavior first.

## Run and report

Start with a focused command such as `PYTHONPATH=src uv run pytest tests/test_consensus.py -q`, then run the full suite for code changes before submission. Use `-k` to isolate a failing case. Run Ruff on edited Python files using the repository configuration.

Separate dependency or environment failures from test assertion failures, and record the command and outcome. No coverage percentage is configured. Documentation-only edits normally need link and snippet checks rather than a full scientific test run.
