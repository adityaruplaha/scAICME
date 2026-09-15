---
name: scaicme-examples
description: Create or update scAICME dataset examples as one self-contained run.py per dataset that reproduces the corresponding analysis notebook exactly, with a README recording data layout and parity evidence.
---

# Dataset examples

Use this when adding or changing a dataset workflow under `examples/`. The
repository's `AGENTS.md` applies. Examples on this branch are notebook
reproductions: the goal is identical outputs, not improved analysis.

## Layout

One lowercase directory per dataset under `examples/` (`pbmc68k`, `gse225475`, ...)
containing a self-contained `run.py` and a `README.md`. `src/icme_examples.py`
discovers any directory with a `run.py`; do not add shared helper modules under
`examples/`—reusable steps belong in the package (`scaicme.pp`, `scaicme.evaluation`,
strategies). Outputs go to `examples/<dataset>/outputs/` and data to
`data/<dataset>/`; both are git-ignored (`.gitignore` keeps `*.py` and `README.md`).

`run.py` reads its data directory from an environment variable with a sensible
default (`SCAICME_<DATASET>_DIR`, default `data/<dataset>`), fails with a clear
message when inputs are missing, and prints the same diagnostics the notebook
printed so runs can be compared. Configuration lives in module-level constants at
the top of the script (samples, marker panel, per-stage settings), each annotated
with the notebook cell it comes from.

## Reproducing a notebook

Transcribe the notebook stage by stage onto package strategies, passing every
setting explicitly rather than relying on package defaults. Where the package lacks
a behavior the notebook has, add it to the package (a strategy option or a new
strategy) instead of special-casing the example. Keep the notebook's own functions
verbatim in `tests/reference_<dataset>_notebook.py` and add exact-equality parity
tests in `tests/test_<dataset>_notebook_parity.py` on the synthetic fixture.

Validate on the real data: compare printed counts with the notebook output, and
when they differ, run the reference functions on the same in-memory data to
separate a port error from library-version drift. Record both in the README's
parity section with the environment used. Do not tune settings to close a gap that
the reference functions reproduce.

## Documenting

The README gives the data source and exact extraction layout (with a download
snippet), the run command, output files, a stage-by-stage mapping from notebook cells
to package calls, deliberate deviations (for example a fixed `random_state` the
notebook lacked), and the parity record. Refer to the source as "the <dataset>
notebook" and cite its content hash; do not name notebook files, directories, or the
original execution environment. Keep downloaded data and generated outputs out of
git. Update `README.md` at the repository root when adding an example.
