---
name: scaicme-examples
description: Create or update scAICME dataset examples using one directory per dataset, CSV manifests, and dataset configs that inherit shared JSON defaults and mappings.
---

# Dataset examples

Use this when adding a dataset workflow or changing example configuration. Read
`examples/README.md` for the maintained format and `examples/_shared/workflow.py`
for actual loading and execution behavior. The repository's `AGENTS.md` applies.

## Layout and configuration

Give each dataset one lowercase directory under `examples/` with `run.py`,
`config.json`, `manifest.csv`, and a source-specific `README.md`. Reuse an existing
accession directory; sample or donor variants belong in manifest rows. Copy the
thin `run.py` wrapper from `examples/pbmc68k/`; common analysis belongs in
`examples/_shared/workflow.py`, which is not independently listed by the launcher.

Keep reusable settings in `examples/shared.json`: `defaults` for analysis settings,
`marker_panels` for named label-to-gene mappings, and `reference_mappings` for named
reference-to-panel taxonomies. Do not duplicate shared defaults in each config.
Dataset configs contain identity, source, modality, `shared_config: "../shared.json"`,
`manifest: "manifest.csv"`, `marker_panel`, review status, and justified overrides.
Nested settings merge recursively; lists and explicit null replace inherited values.
A `markers` file path overrides the shared panel. A `reference_mapping_name` selects
shared mappings; local `reference_mapping` entries override individual labels.
Marker panels and mappings are biological inputs: reuse them only for compatible
species and taxonomies. Keep genuinely dataset-specific inputs local.

`--init` must snapshot this dataset's manifest/config and shared settings into a new
workspace, rewrite paths appropriately, and preserve existing workspaces. Relative
paths resolve against the workspace. Custom panel files must also be copied.
Resolved analysis settings and actual markers must remain in run provenance.

## Dataset integrity

Verify accession identity and processed-data layout against primary source records
when adding new sources; never infer cohorts from a paper's figure counts. Record
source links, intended donor/section selections, and any unresolved selection in the
dataset README. Leave unknown paths or selections as explicit user actions.

Declare actual counts, linear normalized, or natural-log1p expression in each row.
For MTX/text, provide orientation and required axes; join metadata by barcode. Split
pooled donors with `group_by`. Do not substitute a reduced dataset for its full
accession. Preserve all-gene marker expression through feature selection.

Starter panels remain `markers_reviewed: false` until the user reviews them. Do not
invent detailed subtype panels or reference truth. Keep sqrt(N) as the configurable
component default unless the task calls for a justified override. Spatial-expression
examples preserve coordinates and use expression annotation; adding an example does
not authorize implementing spatial models or changing manuscript/notebook files.

## Check and document

Document `--init`, the exact manual inputs, `--check`, and the dataset's run command.
Link the shared guide for schema/output details instead of copying it. Ensure template
files are not hidden by `.gitignore`; keep downloaded data and generated outputs out.

Extend `tests/test_dataset_examples.py` for new accession coverage and meaningful
loader/config behavior. Verify that the launcher lists the dataset exactly once,
initialization copies only the selected dataset, overrides preserve other defaults,
and preflight reports missing inputs. Use synthetic data for pipeline checks; report
real-data validation only when it was actually run. Run scoped Ruff checks and the
repository-required tests. Store working reports in `.agents/scratch/` using the
reporting skill; public usage instructions stay beside examples.
