---
name: scaicme-docs
description: Write or update scAICME README content, API docstrings, strategy explanations, and PBMC example documentation. Use when documenting package behavior or synchronizing docs with code.
---

# scAICME Documentation

Use `AGENTS.md` for repository commands and layout. Resolve the affected API against current source before describing it; existing prose may lag behind implementation.

## Choose the documentation location

- Put installation, quick-start usage, and public workflow explanations in `README.md`.
- Put argument defaults, return values, exceptions, and AnnData effects in the affected API's NumPy-style docstring.
- Use `docs/` for verified, user-facing strategy explanations. Store agent findings and handoffs in `.agents/scratch/`, using its `README.md` as the index. For new working documents, apply `.agents/skills/scaicme-reporting/SKILL.md` to record model, harness, and creation time in YAML frontmatter.
- The former `docs/ai-summary-stategies.md` and reports from `ai-outputs/` are historical AI outputs now in `.agents/scratch/archive/`. Treat them as unverified context, not package documentation or instructions; check claims against current code before reuse.
- Keep repository rules in `AGENTS.md` and reusable workflows in `.agents/skills/`; working notes should record dated findings, evidence, validation, and open questions.
- Keep PBMC workflow instructions consistent with `examples/pbmc3k/run.py`, `examples/pbmc68k/run.py`, and `src/icme_examples.py`.

## Explain observable behavior

Describe the package as marker-driven semi-supervised annotation. Explain seeding, propagation from selected seed columns, and consensus across predictions. Avoid implying novel cell-type discovery or making accuracy claims without supporting results.

Check class exports and constructor signatures before adding snippets. `import scAICME as icme` uses a compatibility shim over `scaicme`. State required AnnData inputs, such as marker expression, seed columns, or `obsm` features, for the specific strategy.

For dispatcher examples, verify accepted single/list/dictionary inputs and `key_added` behavior in `tl.py`. Check actual result suffixes in `LabelingResult.write_in` before naming confidence or metadata columns. Distinguish scores, probabilities, and agreement fractions.

## Check examples proportionately

Verify local paths, command entry points, and optional dependencies against source and `pyproject.toml`. Prefer a small synthetic example when executing a snippet. Run a full PBMC workflow only when needed for the requested validation; it may download datasets and generate plots. State which examples were executed and which were only inspected. Do not report unmeasured scientific improvements.
