---
name: scaicme-current-state
description: Maintain a concise scAICME current-state note describing observed behavior, experimental areas, and open questions. Use when asked to capture project status or when substantial work changes an existing status note.
---

# Current-State Notes

Keep a short, evidence-based orientation in `.agents/scratch/knowledge/<branch>/current-state.md` (the scratch directory is git-ignored and shared across checkouts, so notes are kept per branch; use `git branch --show-current`). Read any existing note before updating it. Use [scaicme-reporting](../scaicme-reporting/SKILL.md) for creation and revision provenance.

## Capture What Matters Now

Inspect the relevant source, current diff, and available validation results. Describe:

- What was observed to work, with a source path or actual check supporting the claim.
- What remains experimental, incomplete, or unverified.
- The few open questions most relevant to the current task.

Identify the inspected revision when available and mention relevant uncommitted changes. A commit hash alone does not describe a dirty working tree. Distinguish passing software checks from evidence of biological annotation quality.

Prefer a few paragraphs or a compact list. Link detailed reports instead of reproducing them. A useful note might distinguish verified dispatcher behavior from an unmeasured change to seed selection; do not invent either finding without inspecting evidence.

## Keep It Lightweight

Update only claims affected by the work. Mark uncertain or outdated statements accordingly; do not rerun expensive experiments merely to populate the note. This is an orientation aid, not a release checklist, maturity score, or roadmap commitment.

Link a newly created note from `.agents/scratch/README.md`. Check its frontmatter and links. Do not create a status report for every small edit.
