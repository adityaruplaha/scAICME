---
name: scaicme-handoff
description: Write or refresh a concise scAICME task handoff when a user requests a checkpoint or substantial work needs continuation across sessions. Use to preserve actionable task context without imposing a report on routine edits.
---

# Task Handoffs

Store a handoff in `.agents/scratch/knowledge/handoff-<topic>.md`, using a short descriptive topic. Update an existing handoff for the same task rather than creating competing copies. Follow [scaicme-reporting](../scaicme-reporting/SKILL.md) for provenance.

## Leave Enough Context to Resume

Capture only what the next session needs:

- The user's objective and constraints that still apply.
- Completed work and relevant file paths; distinguish saved edits from proposals.
- Validation commands and actual outcomes, including checks not run when relevant.
- Unfinished work, concrete blockers, and the next useful action.

Inspect the current diff before describing changed files. Separate work performed in this session from unrelated pre-existing edits. Identify the branch or revision when useful, and mention uncommitted work. Preserve user decisions without turning a proposed next step into authorization for an external action.

Avoid transcripts, exhaustive tool logs, and copied code that is already available in the repository. Link supporting notes or specific files. Record temporary paths only when their contents are still available and needed.

## Resume or Close

When resuming, compare the handoff with current files before acting; another session may have changed them. When the task is completed, mark the handoff complete and remove stale next actions. Archive it only when it no longer serves as active context, preserving its provenance.

Link new handoffs from `.agents/scratch/README.md` and check links and frontmatter. Creating a handoff does not itself justify stopping authorized work; use it at a useful checkpoint or when requested.
