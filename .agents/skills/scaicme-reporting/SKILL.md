---
name: scaicme-reporting
description: Create or update agent reports, knowledge notes, and handoffs in .agents/scratch/ with YAML provenance naming the generating model, harness, and current timestamp. Use for persistent agent working documents, including newly written archive reports.
---

# Agent Reporting Convention

Apply this convention to new Markdown working documents in `.agents/scratch/`, including indexes, knowledge notes, reports, and handoffs. `.claude/scratch/` resolves to the same location. Repository rules, skill definitions, public package documentation, and verbatim historical archives retain their own formats.

## Required YAML Frontmatter

Start each new document with a YAML block before its Markdown title:

```yaml
---
model: "<generating model reported by the active runtime>"
harness: "<active agent application or execution harness>"
generated_at: "<current ISO 8601 UTC timestamp>"
---
```

Replace every placeholder before saving. Use these exact field names and quote all values.

- `model`: use the most specific model identifier explicitly available in the active session or trusted runtime metadata. Preserve its spelling; do not infer a model from the harness, a configured default, or another agent's identity. If unavailable, use `"unknown"` and explain the limitation briefly in the report.
- `harness`: name the actual execution application, such as `"Codex"` or `"Claude Code"`, when established by the session. Include a version only if observed. Use `"unknown"` if the harness cannot be established.
- `generated_at`: read the clock immediately before creating the document. Use UTC with seconds and a `Z` suffix, for example the output of `date -u +%Y-%m-%dT%H:%M:%SZ`. Do not copy a timestamp from this skill, the conversation start, or a previous report.

The generating model is the agent writing the document. When incorporating another agent's findings, attribute those contributions in the body without replacing the document author's provenance. Never invent an exact model version or original generation time.

## Maintain Provenance

Preserve the original three fields on subsequent edits. For substantive revisions, add or refresh `updated_at` with the current UTC timestamp and `updated_by` as a mapping containing the revising agent's `model` and `harness`. This records the latest revision; Git can preserve the full history.

Do not backfill historical archives with the current agent's identity or timestamp. Moving a report does not regenerate it. If adding provenance to an older note is explicitly requested and its origin is unavailable, record unknown original values and separately identify the current revision.

## Report Content and Validation

Keep the body proportional to the task. Record the question or scope, findings with source paths or other evidence, validation actually performed, unresolved questions, and useful next steps. Distinguish observations from inferences and historical claims. Update `.agents/scratch/README.md` when adding a maintained topic.

Before finishing, parse the frontmatter as YAML and verify the required nonempty string fields, no remaining template placeholders, and an ISO 8601 UTC timestamp. Check that the identities match available session evidence. Check local Markdown links. A valid header establishes provenance, not factual correctness.
