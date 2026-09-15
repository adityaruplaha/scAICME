---
name: scaicme-decision-notes
description: Record scAICME design or scientific reasoning that would otherwise be lost, including assumptions, considered alternatives, and reasons to revisit a choice. Use for consequential or surprising decisions, not routine implementation details.
---

# Lightweight Decision Notes

Use `.agents/scratch/knowledge/decision-<topic>.md` when the reasoning behind a choice will help later work. Follow [scaicme-reporting](../scaicme-reporting/SKILL.md) for provenance. Read an existing topic note before adding another.

## Explain the Choice

Keep the note short and include the relevant parts of:

- The concrete question or problem.
- The choice and its status: proposed, implemented, or superseded.
- The reason, assumptions, and evidence supporting it.
- Alternatives actually considered and the material tradeoff.
- What evidence or changed requirement would justify revisiting it.

For scAICME, useful topics could include treating marker scores as evidence rather than probabilities, preserving seed identities, or handling consensus abstentions. These are examples of decisions to document when made, not prescribed algorithm choices.

Link implementation paths, relevant tests, or experiment records. Distinguish user direction from an agent's proposal and observed results from a hypothesis. A note should not claim scientific validation simply because an implementation passes tests.

## Maintain Without Ceremony

Do not require numbering, approval meetings, or a note for every refactor. Do not invent a history of alternatives after the fact. If the original rationale is unknown, say so and label any reconstruction as an inference.

Update the note when the decision changes, preserving creation provenance and briefly explaining the reversal. Link a replacement when superseding a note. Add new topics to `.agents/scratch/README.md` and check frontmatter and links. Decision notes provide context; they do not override current user instructions or repository rules.
