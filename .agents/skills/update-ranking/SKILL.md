---
name: update-ranking
description: Compatibility router for PFMval result rankings. Use when the user invokes the former update-ranking Skill or asks to update experiment comparisons, leaderboards, or best-result summaries.
---

# Update Ranking Compatibility Router

Read `../_shared/legacy-compatibility-policy.md` completely.

Lifecycle: `retired-router`.

Derive rankings only from accepted, comparable experiment registry records and
their validated result envelopes. Preserve metric direction, cohort, phase,
protocol, exclusions, and uncertainty. Pending, explore, diagnostic, rejected,
or historical-only artifacts may be shown in separate labeled sections but
cannot change the accepted ranking.
