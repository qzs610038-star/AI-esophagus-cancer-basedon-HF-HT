---
name: compare
description: Compatibility router for comparing PFMval experiments, plans, checkpoints, or results. Use when the user invokes the former compare Skill or requests a current evidence-backed comparison.
---

# Compare Compatibility Router

Read `../_shared/legacy-compatibility-policy.md` completely.

Lifecycle: `retired-router`.

Use `pfmval-audit` for the comparison. Read the experiment registry and actual
imported result envelopes, distinguish accepted, pending, explore, historical,
and missing evidence, and compare only like-for-like protocol scopes. A file,
checkpoint, plot, or report that has not passed current import and acceptance
rules must not be ranked as accepted evidence.
