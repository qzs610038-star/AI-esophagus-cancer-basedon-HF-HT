---
name: parallel-extract
description: Compatibility router for parallel PFMval extraction. Use when the user invokes the former parallel-extract Skill or asks to parallelize extraction jobs.
---

# Parallel Extract Compatibility Router

Read `../_shared/legacy-compatibility-policy.md` completely.

Lifecycle: `retired-router`.

Translate the request into explicit experiment, workspace, input partition,
resource, run-count, output, and acceptance contracts. Check that parallelism
does not duplicate samples or bypass an embargo. This router may prepare a
reviewable plan, but it must not spawn extraction workers or hide multiple
result-bearing runs inside one approval.
