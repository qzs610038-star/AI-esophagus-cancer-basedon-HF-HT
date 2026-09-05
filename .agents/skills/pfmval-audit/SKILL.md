---
name: pfmval-audit
description: Independently review PFMval plans, code, experiments, results, or completion claims when the user asks for verification or a GO/NO-GO judgment.
---

# PFMval Independent Audit

Use current, task-relevant evidence and remain read-only unless the user also asks for a repair. Do not treat a chat summary, generated view, file presence, implementation, or passing unit test as proof of an experimental claim.

## Method

1. State the claim and the decision it would support.
2. Inspect only the relevant source, configuration, experiment record, result, and user directive.
3. Separate observed facts, accepted results, pending or exploratory material, and missing evidence.
4. Report checks as `PASS`, `WARN`, or `FAIL`, then give a bounded verdict: `GO`, `CONDITIONAL GO`, `NO-GO`, or `NOT APPLICABLE`.

Do not create worktrees without an explicit user instruction or run routine pre-experiment Git checks. The current workflow uses independent packages and manual returns; Gitee is paused. Use `python deploy/pfmval_ops.py agent start-check` only for a task-specific diagnostic need; `--strict`, Registry alignment, W### identity, transport choice, approvals stored as files, and hashes are not universal prerequisites. Treat them as warnings unless the claim specifically depends on them. Hashing is optional and should be used only when the user requests identity verification of a named critical artifact.

The audit may block a positive conclusion only when evidence for that exact claim is missing or contradictory, the action exceeds user authorization, it risks destructive changes, it introduces scientific leakage, or the claimed operation/result is not real. Historical workflow deviations alone do not justify `NO-GO`.

For evidence terminology and common non-equivalences, read [references/evidence-boundaries.md](references/evidence-boundaries.md) only when the claim needs formal evidence classification. Use [assets/audit-report-template.md](assets/audit-report-template.md) when a structured audit packet is useful.
