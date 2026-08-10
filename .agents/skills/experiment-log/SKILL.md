---
name: experiment-log
description: Compatibility router for PFMval experiment logging. Use when the user invokes the former experiment-log Skill or asks to register attempts, results, status, or experiment history.
---

# Experiment Log Compatibility Router

Read `../_shared/legacy-compatibility-policy.md` completely.

Lifecycle: `retired-router`.

Route experiment registration and result status through
`experiments/experiment_registry.json`, immutable job/result envelopes, and
`pfmval_ops.py`. Before a result-bearing execution, ask the user for the actual
allowed run count. Do not append an alternative table or treat a chat report as
an imported result.

Represent an allocated but abandoned number as `abandoned_reserved` with a null
`experiment_id`, null lease/active attempt, its reservation directive, and the
preserved branch/HEAD/path. It is not an experiment and cannot receive an
approval, job, attempt, result, or later reassignment. The next experiment must
consume the monotonic `next_workspace_number`.
