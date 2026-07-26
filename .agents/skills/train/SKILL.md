---
name: train
description: Compatibility router for PFMval training requests. Use when the user invokes the former train Skill or asks to prepare, approve, dispatch, troubleshoot, or run a training experiment.
---

# Train Compatibility Router

Read `../_shared/legacy-compatibility-policy.md` completely.
Read `project_state/plans/gitee_numbered_workspace_protocol_v001_20260726.md`
before proposing a server workflow.

Lifecycle: `retired-router`; this Skill is not a direct launcher.

Require an active experiment, explicit `W###` binding, critical contract,
registered inputs, phase, and a positive user-specified count of actual
result-bearing runs. Distinguish preflight/E1 compatibility work from the
`EXPERIMENT_STARTED` boundary. Dispatch or training is permitted only when the
currently implemented job approval and Gitee workflow also authorize it.
