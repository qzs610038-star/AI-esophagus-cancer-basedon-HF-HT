---
name: post-train
description: Compatibility router for PFMval post-training processing. Use when the user invokes the former post-train Skill or asks to package, validate, import, review, or close a completed experiment.
---

# Post Train Compatibility Router

Read `../_shared/legacy-compatibility-policy.md` completely.

Lifecycle: `retired-router`.

Route artifacts through the current result envelope, Gitee return, quarantine,
validation, idempotent import, and acceptance boundaries. Separate critical,
supporting, and diagnostic artifacts. Do not write an alternative experiment
log, accept a result from its presence alone, or remove the workspace during
post-processing.
