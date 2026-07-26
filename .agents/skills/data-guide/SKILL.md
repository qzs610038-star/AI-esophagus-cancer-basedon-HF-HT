---
name: data-guide
description: Compatibility router for PFMval data guidance. Use when the user invokes the former data-guide Skill or asks how current datasets, splits, manifests, or preprocessing should be used.
---

# Data Guide Compatibility Router

Read `../_shared/legacy-compatibility-policy.md` completely.

Lifecycle: `reference-router`.

Build guidance only from active manifests, registered paths, current plans, and
accepted preprocessing contracts. Preserve the rule that supervised
preprocessing is fitted on the training set and then applied to validation and
external test data. Label unverified paths, manifests, labels, or derived assets
as gaps instead of filling them from historical notes.
