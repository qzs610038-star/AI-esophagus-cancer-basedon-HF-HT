---
name: doc-registry
description: Compatibility router for PFMval document lifecycle work. Use when the user invokes the former doc-registry Skill or requests document registration, supersession, lifecycle review, or registry consistency checks.
---

# Document Registry Compatibility Router

Read `../_shared/legacy-compatibility-policy.md` completely.

Lifecycle: `retired-router`.

Treat `project_state/document_registry.json` and its governed scanner as the
only document lifecycle path. Review with `docs scan` and current state
validation; perform writes only through the active `pfmval_ops.py` interface
when explicitly authorized. Preserve append-only supersession relationships and
never invent an independent local registry.
