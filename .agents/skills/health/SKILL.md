---
name: health
description: Compatibility router for PFMval health checks. Use when the user invokes the former health Skill or requests a current local, state, path, worktree, job, or result health assessment.
---

# Health Compatibility Router

Read `../_shared/legacy-compatibility-policy.md` completely.

Lifecycle: `retired-router`.

Run the current read-only start check, state validation, path validation, and
scope-appropriate registry checks. Use only the allowlisted diagnostic route for
server-related evidence. Report PASS, WARN, and FAIL separately; do not turn a
warning into a success claim or use a health check to dispatch work.
