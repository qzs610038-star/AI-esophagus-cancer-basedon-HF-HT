---
name: onboard
description: Compatibility router for onboarding to PFMval. Use when the user invokes the former onboard Skill or asks an agent to orient itself to the current project safely.
---

# Onboard Compatibility Router

Read `../_shared/legacy-compatibility-policy.md` completely.

Lifecycle: `retired-router`.

Follow the read order in `AGENTS.md`, run the appropriate read-only start check,
and summarize current directives, active plans, experiment state, and blockers.
If the task will modify experiment code, require a numbered-workspace binding
before writing. Onboarding never authorizes extraction, training, result import,
or project-state mutation.
