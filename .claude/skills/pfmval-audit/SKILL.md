---
name: pfmval-audit
description: Thin Claude adapter for the tracked PFMval independent audit Skill.
disable-model-invocation: true
allowed-tools: Read
---

Read and follow [the canonical Skill](../../../.agents/skills/pfmval-audit/SKILL.md)
completely. This adapter adds no policy. If the target is missing or unreadable,
report a warning and stop; do not use any historical fallback.
