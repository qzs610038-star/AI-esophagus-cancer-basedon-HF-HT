---
name: update-ranking
description: Thin Claude adapter for the tracked PFMval update-ranking compatibility Skill.
allowed-tools: Read
---

Read and follow [the canonical Skill](../../../.agents/skills/update-ranking/SKILL.md)
completely. This adapter adds no policy. If the target is missing or unreadable,
report a warning and stop; do not use any historical fallback.
