---
name: sync-server
description: Thin Claude adapter for the tracked PFMval sync-server compatibility Skill.
allowed-tools: Read
---

Read and follow [the canonical Skill](../../../.agents/skills/sync-server/SKILL.md)
completely. This adapter adds no policy. If the target is missing or unreadable,
report a warning and stop; do not use any historical fallback.
