---
name: extract
description: Thin Claude adapter for the tracked PFMval extract compatibility Skill.
disable-model-invocation: true
allowed-tools: Read
---

Read and follow [the canonical Skill](../../../.agents/skills/extract/SKILL.md)
completely. This adapter adds no policy. If the target is missing or unreadable,
report a warning and stop; do not use any historical fallback.
