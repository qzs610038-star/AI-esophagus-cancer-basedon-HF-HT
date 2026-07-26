---
name: doc-registry
description: Thin Claude adapter for the tracked PFMval doc-registry compatibility Skill.
disable-model-invocation: true
allowed-tools: Read
---

Read and follow [the canonical Skill](../../../.agents/skills/doc-registry/SKILL.md)
completely. This adapter adds no policy. If the target is missing or unreadable,
report a warning and stop; do not use any historical fallback.
