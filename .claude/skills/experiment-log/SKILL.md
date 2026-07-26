---
name: experiment-log
description: Thin Claude adapter for the tracked PFMval experiment-log compatibility Skill.
disable-model-invocation: true
allowed-tools: Read
---

Read and follow [the canonical Skill](../../../.agents/skills/experiment-log/SKILL.md)
completely. This adapter adds no policy. If the target is missing or unreadable,
report a warning and stop; do not use any historical fallback.
