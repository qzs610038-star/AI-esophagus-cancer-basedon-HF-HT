---
name: sync-server
description: Compatibility router for PFMval server synchronization. Use when the user invokes the former sync-server Skill or asks to dispatch, fetch, troubleshoot, or return server code and small results.
---

# Server Sync Compatibility Router

Read `../_shared/legacy-compatibility-policy.md` completely.
Read `project_state/plans/gitee_numbered_workspace_protocol_v001_20260726.md`
before proposing any operation.

Lifecycle: `retired-router`; the numbered-workspace automation is still
code-pending.

Use only the configured Gitee transport and local/server single-writer refs.
Require a bound `W###`, exact source commit, clean locus, immutable job/result
envelope, remote SHA verification, and quarantine import. A simple server
adaptation must return to the local canonical writer for review; repeated
transfer conflicts are never solved by silently retraining or overwriting refs.
