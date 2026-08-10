---
name: sync-server
description: Compatibility router for PFMval server synchronization. Use when the user invokes the former sync-server Skill or asks to dispatch, fetch, troubleshoot, or return server code and small results.
---

# Server Sync Compatibility Router

Read `../_shared/legacy-compatibility-policy.md` completely.
Read `project_state/plans/gitee_numbered_workspace_protocol_v001_20260726.md`
before proposing any operation.

Lifecycle: `retired-router`; use the implemented local CLI and schemas, while
requiring live server verification before claiming a real operation succeeded.

Use only the configured Gitee transport and local/server single-writer refs.
Require a bound `W###`, exact source commit, clean locus, immutable job/result
envelope, remote SHA verification, and quarantine import. A simple server
adaptation must return to the local canonical writer for review; repeated
transfer conflicts are never solved by silently retraining or overwriting refs.

Resolve server paths only by registered IDs: `server_governance_checkout`,
`server_experiment_worktrees`, `server_experiment_runs`,
`server_result_returns`, `server_diagnostics`, and `server_runtime_bundles`.
Treat `server_repo_worktree` as legacy/no-new-output. Use `git fetch` plus exact
ref/SHA checks; do not use `git pull`, implicit merge, SSH, SCP, HTTP remote
commands, or tunnels. For troubleshooting, use only a generated diagnostic
request and fixed allowlisted runner. Diagnostic output is non-evidence and is
checked by path, size, byte contract, and Git tree closure rather than a
per-file hash.
