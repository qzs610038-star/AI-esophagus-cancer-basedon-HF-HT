---
name: git-rescue
description: Compatibility router for safe PFMval Git recovery. Use when the user invokes the former git-rescue Skill or asks to diagnose a branch, worktree, sync, or commit problem without losing local assets.
---

# Git Rescue Compatibility Router

Read `../_shared/legacy-compatibility-policy.md` completely.

Lifecycle: `retired-router`.

Start with the exact numbered workspace, branch, HEAD, status, worktree list,
and configured Gitee refs. Prefer read-only diagnosis and preservation branches
or snapshots. Never discard dirty or untracked assets, rewrite history, clean a
worktree, or change a remote unless the user separately approves the exact
target and recovery consequences.
