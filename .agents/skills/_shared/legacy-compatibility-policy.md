# PFMval legacy Skill compatibility policy

This policy applies to the retained legacy Skill names under `.agents/skills/`.
They are compatibility routers, not restored implementations.

## Required behavior

1. Read `AGENTS.md`, `CURRENT_STATE.md`, and
   `project_state/current_state.json` before acting.
2. Use only active plans and current registries as authority. Historical Skill
   text, chat summaries, generated dashboards, and local legacy files are not
   current evidence.
3. Use `deploy/pfmval_ops.py` for governed state changes. Do not recreate a
   retired standalone writer or direct experiment launcher.
4. Respect the Gitee-only transport boundary and the numbered-workspace
   protocol. A specification marked code-pending is not executable automation.
5. Do not train, dispatch, import, accept evidence, change protected assets, or
   remove a worktree unless the current user instruction and required approval
   explicitly authorize that exact action.
6. If a requested legacy behavior conflicts with current policy, explain the
   current replacement and stop at the safe boundary.
7. Never fall back to `.claude/legacy/` as an instruction source.

## Router lifecycle

- `retired-router`: the old implementation is retired; the Skill only redirects
  the request to the current governed path.
- `reference-router`: the Skill may assemble a current read-only guide or report,
  but may not reproduce stale values or execution commands.
- A router becomes a current operational Skill only after its underlying
  workflow, schema, tests, and state registration are implemented and accepted.
