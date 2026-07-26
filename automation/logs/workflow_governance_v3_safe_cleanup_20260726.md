# Workflow governance v3 safe-cleanup record

> date: 2026-07-26
> authorization: `DIR-20260726-002`
> code baseline: `main@a96bac88`
> scope: local legacy Skill/Workflow quarantine, safety-message repair, and tracked audit Skill creation
> evidence role: maintenance record; not experiment evidence

## Boundaries held

- No training or result import.
- No local or server experiment dispatch.
- No server connection or worktree creation/deletion.
- No modification to `histogene/`, `egnv1/`, or `egnv2/`.
- No acceptance of Explore or pending experiment results.
- No remote push, branch rewrite, cache cleanup, or asset migration.

## Tracked changes

- Added the active safe-cleanup plan:
  `project_state/plans/workflow_governance_v3_20260726.md`.
- Added the review-only structural refactor packet:
  `project_state/plans/workflow_governance_v3_refactor_review_20260726.md`.
- Added `.agents/skills/pfmval-audit/` with:
  - `SKILL.md`
  - `agents/openai.yaml`
  - `references/evidence-boundaries.md`
  - `assets/audit-report-template.md`
- Added the Skill read-order pointer to `AGENTS.md`.
- Registered only `workflow_governance_v3` as an active plan. The structural
  refactor packet remains pending user review and is not an execution authority.

## Ignored local adapter cleanup

`.claude/` is ignored by `.gitignore`; the following changes are local,
reversible, and require path/content checks rather than `git diff`.

### Quarantined legacy Skills

The 16 directories below moved without deletion from `.claude/skills/` to
`.claude/legacy/skills-20260726/`:

- `compare`
- `data-guide`
- `doc-registry`
- `experiment-log`
- `extract`
- `git-rescue`
- `health`
- `meeting-report`
- `onboard`
- `parallel-extract`
- `post-train`
- `sync-server`
- `train`
- `train-guide`
- `update-ranking`
- `viz-guide`

The old `.claude/agents/code-reviewer.md` moved to
`.claude/legacy/agents/code-reviewer-20260726.md`. A new
`.claude/skills/README.md` points to the tracked authority.

### Hook repair

- `block-git-clean.py` still blocks `git clean`, but no longer recommends
  `git fetch --force && git reset --hard origin/main`.
- Both active safety hooks now cite `AGENTS.md` fixed boundaries.
- Matching behavior, protected-directory list, exit codes, and fail-open
  behavior were not structurally changed.

### Historical workflow labels

Added a historical-local-reference banner to:

- `.claude/git-cloud-sync-options.md`
- `.claude/he-ssgsea-greedy-river.md`
- `.claude/deployment-gotchas.md`

## Deferred structural work

- Do not mechanically prune `.claude/settings.json`; its broad allowlist needs
  a separate permission audit.
- Do not replace the protected-directory Hook until a tracked asset-policy
  evaluator has passed shadow tests.
- Do not implement experiment workspaces, attempt leases, gate v2, asset
  registry, protected-dir declassification, or physical cleanup before the user
  approves the review packet.
- Current state/document schema cannot faithfully register a
  `pending_review` plan or a tracked Skill. This gap is specified in R5b of the
  refactor packet; it is not silently worked around by marking the review packet
  active.

## Fresh verification

| Check | Result |
|---|---|
| `pfmval-audit` `quick_validate.py` | PASS |
| Independent forward-test against a false Explore-acceptance claim | PASS; returned `NO-GO`, kept `explore_only/pending_review`, made no writes |
| Legacy Skill quarantine | PASS; legacy directories `16`, active legacy directories `0` |
| Legacy code-reviewer quarantine | PASS; archived file present |
| Historical workflow banners | PASS; `3/3` files |
| Hook AST parse | PASS |
| Dangerous Hook recommendation search | PASS; `reset --hard` / `fetch --force` hits `0` |
| `git clean -fd` Hook smoke | PASS; exit `2` |
| `git status --short` Hook smoke | PASS; exit `0` |
| protected-directory write Hook smoke | PASS; exit `2` |
| `agent start-check --strict` | `PASS=6 WARN=6 FAIL=0` |
| `paths validate` | `PASS=0 WARN=1 FAIL=0` |
| existing v1 job manifest validation | PASS |
| document dry scan | total `164`, live `132`, active `15`, superseded `11`, historical `106`, missing `32` |
| full project pytest | `76 passed in 41.59s` |
| `git diff --check` | PASS |

The first two sandboxed pytest attempts produced setup/cleanup ACL errors in the
temporary directory and no assertion failures. The authoritative rerun used a
normal local temporary directory, passed all 76 tests, and its exact temporary
directory was removed afterward.

The six strict-gate warnings are pre-existing: five accepted legacy results lack
complete result envelopes, and the standardized MPP labels contain
duplicate/conflicting barcodes. The path warning is the same protected MPP
label condition. None were changed or suppressed.

The final synchronization must leave the document registry exactly one revision
behind current state, which is the supported generated-view lag.

## Compatibility correction after `DIR-20260726-003`

The user subsequently required every former `.claude` Skill name to remain
available as a direct cross-agent adapter instead of leaving the active
directory empty.

- The 16 original implementations remain preserved under
  `.claude/legacy/skills-20260726/`; none is deleted or copied back.
- Safe same-name compatibility routers now live under `.agents/skills/`.
- Tracked `.claude/skills/<name>/SKILL.md` files directly reference the matching
  `.agents` Skill and add no executable policy.
- `pfmval-audit` has the same adapter shape.
- The earlier “active legacy directories 0” check is retained above as the
  pre-correction snapshot; the final check must instead assert 17 thin adapters,
  17 matching canonical targets, and 16 preserved historical implementations.

### Final correction verification

| Check | Result |
|---|---|
| Canonical Skill metadata | PASS; `17/17` passed `quick_validate.py` |
| Claude adapter mapping | PASS; `17` thin adapters, `17` existing same-name canonical targets |
| Historical preservation | PASS; `16` original implementations remain under the legacy directory |
| Active adapter safety | PASS; Read-only metadata, direct same-name target, no command block or retired transport/write token |
| Fresh-agent `train` forward-test | PASS; three seeds counted as three run units; missing W/approval/contracts returned NO-GO |
| Fresh-agent `sync-server` forward-test | PASS; rejected pull/overwrite/retrain and returned fetch/exact-ref/adaptation-review flow |
| Skill/review lifecycle TDD | PASS; red tests failed before implementation and green tests passed afterward |
| Lifecycle registration | PASS; canonical Skill=`active/normative`, adapter=`active/reference`, numbered protocol=`approved_design`, refactor packet=`pending_review` |
| Document scan | total `187`, live `171`, active `49`, approved-design `1`, pending-review `1`, superseded `11`, historical `109`, missing `16` |
| Full project test scope | `78 passed in 40.11s` using a repository-external ASCII basetemp |
| Strict start gate | `PASS=6 WARN=6 FAIL=0` |
| Path validation | `PASS=0 WARN=1 FAIL=0` |
| Existing v1 job validation | PASS |
| Hook AST and smoke tests | PASS; four files parsed, protected write and destructive clean blocked, read-only status allowed |

Three earlier pytest attempts were non-authoritative environment probes:

- the repository root collected third-party OpenMidnight examples with optional
  dependencies and sample files;
- the default Windows pytest temp root had an inherited ACL failure;
- a basetemp inside the repository inherited the parent Git repository and a
  non-ASCII path, invalidating seven fixtures by design.

No assertion failure remained when the existing project seam was run from an
external ASCII temp root. All uniquely created temporary directories were
removed after verification.
