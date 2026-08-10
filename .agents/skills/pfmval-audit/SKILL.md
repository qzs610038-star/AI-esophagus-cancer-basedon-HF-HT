---
name: pfmval-audit
description: Independently audit PFMval plans, completion claims, code changes, experiment results, project state, and GO or NO-GO decisions. Use when the user asks to verify, review, reverse-check, accept or reject, judge feasibility, or confirm that a PFMval claim is current and supported by the required evidence.
---

# PFMval Independent Audit

Audit from current repository evidence, not from a handoff, chat summary, report,
or prior assistant claim. Stay read-only unless the user separately authorizes a
specific repair.

## Required read order

1. Read `AGENTS.md`.
2. Read `CURRENT_STATE.md`.
3. Read `project_state/current_state.json`.
4. When a claim depends on user approval, supersession, or instruction meaning,
   read `project_state/directives.jsonl`.
5. For experiments, performance, or next-step decisions, read
   `experiments/experiment_registry.json`.
6. Read `project_state/document_registry.json` and use only records whose
   lifecycle is `active` as current execution authority.
7. For server paths or synchronization, also read `configs/server_paths.yaml`
   and `01_指南与解读/部署方案/服务器路径索引_20260701.md`.

Record the repository locus before auditing:

```powershell
git status --short --branch
git rev-parse HEAD
```

Run the strict read-only preflight before issuing a project conclusion:

```powershell
python deploy/pfmval_ops.py agent start-check --strict
```

For an allowlisted, non-evidentiary server diagnostic, use the diagnostic
preflight instead. Do not generalize this exception to training or result
acceptance.

```powershell
python deploy/pfmval_ops.py agent start-check --task diagnostic
```

## Audit workflow

### 1. Normalize the claim

Write the claim as a testable sentence. Identify:

- the claimed object;
- the claimed lifecycle state;
- the claimed commit, experiment, job, attempt, or result identifiers;
- what decision the claim would authorize.

### 2. Classify the evidence

Classify every relevant item as one of:

- `current`
- `accepted`
- `pending_review`
- `diagnostic_only`
- `explore_only`
- `historical`
- `missing`

Read [references/evidence-boundaries.md](references/evidence-boundaries.md) for
the acceptance boundaries. Never promote an item merely because the file exists,
a test passed, or a report says it was completed.

### 3. Reverse-verify the real artifacts

Inspect the actual files named by the claim. Depending on scope, verify:

- source symbols and configuration values;
- experiment, job, approval, and result bindings;
- critical-input and critical-output hashes;
- reported metrics and exclusions;
- generated-view consistency with the machine-readable source;
- evidence scope, especially host/offline, simulation, server, board, external
  test, and accepted-result boundaries.
- numbered workspace closure: physical/Git worktrees, Registry records,
  experiment ids, branch, HEAD, dirty state, lifecycle, lease, and jobs;
- server path separation: governance checkout, persistent `W###` source,
  external run, return, diagnostic, and runtime-bundle roots;
- return closure: terminal JSON, original training CSV/TXT, every required raw
  prediction split, exact force-added staged tree, and immutable result ref;
- hash-policy/runtime agreement: SHA-256 only remains HARD for critical evidence;
  supporting/diagnostic files are not accidentally re-hashed or allowed to
  source accepted metrics.

Source inspection proves implementation, not runtime success. Tests prove only
the exercised behavior. A result bundle proves only what its validated envelope
and imported status cover.

### 4. Separate machine checks from human evidence

Record separately:

- commands run and their fresh outputs;
- static source or schema evidence;
- runtime or experimental evidence;
- manual observations supplied by the user;
- gaps that have not been run or observed.

Do not infer PASS from an absent result, an old report, source readability, or a
historical artifact.

### 5. Issue a bounded verdict

Use `PASS`, `WARN`, and `FAIL` for individual checks. Then give exactly one
overall verdict:

- `GO`: all required evidence for the stated decision is current and accepted.
- `CONDITIONAL GO`: no blocking failure, but explicitly named conditions remain.
- `NO-GO`: at least one required condition failed or is missing.
- `NOT APPLICABLE`: the request does not authorize a GO or NO-GO decision.

Use [assets/audit-report-template.md](assets/audit-report-template.md) as the
default output shape. Preserve any user-specified audit packet or PASS/FAIL
format literally.

## Safety boundaries

- Do not train, dispatch jobs, operate a server, import results, accept evidence,
  edit registries, or repair code while performing an audit.
- Do not use SSH, SCP, HTTP remote commands, tunnels, or unregistered transfer
  paths.
- Treat `histogene/`, `egnv1/`, and `egnv2/` according to the current recorded
  directive. `DIR-20260809-002` removes their special protection but does not
  authorize migration or deletion before the exact review list is approved.
- Do not close directives or reinterpret an explicit user approval.
- Treat Explore outputs as candidates unless a later explicit directive accepts
  them.
- Report a failing or unavailable memory service as `WARN`; it is not a reason
  to skip repository verification.

If a repair is requested in the same task, finish and report the read-only audit
first, then implement only the specifically authorized repair and rerun fresh
verification.
