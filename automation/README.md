# PFMval Gitee job/result exchange

This directory contains small, reviewable envelopes transported on dedicated
Gitee branches. It does not contain model weights, datasets or unrestricted
shell commands.

## Branch ownership

- Local job/fix owner: `automation/local/<job_id>`
- Server status/result owner: `automation/server/<job_id>`
- Never have both sides write the same branch.

## Local dispatch

Before dispatch, commit all tracked code/state changes. Untracked checkpoints may
remain, but the dispatcher refuses a dirty tracked tree so `source_commit` is
reproducible and contains the maintenance CLI/state package.

```powershell
python deploy/pfmval_ops.py agent start-check --strict --task general
python deploy/pfmval_ops.py job dispatch --job-id <id> --experiment-id <registry-id> --phase preflight --command-id state_preflight --path-id mpp_standard_splits
git switch -c automation/local/<id>
git add automation/jobs/<id>/job.json
git commit -m "dispatch <id>"
git push gitee automation/local/<id>
```

For a formal job, create an approval JSON that binds the exact `job_id` and
`source_commit`, sets `approved: true`, and uses
`source: explicit_user_instruction`. The approval must be passed to
`job dispatch --approval`; the agent/watcher may not create it itself.

## Server execution

After manually fetching the first job branch, validate before execution:

```powershell
python deploy/pfmval_ops.py agent start-check --strict --task server
python deploy/pfmval_ops.py paths validate --task server
python deploy/pfmval_ops.py job validate --manifest automation/jobs/<id>/job.json
python deploy/pfmval_ops.py job run --manifest automation/jobs/<id>/job.json --dry-run
```

The `server` task deliberately skips `required_on: local` legacy paths. Do not
copy local partner-label directories into a server worktree to satisfy that
check.

Remove `--dry-run` only after preflight passes. Actual execution creates a
detached worktree under path id `server_automation_worktrees` at the pinned
`source_commit`. Reuse requires both the exact HEAD and a clean worktree.
`standard_training` resolves the script from the experiment Registry and
accepts only validated flag/value parameters; arbitrary shell command strings
are unsupported. MPP path flags cannot be supplied by the job: the runner
injects them from registered path IDs, while standard splits come from the
pinned worktree.

## Server result package

```powershell
python deploy/pfmval_ops.py job pack --manifest automation/jobs/<id>/job.json --status success --artifact <small-file> --metrics-json <metrics.json> --output automation/results/<id>/<attempt>
git switch -c automation/server/<id>
git add automation/results/<id>/<attempt>
git commit -m "return <id>"
git push gitee automation/server/<id>
```

Each file is capped at 20 MiB and each bundle at 50 MiB. Checkpoints and large
features remain on the server; record path, size and SHA-256 in result metadata.
The output directory must be empty, and import rejects files that are not listed
in `result.json`.

## Local import

```powershell
python deploy/pfmval_ops.py result import --bundle automation/results/<id>/<attempt>
```

Only a validated import may update Registry, Dashboard and CURRENT_STATE.
The result must match its dispatched `automation/jobs/<job_id>/job.json`; a
hand-crafted or phase-mismatched result is rejected. Packaging uses a temporary
directory, and interrupted multi-file imports are finalized or rolled back from
their transaction backups on the next import.
Preflight envelopes are stored as `last_preflight` and never replace an accepted
training result. Successful smoke/formal envelopes require `data_manifest_id`,
`path_index_version`, `best_epoch` and at least one selection/evaluation metric
before they can become accepted evidence. Smoke jobs require an explicit budget
of at most 3 epochs and never enter `latest_accepted_result_ids`; only a validated
formal result can become the latest accepted result.
Automated polling/repair is intentionally deferred to a separate task.
