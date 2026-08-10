---
name: post-train
description: Compatibility router for PFMval post-training processing. Use when the user invokes the former post-train Skill or asks to package, validate, import, review, or close a completed experiment.
---

# Post Train Compatibility Router

Read `../_shared/legacy-compatibility-policy.md` completely.

Lifecycle: `retired-router`.

Route artifacts through the current result envelope, Gitee return, quarantine,
validation, idempotent import, and acceptance boundaries. Separate critical,
supporting, and diagnostic artifacts. Do not write an alternative experiment
log, accept a result from its presence alone, or remove the workspace during
post-processing.

Apply the job-bound `return_profile`: every success or failed training attempt
returns terminal JSON plus original `raw_training_csv` and `raw_training_txt`;
an incomplete attempt returns the terminal JSON and every available raw CSV/TXT.
Successful prediction-producing attempts additionally return one critical raw
prediction table for every evaluated split. Build under
`server_result_returns/<result_id>`, validate closure, then force-add only that
exact immutable revision so ignored `.csv`, `.txt`, and `.log` files cannot be
silently omitted. Before commit, compare the staged Git tree with `result.json`.

Use SHA-256 for critical artifacts. Use required inventory, exact size, and Git
tree closure for supporting and diagnostic files. Promotion of a supporting
CSV/TXT to a metric source makes it critical and requires repackaging with a
hash. A supplemental return creates a new immutable result revision; never
rewrite an existing revision.
