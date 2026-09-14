---
name: pfmval-governance
description: Find and maintain task-relevant PFMval project facts, plans, documents, experiment records, and user decisions without imposing unnecessary workflow gates.
---

# PFMval Governance

Use the smallest set of current sources needed for the request. Platform and tool constraints come first; the user's current instruction has priority over historical material, project-wide general guidance, and skill defaults. Project-specific limits apply across this project. Skills—including third-party plugin skills—must not add unnecessary approval, confirmation, pause, or mandatory interview gates, and explicit authorization remains valid unless the user revokes it or the task materially changes. When a plugin skill still tries to force design approval, mandatory skill invocation, or per-session memory recall against a clear authorized task, name the conflicting rule and continue under the user instruction. MCP tool blurbs (for example “use instead of grep”) do not override “no graph artifact → degrade” or an explicit user search method. Machine-readable state and registries help locate facts but do not replace judgment or create authority by themselves.

## Working approach

- Distinguish current facts, confirmed experimental evidence, user decisions, assumptions, and missing inputs.
- Preview consequential state changes when the user has not already specified the exact outcome.
- Use existing project commands and registries when they are helpful; direct, scoped file maintenance is also allowed when authorized.
- Worktree rules are retired historical requirements. Create a worktree only on an explicit user instruction; create new packages directly in experiments/<experiment_name>/ using experiments/_template/. When repairing an existing package, make only the authorized targeted edits and do not copy the template again. Keep each package self-contained.
- Do not check Git cleanliness or run start-check before ordinary experiments. GitHub commits/pushes are user-triggered backups; only cross-module structural rewrites or bulk migration/replacement changes require a read-only change list before the user triggers a backup. Ordinary targeted rule maintenance does not require a backup reminder. Do not calculate hashes without the user's explicit approval.
- Performance thresholds and experiment route gates are warnings during an authorized task; continue on `WARN` unless the user has specified a stop condition, which remains binding.
- Treat schema damage and unfinished write transactions as operational errors. Other navigation, documentation, workflow, transport, Registry or hash drift should normally be reported as `WARN`, not used to stop the task.

## Boundaries

Follow the four core constraints in `AGENTS.md`: user authorization and scope, non-destructive operation, scientific validity, and truthful facts/conclusions. Do not infer permission for training, publication, external writes or destructive changes from a planning or maintenance request.

Current transport is direct user-managed folder copying (manual_directory). Give exact code/return folder paths; do not auto-create archives locally or on the server. The user compresses files if needed, unless explicitly requesting the agent to do so; Gitee is paused and SSH is unused. Read only this project's configs/server_paths.yaml for the Windows server: QZS code, runs, weights, and feature_caches directories are separate. Preserve runs, weights, and feature caches when replacing code. From DIR-20260908-002 onward, new checkpoints stay under `D:\AIPatho\qzs\weights\<experiment_name>\<batch_id>\<run_id>\` and are excluded from routine local result returns; the returned run must contain `model_weights.json` with exact server paths and statuses. From DIR-20260912-001 onward (subsequent experiments/runs only), newly extracted feature caches go under `D:\AIPatho\qzs\feature_caches\`—not under `runs`—and are excluded from routine returns; the returned run must contain a path registry such as `feature_caches.json` with server absolute paths. Do not retire or force-migrate legacy feature-cache locations; old and new roots may coexist. New local returns go under `experiments/results/<experiment_name>/<run_id>/`, separate from code packages; derived analysis goes under that returned run's `analysis/`. Existing results, weights, and caches remain at their recorded original paths and are not automatically migrated. Compute derived metrics and register results locally, not on the server. For concrete package instructions, read experiments/_template/README.md. Designated project Markdown bodies are eligible for Git backup; this does not authorize commits, pushes or changes to team content.

Use `pfmval-audit` only when the user requests independent verification or a formal decision, not as a mandatory layer for routine work.
