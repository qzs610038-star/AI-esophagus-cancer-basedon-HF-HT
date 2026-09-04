---
name: pfmval-governance
description: Find and maintain task-relevant PFMval project facts, plans, documents, experiment records, and user decisions without imposing unnecessary workflow gates.
---

# PFMval Governance

Use the smallest set of current sources needed for the request. The user's current instruction has priority; machine-readable state and registries help locate facts but do not replace judgment or create authority by themselves.

## Working approach

- Distinguish current facts, confirmed experimental evidence, user decisions, assumptions, and missing inputs.
- Preview consequential state changes when the user has not already specified the exact outcome.
- Use existing project commands and registries when they are helpful; direct, scoped file maintenance is also allowed when authorized.
- Treat W### worktrees, implementation plans, Registry entries, lifecycle labels, `start-check`, source commits and hashes as optional organization or audit mechanisms unless the task specifically requires one.
- Treat schema damage and unfinished write transactions as operational errors. Other navigation, documentation, workflow, transport, Registry or hash drift should normally be reported as `WARN`, not used to stop the task.

## Boundaries

Follow the four core constraints in `AGENTS.md`: user authorization and scope, non-destructive operation, scientific validity, and truthful facts/conclusions. Do not infer permission for training, publication, external writes or destructive changes from a planning or maintenance request.

Synchronization is configurable. Gitee is the currently configured channel, not a permanent exclusive channel; user-managed archives, SSH or other channels may be added later. Do not require synchronization hashes for ordinary transfers or maintenance.

Use `pfmval-audit` only when the user requests independent verification or a formal decision, not as a mandatory layer for routine work.
