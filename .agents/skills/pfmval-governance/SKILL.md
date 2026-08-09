---
name: pfmval-governance
description: Route PFMval planning, project facts, long-term documents, learning guides, lifecycle review, workflow discovery, paper-output interfaces, and governed experiment intake through current repository truth and the existing local CLI.
---

# PFMval Governance

Use this Skill when the user wants to plan or maintain PFMval work rather than
independently audit a completion claim. Read current repository evidence before
choosing a route; never treat chat, a generated view, or an explore artifact as
accepted project truth.

## Required read order

1. `AGENTS.md`
2. `CURRENT_STATE.md`
3. `project_state/current_state.json`
4. `project_state/directives.jsonl` when approval or instruction meaning matters
5. `project_state/document_registry.json` for active document authority
6. `experiments/experiment_registry.json` only when an experiment or result is in scope

Run the strict local start gate before generating a project conclusion:

```powershell
python deploy/pfmval_ops.py agent start-check --strict
```

## Decision flow

1. Separate current facts, accepted evidence, user decisions, assumptions, and
   missing inputs.
2. Classify the request using the route table below.
3. Preview the result before any governed state change.
4. Ask only for a decision that changes the route or confirms a real fact.
5. Make state changes only through `deploy/pfmval_ops.py`.
6. Explain the resulting lifecycle and evidence boundary. Do not claim a real
   operation ran when only a fixture, preview, or interface was exercised.

## Stable routes

| User entry | Technical route | Local command boundary | Correct stopping point |
|---|---|---|---|
| Non-experimental project fact | `project_fact` | `knowledge fact-preview` or `knowledge profile --profile project_fact` | confirmation preview |
| Long-term meeting summary | `meeting_brief` | `knowledge profile --profile meeting_brief` | draft brief with accepted/explore/failed separated |
| Long-term research path | `research_path` | `knowledge profile --profile research_path` | draft with uncertainty, negative routes, and stop reasons |
| Learning guide maintenance | `learning_guide` | `knowledge profile --profile learning_guide` or `knowledge freshness --document-id ...` | active Registry learning guide returns `fresh`/`review_due`; unregistered intake remains `interface_reserved` |
| Document/constraint lifecycle | `lifecycle_review` | `knowledge profile --profile lifecycle_review` or `knowledge freshness` | `review_due` or current preview |
| Workflow discovery/evolution | `workflow_discovery` | `workflow discover --scope-manifest ...` | unrecorded preview or candidate only |
| Paper writing/output | `paper_output` | `knowledge profile --profile paper_output` | `template_pending` |

The long-term-document user entry deliberately routes to two profiles, so six
user workflows produce seven technical routes. Use templates from
`templates/`; every shipped payload is synthetic and must be replaced or
explicitly confirmed before it can represent a real project fact or document.

For knowledge-only maintenance, use `agent start-check --strict --task knowledge`.
This task profile may downgrade experiment-workspace HEAD drift and generated
navigation drift to WARN, but it never weakens path/branch identity, experiment
Registry, active normative document, server transport, approval, result, or
protected-asset failures.

## Experiment intake boundary

`templates/experiment_intake_interface.json` collects a future experiment
question, hypothesis, variables, run budget, evidence tier, stop rule, and
required `W###` binding. It does not register an experiment, create a job,
dispatch work, import a result, or consume a run unit. Those actions remain
subject to their existing explicit approval and workspace gates.

## Safety boundaries

- No direct server access, training, promotion of results, deletion, migration,
  unprotection, external-platform integration, cloud synchronization, or
  remote publication.
- Project facts require preview, conflict review, and separate explicit
  confirmation for each real statement.
- Lifecycle review produces impact information only; it never changes a
  directive or document lifecycle automatically.
- New discoveries remain candidates until a later explicit review and
  activation. Frequency never substitutes for approval.
- Paper output indexes project materials and waits for the senior author's
  template. It does not create manuscript prose.
- Use `pfmval-audit` for an independent GO/NO-GO or completion review.
