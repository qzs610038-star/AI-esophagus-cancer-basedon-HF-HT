# PFMval evidence boundaries

## Authority order

1. Explicit current user directive and the project's current constraints.
2. The authoritative source for the **kind of fact being claimed**.
3. A later, same-object record at the same or a stronger lifecycle level.
4. Generated views, reports, handoffs, local notes, and historical artifacts.

There is no single global ordering that lets a state-machine row override every
other source. Resolve conflicts by object and fact class first:

| Claim or decision | Primary source | Conflict rule |
|---|---|---|
| User choice, route, or permission | Current user directive and project directives | The later explicit directive governs the stated scope. |
| Experiment status or accepted conclusion | `experiments/experiment_registry.json` plus the matching acceptance/review record | A later `accepted`/`rejected` record for the same experiment and result governs an earlier `planned`/`pending_review` note. |
| Operational state and active plan routing | `project_state/current_state.json` and its current machine sources | A stale document header does not create an active plan; a state row also cannot erase a later accepted experiment record. |
| Server paths and delivery rules | `configs/server_paths.yaml`, `AGENTS.md`, and current governance documents | Current project constraints govern old transport or deployment prose. |
| Document lifecycle | `project_state/document_registry.json` and explicit lifecycle overrides | Registry lifecycle describes document use; it does not upgrade scientific evidence or override a later experiment acceptance record. |
| Explanation, teaching, or historical context | The document itself, labelled by lifecycle | It may explain or preserve history but cannot silently become current evidence. |

When records for the same object have different dates or scopes, compare the
date, protocol/revision, and declared scope. Do not let an early
`pending_review` sentence override a later accepted result, and do not treat a
newer file date alone as proof that an older result or plan is obsolete.

## Evidence classes

| Class | Meaning | May authorize a project decision |
|---|---|---|
| `accepted` | Explicitly accepted in the experiment Registry within its recorded protocol and scope; registration/import alone is insufficient | Yes, within its declared scope |
| `current` | Current source, state, configuration, or plan | Only for the fact it directly proves |
| `pending_review` | Produced but awaiting explicit review or import | No |
| `diagnostic_only` | Non-evidentiary troubleshooting output | No |
| `explore_only` | Local exploratory output or hypothesis | No |
| `historical` | Retained for lineage or reference | No |
| `missing` | Required evidence is absent or unreadable | No |

## Common non-equivalences

- File present is not file validated.
- Source implemented is not runtime passed.
- Unit test passed is not experiment accepted.
- Result packed is not result imported.
- Result imported is not approval for a different protocol revision.
- Raw server output can establish an observed artifact; result registration records it, while scientific acceptance requires an explicit accepted Registry status.
- Explore result is not comparable experiment evidence.
- Dashboard text is not the experiment fact source.
- A hash match proves identity, not scientific validity.
- A manual observation proves only the observation and stated conditions.

## Required identifiers

When applicable, bind the claim to:

- `source_commit`
- `experiment_id`
- `job_id`
- attempt or protocol revision
- approval record
- result envelope and import status
- optional artifact identity evidence when explicitly requested

Only identifiers required by the exact claim are mandatory. Missing historical
workflow metadata is a `WARN` unless it makes that claim unverifiable.
