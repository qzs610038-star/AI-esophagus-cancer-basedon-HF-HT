# PFMval evidence boundaries

## Authority order

1. Explicit current user directive.
2. Machine-readable current state.
3. Active registered plan.
4. Experiment, job, approval, result, and asset records at their declared
   lifecycle level.
5. Generated views.
6. Reports, handoffs, local notes, and historical artifacts.

A lower item cannot override a higher item.

## Evidence classes

| Class | Meaning | May authorize a project decision |
|---|---|---|
| `accepted` | Validated and explicitly imported or accepted under the current protocol | Yes, within its declared scope |
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
- Server output is not local Registry truth until the governed import completes.
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
