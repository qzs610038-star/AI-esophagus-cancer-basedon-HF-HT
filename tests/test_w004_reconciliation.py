import json
import shutil
from pathlib import Path


from scripts.pfmval_w004_reconciliation import reconcile_w004


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = (
    PROJECT_ROOT
    / "project_state"
    / "governance"
    / "w004_stage2_reconciliation_v002_20260827.json"
)
BASELINE_FILES = (
    "experiments/experiment_registry.json",
    "project_state/workspace_registry.json",
    "project_state/experiment_approvals.jsonl",
    "project_state/attempt_events.jsonl",
    "project_state/result_import_events.jsonl",
    "project_state/scientific_records.jsonl",
)


def _copy_baseline(tmp_path: Path) -> Path:
    root = tmp_path / "target"
    for relative in BASELINE_FILES:
        source = PROJECT_ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return root


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_w004_check_is_read_only_and_apply_is_atomic_idempotent(tmp_path):
    root = _copy_baseline(tmp_path)
    before_check = _tree_bytes(root)

    checked = reconcile_w004(
        root,
        MANIFEST,
        mode="check",
        evidence_root=PROJECT_ROOT,
        history_repo=PROJECT_ROOT,
    )

    assert checked["status"] == "ready"
    assert checked["validated_bundle_count"] == 4
    assert checked["historical_attempt_event_count"] == 15
    assert checked["historical_job_match_count"] == 4
    assert _tree_bytes(root) == before_check

    applied = reconcile_w004(
        root,
        MANIFEST,
        mode="apply",
        evidence_root=PROJECT_ROOT,
        history_repo=PROJECT_ROOT,
        recorded_at="2026-08-27T08:00:00+00:00",
    )

    assert applied["status"] == "applied"
    registry = json.loads(
        (root / "experiments" / "experiment_registry.json").read_text(
            encoding="utf-8"
        )
    )
    matching = [
        item
        for item in registry["experiments"]
        if item["id"]
        == "phase3_dual_baseline_spatial_pathway_transfer_v001_20260804"
    ]
    assert len(matching) == 1
    experiment = matching[0]
    assert experiment["protocol_revision"] == 2
    assert experiment["evidence_status"] == "accepted"
    assert experiment["evidence_scope"] == "COMPATIBILITY_ONLY"
    assert experiment["raw_mpp2_claim"] is False
    assert experiment["result_group"]["group_id"] == (
        "W004-STAGE2-A001-A004-GROUP-V001"
    )
    assert experiment["result_ids"] == [
        "W004-A001-R002-result-repack-v001",
        "W004-A002-result-repack-R001",
        "W004-A003-result-repack-R001",
        "W004-A004-result-repack-R001",
    ]

    imports = _read_jsonl(
        root / "project_state" / "result_import_events.jsonl"
    )
    w004_imports = [
        item for item in imports if item.get("workspace_id") == "W004"
    ]
    assert len(w004_imports) == 5
    assert sum(
        item["event_type"] == "RESULT_IMPORT_REVERIFIED"
        for item in w004_imports
    ) == 1
    assert {
        item["attempt_id"]
        for item in w004_imports
        if item["event_type"] == "RESULT_IMPORT_VERIFIED"
        and item["recorded_at"] == "2026-08-27T08:00:00+00:00"
    } == {"A002", "A003", "A004"}
    assert (
        root
        / "project_state"
        / "governance"
        / "phase3_stage2_v002_critical_contract_20260807.json"
    ).is_file()
    assert not (root / "automation" / "jobs" / "W004-A001" / "job.json").exists()

    after_first_apply = _tree_bytes(root)
    replay = reconcile_w004(
        root,
        MANIFEST,
        mode="apply",
        evidence_root=PROJECT_ROOT,
        history_repo=PROJECT_ROOT,
        recorded_at="2026-08-27T09:00:00+00:00",
    )

    assert replay["status"] == "already_applied"
    assert _tree_bytes(root) == after_first_apply
