import json

from scripts.pfmval_assets import compile_asset_policy, evaluate_asset_writes


def test_asset_policy_shadow_reports_but_never_blocks_or_writes_registry(tmp_path):
    root = tmp_path / "repo"
    registry_path = root / "project_state" / "asset_registry.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "updated_at": "2026-07-27T00:00:00+00:00",
                "assets": [
                    {
                        "asset_id": "protected-histogene",
                        "path": "histogene",
                        "era": "legacy_mixed",
                        "custody": "project",
                        "asset_class": "code_root",
                        "lifecycle": "protected",
                        "mutability": "approval_required",
                        "evidence_role": "shared_dependency",
                        "tracking": "tracked",
                        "hash_policy": "metadata_only",
                        "owner_plan_id": "workflow-governance-v3",
                        "owner_experiment_id": None,
                        "replacement": None,
                        "retention_policy": "retain",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    before = registry_path.read_bytes()
    policy = compile_asset_policy(root)
    report = evaluate_asset_writes(
        root,
        policy,
        [
            "histogene/utils.py",
            str(root / "HISTOGENE" / "utils.py"),
            "../outside.txt",
        ],
        mode="shadow",
    )

    assert report["blocked"] is False
    assert len(report["would_block"]) == 3
    assert registry_path.read_bytes() == before
