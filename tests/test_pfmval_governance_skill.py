import json
from pathlib import Path

from scripts.pfmval_knowledge import build_document_profile
from scripts.pfmval_workflows import discover_workflows


ROOT = Path(__file__).resolve().parent.parent
SKILL_ROOT = ROOT / ".agents" / "skills" / "pfmval-governance"


def test_g08_canonical_skill_and_thin_adapter_expose_seven_safe_routes():
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    adapter = (
        ROOT / ".claude" / "skills" / "pfmval-governance" / "SKILL.md"
    ).read_text(encoding="utf-8")

    assert skill.startswith("---\nname: pfmval-governance\n")
    assert adapter.count("../../../.agents/skills/pfmval-governance/SKILL.md") == 1
    assert "deploy/pfmval_ops.py" not in adapter
    assert "scripts/" not in adapter
    for route in (
        "project_fact",
        "meeting_brief",
        "research_path",
        "learning_guide",
        "lifecycle_review",
        "workflow_discovery",
        "paper_output",
    ):
        assert f"`{route}`" in skill
    assert "active Registry learning guide" in skill
    assert "--task knowledge" in skill

    forbidden_direct_actions = (
        "ssh ",
        "scp ",
        "git push",
        "result accept",
        "fact-confirm",
        "close --execute",
        "rm -rf",
        "git clean",
    )
    assert all(value not in skill.lower() for value in forbidden_direct_actions)


def test_g08_skill_ships_all_approved_local_intake_templates():
    expected = {
        "hypothesis_plan_intake.json",
        "project_fact_intake.json",
        "meeting_brief_intake.json",
        "research_path_intake.json",
        "learning_guide_intake.json",
        "document_lifecycle_review_intake.json",
        "workflow_discovery_scope.json",
        "paper_output_intake.json",
        "experiment_intake_interface.json",
    }
    actual = {
        path.name for path in (SKILL_ROOT / "templates").glob("*.json")
    }
    assert actual == expected
    for path in (SKILL_ROOT / "templates").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["fixture_only"] is True
        assert payload["real_operation"] == "NOT RUN"


def test_g08_seven_technical_routes_use_synthetic_templates_without_real_writes(
    tmp_path,
):
    root = tmp_path / "repo"
    templates = SKILL_ROOT / "templates"
    (root / "experiments").mkdir(parents=True)
    (root / "experiments" / "experiment_registry.json").write_text(
        json.dumps(
            {
                "version": 1,
                "experiments": [
                    {
                        "id": "fixture-accepted",
                        "result_id": "result-accepted",
                        "evidence_status": "accepted",
                        "status": "done",
                    },
                    {
                        "id": "fixture-failed",
                        "result_id": "result-failed",
                        "evidence_status": "rejected",
                        "status": "failed",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "project_state").mkdir()
    (root / "project_state" / "document_registry.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "documents": [
                    {
                        "doc_id": "fixture-active-document",
                        "path": "fixture.md",
                        "lifecycle": "active",
                        "freshness": "fresh",
                        "dependency_fingerprints": {"source": "fixture-old"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def load(name):
        return json.loads((templates / name).read_text(encoding="utf-8"))

    outputs = {
        "project_fact": build_document_profile(
            root,
            profile="project_fact",
            payload=load("project_fact_intake.json"),
        ),
        "meeting_brief": build_document_profile(
            root,
            profile="meeting_brief",
            payload=load("meeting_brief_intake.json"),
        ),
        "research_path": build_document_profile(
            root,
            profile="research_path",
            payload=load("research_path_intake.json"),
        ),
        "learning_guide": build_document_profile(
            root,
            profile="learning_guide",
            payload=load("learning_guide_intake.json"),
        ),
        "lifecycle_review": build_document_profile(
            root,
            profile="lifecycle_review",
            payload=load("document_lifecycle_review_intake.json"),
        ),
        "paper_output": build_document_profile(
            root,
            profile="paper_output",
            payload=load("paper_output_intake.json"),
        ),
    }
    assert outputs["project_fact"]["status"] == "confirmation_required"
    assert outputs["learning_guide"]["status"] == "interface_reserved"
    assert outputs["lifecycle_review"]["status"] == "review_due"
    assert outputs["paper_output"]["status"] == "template_pending"
    assert "探索材料（非 accepted）" in outputs["meeting_brief"]["body"]
    assert "停止原因" in outputs["research_path"]["body"]

    scope = load("workflow_discovery_scope.json")
    marker = (
        '<!-- pfmval-workflow: {"purpose":"fixture","trigger":"fixture",'
        '"inputs":[],"outputs":[]} -->\n'
    )
    for relative in scope["selected_paths"]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(marker, encoding="utf-8")
    discovery = discover_workflows(
        root,
        selected_paths=scope["selected_paths"],
        tracked_paths=scope["selected_paths"],
    )
    assert discovery["status"] == "candidate_review_required"
    assert all(
        item["lifecycle"] == "candidate"
        for item in discovery["candidates"]
    )
    assert not (root / "project_state" / "project_facts.jsonl").exists()
    assert not (root / "project_state" / "workflow_catalog.json").exists()


def test_g08_approved_manifest_has_six_user_entries_and_seven_routes():
    manifest = json.loads(
        (
            ROOT
            / "project_state"
            / "workflow_manifests"
            / "g08_approved_workflows.json"
        ).read_text(encoding="utf-8")
    )
    expected_ids = {
        "WF-PROJECT-FACT",
        "WF-LONGTERM-DOCS",
        "WF-LEARNING-GUIDE",
        "WF-DOCUMENT-LIFECYCLE",
        "WF-WORKFLOW-DISCOVERY",
        "WF-PAPER-OUTPUT",
    }
    assert {item["workflow_id"] for item in manifest["workflows"]} == expected_ids
    routes = [
        route
        for item in manifest["workflows"]
        for route in item["technical_routes"]
    ]
    assert len(routes) == 7
    assert len(set(routes)) == 7
    assert all(item["version"] == "0.1.0" for item in manifest["workflows"])
    assert all(
        item["real_operation_status"] == "NOT RUN"
        for item in manifest["workflows"]
    )


def test_g08_repository_catalog_admits_only_the_six_approved_workflows():
    catalog = json.loads(
        (ROOT / "project_state" / "workflow_catalog.json").read_text(
            encoding="utf-8"
        )
    )
    expected_ids = {
        "WF-PROJECT-FACT",
        "WF-LONGTERM-DOCS",
        "WF-LEARNING-GUIDE",
        "WF-DOCUMENT-LIFECYCLE",
        "WF-WORKFLOW-DISCOVERY",
        "WF-PAPER-OUTPUT",
    }
    assert {item["workflow_id"] for item in catalog["entries"]} == expected_ids
    assert all(item["lifecycle"] == "active" for item in catalog["entries"])
    assert all(item["standardized"] is False for item in catalog["entries"])
    assert all(item["version"] == "0.1.0" for item in catalog["entries"])
    assert all(
        item["authorization_ref"] == "DIR-20260727-004"
        for item in catalog["entries"]
    )
    assert len(catalog["scans"]) >= 1
