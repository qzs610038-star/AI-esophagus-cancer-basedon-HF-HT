"""One-shot, hash-free maintenance authorized in the 2026-09-19 health task.

This script changes metadata and navigation only. It never accepts a result.
The revision guard prevents accidentally replaying this maintenance.
"""
from __future__ import annotations

import copy
import csv
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "project_state/health_checks/20260919"
NOW = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
REV = 265
REPORT = "maintenance_logs/项目事实与规则治理完成与待决事项_20260919.md"
MODEL = "团队项目进度与结论/qzs/Phase2最终模型_Phase3交接包_20260918"
HANDOFF = "团队项目进度与结论/方案共享/Phase2完整交接包_20260919_首版"
TEAM = "团队项目进度与结论/ljq"
TEAM_ID = "mpp2_task23_scratch_ablation_20260915"
TEAM_RESULT = f"{TEAM}/result/{TEAM_ID}"
BATCH2 = "phase2_backbone_spatial_ablation_batch2_20260919"
REUSE = "mpp2_uni2h_mlp_baseline_20260906"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def write(path, value):
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name + ".governance-tmp")
    staging.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    staging.replace(target)


state = read("project_state/current_state.json")
docs = read("project_state/document_registry.json")
registry = read("experiments/experiment_registry.json")
assert state["state_revision"] == 264, "Recheck concurrent state changes before applying."
before = copy.deepcopy({"state": state, "docs": docs, "registry": registry})
experiments = {x["id"]: x for x in registry["experiments"]}
assert all(x not in experiments for x in [BATCH2, REUSE, TEAM_ID])
fullfov = experiments["phase2_fullfov_hpo_v1"]
assert fullfov["evidence_status"] == "accepted" and fullfov["accepted"] is True

# Register code existence without inferring approval, server readiness or runs.
for name, display, role in [
    (BATCH2, "Phase2 第二批全视野空间头基础模型替换", "planned_new_experiment_code"),
    (REUSE, "MPP2 UNI2-h MLP 独立复用代码包", "reusable_code_not_new_accepted_baseline"),
]:
    package = read(f"experiments/{name}/package.json")
    record = {
        "id": name, "display_name": display, "family": "phase2_supplemental",
        "status": "planned", "evidence_status": "planned", "accepted": False,
        "registration_status": "registered", "registered_at": NOW, "updated_at": NOW,
        "registration_scope": "code_package_metadata_only_no_result_or_training_authorization",
        "experiment_package": f"experiments/{name}", "code_version": package["code_version"],
        "entrypoint": package["entrypoint"], "code_status": "local_package_present",
        "package_role": role, "metric_registration_status": "no_returned_result_registered",
        "source_evidence": [f"experiments/{name}/package.json", f"experiments/{name}/README.md"],
        "training_authorized_by_this_registration": False,
        "next_action": "retain_code_registration_and_follow_separate_execution_instruction",
        "governance_record": REPORT,
    }
    if "protocol_version" in package:
        record["protocol_version"] = package["protocol_version"]
    if name == BATCH2:
        record["baseline_experiment_id"] = "phase2_fullfov_hpo_v1"
    else:
        record["baseline_experiment_id"] = "mpp2_barcode_repair_v003_frozen_baseline_20260711"
        record["notes"] = "本条只登记独立复用包，不替代原 accepted 基线，也不把默认 smoke 视为正式训练。"
    registry["experiments"].append(record)

# Two physically returned runs remain explicitly unaccepted, with raw metrics
# referenced rather than recopied into a purported accepted performance table.
team_runs = []
for run_dir in sorted((ROOT / TEAM_RESULT).iterdir()):
    summary_file = run_dir / "batch_summary.json"
    if not summary_file.is_file():
        continue
    summary = json.loads(summary_file.read_text(encoding="utf-8-sig"))
    metrics_file = run_dir / "summary_metrics.csv"
    with metrics_file.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    run_kind = summary["run_kind"]
    assert run_kind in {"full", "smoke"} and len(rows) == 10
    run_path = run_dir.relative_to(ROOT).as_posix()
    team_runs.append({
        "run_id": run_dir.name, "run_kind": run_kind, "result_root": run_path,
        "summary_path": f"{run_path}/batch_summary.json",
        "metrics_path": f"{run_path}/summary_metrics.csv",
        "reported_package_check_status": summary["status"],
        "evidence_status": "pending" if run_kind == "full" else "diagnostic",
        "accepted": False, "metric_rows": len(rows),
        "arms": sorted({r["arm_id"] for r in rows}),
        "splits": sorted({r["split"] for r in rows}),
        "reported_optimizer_updates": sorted({int(r["actual_optimizer_updates"]) for r in rows}),
        "verification_scope": "local_metadata_and_csv_presence_only_not_independent_scientific_acceptance",
    })
assert {r["run_kind"] for r in team_runs} == {"full", "smoke"}
pending_id = "mpp2-task23-scratch-ablation-20260915-full-pending-result"
team_record = {
    "id": TEAM_ID, "display_name": "LJQ 任务2/3 从零训练回传（待审）",
    "family": "phase2_team_return", "status": "returned_pending_review",
    "registration_status": "registered", "evidence_status": "pending", "accepted": False,
    "registered_at": NOW, "updated_at": NOW,
    "experiment_package": f"{TEAM}/{TEAM_ID}", "result_root": TEAM_RESULT,
    "result_id": pending_id, "result_phase": "returned_full_run_not_accepted",
    "registration_scope": "returned_run_inventory_and_metric_paths_only_no_scientific_acceptance",
    "returned_runs": team_runs,
    "related_experiment_ids": ["phase2_gene_reconstruction_comparison_v001_20260904", "phase2_spatial_density_comparison_v001_20260904"],
    "metric_definitions_status": "raw_columns_preserved_aggregation_and_target_contract_pending_review",
    "reported_pcc_columns": ["patient_pathway_macro_pcc", "flattened_pcc"],
    "metric_source_boundary": "数值留在各运行 summary_metrics.csv；未跨任务、标签或种子合并，未生成 accepted 指标。",
    "review_required": ["任务2标签、背景基因集合与评分合同", "任务3旧输入协议与新全视野方案的比较边界", "full 与 smoke 分开复核"],
    "evidence_paths": [f"{TEAM}/任务2和任务3消融实验报告_20260915.docx", f"{TEAM}/任务2对照组与历史基线差异说明_20260916.docx"],
    "next_action": "user_decides_target_contract_and_bounded_acceptance_review",
    "training_authorized_by_this_registration": False, "governance_record": REPORT,
    "notes": "包内 verified 仅是包自身检查状态，不等于 Registry accepted；本条不替代中央任务2/3的计划记录。",
}
registry["experiments"].append(team_record)
fullfov["registration_scope"] = "completed_run_and_metrics_reviewed_and_accepted_with_recorded_limitations"
ridge = experiments["mpp2_pathway_ridge_calibration_v001_20260717"]
ridge["historical_next_action"] = ridge["next_action"]
ridge["next_action"] = "closed_failed_rejected_no_automatic_rerun_or_deployment"
ridge["governance_note"] = "2026-09-19 对齐既有 closed_failed 方案和 rejected 记录；保留旧授权与实验数值，不再把旧 replacement job 建议作为当前待办。"

# Align duplicated current policy with the existing September 12 directive.
old_policy = copy.deepcopy(registry["current_mpp_policy"])
registry.setdefault("current_mpp_policy_history", []).append({**old_policy, "status": "historical_partially_superseded", "superseded_clauses_by": "DIR-20260912-002", "retained_clauses": ["MPP2 selected", "all metrics retained", "six-fold deferred"]})
registry["current_mpp_policy"] = {
    "effective_date": "2026-09-12", "selected_mpp": 2,
    "decision": "固定经典空间残差，暂停方法改进；推进补充实验合同、团队交接与论文准备。",
    "rationale": "同步已存在 DIR-20260912-002；本轮没有新增路线决定。",
    "next_recommended_experiment": "按已登记补充计划和后续明确指令推进；此治理登记不授权训练。",
    "reference_doc": "project_state/governance/Phase2组会决策与论文后续任务_20260912.md",
    "directive_id": "DIR-20260912-002",
    "governance_refreshed_at": NOW,
}
program = registry["phase2_supplemental_program"]
contract = program["metric_registration_contract"]
contract["aggregation_protocol_history"] = [{
    "recorded_under_contract": contract["id"],
    "status": "historical_definition_not_current_pcc_display",
    "aggregation_protocol": copy.deepcopy(contract["aggregation_protocol"]),
    "note": "历史 Fisher 汇总数值保持原定义，不按新展示规则重命名或覆盖。",
}]
contract["aggregation_protocol"]["unit_order"] = "within_patient_and_pathway_then_equal_arithmetic_mean"
contract["aggregation_protocol"]["pcc_across_patients"] = "equal_arithmetic_mean_after_within_patient_pathway_pearson"
contract["aggregation_protocol"]["pcc_display_directive_ids"] = ["DIR-20260908-003", "DIR-20260908-004"]
contract["aggregation_protocol"]["historical_metric_boundary"] = "Fisher 汇总、患者—通路等权算术平均和整体展平是不同指标；旧值保持原义。展示修订不改变既定选模指标。"

# Current state: synchronize facts, retain limitations and unconfirmed delivery.
state["state_revision"] = REV
state["updated_at"] = NOW
for directive_id in ["DIR-20260908-003", "DIR-20260908-004"]:
    if directive_id not in state["active_directive_ids"]:
        state["active_directive_ids"].append(directive_id)
for key, reason in [
    ("mpp_training", "历史 MPP 训练合同；当前交付与方法入口已转为独立包及 Phase2 经典空间路线，旧 accepted 基线保留。"),
    ("mpp2_pathway_ridge_calibration", "方案已有 closed_failed 结案，实验 rejected，不再作为活跃执行计划。"),
]:
    old = state["active_plans"].pop(key)
    state["historical_plans"][key] = {**old, "status": "historical", "archived_at": NOW, "archive_reason": reason}
state["pending_result_ids"] = [x for x in state["pending_result_ids"] if x != fullfov["result_id"]]
state["pending_result_ids"].append(pending_id)
for exp_id in ["phase2_softlink_local_v2", "phase2_softlink_contrastive_v3", "phase2_spatial_warmstart_v1", "phase2_fullfov_hpo_v1"]:
    record = experiments[exp_id]
    assert record["evidence_status"] == "accepted"
    if record["result_id"] not in state["latest_accepted_result_ids"]:
        state["latest_accepted_result_ids"].append(record["result_id"])
p = state["phase2_program"]
p["updated_at"] = NOW
p["current_method"] = "经典空间残差方法保持固定；当前模型交付采用已接纳 full-FOV 批次的 UNI2-h spatial seed45，第一轮 spatial 保留为方法来源。"
p["supplementary_execution_status"] = "fullfov_accepted_handoff_materials_ready_task3_code_ready_team_return_pending_review"
p["latest_result_review"].update({"status": "accepted", "accepted_at": fullfov["accepted_at"], "finding": fullfov["accepted_conclusion"], "review_record_path": fullfov["review_record_path"]})
p["latest_result_review"]["limitations"] = [
    "外部仅 XZY 一名患者，且不是全新盲测",
    "patch_coverage_size 缺失，物理图像支持域与划分无重叠尚未独立确认",
    "原训练批次 Phase3 合同导出未回传；后续独立模型交接包已在本地生成，两者分别记录",
]
p["current_model_delivery"] = {
    "path": MODEL, "entry": f"{MODEL}/README.md", "package_version": "1.1.0",
    "source_experiment_id": "phase2_fullfov_hpo_v1", "source_result_id": fullfov["result_id"],
    "model": "UNI2-h", "arm": "spatial", "seed": 45, "selected_formal_epoch": 28,
    "local_checkpoint": f"{MODEL}/weights/formal_best.pt",
    "local_checkpoint_exists": (ROOT / MODEL / "weights/formal_best.pt").is_file(),
    "local_checkpoint_size_bytes": (ROOT / MODEL / "weights/formal_best.pt").stat().st_size,
    "validation_evidence": [f"{MODEL}/audit/handoff_validation.json", f"{MODEL}/audit/experiment_consistency.json"],
    "validation_boundary": "记录既有本地交付验证与实体存在；不代表真实 Phase3 队列已运行或服务器现存路径已核验。",
    "team_delivery_status": "not_confirmed",
}
p["task_2_3_handoff"].update({
    "status": "task2_contract_pending_task3_new_code_ready_team_return_pending_review",
    "task3_experiment_id": "phase2_spatial_density_comparison_v001_20260904",
    "task3_current_package": "experiments/phase2_task3_fullfov_density_ablation_v1",
    "task3_code_status": experiments["phase2_spatial_density_comparison_v001_20260904"]["code_status"],
    "team_return_experiment_id": TEAM_ID, "team_return_evidence_status": "pending",
    "task2_diagnostic_report": "03审计报告/Phase2任务二_319基因ssGSEA重构通路与BarcodeRepairV003标签差异_专项审计报告_20260919.md",
    "task2_diagnostic_merge_status": "pending_user_decision_do_not_merge_into_handoff_plan",
})
p["pending_inputs"] = ["task2_target_and_scoring_contract", "team_return_acceptance_scope", "handoff_delivery_and_member_confirmation", "final_publication_metric_selection", "paper_code_and_architecture_figure_plan"]
p["paper_preparation"].update({"status": "first_handoff_materials_ready_team_review_and_paper_writing_pending", "handoff_entry": f"{HANDOFF}/00_从这里开始.md", "delivery_check": f"{HANDOFF}/版本与交付检查.md", "team_delivery_status": "not_confirmed", "teammate_review_status": "not_confirmed"})
p["registered_code_only_experiment_ids"] = [BATCH2, REUSE]
for task in state["near_term_tasks"]:
    if task["id"] == "20260918_phase2_baseline_team_handoff":
        task.update({"status": "materials_ready_delivery_unconfirmed", "materials_status": "ready", "delivery_status": "not_confirmed", "materials_path": MODEL, "note": "全视野结果已于 9月18日接纳；本地正式权重与配置交接包存在。没有实际发送或队友验收的确认记录。"})
    elif task["id"] == "20260918_19_phase2_exploration_materials":
        task.update({"status": "first_version_ready_delivery_and_review_unconfirmed", "materials_status": "first_version_ready", "delivery_status": "not_confirmed", "materials_path": HANDOFF, "note": "9月19日首版交接包已生成并完成既有文字修订；实际发送、队友评审及论文填写未确认。"})
state["governance_maintenance"] = {"updated_at": NOW, "record": REPORT, "scope": "confirmed_registration_and_document_governance_only", "new_result_acceptances": 0}
for key in ["updated_at", "current_method", "supplementary_execution_status", "task_2_3_handoff", "paper_preparation", "current_model_delivery", "registered_code_only_experiment_ids"]:
    program[key] = copy.deepcopy(p[key])
registry["updated_at"] = NOW

# Document metadata: detailed reviewed candidates, plus explicit bundle coverage.
tracked = set(subprocess.run(["git", "-c", "core.quotepath=false", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True).stdout.decode("utf-8").split("\0"))
by_path = {x["path"]: x for x in docs["documents"]}
by_id = {x["doc_id"]: x for x in docs["documents"]}
counter = 0


def register(path, doc_id=None, lifecycle="pending_review", authority="reference", reason="已核实文件存在；本轮未逐条审核正文，不作为已确认结论或当前执行授权。", role="reference_document", category=None):
    global counter
    assert (ROOT / path).is_file(), path
    old = by_path.get(path)
    if old:
        return old
    if doc_id is None:
        counter += 1
        doc_id = f"governance-supplement-20260919-{counter:03d}"
    assert doc_id not in by_id, doc_id
    item = {"doc_id": doc_id, "path": path, "category": category or path.split('/')[0],
            "scope": "phase2_handoff_governance", "authority": authority, "lifecycle": lifecycle,
            "availability": "tracked" if path in tracked else "local_only", "verified_at": NOW,
            "state_revision": REV, "supersedes": [], "superseded_by": [],
            "truth_sources": ["project_state/current_state.json", "experiments/experiment_registry.json", "project_state/directives.jsonl"],
            "doc_role": role, "freshness": "review_due", "freshness_reason": reason,
            "governance_record": REPORT, "verification_scope": "existence_and_document_role_not_automatic_content_acceptance"}
    docs["documents"].append(item)
    by_path[path] = item
    by_id[doc_id] = item
    return item


reviewed = read("project_state/health_checks/20260919/reviewed_document_actions.json")["items"]
for item in reviewed:
    if item["disposition"] == "covered_by_package":
        continue
    life = item["lifecycle_suggestion"]
    if item["number"] in {14, 17, 20, 27}:
        life = "historical" # superseded purpose / dated scope, not invalid evidence
    doc = register(item["path"], item["candidate_doc_id"], life, item["authority"], item["reason"])
    doc["related_docs"] = item["related_document_candidates"]
    if item["number"] == 29:
        doc["superseded_by"] = ["phase2-final-model-handoff-20260918"]
        doc["replacement_scope"] = "当前模型交付入口；不否定旧实验的 accepted 结果。"
    if item["number"] == 18:
        doc["handoff_plan_merge_status"] = "pending_user_decision_existing_hold_preserved"

# Add traceable actual result review without inventing a new result acceptance.
register(fullfov["report_path"], "phase2-fullfov-accepted-results-20260919", "active", "derived", "实验已接纳，限已登记批次；标题保留审核前原名，状态见接纳记录。", "experiment_report")
register(fullfov["review_record_path"], "phase2-fullfov-acceptance-record-20260918", "active", "reference", "本条定位已存在的 9月18日接纳记录，本轮未新增接纳。", "acceptance_record")

# Package-level coverage deliberately records every member path; cached tool
# READMEs are not documents. Existing independently registered members remain.
bundles = [
    (MODEL, f"{MODEL}/README.md", "active", "model_handoff_package"),
    (HANDOFF, f"{HANDOFF}/00_从这里开始.md", "active", "paper_handoff_package"),
    (f"{TEAM}/{TEAM_ID}", f"{TEAM}/{TEAM_ID}/README.md", "pending_review", "returned_experiment_package"),
    (f"{TEAM}/mpp2_softlink_spatial_task23_20260912/mpp2_softlink_spatial_task23_20260912", f"{TEAM}/mpp2_softlink_spatial_task23_20260912/mpp2_softlink_spatial_task23_20260912/README.md", "historical", "retired_finetuning_package"),
    ("团队项目进度与结论/qzs/Phase3_方案A经典空间残差_开箱包_20260912", "团队项目进度与结论/qzs/Phase3_方案A经典空间残差_开箱包_20260912/README.md", "superseded", "old_model_handoff_package"),
    ("03审计报告/20260821_Phase2-MPP2-PCC瓶颈", "03审计报告/20260821_Phase2-MPP2-PCC瓶颈/00_总管任务包.md", "historical", "historical_audit_bundle"),
    ("03审计报告/待审核/W007_精简结论包_20260821", "03审计报告/待审核/W007_精简结论包_20260821/审核入口.md", "pending_review", "pending_audit_bundle"),
]
covered = {}
for folder, entry, life, role in bundles:
    doc = register(entry, lifecycle=life, role=role)
    members = sorted(f.relative_to(ROOT).as_posix() for f in (ROOT/folder).rglob("*.md") if not {".pytest_cache", "__pycache__", ".git"}.intersection(f.parts) and f.relative_to(ROOT).as_posix() != entry)
    doc.update({"doc_role": role, "coverage_scope": "enumerated_local_bundle_members", "covered_documents": members, "coverage_policy": "包内页及复制件通过本入口定位；不因此取得独立规范性权威或科学接纳状态。"})
    covered.update({path: doc["doc_id"] for path in members})

log_index = register("maintenance_logs/README.md", "maintenance-log-index", "active", "reference", "维护摘要入口，日志不覆盖机器事实或用户授权。", "maintenance_index")
log_members = sorted(f.relative_to(ROOT).as_posix() for sub in ["daily", "weekly"] for f in (ROOT/"maintenance_logs"/sub).rglob("*.md"))
log_index["covered_documents"] = log_members
log_index["coverage_policy"] = "定期维护摘要和模板按目录入口登记；不是当前结论或批准记录。"
covered.update({path: log_index["doc_id"] for path in log_members})
figure_parent = by_id["phase2-softlink-v21-results-mechanism-report-20260908"]
figure_members = sorted(f.relative_to(ROOT).as_posix() for f in (ROOT/"01_指南与解读/分析报告/report_figures/phase2_softlink_v21_20260908").glob("*.md"))
figure_parent["covered_documents"] = figure_members
figure_parent["coverage_policy"] = "图示说明和历史底稿的导航覆盖；底稿仍是历史文档。"
covered.update({path: figure_parent["doc_id"] for path in figure_members})

# Resolve the complete raw path diff from the scan, conservatively labelling
# unreviewed material, and preserving obsolete execution material as history.
diff = read("project_state/health_checks/20260919/filesystem_document_diff.json")
dispositions = []
for directory in diff["directories"]:
    for path in directory["unregistered_paths"]:
        if {".pytest_cache", "__pycache__", ".git"}.intersection(Path(path).parts):
            dispositions.append({"path": path, "disposition": "excluded_tool_cache"})
            continue
        if path in covered and path not in by_path:
            dispositions.append({"path": path, "disposition": "covered_by_registered_bundle", "doc_id": covered[path]})
            continue
        if path not in by_path:
            historical = path.startswith("project_state/governance/") or (path.startswith("maintenance_logs/") and path != REPORT)
            historical = historical or (path.startswith("01_指南与解读/") and any(x in path for x in ["20260905", "20260906", "20260907", "20260908", "20260909", "20260910", "20260911"]))
            life = "historical" if historical else "pending_review"
            reason = "旧路线、旧回传流程或已发生事件的历史材料；保留当时证据，不作为当前执行授权。" if historical else "已核实文件存在；正文现行范围尚未逐项复核，保守待审。"
            if path in [f"{TEAM}/Phase2任务2与任务3_智能体交接合同_20260914.md", f"{TEAM}/Phase2任务2与任务3_队友交接说明_20260914.md"]:
                life = "active"
                reason = "9月14日队友从零训练合同入口；对实际9月15日回传的适用性与科研结论仍待审。"
            if path == "03审计报告/GIT_INDEX.md":
                life = "historical"
                reason = "8月目录状态快照，当前审计约束与入口以 AGENTS.md、pfmval-audit 和 03审计报告/README.md 为准。"
            register(path, lifecycle=life, reason=reason)
        dispositions.append({"path": path, "disposition": "independently_registered", "doc_id": by_path[path]["doc_id"]})

for record in registry["experiments"]:
    package = record.get("experiment_package")
    if package and (ROOT/package/"README.md").is_file():
        doc = register(f"{package}/README.md", lifecycle="active" if record["id"] in [BATCH2, REUSE, "phase2_fullfov_hpo_v1", "phase2_spatial_density_comparison_v001_20260904"] else "historical", role="experiment_package_readme", reason="仅登记实际代码包使用说明；实验、训练与接纳状态分别以对应 Registry 记录为准。")
        doc["related_experiment_id"] = record["id"]
    for key in ["report_path", "review_record_path"]:
        path = record.get(key)
        if path and (ROOT/path).is_file() and Path(path).suffix.lower() in {".md", ".pdf", ".docx"}:
            doc = register(path, lifecycle="active" if record.get("evidence_status") == "accepted" else "pending_review", role="experiment_evidence_document")
            doc["related_experiment_id"] = record["id"]

for path in [".agents/skills/pfmval-audit/references/minimal-review.md", ".agents/skills/pfmval-audit/assets/minimal-review-record.md"]:
    register(path, lifecycle="active", reason="当前 pfmval-audit 主技能直接使用的操作参考；随技能主源适用，不增加独立审批。", role="skill_reference")
register("experiments/server_load_check_20260905/README.md", "server-load-check-package-20260905", "historical", "reference", "仅服务器抽样读取诊断包；已有47 PASS、3 WARN记录，不等于科研结果或完整训练就绪。", "diagnostic_package_readme")

for doc_id in ["plan-mpp-training", "doc-2f635595aea6", "doc-9328df98eb69", "doc-21798b23e36d", "doc-9e8fc4dbabef", "doc-f273703af094"]:
    d = by_id[doc_id]
    d.update({"lifecycle": "historical", "authority": "reference", "verified_at": NOW, "state_revision": REV, "freshness": "review_due", "freshness_reason": "已被后续明确决策或结案取代其当前执行用途；保留历史内容和既有科研接纳边界。", "governance_record": REPORT})
for doc_id in ["doc-eea543c941ec", "doc-3bbbee463565"]:
    d = by_id[doc_id]
    d.update({"lifecycle": "active", "authority": "reference", "scope": "skill_reference:pfmval-audit", "doc_role": "skill_reference", "verified_at": NOW, "state_revision": REV, "freshness": "fresh", "freshness_reason": "当前审查技能实际引用，不能误标为退役技能；以主技能及 AGENTS.md 范围为准。"})
figure_parent.update({"lifecycle": "active", "freshness": "review_due", "freshness_reason": "对应实验9月11日已接纳描述性结果；历史待审字样已由后续接纳记录推进，机制因果解释不自动获接纳。", "verified_at": NOW, "state_revision": REV})

register("maintenance_logs/项目事实与规则健康扫描_20260919.md", "governance-health-scan-20260919", "historical", "derived", "这是治理执行前的状态快照；已修复项以同日完成记录为准。", "maintenance_report")
assert (ROOT/REPORT).is_file(), "Create the report before registering it."
register(REPORT, "governance-health-refresh-20260919", "active", "derived", "本次确定治理工作的执行记录和待用户裁决项。", "maintenance_report")

# Reference candidates pointing at covered pages resolve to the parent entry.
covered_ids = {i["candidate_doc_id"]: covered[i["path"]] for i in reviewed if i["disposition"] == "covered_by_package"}
for d in docs["documents"]:
    for key in ["related_docs", "source_refs", "superseded_by", "supersedes"]:
        if key in d:
            d[key] = list(dict.fromkeys(covered_ids.get(x, x) for x in d[key]))
            assert all(x in by_id for x in d[key]), (d["doc_id"], key, d[key])
    if (ROOT/d["path"]).is_file():
        d["availability"] = "tracked" if d["path"] in tracked else "local_only"
for d in docs["documents"]:
    if d.get("authority") == "derived" and d["path"] in ["CURRENT_STATE.md", "PROJECT_GUIDE.md", "README.md", ".claude/next-steps.md", ".claude/session-brief.md", "experiments/experiment_progress.md", "experiments/experiment_dashboard.md"]:
        d.update({"verified_at": NOW, "state_revision": REV, "freshness": "fresh", "freshness_reason": "本轮定向刷新对应机器事实的导航状态。"})
docs["state_revision"] = REV
docs["updated_at"] = NOW
counts = Counter(x["lifecycle"] for x in docs["documents"])
docs["summary"] = {"total": len(docs["documents"]), "live": sum((ROOT/d["path"]).is_file() for d in docs["documents"]), **{key: counts[key] for key in ["missing", "active", "superseded", "historical", "pending_review", "approved_design", "draft"]}}

# Validate the proposed in-memory transaction before any canonical write.
import jsonschema
jsonschema.validate(state, read("project_state/schemas/current_state.schema.json"))
jsonschema.validate(docs, read("project_state/schemas/document_registry.schema.json"))
assert len(by_path) == len(by_id) == len(docs["documents"])
old_experiments = {x["id"]: x for x in before["registry"]["experiments"]}
allowed_changes = {"phase2_fullfov_hpo_v1": {"registration_scope"}, "mpp2_pathway_ridge_calibration_v001_20260717": {"next_action", "historical_next_action", "governance_note"}}
experiment_changes = []
for item in registry["experiments"]:
    if item["id"] not in old_experiments:
        assert item["accepted"] is False
        continue
    old = old_experiments[item["id"]]
    keys = [k for k in set(old)|set(item) if old.get(k) != item.get(k)]
    assert set(keys) <= allowed_changes.get(item["id"], set()), (item["id"], keys)
    if keys:
        experiment_changes.append({"id": item["id"], "fields": {k: {"before": old.get(k), "after": item.get(k)} for k in sorted(keys)}})
old_docs = {d["doc_id"]: d for d in before["docs"]["documents"]}
doc_changes = []
for d in docs["documents"]:
    old = old_docs.get(d["doc_id"])
    if old and old != d:
        fields = {k: {"before": old.get(k), "after": d.get(k)} for k in set(old)|set(d) if old.get(k) != d.get(k)}
        doc_changes.append({"doc_id": d["doc_id"], "path": d["path"], "fields": fields})
journal = {"applied_at": NOW, "state_revision_before": 264, "state_revision_after": REV,
           "authorization": "本任务用户明确要求先完成能够确定的补登记与文档治理；不确定项呈报。",
           "new_document_count": len(docs["documents"])-len(before["docs"]["documents"]),
           "new_documents": [d for d in docs["documents"] if d["doc_id"] not in old_docs],
           "updated_documents": doc_changes, "old_experiment_metadata_changes": experiment_changes,
           "new_experiment_ids": [BATCH2, REUSE, TEAM_ID], "new_result_acceptances": 0,
           "all_preexisting_experiment_metrics_unchanged": True,
           "source_diff_dispositions": dispositions, "summary": docs["summary"]}
write("project_state/health_checks/20260919/governance_changes.json", journal)
write("experiments/experiment_registry.json", registry)
write("project_state/current_state.json", state)
write("project_state/document_registry.json", docs)
print(json.dumps({"new_documents": journal["new_document_count"], "documents": docs["summary"], "experiments": len(registry["experiments"]), "dispositions": dict(Counter(d["disposition"] for d in dispositions)), "state_revision": REV}, ensure_ascii=False))
