from pathlib import Path
import json
import re
import subprocess

root = Path(__file__).resolve().parents[1]
state_path = root / "project_state/current_state.json"
state = json.loads(state_path.read_text(encoding="utf-8"))
decision = "DIR-20260905-002"
guide = "project_state/governance/独立实验包与手动回传_20260905.md"
historical_paths = [
    "project_state/plans/server_maintenance.md",
    "project_state/plans/gitee_numbered_workspace_protocol_v001_20260726.md",
    "01_指南与解读/部署方案/服务器零训练Gitee往返试点检查方案_20260810.md",
]
for path in historical_paths:
    file = root / path
    original = file.read_text(encoding="utf-8-sig")
    notice = "> **历史流程，2026-09-05 起不再执行。** 原文保留追溯；当前使用独立实验包与用户手动回传，Gitee 暂停、工作树仅用户明确要求时创建。当前说明见项目根目录 `" + guide + "`。\n\n"
    if not original.startswith(notice):
        file.write_text(notice + original, encoding="utf-8")
old = state["active_plans"].pop("server_maintenance")
old.update(status="historical", superseded_by=decision)
state["historical_plans"]["server_maintenance"] = old
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

index_paths = [p + "/GIT_INDEX.md" for p in ["01_指南与解读/学习指南", "01_指南与解读/部署方案", "02_组会汇报", "团队项目进度与结论"]]
registry_path = root / "project_state/document_registry.json"
registry = json.loads(registry_path.read_text(encoding="utf-8"))
obsolete = set(historical_paths + [v["path"] for v in state["historical_plan_reviews"].values()])
historical_ids = []
for doc in registry["documents"]:
    if doc["path"] in obsolete:
        doc.update(lifecycle="historical", lifecycle_override="historical", authority="reference", state_revision=state["state_revision"], verified_at=state["updated_at"])
        doc["superseded_by"] = list(dict.fromkeys(doc.get("superseded_by", []) + ["manual-experiment-delivery-20260905"]))
    if doc["path"] in index_paths:
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", doc["path"]], cwd=root, capture_output=True).returncode == 0
        doc.update(lifecycle="active", lifecycle_override="active", authority="reference", state_revision=state["state_revision"], verified_at=state["updated_at"], availability="tracked" if tracked else "local_only", superseded_by=[])
        doc["git_tracking_policy"] = "markdown_bodies_eligible_user_triggered_commit"
    if doc["lifecycle"] == "historical":
        historical_ids.append(doc["doc_id"])
registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

navigation_path = root / "PROJECT_GUIDE.md"
navigation = navigation_path.read_text(encoding="utf-8")
navigation = navigation.replace("### Active 部署方案\n\n- [doc-c1e997e52610](01_指南与解读/部署方案/服务器零训练Gitee往返试点检查方案_20260810.md)", "### 当前部署方式\n\n- [独立实验包与手动回传](" + guide + ")\n- [零训练模板与运行说明](experiments/_template/README.md)\n- 原 Gitee 往返试点仅作历史追溯，不是当前部署方案。")
# Remove historical documents only from the active-document section, preserving later historical lists.
start = navigation.index("## 当前 active 文档")
next_section = navigation.find("\n## ", start + 1)
end = next_section if next_section != -1 else len(navigation)
active = navigation[start:end]
active = "\n".join(line for line in active.split("\n") if not any(line.startswith("- [" + i + "]") for i in historical_ids))
active += "\n- [独立实验包与手动回传](" + guide + ")\n- [独立实验包模板](experiments/_template/README.md)\n"
navigation_path.write_text(navigation[:start] + active + navigation[end:], encoding="utf-8")

current_path = root / "CURRENT_STATE.md"
current = current_path.read_text(encoding="utf-8")
current = current.replace("- `server_maintenance`: [project_state/plans/server_maintenance.md](project_state/plans/server_maintenance.md)\n", "")
current_path.write_text(current, encoding="utf-8")

brief = root / ".claude/next-steps.md"
text = brief.read_text(encoding="utf-8")
text = text.replace("当前通道为 gitee；同步通道可按用户指令扩展。", "手动复制/压缩包回传；Gitee 暂停。独立包模板：experiments/_template/；工作树仅用户明确要求时创建。")
brief.write_text(text, encoding="utf-8")

# Include all follow-up metadata edits in the same user directive's affected-file inventory.
path = root / "project_state/directives.jsonl"
lines = path.read_text(encoding="utf-8").splitlines()
for index, line in enumerate(lines):
    event = json.loads(line)
    if event.get("directive_id") == decision:
        event["affected_files"] = list(dict.fromkeys(event["affected_files"] + historical_paths + ["project_state/plans/workflow_governance_v3_20260726.md", ".claude/next-steps.md"]))
        lines[index] = json.dumps(event, ensure_ascii=False)
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("Historical transport plans and current document navigation aligned.")
