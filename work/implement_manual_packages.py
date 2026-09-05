"""One-time, scoped maintenance for the user-approved manual package workflow."""
from datetime import datetime
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
changed = []
stamp = datetime.now().astimezone().isoformat(timespec="seconds")
decision = "DIR-20260905-002"
guide = "project_state/governance/独立实验包与手动回传_20260905.md"


def read(path):
    return (ROOT / path).read_text(encoding="utf-8-sig")


def write(path, text):
    file = ROOT / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8")
    changed.append(path)


def replace(path, old, new):
    text = read(path)
    assert old in text, (path, old)
    write(path, text.replace(old, new))


def dump(path, obj):
    write(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


rules = """## 当前实验交付方式（2026-09-05）

- 工作树规则彻底退役，仅作历史追溯；只有用户明确要求才创建工作树，不能自动创建或推荐为默认步骤。旧工作树和结果保留，不自主清理。
- 直接在当前项目 `experiments/<实验名>/` 内编写本轮全部代码，复制 `experiments/_template/` 开始；代码包相对封闭，包含必要项目代码、小型配置与划分，外部大数据及权重由配置引用。
- Windows 服务器只使用本项目 `configs/server_paths.yaml`：`D:\\AIPatho\\qzs\\code` 放代码，`D:\\AIPatho\\qzs\\runs` 放分实验、分运行的日志和原始结果。用户手动上传/回传，修复只替换对应代码目录，保留运行目录。
- Gitee 同步暂停；旧通道和脚本仅供追溯。GitHub 仅用于备份和网页端探索，提交与推送由用户触发。实验前不主动检查 Git 干净状态、不运行 start-check；大规模改动前先安排用户触发备份提交，不自动提交或推送。
- 服务器仅产出原始训练输出及训练、早停、模型选择必需指标；可由原始数据推导的其余指标、图表与登记在本地完成。既有“全部指标保留”继续有效，仅改变计算位置。
- 学习、探索、决策、团队贡献等指定文档目录中的 Markdown 正文允许纳入 Git；跟踪不代表提交已发生，也不代表科研结论获确认。团队正文仍只根据用户指令修改。
- 操作说明见 [独立实验包与手动回传](project_state/governance/独立实验包与手动回传_20260905.md)。

"""
replace("AGENTS.md", "除下列核心约束外，项目内既有流程、Registry、工作树、预检、哈希和同步约定均为建议或警告，不得自行阻断用户已明确授权的工作。", "旧流程不覆盖下列核心约束及当前实验交付方式；Registry、诊断等按需使用，不得自行阻断用户已明确授权的工作。")
replace("AGENTS.md", "## 任务相关事实源与建议阅读", rules + "## 任务相关事实源与建议阅读")
replace("AGENTS.md", "推荐流程：保留原始运行输出和必要指标；按", "手动回传时先保存原始输出，在本地补充派生指标并登记；服务器不生成旧登记信封。旧导入器若要求 W###、提交编号或哈希，不伪造字段、不自动计算哈希，可根据用户授权直接维护相关登记，结果未经确认不得标为 accepted。\n\n历史兼容导入流程（已有对应元数据时按需使用）：保留原始运行输出和必要指标；按")
replace("AGENTS.md", "- `project_state/current_state.json`、实验 Registry、文档 Registry、W### 工作树和 `start-check` 是可选的组织、检索与诊断工具，不是通用前置门禁。结构损坏或未完成写事务可阻止继续写入，其余不一致原则上报告为 `WARN`。", "- 当前状态、实验与文档 Registry 用于按需检索。工作树仅保留历史，创建须用户明确指令；start-check 不作为日常实验前步骤。结构损坏或未完成写事务可阻止继续写入，其余不一致原则上报告为 `WARN`。")
replace("AGENTS.md", "- 同步方式按当前配置和用户指令选择。当前已配置的可用通道是 Gitee；未来可增加用户手动压缩包、SSH 或其他通道，不设永久唯一通道。", "- 当前通道是用户手动复制/压缩包；Gitee 暂停，SSH 暂不使用；不自动进行任何远端同步。")
replace("AGENTS.md", "- 哈希只在用户要求核验特定关键实验产物，或具体格式无法取消该字段时作为可选完整性证据；不得把同步、普通文档、导航视图或一般维护哈希作为硬门禁。", "- 不默认计算哈希；确需启用时先说明原因、必要性及影响，得到用户明确批准后再做。旧格式有哈希字段不构成自动授权。")

replace(".agents/skills/pfmval-governance/SKILL.md", "- Treat W### worktrees, implementation plans, Registry entries, lifecycle labels, `start-check`, source commits and hashes as optional organization or audit mechanisms unless the task specifically requires one.", "- Worktree rules are retired historical requirements. Create a worktree only on an explicit user instruction; work directly in experiments/<experiment_name>/ using experiments/_template/. Keep each package self-contained.\n- Do not check Git cleanliness or run start-check before ordinary experiments. GitHub commits/pushes are user-triggered backups; arrange a user-triggered backup commit before large changes. Do not calculate hashes without the user's explicit approval.")
replace(".agents/skills/pfmval-governance/SKILL.md", "Synchronization is configurable. Gitee is the currently configured channel, not a permanent exclusive channel; user-managed archives, SSH or other channels may be added later. Do not require synchronization hashes for ordinary transfers or maintenance.", "Current transport is user-managed copying/archives; Gitee is paused and SSH is unused. Read only this project's configs/server_paths.yaml for the Windows server: QZS code and runs directories are separate. Preserve runs when replacing code. Compute derived metrics and register results locally, not on the server. For concrete package instructions, read experiments/_template/README.md. Designated project Markdown bodies are eligible for Git backup; this does not authorize commits, pushes or changes to team content.")
replace(".agents/skills/pfmval-audit/SKILL.md", "Use `python deploy/pfmval_ops.py agent start-check` when its diagnostics help;", "Do not create worktrees without an explicit user instruction or run routine pre-experiment Git checks. The current workflow uses independent packages and manual returns; Gitee is paused. Use `python deploy/pfmval_ops.py agent start-check` only for a task-specific diagnostic need;")

old_paths = ["experiments/workspaces/_template/README.md", "automation/README.md", "deploy/SYNC_GUIDE.md"]
for path in old_paths:
    header = "> **历史流程，已由 2026-09-05 用户决定取代。** 下文仅供追溯，不是当前操作指令。新实验使用 `experiments/_template/`，手动上传/回传，工作树仅在用户明确要求时创建，Gitee 暂停。当前说明：`project_state/governance/独立实验包与手动回传_20260905.md`。\n\n"
    write(path, header + read(path))

# Preserve historical transport/path records; add new roots without redirecting legacy scripts.
path = "configs/server_paths.yaml"
text = read(path)
start = text.index("transport:\n")
end = text.index("paths:\n", start)
old_transport = text[start:end].replace("transport:\n", "legacy_transport:\n", 1)
new_transport = """transport:
  mode: configurable
  current_channel: manual_archive
  active_channels: [manual_archive]
  paused_channels: [gitee]
  candidate_channels: [ssh]
  remote: null
  fetch_exact_commit_only: false
  force_push_allowed: false
  code_root_key: server_manual_code
  runs_root_key: server_manual_runs
"""
text = text[:start] + new_transport + "# Historical Gitee configuration; not the current execution policy.\n" + old_transport + text[end:]
text = text.replace('updated_at: "2026-08-10T00:00:00+08:00"', f'updated_at: "{stamp}"', 1)
text = text.replace("paths:\n", r"""paths:
  server_manual_code:
    path: 'D:\AIPatho\qzs\code'
    kind: directory
    status: active
    required_on: server
    stability: user_selected
    role: manual_experiment_package_root
    write_policy: user_replaces_experiment_code_only
    live_verification_required: true
    note: User-selected destination; not created or verified on the server by this change.
  server_manual_runs:
    path: 'D:\AIPatho\qzs\runs'
    kind: directory
    status: active
    required_on: server
    stability: user_selected
    role: manual_experiment_run_root
    write_policy: new_run_directory_only
    live_verification_required: true
    note: Preserve existing runs when replacing code; not created on the server by this change.
""", 1)
for key in ["server_governance_checkout", "server_experiment_worktrees", "server_experiment_runs", "server_result_returns", "server_runtime_bundles"]:
    pattern = rf"(  {key}:\n.*?)(?=\n  \w|\Z)"
    text = re.sub(pattern, lambda m: m[0].replace("status: active", "status: legacy"), text, flags=re.S)
write(path, text)

# Remove the retired index-only policy blocks, then add directory-scoped Markdown rules.
path = ".gitignore"
text = read(path)
start = text.index("# Learning guides: synchronize only")
end = text.index("# Historical reference package", start)
text = text[:start] + text[end:]
text = text.replace("# Deployment material is local-only; synchronize its directory index only.", "# Deployment non-Markdown attachments stay local; Markdown rules are below.")
text = text.replace("# Analysis reports are fully version-controlled and synchronize with worktrees.", "# Preserve existing analysis-report tracking.")
text += "\n# 2026-09-05: project knowledge Markdown bodies are eligible for user-triggered Git backup.\n"
for folder in ["01_指南与解读", "02_组会汇报", "团队项目进度与结论", "docs", "archive", "Ai病理项目文献汇总"]:
    text += f"!/{folder}/\n/{folder}/**\n!/{folder}/**/\n!/{folder}/**/*.md\n"
text += """
# Preserve the prior analysis-report policy; runtime exclusions still apply.
!/01_指南与解读/分析报告/**
!experiments/explorations/**/
!experiments/explorations/**/*.md

# New experiment packages remain source-controlled; raw runs/derived output are local.
experiments/**/runs/
experiments/**/analysis/
experiments/**/checkpoints/
experiments/**/*.pt
experiments/**/*.pth
experiments/**/*.ckpt
experiments/**/__pycache__/
experiments/**/*.py[cod]
experiments/_template/tests/work/
work/manual_package_validation/
**/*.log
"""
write(path, text)

for folder, title, purpose in [
    ("01_指南与解读/学习指南", "学习指南", "学习、探索、实验复盘和技术理解"),
    ("01_指南与解读/部署方案", "部署方案", "服务器配置说明、手动上传回传和历史排障"),
    ("02_组会汇报", "组会汇报", "会议记录、项目讨论和团队决定"),
    ("团队项目进度与结论", "团队协作", "成员进度、探索、决策和贡献"),
]:
    write(folder + "/GIT_INDEX.md", f"""# {title}目录索引

> 2026-09-05 更新：本目录及子目录 Markdown 正文允许纳入 Git，不再只跟踪索引。依据 DIR-20260905-002；旧索引规则由本次决定取代，历史可从 Git 追溯。

本目录保存{purpose}。GitHub 用于项目备份与网页端查阅；提交、推送由用户触发，本次只调整跟踪规则。附件、数据、权重及运行产物继续按忽略规则保留在本地。

当前实验使用 `experiments/<实验名>/` 独立包，服务器代码与结果由用户手动传输；Gitee 暂停，不因阅读或提交文档创建工作树。服务器路径以项目 `configs/server_paths.yaml` 为准。

历史文档用于追溯，不覆盖用户最新指令；正文纳入 Git 不自动使探索结果成为已确认结论。团队目录仍仅根据用户指令修改，本次不改写成员材料正文。
""")

state = json.loads(read("project_state/current_state.json"))
revision = state["state_revision"] + 1
state.update(state_revision=revision, updated_at=stamp)
state["active_directive_ids"].append(decision)
state["legacy_server_transport"] = state["server_transport"]
state["server_transport"] = {
    "mode": "configurable", "current_channel": "manual_archive", "active_channels": ["manual_archive"],
    "paused_channels": ["gitee"], "candidate_channels": ["ssh"], "remote_name": None,
    "allowed_operations": ["user_manual_copy", "user_manual_return"],
    "canonical_code_writer": "local_experiment_package", "return_mode": "user_owned_run_directory",
    "code_root_key": "server_manual_code", "runs_root_key": "server_manual_runs", "force_push_allowed": False,
}
state["experiment_delivery"] = {
    "decision_id": decision, "guide": guide, "template": "experiments/_template",
    "local_package_pattern": "experiments/<experiment_name>",
    "worktree_policy": "historical_only_create_on_explicit_user_instruction",
    "routine_git_precheck": False, "github_role": "backup_and_web_reading",
    "commit_push_policy": "user_triggered_only", "before_large_changes": "arrange_user_triggered_backup_commit",
    "server_outputs": "raw_training_outputs_and_training_selection_metrics",
    "derived_metrics_and_registration": "local", "preserve_all_previously_retained_metrics": True,
    "replace_code_preserve_runs": True, "default_resume": False,
}
state.setdefault("historical_plans", {})["workflow_governance_v3"] = state["active_plans"].pop("workflow_governance_v3")
state["historical_plans"]["workflow_governance_v3"].update(status="historical", superseded_by=decision)
for key, item in list(state.get("pending_plan_reviews", {}).items()):
    if key.startswith(("gitee_numbered_workspace", "workflow_governance_v3_refactor")):
        item.update(status="historical", superseded_by=decision)
        state.setdefault("historical_plan_reviews", {})[key] = state["pending_plan_reviews"].pop(key)
state["active_plans"]["manual_experiment_delivery"] = {"doc_id": "manual-experiment-delivery-20260905", "path": guide, "status": "active", "approved_at": stamp}
state["superseded_conclusions"] += ["Worktrees are an optional default experiment organization tool; new creation now requires an explicit user instruction.", "Gitee is the current server transport; it is now paused in favor of manual packages.", "Learning, deployment and team Markdown bodies are local-only; designated directories now permit Git backup."]
dump("project_state/current_state.json", state)

summary = "用户批准独立实验包与手动回传策略：工作树要求彻底退役，仅用户明确指令可创建；新实验在 experiments/<实验名> 自包含代码包内维护；PFMval Windows 服务器 D:\\AIPatho\\qzs 下分 code/runs，手动上传与回传，修复替换代码保留运行结果；Gitee 暂停，GitHub 仅备份和网页查阅，提交推送由用户触发，不做日常实验前 Git 预检；服务器只产出原始结果与训练选择必需指标，其余指标和登记移到本地；指定文档目录 Markdown 正文允许 Git 跟踪。本次仅规则、配置、通用零训练模板，不迁移旧实验、不训练、不操作服务器、不提交推送。"
directives = read("project_state/directives.jsonl")
assert decision not in directives
entry = {"event_type": "directive", "directive_id": decision, "issued_at": stamp, "summary": summary,
         "scope": "project_maintenance,experiment_delivery,server_transport,git_backup", "topic": "manual_self_contained_experiment_packages_v1",
         "status": "active", "supersedes": [], "amends": ["DIR-20260904-004", "DIR-20260905-001"],
         "effective_from_revision": revision, "decision_record": guide, "source": "explicit_user_instruction",
         "affected_files": list(dict.fromkeys(changed + [guide, "experiments/_template", "project_state/document_registry.json", "README.md", "CURRENT_STATE.md", "PROJECT_GUIDE.md"]))}
write("project_state/directives.jsonl", directives.rstrip() + "\n" + json.dumps(entry, ensure_ascii=False) + "\n")

replace("README.md", "- 服务器通信：当前通道为 **gitee**；后续可按用户指令扩展。", "- 服务器通信：**手动复制/压缩包回传**；Gitee 暂停。新实验使用 [独立包模板](experiments/_template/README.md)，工作树仅在用户明确要求时创建。\n- GitHub：仅备份与网页查阅，提交推送由用户触发；日常实验不进行 Git 干净状态预检。")
text = read("README.md")
write("README.md", re.sub(r"状态版本：`\d+`", f"状态版本：`{revision}`", text, count=1))
text = read("CURRENT_STATE.md")
text = re.sub(r"State revision: `\d+` \| Updated: `[^`]+`", f"State revision: `{revision}` | Updated: `{stamp}`", text, count=1)
text = text.replace("本次为会议决策的范围内更新。", "本次补充独立实验包与手动回传决定。")
text = text.replace("## Current directives\n", f"## Current directives\n\n- `{decision}`：{summary}\n")
text = text.replace("- Mode: **configurable**; current channel: `gitee`.\n- Active channels: gitee; candidate channels: manual_archive, ssh.", "- 当前通道：`manual_archive`（用户手动复制/压缩包）；Gitee 暂停，SSH 暂不使用。\n- 服务器：PFMval Windows，`D:\\AIPatho\\qzs\\code` 与 `D:\\AIPatho\\qzs\\runs`；未在本次连接或创建目录。\n- 新实验模板：`experiments/_template/`；可推导指标与结果登记在本地完成。")
text = text.replace("- `workflow_governance_v3`: [project_state/plans/workflow_governance_v3_20260726.md](project_state/plans/workflow_governance_v3_20260726.md)", f"- `manual_experiment_delivery`: [{guide}]({guide})\n- 原 workflow_governance_v3 已退为历史，其工作树/Gitee 规则不再执行。")
write("CURRENT_STATE.md", text)
text = read("PROJECT_GUIDE.md").replace("## 我要找什么", "## 当前交付方式（2026-09-05）\n\n新实验在 `experiments/<实验名>/` 独立包维护；工作树仅用户明确要求时创建；Gitee 暂停，Windows QZS 服务器手动上传代码、回传 runs。GitHub 只备份和网页查阅，提交推送由用户触发。\n\n- [模板与一键运行](experiments/_template/README.md)\n- [完整操作说明](" + guide + ")\n\n## 我要找什么")
text = text.replace("| 工作树身份与 lease |", "| 历史工作树身份（仅追溯） |")
text = text.replace("| `experiments/` | 实验事实与两类生成视图 |", "| `experiments/` | 独立实验包、实验事实与派生视图 |")
text = text.replace("| CLI-only |", "| 按用户授权维护 |")
text = text.replace("| CLI-only/generated |", "| 实验包直接维护，视图按需更新 |")
write("PROJECT_GUIDE.md", text)

registry = json.loads(read("project_state/document_registry.json"))
registry.update(updated_at=stamp, state_revision=revision)
for doc in registry["documents"]:
    if doc["path"] in old_paths or doc["path"] == "project_state/plans/workflow_governance_v3_20260726.md":
        doc.update(lifecycle="historical", lifecycle_override="historical", authority="reference", state_revision=revision, verified_at=stamp)
        doc["superseded_by"] = list(dict.fromkeys(doc.get("superseded_by", []) + ["manual-experiment-delivery-20260905"]))
for doc_id, path in [("manual-experiment-delivery-20260905", guide), ("manual-experiment-template-20260905", "experiments/_template/README.md")]:
    registry["documents"].append({"doc_id": doc_id, "path": path, "category": "实验交付", "scope": "manual_experiment_delivery",
        "authority": "normative", "lifecycle": "active", "availability": "local_only", "git_tracking_policy": "eligible_user_triggered_commit",
        "verified_at": stamp, "state_revision": revision, "supersedes": [], "superseded_by": [],
        "truth_sources": ["AGENTS.md", "configs/server_paths.yaml", "project_state/directives.jsonl"], "decision_id": decision})
dump("project_state/document_registry.json", registry)
dump("work/manual_package_changed_files.json", list(dict.fromkeys(changed)))
print(json.dumps({"decision": decision, "revision": revision, "changed_files": len(set(changed))}, ensure_ascii=False))
