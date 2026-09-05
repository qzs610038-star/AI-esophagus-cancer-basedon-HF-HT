from pathlib import Path
import json
import subprocess
import yaml
import jsonschema

root = Path(__file__).resolve().parents[1]
state_file = root / "project_state/current_state.json"
state = json.loads(state_file.read_text(encoding="utf-8"))
for key, item in list(state["pending_plan_reviews"].items()):
    if item["status"] == "historical":
        state.setdefault("historical_plan_reviews", {})[key] = state["pending_plan_reviews"].pop(key)
state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# Preserve the complete original plan body, clearly labeling it historical.
historical = root / "project_state/plans/workflow_governance_v3_20260726.md"
notice = "> **历史方案（2026-09-05 退役）**：下文保留原始设计供追溯。工作树/Gitee/固定预检流程不再生效；当前按 [独立实验包与手动回传](../governance/独立实验包与手动回传_20260905.md) 执行。创建工作树仅限用户明确指令。\n\n"
body = historical.read_text(encoding="utf-8-sig")
if not body.startswith(notice):
    historical.write_text(notice + body, encoding="utf-8")

checks = []
for name, schema in [("current_state.json", "current_state.schema.json"), ("document_registry.json", "document_registry.schema.json")]:
    jsonschema.validate(json.loads((root / "project_state" / name).read_text(encoding="utf-8")),
                        json.loads((root / "project_state/schemas" / schema).read_text(encoding="utf-8")))
    checks.append(name + ": schema PASS")
server = yaml.safe_load((root / "configs/server_paths.yaml").read_text(encoding="utf-8"))
package_config = json.loads((root / "experiments/_template/config.json").read_text(encoding="utf-8"))
assert server["paths"]["server_manual_runs"]["path"] == r"D:\AIPatho\qzs\runs" == package_config["runs_root"]
assert server["paths"]["server_manual_code"]["path"] == r"D:\AIPatho\qzs\code"
assert server["runtime"]["python_interpreter"] == package_config["python_interpreter"]
assert server["transport"]["current_channel"] == state["server_transport"]["current_channel"] == "manual_archive"
checks.append("Windows QZS paths, interpreter and manual transport: PASS")
events = [json.loads(line) for line in (root / "project_state/directives.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
assert sum(e.get("directive_id") == "DIR-20260905-002" for e in events) == 1
checks.append("Directive JSONL and unique new decision: PASS")

eligible = [
    "experiments/_template/run.ps1", "experiments/_template/runner.py",
    "experiments/new_example/src/train.py",
    "01_指南与解读/学习指南/Phase2_空间转录组核心评估指标体系与论文对比学习指南_20260904.md",
    "01_指南与解读/部署方案/服务器路径索引_20260701.md",
    "团队项目进度与结论/qzs/贡献.md", "02_组会汇报/2026/会议记录.md",
    "团队项目进度与结论/lzd/Phase2与Phase3工作计划原文_20260903.md",
    "docs/决策/记录.md", "archive/决策/记录.md", "Ai病理项目文献汇总/方法.md",
    "experiments/explorations/探索/记录.md",
]
ignored = [
    "experiments/new_example/runs/run1/raw/predictions.csv",
    "experiments/new_example/runs/run1/README.md",
    "experiments/new_example/analysis/run1/figure.png",
    "experiments/new_example/checkpoints/best.pth",
    "experiments/new_example/src/__pycache__/train.pyc",
    "experiments/new_example/data.pt", "experiments/new_example/package.zip",
    "01_指南与解读/学习指南/附件.pdf", "团队项目进度与结论/qzs/附件.xlsx",
    "data_new_3ST/train/image.tif", "deploy/secrets.sh", "work/manual_package_validation/test/run.json",
    "01_指南与解读/分析报告/cache/output.pt",
    "01_指南与解读/分析报告/run_output.zip",
]
for paths, expected in [(eligible, 1), (ignored, 0)]:
    for path in paths:
        result = subprocess.run(["git", "check-ignore", "--no-index", "-q", "--", path], cwd=root)
        assert result.returncode == expected, (path, result.returncode, expected)
checks.append(f"Git ignore behavior: PASS ({len(eligible)} eligible, {len(ignored)} excluded)")

for path in ["AGENTS.md", "README.md", ".agents/skills/pfmval-governance/SKILL.md", "CURRENT_STATE.md"]:
    content = (root / path).read_text(encoding="utf-8")
    assert "当前通道为 **gitee**" not in content
    assert "Gitee is the currently configured channel" not in content
checks.append("Current entrypoints no longer advertise default Gitee: PASS")
report = root / "work/manual_package_validation/config_checks.json"
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("\n".join(checks))
