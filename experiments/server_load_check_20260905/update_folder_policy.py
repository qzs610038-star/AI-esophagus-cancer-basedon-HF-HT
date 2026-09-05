"""Record the user's folder-transfer preference and review the returned diagnostic."""
from datetime import datetime
import json
from pathlib import Path
import re
import yaml

here = Path(__file__).resolve().parent
root = here.parents[1]
stamp = datetime.now().astimezone().isoformat(timespec="seconds")
returned = here / "deliverables/20260905_234330_776_3c6878e3"
report = json.loads((returned / "report.json").read_text(encoding="utf-8-sig"))
run = json.loads((returned / "run.json").read_text(encoding="utf-8-sig"))
counts = {s: sum(c["status"] == s for c in report["checks"]) for s in ["PASS", "WARN", "FAIL"]}
assert report["complete"] and counts == report["counts"] == {"PASS": 47, "WARN": 3, "FAIL": 0}
assert run["exit_code"] == 0 and run["status"] == "succeeded"
checks = {c["id"]: c for c in report["checks"]}
assert len(checks) == 50


def change(path, replacements):
    file = root / path
    text = file.read_text(encoding="utf-8-sig")
    for old, new in replacements:
        assert old in text, (path, old)
        text = text.replace(old, new)
    file.write_text(text, encoding="utf-8")


change("AGENTS.md", [("- 当前通道是用户手动复制/压缩包；Gitee 暂停，SSH 暂不使用；不自动进行任何远端同步。",
    "- 当前通道是用户直接复制文件夹（manual_directory）；交付时给出需要复制的准确文件夹路径。默认不在本地或服务器生成压缩包，需要时由用户自行压缩，或在用户明确要求后再生成。Gitee 暂停，SSH 暂不使用；不自动进行任何远端同步。")])
change(".agents/skills/pfmval-governance/SKILL.md", [("Current transport is user-managed copying/archives;", "Current transport is direct user-managed folder copying (manual_directory). Give exact code/return folder paths; do not auto-create archives locally or on the server. The user compresses files if needed, unless explicitly requesting the agent to do so;")])
change("README.md", [("**手动复制/压缩包回传**", "**直接复制文件夹回传，不自动压缩**")])
change("experiments/_template/README.md", [("完成后将所需运行目录整体打包，放回本地", "完成后直接复制所需运行文件夹，放回本地"),
    ("本地登记按原始证据建立待登记记录。", "交付和回传均直接提供文件夹路径，不自动创建压缩包；需要时由用户自行压缩。\n\n本地登记按原始证据建立待登记记录。")])
change("project_state/governance/独立实验包与手动回传_20260905.md", [
    ("当前传输为用户手动复制或压缩包，", "当前传输为用户直接复制文件夹，不自动生成压缩包；需要时由用户自行压缩。"),
    ("用户将运行目录打包回传至", "用户将运行文件夹直接复制回传至")])

state_file = root / "project_state/current_state.json"
state = json.loads(state_file.read_text(encoding="utf-8"))
revision = state["state_revision"] + 1
directive_file = root / "project_state/directives.jsonl"
original_events = directive_file.read_text(encoding="utf-8")
events = [json.loads(line) for line in original_events.splitlines() if line.strip()]
prefix = "DIR-" + stamp[:10].replace("-", "") + "-"
number = max([int(e["directive_id"].split("-")[-1]) for e in events if e.get("directive_id", "").startswith(prefix)] or [0]) + 1
decision = prefix + f"{number:03d}"
state.update(state_revision=revision, updated_at=stamp)
state["active_directive_ids"].append(decision)
state["server_transport"].update(current_channel="manual_directory", active_channels=["manual_directory"],
                                 return_mode="user_owned_run_directory", auto_archive=False)
state["experiment_delivery"].update(transfer_preference_directive_id=decision, default_transfer="directory_copy", auto_archive=False)
relative_report = str((returned / "report.json").relative_to(root)).replace("\\", "/")
observation = {"source_report": relative_report, "run_id": run["run_id"], "server_run_ended_at": run["ended_at"],
               "reviewed_at": stamp, "counts": counts, "scope": "sample_loading_only_not_full_training_readiness",
               "python": checks["runtime"]["detail"]["python"], "torch": checks["cuda"]["detail"]["torch"],
               "cuda": checks["cuda"]["detail"]["compiled_cuda"], "gpu": checks["cuda"]["detail"]["devices"][0]["name"],
               "warning_ids": [c["id"] for c in report["checks"] if c["status"] == "WARN"]}
state["last_server_load_check"] = observation
state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

server_file = root / "configs/server_paths.yaml"
text = server_file.read_text(encoding="utf-8")
text = text.replace("  current_channel: manual_archive\n  active_channels: [manual_archive]", "  current_channel: manual_directory\n  active_channels: [manual_directory]\n  auto_archive: false", 1)
text = re.sub(r'^updated_at: .*$', 'updated_at: "' + stamp + '"', text, count=1, flags=re.M)
text += "\n# User-returned diagnostic observation; does not promote every recorded path to verified.\n" + yaml.safe_dump({"last_load_check": observation}, allow_unicode=True, sort_keys=False)
server_file.write_text(text, encoding="utf-8")

summary = "用户确认本地与服务器直接传递文件夹；后续给出准确代码/运行文件夹路径，不自动生成本地或服务器压缩包，必要时用户自行压缩。取消诊断包的自动回传ZIP，保留原始回传和旧文件。已读取用户回传的v001诊断：47 PASS、3 WARN、0 FAIL，仅作为服务器抽样读取观察，不作完整训练就绪或科研结果确认。"
affected = ["AGENTS.md", ".agents/skills/pfmval-governance/SKILL.md", "README.md", "CURRENT_STATE.md", "PROJECT_GUIDE.md",
            "experiments/_template/README.md", "configs/server_paths.yaml", "project_state/current_state.json",
            "project_state/governance/独立实验包与手动回传_20260905.md", "experiments/server_load_check_20260905"]
event = dict(event_type="directive", directive_id=decision, issued_at=stamp, summary=summary,
             scope="project_maintenance,server_transport", topic="direct_folder_transfer_without_automatic_archives",
             status="active", supersedes=[], amends=["DIR-20260905-002"], effective_from_revision=revision,
             affected_files=affected, source="explicit_user_instruction")
directive_file.write_text(original_events.rstrip() + "\n" + json.dumps(event, ensure_ascii=False) + "\n", encoding="utf-8")

current = root / "CURRENT_STATE.md"
text = current.read_text(encoding="utf-8")
text = re.sub(r"State revision: `\d+` \| Updated: `[^`]+`", f"State revision: `{revision}` | Updated: `{stamp}`", text, count=1)
text = text.replace("## Current directives\n", "## Current directives\n\n- `" + decision + "`：" + summary + "\n")
text = text.replace("`manual_archive`（用户手动复制/压缩包）", "`manual_directory`（直接复制文件夹，不自动压缩）")
current.write_text(text, encoding="utf-8")
readme = root / "README.md"
readme.write_text(re.sub(r"状态版本：`\d+`", f"状态版本：`{revision}`", readme.read_text(encoding="utf-8"), count=1), encoding="utf-8")
change("PROJECT_GUIDE.md", [("Windows QZS 服务器手动上传代码、回传 runs。", "Windows QZS 服务器直接复制代码和 runs 文件夹；默认不生成压缩包。")])

review = f"""# 首次服务器回传判读

来源：[原始报告](deliverables/{run['run_id']}/REPORT.md)，运行版本 v001，结束于 {run['ended_at']}。原始回传目录保持不变。

50 项中 47 PASS、3 WARN、0 FAIL，运行退出码 0。CUDA 小矩阵、UNI2-h 权重 CPU 读取、MPP2 七位患者各一份图像/标签/1536维缓存均通过。

实际环境：Windows 10，Python 3.13.5，PyTorch 2.6.0+cu124，CUDA 12.4，RTX 4080（约16GB）。这是 PFMval 服务器，不是本机环境。

三项警告：

- `h5py` 未安装。
- `anndata` 未安装。
- Phase3 原始分数目录未在四层搜索范围内找到 `.pt/.pth` 样本；这不证明目录没有其他格式的数据，也不能据此判定数据丢失。

本次仅判读，不安装依赖、不修改数据、不启动新服务器任务。HDF5/AnnData 若被后续实际实验使用，再处理对应依赖。权重未验证完整模型构造和推理；数据仅抽样，未验证全量覆盖和图像标签配对。

用户随后要求直接传递文件夹：v002 仅取消自动压缩，检查内容不变；无需为这项交付修改重跑服务器。后续复制 `deliverables/server_load_check_20260905/` 中的代码；回传终端提示的运行目录。
"""
(here / "SERVER_RETURN_REVIEW.md").write_text(review, encoding="utf-8")
print(json.dumps({"decision": decision, "revision": revision, "returned_counts": counts}, ensure_ascii=False))
