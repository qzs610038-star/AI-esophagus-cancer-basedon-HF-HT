"""Deterministic user-facing views derived from PFMval registries."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


PROGRESS_COLUMNS = (
    "可读名称",
    "experiment ID",
    "result ID",
    "W###",
    "目的/比较",
    "阶段",
    "状态",
    "证据等级",
    "关键结果",
    "当前结论",
    "下一步",
    "更新时间",
)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _cell(value: Any) -> str:
    if value in (None, ""):
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_target(value: str) -> str:
    return (
        value.replace("%", "%25")
        .replace(" ", "%20")
        .replace("(", "%28")
        .replace(")", "%29")
    )


def _key_metrics(experiment: Mapping[str, Any]) -> str:
    decision = experiment.get("decision_summary")
    if isinstance(decision, Mapping):
        control = decision.get("control_value")
        treatment = decision.get("treatment_value")
        delta = decision.get("delta")
        if all(isinstance(value, (int, float)) for value in (control, treatment, delta)):
            return (
                f"Control PCC={float(control):.4f}；"
                f"Treatment PCC={float(treatment):.4f}；"
                f"ΔPCC={float(delta):+.4f}"
            )
    labels = (
        ("best_val_pcc", "Val PCC"),
        ("best_val_loss", "Val loss"),
        ("external_xzy_pcc", "XZY PCC"),
        ("external_xzy_mae", "XZY MAE"),
        ("test_loss", "Test loss"),
    )
    values = []
    for field, label in labels:
        value = experiment.get(field)
        if value is None:
            continue
        if isinstance(value, float):
            rendered = f"{value:.4f}"
        else:
            rendered = str(value)
        values.append(f"{label}={rendered}")
    return "；".join(values) if values else "暂无"


def _conclusion(experiment: Mapping[str, Any]) -> str:
    decision = experiment.get("decision_summary")
    if isinstance(decision, Mapping) and decision.get("statement"):
        return _cell(decision["statement"])
    if experiment.get("conclusion"):
        return _cell(experiment["conclusion"])
    evidence = experiment.get("evidence_status", "pending")
    return {
        "accepted": "已接纳",
        "rejected": "已拒绝",
        "historical": "仅历史参考",
        "superseded": "已被替代",
        "pending": "待审查",
    }.get(str(evidence), "待审查")


def _progress_row(experiment: Mapping[str, Any]) -> str:
    purpose = (
        experiment.get("purpose")
        or experiment.get("comparison")
        or experiment.get("decision_gate")
        or "未登记"
    )
    values = (
        experiment.get("display_name") or experiment.get("id"),
        experiment.get("id"),
        (experiment.get("paired_result") or {}).get("pair_id")
        or experiment.get("result_id")
        or (experiment.get("last_preflight") or {}).get("result_id"),
        experiment.get("workspace_id"),
        purpose,
        experiment.get("phase") or experiment.get("result_phase"),
        experiment.get("status"),
        experiment.get("evidence_status") or "pending",
        _key_metrics(experiment),
        _conclusion(experiment),
        experiment.get("next_action") or "无",
        experiment.get("updated_at")
        or experiment.get("imported_at")
        or experiment.get("completed_at")
        or "—",
    )
    return "| " + " | ".join(_cell(value) for value in values) + " |"


def _table(experiments: Iterable[Mapping[str, Any]], empty_text: str) -> list[str]:
    rows = list(experiments)
    lines = [
        "| " + " | ".join(PROGRESS_COLUMNS) + " |",
        "|" + "|".join("---" for _ in PROGRESS_COLUMNS) + "|",
    ]
    if rows:
        lines.extend(_progress_row(item) for item in rows)
    else:
        lines.append(
            "| "
            + " | ".join([empty_text, *(["—"] * (len(PROGRESS_COLUMNS) - 1))])
            + " |"
        )
    return lines


def build_experiment_progress(registry: Mapping[str, Any]) -> str:
    """Build the concise user view from the Experiment Registry only."""
    experiments = list(registry.get("experiments", []))
    current = []
    accepted = []
    historical = []
    for experiment in experiments:
        evidence = str(experiment.get("evidence_status", "pending"))
        status = str(experiment.get("status", "planned"))
        lifecycle = str(experiment.get("lifecycle", "active"))
        if evidence == "accepted":
            accepted.append(experiment)
        elif (
            evidence in {"historical", "rejected", "superseded"}
            or lifecycle in {"historical", "superseded", "closed"}
            or status in {"failed", "closed", "tombstoned"}
        ):
            historical.append(experiment)
        else:
            current.append(experiment)

    sort_key = lambda item: (
        str(item.get("priority", "P9")),
        str(item.get("id", "")),
    )
    current.sort(key=sort_key)
    accepted.sort(key=sort_key)
    historical.sort(key=sort_key)
    source_hash = _canonical_hash(registry)
    lines = [
        "# 实验进度",
        "",
        "> 此文件由 `experiments/experiment_registry.json` 自动生成，禁止手工维护。",
        f"> source_sha256: `{source_hash}`",
        f"> state_revision: `{registry.get('state_revision', 'N/A')}`；updated_at: `{registry.get('updated_at', 'unknown')}`",
        "",
        "## 当前与待处理",
        "",
        *_table(current, "暂无当前或待处理实验"),
        "",
        "## 已接纳结果",
        "",
        *_table(accepted, "暂无已接纳结果"),
        "",
        "## 历史记录",
        "",
        *_table(historical, "暂无历史记录"),
        "",
    ]
    return "\n".join(lines)


def refresh_experiment_views(root: Path) -> dict[str, str]:
    """Atomically refresh the full and concise views from one registry snapshot."""
    from scripts.finalize_experiment import build_dashboard

    registry_path = root / "experiments" / "experiment_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    dashboard = build_dashboard(registry)
    progress = build_experiment_progress(registry)
    dashboard_path = root / "experiments" / "experiment_dashboard.md"
    progress_path = root / "experiments" / "experiment_progress.md"
    _write_text_atomic(dashboard_path, dashboard)
    _write_text_atomic(progress_path, progress)
    return {
        "dashboard_sha256": hashlib.sha256(
            dashboard.encode("utf-8")
        ).hexdigest(),
        "progress_sha256": hashlib.sha256(
            progress.encode("utf-8")
        ).hexdigest(),
    }


def build_project_guide(root: Path) -> str:
    """Build the tracked user navigation without promoting historical docs."""
    document_registry = json.loads(
        (root / "project_state" / "document_registry.json").read_text(
            encoding="utf-8"
        )
    )
    asset_registry_path = root / "project_state" / "asset_registry.json"
    asset_registry = (
        json.loads(asset_registry_path.read_text(encoding="utf-8"))
        if asset_registry_path.exists()
        else {"assets": []}
    )
    active_docs = sorted(
        (
            item
            for item in document_registry.get("documents", [])
            if item.get("lifecycle") == "active"
            and item.get("path") != "PROJECT_GUIDE.md"
            and (root / str(item.get("path", ""))).is_file()
        ),
        key=lambda item: str(item.get("path", "")),
    )
    historical_docs = sorted(
        (
            item
            for item in document_registry.get("documents", [])
            if item.get("lifecycle") in {"historical", "superseded"}
            and (root / str(item.get("path", ""))).is_file()
        ),
        key=lambda item: str(item.get("path", "")),
    )
    source_hash = _canonical_hash(
        {
            "document_state_revision": document_registry.get(
                "state_revision"
            ),
            "documents": [
                {
                    key: item.get(key)
                    for key in (
                        "doc_id",
                        "path",
                        "category",
                        "scope",
                        "authority",
                        "lifecycle",
                        "truth_sources",
                        "availability",
                    )
                }
                for item in document_registry.get("documents", [])
                if item.get("path") != "PROJECT_GUIDE.md"
            ],
            "assets": asset_registry.get("assets", []),
        }
    )
    lines = [
        "# PFMval 项目导航",
        "",
        "> 面向用户的生成入口；不承载实验事实，也不替代 Registry。",
        f"> source_sha256: `{source_hash}`",
        "",
        "## 我要找什么",
        "",
        "| 目标 | 首选入口 |",
        "|---|---|",
        "| 当前结论与阻塞 | [CURRENT_STATE.md](CURRENT_STATE.md) |",
        "| 简洁实验进度 | [experiments/experiment_progress.md](experiments/experiment_progress.md) |",
        "| 实验机器事实 | [experiments/experiment_registry.json](experiments/experiment_registry.json) |",
        "| 工作树身份与 lease | [project_state/workspace_registry.json](project_state/workspace_registry.json) |",
        "| 资产分类与保护 | [project_state/asset_registry.json](project_state/asset_registry.json) |",
        "| 完整运维视图 | [experiments/experiment_dashboard.md](experiments/experiment_dashboard.md) |",
        "| 服务器路径 | [configs/server_paths.yaml](configs/server_paths.yaml) |",
        "| Agent 专用规则 | [AGENTS.md](AGENTS.md) |",
        "",
        "## 目录地图",
        "",
        "| 目录 | 用途 | 读者 | 事实角色 | 修改方式 | lifecycle |",
        "|---|---|---|---|---|---|",
        "| `project_state/` | 机器治理状态与 schema | agent | source | CLI-only | active |",
        "| `experiments/` | 实验事实与两类生成视图 | shared | source/generated_view | CLI-only/generated | active |",
        "| `01_指南与解读/` | 方案、指南与学习材料 | user | reference | manual/approval-required | active-by-registry |",
        "| `02_组会汇报/` | 组会材料 | user | reference | manual | active-by-registry |",
        "| `.agents/` | Skill 与 Agent 规则 | agent | reference | approval-required | active |",
        "| `archive/` | 历史材料 | user | historical | approval-required | historical |",
        "",
        "## 当前 active 文档",
        "",
    ]
    if active_docs:
        lines.extend(
            f"- [{item.get('doc_id', item['path'])}]({_markdown_target(item['path'])})"
            for item in active_docs
        )
    else:
        lines.append("- 暂无已登记 active 文档。")
    lines.extend(["", "## 历史与过时资料", ""])
    if historical_docs:
        lines.extend(
            f"- [{item.get('doc_id', item['path'])}]({_markdown_target(item['path'])})"
            for item in historical_docs
        )
    else:
        lines.append("- 暂无已登记 historical/superseded 文档。")
    lines.extend(
        [
            "",
            "> historical 不等于 deleted；不得把本节资料作为当前执行依据。",
            "",
        ]
    )
    return "\n".join(lines)
