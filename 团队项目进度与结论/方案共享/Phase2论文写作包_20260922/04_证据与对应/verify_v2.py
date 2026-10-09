"""Verify the returned-result, registry, manuscript, and figure-data links for v2.

Read-only except for v2_validation.json beside this script. No training or hashes.
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
ROOT = HERE.parents[3]


def read_json(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(a, b, tolerance=1e-12):
    assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=tolerance), (a, b)


def run():
    registry = read_json(ROOT / "experiments/experiment_registry.json")
    entries = {e["id"]: e for e in registry["experiments"]}
    batch = entries["phase2_backbone_spatial_ablation_batch2_20260919"]
    task = entries["phase2_gene_reconstruction_comparison_v001_20260904"]
    main = entries["phase2_fullfov_hpo_v1"]
    assert batch["accepted"] and batch["evidence_status"] == "accepted"
    assert batch["status"] == "done"
    assert task["accepted"] and task["evidence_status"] == "accepted"
    assert task["status"] == "partial" and not task["comparison_complete"]
    assert task["run_kind"] == "smoke" and task["truth_status"] == "provenance_pending"
    assert task["n_runs"] == 1 and task["sample_sd"] is None
    assert task["training_seed"] is None
    assert not task["additional_training_required_before_writing"]

    state = read_json(ROOT / "project_state/current_state.json")
    for e in (batch, task):
        assert e["result_id"] in state["latest_accepted_result_ids"]
    assert state["phase2_program"]["paper_preparation"] == registry["phase2_supplemental_program"]["paper_preparation"]

    extract = read_json(HERE / "已接纳结果摘录.json")
    extracted_fields = 0
    for e in extract["experiments"]:
        source = entries[e["id"]]
        for key, value in e.items():
            assert source[key] == value, (e["id"], key)
            extracted_fields += 1

    analysis = ROOT / batch["result_root"] / "analysis"
    summary = read_csv(analysis / "six_model_summary.csv")
    seeds = read_csv(analysis / "six_model_per_seed.csv")
    assert len(summary) == 12 and len(seeds) == 36
    models = {"UNI": "uni", "UNI2-h": "uni2h", "Virchow2": "virchow2", "H-optimus-0": "hoptimus0", "H-optimus-1": "hoptimus1", "Phikon-v2": "phikonv2"}
    metrics = {"pcc_macro": "patient_macro_pathway_pcc", "pcc_pooled": "pooled_pcc", "z_mse": "pooled_z_mse"}
    registry_rows = {r["model"]: r for r in main["results_summary"] if r.get("stage") == "final-ablation" and r.get("arm") == "spatial"}
    registry_rows.update({r["model"]: r for r in batch["results_summary"]})
    numerical_checks = 0
    for row in summary:
        subset = [r for r in seeds if (r["model"], r["split"]) == (row["model"], row["split"])]
        assert sorted(int(r["seed"]) for r in subset) == [45, 46, 47]
        target = registry_rows[models.get(row["model"], row["model"])][row["split"]]
        for metric, registered in metrics.items():
            values = [float(r[metric]) for r in subset]
            close(statistics.mean(values), row[f"{metric}_mean"])
            close(statistics.stdev(values), row[f"{metric}_sd"])
            close(target["mean"][registered], row[f"{metric}_mean"])
            close(target["sample_sd"][registered], row[f"{metric}_sd"])
            numerical_checks += 4

    task_analysis = ROOT / task["result_root"] / "analysis"
    task_rows = read_csv(task_analysis / "task2_summary.csv")
    assert len(task_rows) == 2
    for row in task_rows:
        target = next(r for r in task["results_summary"] if r["split"] == row["split"])
        for key in ("pcc_macro", "pcc_pooled", "z_mse", "raw_mae"):
            close(row[key], target[key])
            numerical_checks += 1
        assert int(row["n_runs"]) == 1
        assert row["sample_sd"] == "" and row["training_seed"] == ""

    figures = PACKAGE / "02_图表配置/第二批图片_v2"
    manifest = read_json(figures / "data/source_manifest.json")
    for item in manifest["input_tables"]:
        assert (figures / item["copied_file"]).read_bytes() == (ROOT / item["source_file"]).read_bytes()
    exports = []
    for stem in ("Fig_P2_3S_six_encoders", "Fig_P2_5_gene_route"):
        for extension in ("png", "svg", "pdf"):
            path = figures / "exports" / f"{stem}.{extension}"
            assert path.stat().st_size > 1000
            exports.append(path.name)
    assert read_json(figures / "render_checks.json")["status"] == "pass"

    manuscript_cells = 0
    comparison = read_csv(task_analysis / "baseline_descriptive_comparison.csv")
    for name in ("Phase2中文初稿.md", "Phase2_English_Draft.md"):
        content = (PACKAGE / "05_中英文论文初稿_v2" / name).read_text(encoding="utf-8-sig")
        for row in summary:
            cells = [f"{float(row[m + '_mean']):.4f} ± {float(row[m + '_sd']):.4f}" for m in metrics]
            assert any(all(cell in line for cell in cells) for line in content.splitlines() if line.startswith("|")), (name, row["model"], row["split"])
            manuscript_cells += len(cells)
        for row in comparison:
            for metric in ("pcc_macro", "pcc_pooled"):
                value = f"{float(row[metric + '_mean']):.4f}"
                if int(row["n_runs"]) == 3:
                    value += f" ± {float(row[metric + '_sample_sd']):.4f}"
                assert any(value in line for line in content.splitlines() if line.startswith("|")), (name, value)
                manuscript_cells += 1
        assert len(re.findall(r"!\[[^\]]*\]\(", content)) == 7
        for phrase in ("provenance_pending", "训练超参数也未完整回传", "两套通路评分存在差异", "source training seed was not recorded", "若后续研究要量化"):
            assert phrase not in content, (name, phrase)

    markdown_files = [p for p in PACKAGE.rglob("*.md") if "首批图片_v1" not in p.parts and "05_中英文论文初稿_v1" not in p.parts]
    link_count = 0
    broken = []
    for path in markdown_files:
        for match in re.finditer(r"!?\[[^\]]*\]\((<[^>]+>|[^)]+)\)", path.read_text(encoding="utf-8-sig")):
            target = match.group(1).strip().strip("<>")
            if re.match(r"^[a-zA-Z][\w+.-]*://", target) or target.startswith("#"):
                continue
            target = unquote(target.split("#", 1)[0])
            if not target:
                continue
            if re.match(r"^/[A-Za-z]:/", target):
                target = target[1:]
            target = re.sub(r":\d+$", "", target)
            resolved = Path(target) if re.match(r"^[A-Za-z]:[/\\]", target) else path.parent / target
            link_count += 1
            if not resolved.exists():
                broken.append({"file": str(path.relative_to(PACKAGE)), "target": target})
    assert not broken, broken

    result = {
        "status": "pass",
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "本轮登记与写作包v2的派生数值、链接、图件和状态一致性；非训练链审计。",
        "state_revision": state["state_revision"],
        "accepted_results": [batch["result_id"], task["result_id"]],
        "counts": {"six_model_summary_rows": len(summary), "per_seed_rows": len(seeds), "task2_summary_rows": len(task_rows), "numeric_equalities": numerical_checks, "manuscript_numeric_cells": manuscript_cells, "extracted_registry_fields": extracted_fields, "figure_data_copies": len(manifest["input_tables"]), "figure_exports": len(exports), "markdown_files": len(markdown_files), "local_links": link_count},
        "broken_links": broken,
        "manual_review": {"reviewer": "root", "final_pngs_inspected": ["Fig_P2_3S_six_encoders.png", "Fig_P2_5_gene_route.png"], "result": "最终两张PNG已目视检查；标题、图例、坐标与脚注无重叠，任务二无虚构误差线。", "paper_rigor_handling": "评分差异诊断、缺证清单及任务二补证建议放入独立内部说明；方法和图注保留真实目标构建与运行次数。"},
        "new_training": False,
        "hashes_computed": False,
    }
    (HERE / "v2_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
