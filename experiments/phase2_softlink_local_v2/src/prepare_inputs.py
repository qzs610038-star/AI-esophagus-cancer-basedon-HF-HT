"""Generate auditable spatial grouping, geometry, and unlabeled XZY point inputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np

from data import find_feature_path, load_feature_vector, parse_xy


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve(package: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else package / path


def _source_root(feature_path: Path) -> Path:
    parent = feature_path.parent
    return parent.parent if parent.name.lower() in {"train", "val", "internal_val"} else parent


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    if not cleaned:
        raise ValueError(f"不能从 {value!r} 生成空间分组ID")
    return cleaned


def _positive_adjacent_differences(coords: list[tuple[float, float]], axis: int) -> list[float]:
    groups: dict[float, set[float]] = {}
    for x, y in coords:
        varying, fixed = (x, y) if axis == 0 else (y, x)
        groups.setdefault(float(fixed), set()).add(float(varying))
    out = []
    for values in groups.values():
        ordered = sorted(values)
        out.extend(right - left for left, right in zip(ordered, ordered[1:]) if right > left)
    return out


def infer_native_step(coords: list[tuple[float, float]]) -> float:
    """Infer the modal adjacent x/y grid step from a complete coordinate table."""
    dx = _positive_adjacent_differences(coords, 0)
    dy = _positive_adjacent_differences(coords, 1)
    if not dx or not dy:
        raise ValueError("坐标不足以同时推断x/y原生步长")

    def mode(values: list[float]) -> float:
        counts = Counter(round(value, 9) for value in values)
        highest = max(counts.values())
        return float(min(value for value, count in counts.items() if count == highest))

    step_x, step_y = mode(dx), mode(dy)
    if not np.isclose(step_x, step_y):
        raise ValueError(f"x/y主步长不一致: x={step_x}, y={step_y}")
    xs = np.asarray([item[0] for item in coords], dtype=np.float64)
    ys = np.asarray([item[1] for item in coords], dtype=np.float64)
    if not np.allclose(np.mod(xs - xs.min(), step_x), 0.0) or not np.allclose(np.mod(ys - ys.min(), step_x), 0.0):
        raise ValueError(f"点坐标未全部落在推断网格 step={step_x}")
    return step_x


def _external_feature_directory(feature_manifest: dict, mpp_id: int, patient: str) -> Path:
    candidates = []
    for root_key in ("flat_cache_root", "partner_cache_root"):
        value = feature_manifest.get(root_key)
        if not value:
            continue
        root = Path(value)
        candidates.extend([root / str(mpp_id) / patient, root / f"MPP{mpp_id}_UNI" / patient])
    present = [path for path in candidates if path.is_dir() and any(path.glob("*.pt"))]
    if not present:
        raise FileNotFoundError(f"没有找到 {patient} 的完整特征目录，候选={candidates}")
    reference = {path.stem for path in present[0].glob("*.pt")}
    for path in present[1:]:
        if {item.stem for item in path.glob("*.pt")} != reference:
            raise ValueError(f"{patient} 同时存在内容不同的特征目录，不能自动选择: {present}")
    return present[0]


def prepare_inputs(config: dict, *, package_dir: str | Path) -> dict:
    package = Path(package_dir).resolve()
    data = config["data"]
    mpp_id = int(data.get("mpp_id", 2))
    input_dim = int(data["input_dim"])
    external_patient = str(data.get("external_patient", "XZY"))
    split_path = _resolve(package, data["split_manifest_file"])
    feature_manifest_path = _resolve(package, data["feature_source_manifest"])
    feature_manifest = _read_json(feature_manifest_path)
    with split_path.open("r", encoding="utf-8-sig", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    expected_internal = int(data["expected_train_points"]) + int(data["expected_internal_val_points"])
    if len(split_rows) != expected_internal:
        raise ValueError(f"内部点数不符: {len(split_rows)} != {expected_internal}")

    partner = feature_manifest.get("partner_cache_root")
    flat = feature_manifest.get("flat_cache_root")
    by_patient: dict[str, list[dict]] = {}
    roots: dict[str, set[Path]] = {}
    for row in split_rows:
        patient, stem = str(row["patient"]), str(row["patch_stem"])
        feature_path = find_feature_path(stem, patient, mpp_id=mpp_id, partner_cache=partner, flat_cache=flat)
        if feature_path is None:
            raise FileNotFoundError(f"内部特征缺失: patient={patient}, spot={stem}")
        by_patient.setdefault(patient, []).append(row)
        roots.setdefault(patient, set()).add(_source_root(feature_path).resolve())
    for patient, source_roots in roots.items():
        if len(source_roots) != 1:
            raise ValueError(f"patient={patient} 命中多个特征来源组，不能自动视为一张空间切片: {sorted(map(str, source_roots))}")

    mapping_rows = []
    geometry_rows = []
    expected_steps = (config.get("preparation") or {}).get("expected_native_step") or {}
    for patient in sorted(by_patient):
        slide_id = f"MPP{mpp_id}_{_safe_name(patient)}_source01"
        source_root = next(iter(roots[patient]))
        coords = [(float(row["x"]), float(row["y"])) for row in by_patient[patient]]
        step = infer_native_step(coords)
        if patient in expected_steps and not np.isclose(step, float(expected_steps[patient])):
            raise ValueError(f"patient={patient} 推断步长 {step} 与v2.1固定值 {expected_steps[patient]} 不一致")
        mapping_rows.append({
            "patient_id": patient,
            "slide_id": slide_id,
            "status": "verified",
            "evidence_source": str(source_root),
            "notes": "单一完整特征来源组；slide_id是可审计空间分组ID，不冒充物理切片条码",
        })
        geometry_rows.append({
            "patient_id": patient,
            "slide_id": slide_id,
            "s": step,
            "coordinate_unit": "source_pixel_coordinate",
            "patch_coverage_size": "",
            "source": f"complete_split_manifest:{split_path}",
            "status": "verified_observed_grid",
        })

    external_dir = _external_feature_directory(feature_manifest, mpp_id, external_patient)
    external_files = sorted(external_dir.glob("*.pt"))
    expected_external = int(data.get("expected_external_points", 1039))
    if len(external_files) != expected_external:
        raise ValueError(f"外部点数不符: {len(external_files)} != {expected_external}")
    external_coords = []
    external_rows = []
    external_slide = f"MPP{mpp_id}_{_safe_name(external_patient)}_source01"
    for path in external_files:
        x, y = parse_xy(path.stem)
        if x is None or y is None:
            raise ValueError(f"外部特征文件名不能解析坐标: {path.name}")
        load_feature_vector(path, expected_dim=input_dim)
        external_coords.append((float(x), float(y)))
        external_rows.append({
            "patient_id": external_patient,
            "slide_id": external_slide,
            "spot_id": path.stem,
            "x": x,
            "y": y,
            "feature_path": str(path.resolve()),
        })
    external_step = infer_native_step(external_coords)
    mapping_rows.append({
        "patient_id": external_patient,
        "slide_id": external_slide,
        "status": "verified",
        "evidence_source": str(external_dir.resolve()),
        "notes": "完整无标签外部特征来源组；未读取XZY标签",
    })
    geometry_rows.append({
        "patient_id": external_patient,
        "slide_id": external_slide,
        "s": external_step,
        "coordinate_unit": "source_pixel_coordinate",
        "patch_coverage_size": "",
        "source": f"complete_external_feature_directory:{external_dir.resolve()}",
        "status": "verified_observed_grid",
    })

    mapping_path = _resolve(package, data["slide_mapping_file"])
    geometry_path = _resolve(package, data["slide_geometry_file"])
    external_path = package / "inputs" / "external_point_table.csv"
    _write_csv(mapping_path, ["patient_id", "slide_id", "status", "evidence_source", "notes"], mapping_rows)
    _write_csv(geometry_path, ["patient_id", "slide_id", "s", "coordinate_unit", "patch_coverage_size", "source", "status"], geometry_rows)
    _write_csv(external_path, ["patient_id", "slide_id", "spot_id", "x", "y", "feature_path"], external_rows)

    saved_config = deepcopy(config)
    saved_config["data"]["external_point_table"] = "inputs/external_point_table.csv"
    _write_json(package / "config.json", saved_config)
    status_path = _resolve(package, data["slide_mapping_status_file"])
    _write_json(status_path, {
        "status": "verified",
        "method": "one_complete_feature_source_group_per_patient_plus_complete_coordinate_grid",
        "physical_slide_accession_claimed": False,
        "n_internal_patients": len(by_patient),
        "external_patient": external_patient,
        "note_zh": "slide_id是按实际完整特征来源目录生成的空间分组ID；用于阻止跨来源连边，不声称恢复物理切片条码。",
    })
    input_status_path = _resolve(package, data["input_status_file"])
    input_status = _read_json(input_status_path) if input_status_path.is_file() else {"items": {}}
    input_status.setdefault("items", {})["slide_mapping"] = {"status": "generated_verified_source_groups", "path": str(mapping_path)}
    input_status["items"]["external_point_table"] = {
        "status": "generated_unlabeled",
        "path": str(external_path),
        "n_points": len(external_rows),
        "labels_read": False,
    }
    _write_json(input_status_path, input_status)
    report = {
        "status": "prepared",
        "internal_points": len(split_rows),
        "internal_patients": sorted(by_patient),
        "external_patient": external_patient,
        "external_points": len(external_rows),
        "mapping_file": str(mapping_path),
        "geometry_file": str(geometry_path),
        "external_point_table": str(external_path),
        "external_labels_read": False,
        "hashes_used": False,
    }
    _write_json(package / "inputs" / "preparation_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="生成Phase2 v2.1空间分组、步长和XZY无标签点表")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--package-dir", type=Path, default=None)
    args = parser.parse_args()
    package = (args.package_dir or Path(__file__).resolve().parent.parent).resolve()
    config_path = args.config or package / "config.json"
    report = prepare_inputs(_read_json(config_path), package_dir=package)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
