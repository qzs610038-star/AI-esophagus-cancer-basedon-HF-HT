"""Read-only inspection of registered legacy UNI2-h CLS assets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from baseline_import import load_reference_entry, validate_reference_files
from config import load_config
from data import parse_coordinates
from errors import IdentityMismatchError
from run_io import utc_now, write_json


def extract_registered_historical_cls(value) -> torch.Tensor:
    """Legacy-only compatibility: a registered 2-D tensor contributes row 0."""

    if not isinstance(value, torch.Tensor):
        raise IdentityMismatchError("历史 UNI2-h 特征必须是直接 Tensor")
    tensor = value.detach().cpu().float()
    if tensor.ndim == 1 and tensor.shape == (1536,):
        cls = tensor
    elif tensor.ndim == 2 and tensor.shape[1] == 1536 and tensor.shape[0] >= 1:
        cls = tensor[0]
    else:
        raise IdentityMismatchError(f"历史 UNI2-h 特征形状非法: {tuple(tensor.shape)}")
    if not torch.isfinite(cls).all():
        raise IdentityMismatchError("历史 UNI2-h CLS 含 NaN/Inf")
    return cls


def describe_registered_historical_feature(value) -> dict:
    """Describe stored evidence without inferring the historical compute mode."""

    extract_registered_historical_cls(value)
    shape = list(value.shape)
    return {
        "shape": shape,
        "source_dtype": str(value.dtype).removeprefix("torch."),
        "mean_patch_available": shape == [265, 1536],
        "historical_compute_precision_fully_recorded": False,
    }


def _candidate_paths(config: dict, patient: str, spot: str) -> list[Path]:
    mpp_id = int(config["data"]["mpp_id"])
    filename = f"{spot}.pt"
    partner = Path(config["paths"]["uni2h_partner_cache_root"])
    flat = Path(config["paths"]["uni2h_flat_cache_root"])
    return [
        partner / f"MPP{mpp_id}_UNI" / patient / filename,
        partner / f"MPP{mpp_id}_UNI" / patient / "train" / filename,
        partner / f"MPP{mpp_id}_UNI" / patient / "val" / filename,
        partner / str(mpp_id) / patient / filename,
        flat / str(mpp_id) / patient / filename,
        flat / f"MPP{mpp_id}_UNI" / patient / filename,
    ]


def _identities_from_inputs(config: dict) -> list[tuple[str, str, str]]:
    identities: list[tuple[str, str, str]] = []
    with Path(config["inputs"]["split_manifest"]).open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            identities.append((str(row["patient"]), Path(str(row["patch_stem"])).stem, str(row["split"])))
    external = str(config["data"]["external_patient"])
    external_dir = Path(config["paths"]["image_root"]) / external / "patch_images"
    if not external_dir.is_dir():
        raise IdentityMismatchError(f"外部图像目录不存在，无法建立 XZY 身份清单: {external_dir}")
    for path in sorted(external_dir.glob("*.png"), key=lambda item: item.stem):
        parse_coordinates(path.stem)
        identities.append((external, path.stem, "external_test"))
    return identities


def inspect_uni2h(config: dict, *, seeds: Sequence[int], output_dir: str | Path | None = None) -> dict:
    identities = _identities_from_inputs(config)
    expected_total = sum(int(value) for value in config["data"]["expected_counts"].values())
    if len(identities) != expected_total:
        raise IdentityMismatchError(f"待核查身份应为 {expected_total}，实际={len(identities)}")
    shape_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    missing: list[dict] = []
    ambiguous: list[dict] = []
    resolved_rows: list[dict] = []
    full_token_count = 0
    for patient, spot, split in identities:
        existing = [path.resolve() for path in _candidate_paths(config, patient, spot) if path.is_file()]
        unique = list(dict.fromkeys(existing))
        if not unique:
            missing.append({"patient": patient, "spot": spot, "split": split})
            continue
        selected = unique[0]
        if len(unique) > 1:
            ambiguous.append({"patient": patient, "spot": spot, "candidates": [str(path) for path in unique]})
        try:
            value = torch.load(selected, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise IdentityMismatchError(f"无法安全读取历史特征 {selected}: {exc}") from exc
        description = describe_registered_historical_feature(value)
        shape = tuple(description["shape"])
        shape_counts[str(shape)] += 1
        dtype_counts[description["source_dtype"]] += 1
        if shape == (265, 1536):
            full_token_count += 1
        source_counts[str(selected.parents[1] if len(selected.parents) > 1 else selected.parent)] += 1
        resolved_rows.append(
            {
                "patient": patient,
                "spot": spot,
                "split": split,
                "selected_path": str(selected),
                "shape": str(shape),
                "source_dtype": description["source_dtype"],
                "mean_patch_available": description["mean_patch_available"],
            }
        )

    references = []
    for seed in seeds:
        entry = load_reference_entry(config["inputs"]["baseline_reference_manifest"], int(seed))
        references.append(
            validate_reference_files(
                entry,
                config["paths"]["historical_reference_root"],
                train_count=int(config["data"]["expected_counts"]["train"]),
                internal_count=int(config["data"]["expected_counts"]["internal_val"]),
                external_count=int(config["data"]["expected_counts"]["external_test"]),
            )
        )
    report = {
        "status": "ok" if not missing else "incomplete",
        "inspected_at": utc_now(),
        "read_only": True,
        "reextraction_started": False,
        "expected_identities": expected_total,
        "resolved_identities": len(resolved_rows),
        "missing_count": len(missing),
        "missing_examples": missing[:50],
        "ambiguous_count": len(ambiguous),
        "ambiguous_examples": ambiguous[:50],
        "shape_counts": dict(shape_counts),
        "source_dtype_counts": dict(dtype_counts),
        "historical_compute_precision_fully_recorded": False,
        "source_counts": dict(source_counts),
        "complete_265_token_count": full_token_count,
        "mean_patch_policy": "available_only_for_exact_[265,1536]; absence_never_triggers_reextraction",
        "references": [
            {
                "seed": item["seed"],
                "status": item["status"],
                "formal_epoch": item["formal_epoch"],
                "source_batch": item["source_batch"],
                "external_target_z_present": item["external_target_z_present"],
            }
            for item in references
        ],
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        with (output / "resolved_feature_paths.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = list(resolved_rows[0]) if resolved_rows else ["patient", "spot", "split", "selected_path", "shape", "mean_patch_available"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(resolved_rows)
        report["output_dir"] = str(output.resolve())
        report["resolved_feature_paths_csv"] = str((output / "resolved_feature_paths.csv").resolve())
        write_json(output / "inspection_uni2h.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="只读核查既有 UNI2-h 缓存与历史 point 参照")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", action="append", type=int, choices=(42, 43, 44))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = inspect_uni2h(load_config(args.config), seeds=args.seed or [42], output_dir=args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
