"""Compare online UNI2-h CLS outputs with the accepted frozen cache.

This is a post-resource-release preflight, not a training command.  It refuses
to run without an explicit acknowledgement that the MPP1/3/4/5 reruns are no
longer occupying server resources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd
import torch
import torch.nn.functional as functional
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_mpp_online import TRAIN_PATIENTS, validate_manifest_frame
from path_registry import get_registered_path
from scripts.pfmval_state import (
    active_mpp_repair,
    validate_state,
    verify_mpp_repair_server_assets,
)
from uni2h.uni2h_utils import load_uni2h_backbone


RESOURCE_RELEASE_ACK = "MPP1_3_4_5_IDLE"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MPP2 online-vs-cache CLS parity gate")
    parser.add_argument("--data_manifest_id", required=True)
    parser.add_argument("--manifest_labels_root", required=True)
    parser.add_argument("--resource_release_ack", required=True)
    parser.add_argument("--splits_root", default="mpp_standard_splits")
    parser.add_argument("--mpp_root", default="")
    parser.add_argument("--cache_root", default="")
    parser.add_argument("--flat_cache_root", default="")
    parser.add_argument("--samples_per_patient", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser


def _stable_sample(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if count < 1:
        raise ValueError("samples_per_patient must be positive")
    ranked = frame.copy()
    ranked["_order"] = ranked.apply(
        lambda row: hashlib.sha256(
            f"{row['patient']}|{row['patch_stem']}|{row['split']}".encode("utf-8")
        ).hexdigest(),
        axis=1,
    )
    return ranked.sort_values("_order").head(count).drop(columns="_order")


def resolve_unique_cache_path(
    cache_root: Path,
    flat_cache_root: Path,
    mpp_id: int,
    patient: str,
    stem: str,
) -> Path:
    candidates = [
        cache_root / f"MPP{mpp_id}_UNI" / patient / f"{stem}.pt",
        cache_root / f"MPP{mpp_id}_UNI" / patient / "train" / f"{stem}.pt",
        cache_root / f"MPP{mpp_id}_UNI" / patient / "val" / f"{stem}.pt",
        cache_root / str(mpp_id) / patient / f"{stem}.pt",
        flat_cache_root / str(mpp_id) / patient / f"{stem}.pt",
        flat_cache_root / f"MPP{mpp_id}_UNI" / patient / f"{stem}.pt",
    ]
    matches = list(dict.fromkeys(path.resolve() for path in candidates if path.is_file()))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one cache for {mpp_id}/{patient}/{stem}, found {matches}"
        )
    return matches[0]


def extract_online_cls_feature(
    backbone: torch.nn.Module,
    images: torch.Tensor,
    feature_dim: int,
) -> torch.Tensor:
    """Return UNI2-h CLS features using the same path as cache extraction.

    UNI2-h ``forward_features`` returns all patch/register tokens with shape
    ``[B, N, D]``.  The accepted cache stores only ``[:, 0, :]``.  Some
    backbones may already return ``[B, D]``; both forms are accepted, while
    other shapes fail before similarity is computed.
    """
    features = backbone.forward_features(images)
    if features.ndim == 3:
        features = features[:, 0, :]
    elif features.ndim != 2:
        raise ValueError(f"online UNI2-h output must be [B,D] or [B,N,D], got {tuple(features.shape)}")
    if features.shape[-1] != feature_dim:
        raise ValueError(
            f"online UNI2-h feature shape mismatch: expected last dim {feature_dim}, "
            f"got {tuple(features.shape)}"
        )
    return features


def _validate_state_and_repair(args: argparse.Namespace) -> dict:
    if args.resource_release_ack != RESOURCE_RELEASE_ACK:
        raise RuntimeError(
            f"resource release acknowledgement must equal {RESOURCE_RELEASE_ACK}"
        )
    report = validate_state(PROJECT_ROOT, strict=True, task="training", host_scope="server")
    if not report.ok:
        report.emit()
        raise RuntimeError("authoritative state blocks cache parity")
    repair = active_mpp_repair(PROJECT_ROOT)
    if repair is None or repair.get("data_manifest_id") != args.data_manifest_id:
        raise RuntimeError("cache parity must bind the active repaired data manifest")
    supplied = Path(args.manifest_labels_root).resolve()
    active = Path(str(repair["server_stage_path"])).resolve()
    if os.path.normcase(str(supplied)) != os.path.normcase(str(active)):
        raise RuntimeError("cache parity repaired label root does not match active staging")
    verify_mpp_repair_server_assets(PROJECT_ROOT, repair)
    return repair


def main() -> int:
    args = build_argparser().parse_args()
    repair = _validate_state_and_repair(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    registry_path = PROJECT_ROOT / "configs" / "server_paths.yaml"
    mpp_root = Path(args.mpp_root) if args.mpp_root else get_registered_path(
        "mpp_data_root", registry_path=registry_path, project_root=PROJECT_ROOT,
    )
    cache_root = Path(args.cache_root) if args.cache_root else get_registered_path(
        "server_mpp_partner_cache", registry_path=registry_path, project_root=PROJECT_ROOT,
    )
    flat_cache_root = Path(args.flat_cache_root) if args.flat_cache_root else get_registered_path(
        "server_mpp_flat_cache", registry_path=registry_path, project_root=PROJECT_ROOT,
    )

    manifest_path = Path(args.splits_root) / "group_2" / "split_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    validate_manifest_frame(manifest)
    selected = pd.concat([
        _stable_sample(manifest[manifest["patient"] == patient], args.samples_per_patient)
        for patient in TRAIN_PATIENTS
    ], ignore_index=True)

    backbone, transform, feature_dim = load_uni2h_backbone(device=device)
    backbone.eval()
    rows: List[dict] = []
    with torch.inference_mode():
        for record in selected.itertuples(index=False):
            image_path = (
                mpp_root / "2" / str(record.patient) / "patch_images"
                / f"{record.patch_stem}.png"
            )
            if not image_path.is_file():
                raise FileNotFoundError(f"manifest PNG missing: {image_path}")
            cache_path = resolve_unique_cache_path(
                cache_root, flat_cache_root, 2, str(record.patient), str(record.patch_stem),
            )
            cached = torch.load(cache_path, map_location="cpu")
            if cached.ndim == 2:
                cached = cached[0]
            cached_dtype = str(cached.dtype)
            cached = cached.float()
            if cached.shape != (feature_dim,):
                raise ValueError(f"cache feature shape mismatch: {cache_path} {cached.shape}")

            with Image.open(image_path) as image:
                online = extract_online_cls_feature(
                    backbone,
                    transform(image.convert("RGB")).unsqueeze(0).to(device),
                    feature_dim,
                ).squeeze(0).detach().cpu().float()
            difference = (online - cached).abs()
            cosine = float(functional.cosine_similarity(online, cached, dim=0))
            threshold = 0.99999 if cached_dtype == "torch.float32" else 0.999
            rows.append({
                "mpp_id": 2,
                "patient": record.patient,
                "patch_stem": record.patch_stem,
                "split": record.split,
                "image_path": str(image_path),
                "cache_path": str(cache_path),
                "cache_dtype": cached_dtype,
                "cosine_similarity": cosine,
                "mean_absolute_error": float(difference.mean()),
                "max_absolute_error": float(difference.max()),
                "threshold": threshold,
                "passed": cosine >= threshold,
            })

    output = Path(args.output)
    summary_path = output.with_suffix(".json")
    if output.exists() or summary_path.exists():
        raise FileExistsError(
            f"refusing to overwrite parity output: {output} / {summary_path}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    summary = {
        "data_manifest_id": args.data_manifest_id,
        "repair_evidence_id": repair.get("evidence_id"),
        "sample_count": len(frame),
        "all_passed": bool(frame["passed"].all()),
        "min_cosine_similarity": float(frame["cosine_similarity"].min()),
        "max_absolute_error": float(frame["max_absolute_error"].max()),
        "result_csv": str(output),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    if not summary["all_passed"]:
        raise RuntimeError("online-vs-cache parity gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
