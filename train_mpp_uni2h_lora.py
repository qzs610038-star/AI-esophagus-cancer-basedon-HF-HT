"""Guarded online MPP2 UNI2-h LoRA trainer.

The initial route is intentionally narrow: repaired MPP2 manifest, fixed
external MPP2/XZY, the accepted ``MPPMLPHead``, and either paired frozen-head
continuation or qkv+proj LoRA rank 8.  External XZY is evaluated only after the
best internal-validation checkpoint has been selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_mpp_online import build_online_manifest_datasets
from histogene.utils import compute_metrics
from lora_utils import freeze_all_parameters
from model_mpp_uni2h_lora import (
    OnlineMPPModel,
    build_paired_optimizer,
    configure_p0_lora,
    load_accepted_mpp_head,
    trainable_parameter_names,
)
from path_registry import get_registered_path
from scripts.pfmval_state import (
    active_mpp_repair,
    validate_state,
    verify_mpp_repair_server_assets,
)
from train_mpp_uni2h_mlp import resolve_manifest_data_roots
from uni2h.uni2h_utils import load_uni2h_backbone


os.environ.setdefault("HF_HUB_OFFLINE", "1")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Guarded online MPP2 UNI2-h frozen/LoRA paired trainer"
    )
    parser.add_argument("--mode", choices=["frozen", "lora"], required=True)
    parser.add_argument("--train_mpp_id", type=int, default=2)
    parser.add_argument("--external_mpp_id", type=int, default=2)
    parser.add_argument("--external_patient", default="XZY")
    parser.add_argument("--mpp_root", default="")
    parser.add_argument("--splits_root", default="mpp_standard_splits")
    parser.add_argument("--manifest_labels_root", required=True)
    parser.add_argument("--data_manifest_id", required=True)
    parser.add_argument("--head_checkpoint", required=True)
    parser.add_argument("--head_checkpoint_sha256", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--output_root", default="checkpoints/mpp_uni2h_lora")

    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--lora_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad_checkpointing", action="store_true", default=False)

    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.3)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_args(args: argparse.Namespace) -> None:
    if args.train_mpp_id != 2 or args.external_mpp_id != 2:
        raise ValueError("approved Phase 0 route fixes train/external MPP to 2")
    if args.external_patient != "XZY":
        raise ValueError("approved Phase 0 route fixes external_patient=XZY")
    if args.num_epochs < 1:
        raise ValueError("num_epochs must be positive")
    if args.num_epochs > 3:
        raise ValueError("current approved smoke budget is capped at 3 epochs")
    if args.seed != 42:
        raise ValueError("Phase 0/P0 is pre-registered for seed=42 only")
    if args.grad_accum_steps < 1 or args.batch_size < 1:
        raise ValueError("batch_size and grad_accum_steps must be positive")
    if args.mode == "frozen" and (args.lora_rank != 8 or args.lora_alpha != 16.0):
        raise ValueError("paired frozen control must retain the registered LoRA metadata")


def _validate_authoritative_state(args: argparse.Namespace) -> Dict:
    report = validate_state(
        _PROJECT_ROOT,
        strict=True,
        task="training",
        host_scope="server",
    )
    if not report.ok:
        report.emit()
        raise RuntimeError("authoritative project state blocks MPP2 LoRA")
    repair = active_mpp_repair(_PROJECT_ROOT)
    if repair is None:
        raise RuntimeError("active repaired-label manifest is required")
    if args.data_manifest_id != repair.get("data_manifest_id"):
        raise RuntimeError(
            f"data manifest mismatch: supplied={args.data_manifest_id} "
            f"active={repair.get('data_manifest_id')}"
        )
    supplied = Path(args.manifest_labels_root).resolve()
    active = Path(str(repair["server_stage_path"])).resolve()
    if os.path.normcase(str(supplied)) != os.path.normcase(str(active)):
        raise RuntimeError(
            f"manifest label root mismatch: supplied={supplied} active={active}"
        )
    verify_mpp_repair_server_assets(_PROJECT_ROOT, repair)
    return repair


def _set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _collect_metadata(dataset) -> List[dict]:
    if isinstance(dataset, ConcatDataset):
        rows: List[dict] = []
        for child in dataset.datasets:
            rows.extend(_collect_metadata(child))
        return rows
    if not hasattr(dataset, "metadata_rows"):
        raise TypeError(f"dataset lacks metadata_rows: {type(dataset).__name__}")
    return dataset.metadata_rows()


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    safe_predictions = np.clip(
        np.nan_to_num(predictions, nan=0.0, posinf=10.0, neginf=-10.0),
        -100.0, 100.0,
    )
    safe_labels = np.clip(
        np.nan_to_num(labels, nan=0.0, posinf=10.0, neginf=-10.0),
        -100.0, 100.0,
    )
    with np.errstate(over="warn", invalid="warn"):
        return compute_metrics(safe_labels, safe_predictions)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler,
    amp_enabled: bool,
    grad_accum_steps: int,
    gradient_clip: float,
) -> Tuple[float, dict]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    predictions: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, targets)
            scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()
        total_loss += loss.item() * images.size(0)
        predictions.append(outputs.detach().cpu())
        labels.append(targets.detach().cpu())

        should_step = (step + 1) % grad_accum_steps == 0 or step + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    predictions_np = torch.cat(predictions).numpy()
    labels_np = torch.cat(labels).numpy()
    return total_loss / len(loader.dataset), _metrics(labels_np, predictions_np)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[float, dict, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    predictions: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, targets)
        total_loss += loss.item() * images.size(0)
        predictions.append(outputs.cpu())
        labels.append(targets.cpu())
    predictions_np = torch.cat(predictions).numpy()
    labels_np = torch.cat(labels).numpy()
    return (
        total_loss / len(loader.dataset),
        _metrics(labels_np, predictions_np),
        predictions_np,
        labels_np,
    )


def _write_predictions(
    path: Path,
    metadata: List[dict],
    target_cols: List[str],
    predictions: np.ndarray,
    labels: np.ndarray,
) -> None:
    if len(metadata) != len(predictions):
        raise ValueError("prediction metadata length mismatch")
    frame = pd.DataFrame(metadata)
    for index, target in enumerate(target_cols):
        frame[f"true_{target}"] = labels[:, index]
        frame[f"pred_{target}"] = predictions[:, index]
    frame.to_csv(path, index=False)


def _raw_scale_metrics(
    target_cols: List[str],
    predictions: np.ndarray,
    labels: np.ndarray,
    params_path: Path,
) -> dict:
    params = json.loads(params_path.read_text(encoding="utf-8"))
    pathway_params = params.get("pathways", {})
    rows = []
    for index, target in enumerate(target_cols):
        entry = pathway_params.get(target)
        if not entry:
            raise ValueError(f"z-score params missing pathway: {target}")
        std = float(entry["std"])
        pred_raw = predictions[:, index] * std + float(entry["mean"])
        label_raw = labels[:, index] * std + float(entry["mean"])
        mae = float(np.mean(np.abs(label_raw - pred_raw)))
        if np.std(label_raw) > 0 and np.std(pred_raw) > 0:
            pcc = float(np.corrcoef(label_raw, pred_raw)[0, 1])
        else:
            pcc = float("nan")
        denominator = float(np.sum((label_raw - label_raw.mean()) ** 2))
        r2 = (
            float(1.0 - np.sum((label_raw - pred_raw) ** 2) / denominator)
            if denominator > 0 else float("nan")
        )
        rows.append({"pathway": target, "pcc": pcc, "mae_raw": mae, "r2_raw": r2})
    return {
        "per_pathway": rows,
        "external_xzy_mae_raw": float(np.mean([row["mae_raw"] for row in rows])),
        "external_xzy_r2_raw": float(np.mean([row["r2_raw"] for row in rows])),
    }


def main() -> int:
    args = build_argparser().parse_args()
    _validate_args(args)
    repair = _validate_authoritative_state(args)

    torch.set_num_threads(args.num_threads)
    generator = _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    mpp_root = Path(args.mpp_root) if args.mpp_root else get_registered_path("mpp_data_root")

    head_checkpoint = Path(args.head_checkpoint)
    if not head_checkpoint.is_file():
        raise FileNotFoundError(f"accepted head checkpoint missing: {head_checkpoint}")
    actual_checkpoint_sha = _sha256_file(head_checkpoint)
    if actual_checkpoint_sha.lower() != args.head_checkpoint_sha256.lower():
        raise ValueError(
            f"head checkpoint SHA-256 mismatch: expected={args.head_checkpoint_sha256} "
            f"actual={actual_checkpoint_sha}"
        )

    output_dir = Path(args.output_root) / args.dataset_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    roots = resolve_manifest_data_roots(
        args.splits_root, args.manifest_labels_root, args.train_mpp_id,
    )
    manifest = pd.read_csv(roots["split_manifest"])
    backbone, transform, feature_dim = load_uni2h_backbone(device=device)
    if args.grad_checkpointing:
        if not hasattr(backbone, "set_grad_checkpointing"):
            raise RuntimeError("backbone does not support set_grad_checkpointing")
        backbone.set_grad_checkpointing(True)

    if args.mode == "lora":
        lora_parameter_count = configure_p0_lora(
            backbone,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
    else:
        freeze_all_parameters(backbone)
        lora_parameter_count = 0

    train_dataset, val_dataset, external_dataset, target_cols = (
        build_online_manifest_datasets(
            manifest_df=manifest,
            mpp_root=mpp_root,
            labels_root=roots["labels"],
            transform=transform,
            train_mpp_id=args.train_mpp_id,
            external_mpp_id=args.external_mpp_id,
        )
    )
    if len(target_cols) != 30:
        raise ValueError(f"expected 30 pathways, got {len(target_cols)}")

    model = OnlineMPPModel(
        backbone=backbone,
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        output_dim=len(target_cols),
        dropout=args.dropout,
    ).to(device)
    checkpoint_meta = load_accepted_mpp_head(model, head_checkpoint)
    optimizer = build_paired_optimizer(
        model,
        mode=args.mode,
        head_lr=args.head_lr,
        lora_lr=args.lora_lr,
        weight_decay=args.weight_decay,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    external_loader = DataLoader(external_dataset, shuffle=False, **loader_kwargs)

    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    trainable_names = set(trainable_parameter_names(model))
    start_time = time.time()
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            amp_enabled, args.grad_accum_steps, args.gradient_clip,
        )
        val_loss, val_metrics, _, _ = evaluate(
            model, val_loader, criterion, device, amp_enabled,
        )
        is_best = val_loss < best_val_loss - args.min_delta
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
                if key in trainable_names
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_pcc": train_metrics["pcc"],
            "val_loss": val_loss,
            "val_pcc": val_metrics["pcc"],
            "val_mae": val_metrics["mae"],
            "val_r2": val_metrics["r2"],
            "is_best": is_best,
        })
        if epochs_without_improvement >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("training produced no finite internal validation checkpoint")
    model.load_state_dict(best_state, strict=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    torch.save({
        "epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": best_state,
        "mode": args.mode,
        "data_manifest_id": args.data_manifest_id,
        "state_format": "trainable_only",
        "base_model": "MahmoodLab/UNI2-h",
        "base_model_weights_frozen": True,
    }, output_dir / "best_checkpoint.pth")

    val_loss, val_metrics, val_predictions, val_labels = evaluate(
        model, val_loader, criterion, device, amp_enabled,
    )
    _write_predictions(
        output_dir / "predictions_internal_val.csv",
        _collect_metadata(val_dataset), target_cols, val_predictions, val_labels,
    )

    # External XZY is touched only after internal checkpoint selection is final.
    external_loss, external_metrics, external_predictions, external_labels = evaluate(
        model, external_loader, criterion, device, amp_enabled,
    )
    _write_predictions(
        output_dir / "predictions_external_xzy.csv",
        _collect_metadata(external_dataset), target_cols,
        external_predictions, external_labels,
    )
    raw_metrics = _raw_scale_metrics(
        target_cols, external_predictions, external_labels, roots["zscore_params"],
    )
    pd.DataFrame(raw_metrics["per_pathway"]).to_csv(
        output_dir / "per_pathway_pcc.csv", index=False,
    )

    summary = {
        "mode": args.mode,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_pcc": val_metrics["pcc"],
        "external_xzy_pcc": external_metrics["pcc"],
        "external_xzy_mae": external_metrics["mae"],
        "external_xzy_r2": external_metrics["r2"],
        "test_loss": external_loss,
        "internal_val": {"loss": val_loss, **val_metrics},
        "external_xzy": {"loss": external_loss, **external_metrics},
        "external_xzy_mae_raw": raw_metrics["external_xzy_mae_raw"],
        "external_xzy_r2_raw": raw_metrics["external_xzy_r2_raw"],
        "train_samples": len(train_dataset),
        "internal_val_samples": len(val_dataset),
        "external_samples": len(external_dataset),
        "target_count": len(target_cols),
        "lora_parameter_count": lora_parameter_count,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "trainable_parameter_names": trainable_parameter_names(model),
        "head_checkpoint_sha256": actual_checkpoint_sha,
        "head_checkpoint_epoch": checkpoint_meta.get("epoch"),
        "data_manifest_id": args.data_manifest_id,
        "repair_evidence_id": repair.get("evidence_id"),
        "elapsed_seconds": time.time() - start_time,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        ),
        "gradient_checkpointing": bool(args.grad_checkpointing),
        "external_evaluated_after_checkpoint_selection": True,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (output_dir / "best_epoch.txt").write_text(
        f"best_epoch={best_epoch}\nbest_val_loss={best_val_loss:.8f}\n",
        encoding="utf-8",
    )
    patient_counts = pd.Series(
        [row["patient"] for row in _collect_metadata(train_dataset)]
    ).value_counts().sort_index().to_dict()
    (output_dir / "training_summary.txt").write_text(
        "\n".join([
            f"Experiment: {args.dataset_name}",
            f"Mode: {args.mode}",
            f"Data manifest: {args.data_manifest_id}",
            f"Train samples: {len(train_dataset)} {patient_counts}",
            f"Internal val samples: {len(val_dataset)}",
            f"External XZY samples: {len(external_dataset)}",
            f"Best epoch: {best_epoch}",
            f"Best val loss: {best_val_loss:.8f}",
            f"External XZY PCC: {external_metrics['pcc']:.6f}",
            f"External XZY MAE raw: {raw_metrics['external_xzy_mae_raw']:.6f}",
            f"External XZY R2 raw: {raw_metrics['external_xzy_r2_raw']:.6f}",
            "External XZY evaluated only after internal checkpoint selection.",
        ]) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
