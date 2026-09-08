from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.manifest import collect_records
from export import write_prediction_table, write_target_spec
from metrics import compute_metrics
from models.protocol import Predictor
from targets.protocol import TargetSpec


def _arrays(model, loader, criterion, device):
    model.train(False)
    losses = []
    preds = []
    labels = []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            output = model(features)
            losses.append(float(criterion(output, targets).item()) * features.size(0))
            preds.append(output.cpu().numpy())
            labels.append(targets.cpu().numpy())
    pred = np.clip(np.nan_to_num(np.concatenate(preds), nan=0.0), -100.0, 100.0)
    truth = np.clip(np.nan_to_num(np.concatenate(labels), nan=0.0), -100.0, 100.0)
    loss = float(sum(losses) / max(len(loader.dataset), 1))
    return loss, compute_metrics(truth, pred), pred, truth


def train_and_export(
    *,
    predictor: Predictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    external_loader: DataLoader,
    train_ds,
    val_ds,
    external_ds,
    spec: TargetSpec,
    sampling_id: str,
    code_version: str,
    run_kind: str,
    num_epochs: int,
    patience: int,
    min_delta: float,
    lr: float,
    seed: int,
    run_dir: Path,
    extra_meta: dict,
) -> dict:
    device = next(predictor.model.parameters()).device
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(predictor.model.parameters(), lr=lr)
    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    idle = 0

    print(f"Training {run_kind}: {num_epochs} epochs, early stop on val_loss", flush=True)
    for epoch in range(1, num_epochs + 1):
        started = time.time()
        predictor.model.train(True)
        running = 0.0
        train_preds = []
        train_labels = []
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            preds = predictor.predict(features)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * features.size(0)
            train_preds.append(preds.detach().cpu())
            train_labels.append(targets.detach().cpu())
        train_loss = running / max(len(train_loader.dataset), 1)
        train_metrics = compute_metrics(
            torch.cat(train_labels).numpy(),
            np.clip(np.nan_to_num(torch.cat(train_preds).numpy(), nan=0.0), -100.0, 100.0),
        )
        val_loss, val_metrics, _, _ = _arrays(predictor.model, val_loader, criterion, device)
        improved = val_loss < best_loss - min_delta
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in predictor.model.state_dict().items()}
            idle = 0
        else:
            idle += 1
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_pcc": round(train_metrics["pcc"], 6),
            "val_loss": round(val_loss, 6),
            "val_pcc": round(val_metrics["pcc"], 6),
            "is_best": improved,
        })
        mark = " best" if improved else f" (no improv. {idle}/{patience})"
        print(
            f"Epoch {epoch:3d}/{num_epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_PCC={train_metrics['pcc']:.4f} val_PCC={val_metrics['pcc']:.4f} "
            f"time={time.time() - started:.1f}s{mark}",
            flush=True,
        )
        if idle >= patience and epoch >= max(patience, 3):
            print(f"Early stop at epoch {epoch}; best_epoch={best_epoch}", flush=True)
            break

    if best_state is not None:
        predictor.model.load_state_dict(best_state)
    checkpoint = {
        "epoch": best_epoch,
        "val_loss": best_loss,
        "model_id": predictor.model_id,
        "in_dim": predictor.in_dim,
        "out_dim": predictor.out_dim,
        "target_id": spec.target_id,
        "target_names": list(spec.names),
        "model_state_dict": {key: value.cpu() for key, value in predictor.model.state_dict().items()},
        "created_at": datetime.now().isoformat(),
        **extra_meta,
    }
    ckpt_path = run_dir / "checkpoints" / "best_checkpoint.pth"
    torch.save(checkpoint, ckpt_path)

    val_loss, val_metrics, val_pred, val_true = _arrays(predictor.model, val_loader, criterion, device)
    ext_loss, ext_metrics, ext_pred, ext_true = _arrays(predictor.model, external_loader, criterion, device)
    write_prediction_table(
        run_dir / "raw" / "predictions_internal_val.csv",
        collect_records(val_ds), val_true, val_pred, spec,
        predictor.model_id, sampling_id, code_version, "checkpoints/best_checkpoint.pth",
    )
    write_prediction_table(
        run_dir / "raw" / "predictions_external.csv",
        collect_records(external_ds), ext_true, ext_pred, spec,
        predictor.model_id, sampling_id, code_version, "checkpoints/best_checkpoint.pth",
    )
    write_target_spec(run_dir / "raw" / "target_spec.json", spec)
    pd_history = __import__("pandas").DataFrame(history)
    pd_history.to_csv(run_dir / "raw" / "training_history.csv", index=False)

    metrics = {
        "run_kind": run_kind,
        "demo_only": False,
        "selection": {
            "metric": "val_loss",
            "best_epoch": best_epoch,
            "best_value": best_loss,
            "split": "internal_val_from_split_manifest",
            "checkpoint": "checkpoints/best_checkpoint.pth",
        },
        "internal_val": {"loss": val_loss, **val_metrics, "n": int(len(val_loader.dataset))},
        "external_reported_after_selection": {"loss": ext_loss, **ext_metrics, "n": int(len(external_loader.dataset))},
        "target_id": spec.target_id,
        "sampling_id": sampling_id,
        "model_id": predictor.model_id,
        "seed": seed,
        "actual_epochs": history[-1]["epoch"] if history else 0,
        "note": "External numbers are reported after checkpoint selection and are not used for model choice.",
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {ckpt_path} (epoch {best_epoch}, val_loss={best_loss:.6f})", flush=True)
    return metrics
