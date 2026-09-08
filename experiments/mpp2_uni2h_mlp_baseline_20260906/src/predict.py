from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.manifest import collect_records
from export import write_prediction_table, write_target_spec
from metrics import compute_metrics
from models.protocol import Predictor
from targets.protocol import TargetSpec


def predict_and_export(
    *,
    predictor: Predictor,
    val_loader: DataLoader,
    external_loader: DataLoader,
    val_ds,
    external_ds,
    spec: TargetSpec,
    sampling_id: str,
    code_version: str,
    checkpoint_source: str,
    run_dir: Path,
    run_kind: str,
) -> dict:
    device = next(predictor.model.parameters()).device
    criterion = torch.nn.MSELoss()
    predictor.model.eval()

    def run(loader):
        preds = []
        labels = []
        total = 0.0
        with torch.no_grad():
            for features, targets in loader:
                features = features.to(device)
                targets = targets.to(device)
                output = predictor.predict(features)
                total += float(criterion(output, targets).item()) * features.size(0)
                preds.append(output.cpu().numpy())
                labels.append(targets.cpu().numpy())
        pred = __import__("numpy").concatenate(preds)
        truth = __import__("numpy").concatenate(labels)
        return total / max(len(loader.dataset), 1), compute_metrics(truth, pred), pred, truth

    val_loss, val_metrics, val_pred, val_true = run(val_loader)
    ext_loss, ext_metrics, ext_pred, ext_true = run(external_loader)
    write_prediction_table(
        run_dir / "raw" / "predictions_internal_val.csv",
        collect_records(val_ds), val_true, val_pred, spec,
        predictor.model_id, sampling_id, code_version, checkpoint_source,
    )
    write_prediction_table(
        run_dir / "raw" / "predictions_external.csv",
        collect_records(external_ds), ext_true, ext_pred, spec,
        predictor.model_id, sampling_id, code_version, checkpoint_source,
    )
    write_target_spec(run_dir / "raw" / "target_spec.json", spec)
    metrics = {
        "run_kind": run_kind,
        "demo_only": False,
        "selection": {
            "metric": "not_selected_in_predict_mode",
            "checkpoint": checkpoint_source,
        },
        "internal_val": {"loss": val_loss, **val_metrics, "n": int(len(val_loader.dataset))},
        "external_reported_after_selection": {"loss": ext_loss, **ext_metrics, "n": int(len(external_loader.dataset))},
        "target_id": spec.target_id,
        "sampling_id": sampling_id,
        "model_id": predictor.model_id,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Predict finished from {checkpoint_source}", flush=True)
    return metrics
