"""Cross-patient 3-fold CV training for UNI2-h + DenseNet121 MLP."""
import argparse
import os
import sys
import signal
import copy
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

signal.signal(signal.SIGINT, signal.SIG_IGN)

from config_utils import load_config, get_device, get_fold_config
from uni2h_des import (
    CachedFeaturePatchDataset,
    DenseNet121StyleRegressor,
    calculate_max_abs_diff_per_target,
    ensure_zscore_csv,
    evaluate,
    extract_and_cache_features,
    load_uni2h_backbone,
    train_one_epoch,
)
from notify_utils import notify_training_complete, notify_training_error


# ── Cross-validation fold config ──
FOLDS = {
    1: {"train": ["JFX0729", "LMZ12939"], "test": "HYZ15040"},
    2: {"train": ["HYZ15040", "LMZ12939"], "test": "JFX0729"},
    3: {"train": ["HYZ15040", "JFX0729"], "test": "LMZ12939"},
}

# Data root paths
PATCHES_BASE = _PROJECT_ROOT / "data_new_3ST" / "patch_noov_spilt"
LABELS_BASE = _PROJECT_ROOT / "data_new_3ST" / "ssGSEA_zscore"
CACHE_BASE = _SCRIPT_DIR / "uni2h_cache_30"


def get_patient_data(patient: str):
    """Return (patches_dir_train, patches_dir_val, labels_csv_raw, labels_csv_zscore, cache_dir) for a patient."""
    patches_base = PATCHES_BASE / f"{patient}_noov_split"
    train_dir = str(patches_base / "train_patches")
    val_dir = str(patches_base / "val_patches")
    labels_raw = str(LABELS_BASE / f"{patient}_ssGSEA_zscore.csv")
    labels_zscore = str(LABELS_BASE / f"{patient}_ssGSEA_zscore.csv")  # already z-scored
    cache_dir = str(CACHE_BASE / patient)
    return train_dir, val_dir, labels_raw, labels_zscore, cache_dir


def build_argparser():
    p = argparse.ArgumentParser(description="Cross-patient 3-fold CV for UNI2-h DenseNet121 MLP")
    p.add_argument("--fold", type=int, default=1, choices=[1, 2, 3],
                   help="Which CV fold (1-3)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--initial_dim", type=int, default=256)
    p.add_argument("--growth_rate", type=int, default=32)
    p.add_argument("--bottleneck_factor", type=int, default=4)
    p.add_argument("--transition_factor", type=float, default=0.5)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--num_targets", type=int, default=30)
    p.add_argument("--target_start_col", type=int, default=1)
    p.add_argument("--hf_local_only", action="store_true", default=True)
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    return p


def main():
    args = build_argparser().parse_args()
    fold_cfg = FOLDS[args.fold]
    train_patients = fold_cfg["train"]
    test_patient = fold_cfg["test"]

    print("=" * 60)
    print(f"UNI2-h Cross-Patient 3-Fold CV — Fold {args.fold}")
    print(f"  Train: {' + '.join(train_patients)}")
    print(f"  Test:  {test_patient}")
    print("=" * 60)

    # ── Load backbone ──
    device = get_device(load_config())
    print(f"Device: {device}")
    backbone, transform, feature_dim = load_uni2h_backbone(
        device=device, local_only=args.hf_local_only
    )
    print(f"UNI2-h loaded. Feature dim={feature_dim}")

    # ── Extract features for all 3 patients ──
    all_patients = train_patients + [test_patient]
    for patient in all_patients:
        train_dir, val_dir, labels_raw, labels_zscore, cache_dir = get_patient_data(patient)
        train_cache = os.path.join(cache_dir, "train")
        val_cache = os.path.join(cache_dir, "val")
        os.makedirs(train_cache, exist_ok=True)
        os.makedirs(val_cache, exist_ok=True)

        n_tr = extract_and_cache_features(
            backbone, transform, train_dir, train_cache, device, rebuild=args.rebuild_cache
        )
        n_vl = extract_and_cache_features(
            backbone, transform, val_dir, val_cache, device, rebuild=args.rebuild_cache
        )
        if n_tr + n_vl > 0:
            print(f"  [{patient}] Extracted {n_tr} train + {n_vl} val features")

    # ── Build datasets ──
    # Train: ConcatDataset of train splits from train_patients
    train_datasets = []
    for patient in train_patients:
        train_dir, val_dir, labels_raw, labels_zscore, cache_dir = get_patient_data(patient)
        labels_csv = ensure_zscore_csv(
            labels_raw, labels_zscore,
            target_start_col=args.target_start_col,
            num_targets=args.num_targets,
        )
        ds = CachedFeaturePatchDataset(
            patches_dir=train_dir,
            labels_csv=labels_csv,
            feature_cache_dir=os.path.join(cache_dir, "train"),
            target_start_col=args.target_start_col,
            num_targets=args.num_targets,
        )
        train_datasets.append(ds)
        print(f"  [{patient}] Train dataset: {len(ds)} samples")

    train_dataset = ConcatDataset(train_datasets)
    print(f"  Combined train: {len(train_dataset)} samples")

    # Test: val_patches from test_patient
    test_train_dir, test_val_dir, test_labels_raw, test_labels_zscore, test_cache_dir = get_patient_data(test_patient)
    test_labels_csv = ensure_zscore_csv(
        test_labels_raw, test_labels_zscore,
        target_start_col=args.target_start_col,
        num_targets=args.num_targets,
    )
    test_dataset = CachedFeaturePatchDataset(
        patches_dir=test_val_dir,
        labels_csv=test_labels_csv,
        feature_cache_dir=os.path.join(test_cache_dir, "val"),
        target_start_col=args.target_start_col,
        num_targets=args.num_targets,
    )
    print(f"  [{test_patient}] Test dataset: {len(test_dataset)} samples")

    # ── DataLoaders ──
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    # ── Model ──
    model = DenseNet121StyleRegressor(
        feature_dim=feature_dim,
        initial_dim=args.initial_dim,
        growth_rate=args.growth_rate,
        bottleneck_factor=args.bottleneck_factor,
        transition_factor=args.transition_factor,
        output_dim=args.num_targets,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # ── Checkpoint paths ──
    dataset_name = f"CrossPatient_Fold{args.fold}_{test_patient}_UNI2h_DenseNet121"
    ckpt_dir = _SCRIPT_DIR / "checkpoints" / dataset_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / "best_model_uni2h.pth"
    resume_ckpt_path = ckpt_dir / "resume_uni2h.pth"
    history_path = ckpt_dir / "best_model_uni2h.history.csv"

    # ── Training state ──
    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = 0
    best_pcc = 0.0
    best_state = None
    history = []
    patience_counter = 0

    if args.resume:
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        patience_counter = ckpt.get("patience_counter", 0)
        best_epoch = ckpt.get("best_epoch", 0)
        best_pcc = ckpt.get("best_pcc", 0.0)
        if "history" in ckpt:
            history = ckpt["history"]
        print(f"[INFO] Resumed from epoch {start_epoch + 1}")

    early_stopped = False
    current_epoch = 0

    try:
        for epoch in range(start_epoch, args.num_epochs):
            current_epoch = epoch + 1
            print(f"\nEpoch {current_epoch}/{args.num_epochs}")

            train_overall, train_full = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_overall, val_full = evaluate(model, test_loader, criterion, device)

            scheduler.step(val_overall["loss"])

            print(
                f"Train | loss={train_overall['loss']:.6f} mae={train_overall['mae']:.6f} "
                f"r2={train_overall['r2']:.6f} pcc={train_overall['pcc']:.6f}"
            )
            print(
                f"Test  | loss={val_overall['loss']:.6f} mae={val_overall['mae']:.6f} "
                f"r2={val_overall['r2']:.6f} pcc={val_overall['pcc']:.6f}"
            )

            history.append({
                "epoch": current_epoch,
                "train_loss": train_overall["loss"],
                "train_mae": train_overall["mae"],
                "train_r2": train_overall["r2"],
                "train_pcc": train_overall["pcc"],
                "val_loss": val_overall["loss"],
                "val_mae": val_overall["mae"],
                "val_r2": val_overall["r2"],
                "val_pcc": val_overall["pcc"],
                "lr": optimizer.param_groups[0]["lr"],
            })

            if val_overall["loss"] < best_val_loss - args.min_delta:
                best_val_loss = val_overall["loss"]
                best_epoch = current_epoch
                best_pcc = val_overall["pcc"]
                best_state = {
                    "epoch": current_epoch,
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "feature_dim": feature_dim,
                    "num_targets": args.num_targets,
                    "initial_dim": args.initial_dim,
                    "growth_rate": args.growth_rate,
                    "bottleneck_factor": args.bottleneck_factor,
                    "transition_factor": args.transition_factor,
                    "dropout": args.dropout,
                    "fold": args.fold,
                    "train_patients": train_patients,
                    "test_patient": test_patient,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "best_pcc": best_pcc,
                }
                patience_counter = 0
                print(f"*** New best test loss: {best_val_loss:.6f} (PCC={best_pcc:.4f}) ***")
            else:
                patience_counter += 1
                print(f"No improvement. Early stop: {patience_counter}/{args.early_stop_patience}")
                if patience_counter >= args.early_stop_patience:
                    print(f"Early stopping at epoch {current_epoch}")
                    early_stopped = True
                    break

    except Exception as e:
        notify_training_error(f"UNI2-h-Cross-Fold{args.fold}", current_epoch, str(e))
        raise

    status = "early_stop" if early_stopped else "completed"
    notify_training_complete(
        f"UNI2-h-Cross-Fold{args.fold}", current_epoch, best_epoch, best_pcc, status
    )

    # ── Save best checkpoint ──
    if best_state is None:
        best_state = {
            "epoch": current_epoch,
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "fold": args.fold,
            "train_patients": train_patients,
            "test_patient": test_patient,
        }

    torch.save(best_state, checkpoint_path)
    print(f"Saved best checkpoint: {checkpoint_path}")
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Saved history: {history_path}")

    # ── Final evaluation ──
    model.load_state_dict(best_state["model_state_dict"])
    final_overall, final_full = evaluate(model, test_loader, criterion, device)
    print(f"\n=== Fold {args.fold} Final Results (Best Epoch {best_epoch}) ===")
    print(f"Test Loss={final_overall['loss']:.6f} MAE={final_overall['mae']:.6f} R²={final_overall['r2']:.6f} PCC={final_overall['pcc']:.6f}")

    # Per-target summary
    print("\nPer-target PCC:")
    for i in range(args.num_targets):
        tk = f"target_{i}"
        if tk in final_full.get("per_target", {}):
            print(f"  T{i:02d}: PCC={final_full['per_target'][tk]['pcc']:.4f}")

    # ── Generate predictions CSV ──
    try:
        test_target_cols = test_dataset.target_cols if hasattr(test_dataset, 'target_cols') else [f"target_{i}" for i in range(args.num_targets)]
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(device)
                outputs = model(features)
                all_preds.append(outputs.cpu())
                all_labels.append(targets.cpu())
        preds_cat = torch.cat(all_preds).numpy()
        labels_cat = torch.cat(all_labels).numpy()

        pred_df = pd.DataFrame()
        for i, col in enumerate(test_target_cols):
            pred_df[f"true_{col}"] = labels_cat[:, i]
            pred_df[f"pred_{col}"] = preds_cat[:, i]

        from visualize_results import generate_full_report
        vis_dir = str(ckpt_dir / "results_vis")
        actual_vis_dir = generate_full_report(
            model_name=f"UNI2-h DenseNet121 Fold{args.fold}",
            history_csv=str(history_path),
            predictions_csv=None,
            output_dir=vis_dir,
            params={
                "fold": args.fold,
                "train_patients": "+".join(train_patients),
                "test_patient": test_patient,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "learning_rate": args.learning_rate,
                "initial_dim": args.initial_dim,
                "growth_rate": args.growth_rate,
                "dropout": args.dropout,
                "num_targets": args.num_targets,
                "best_epoch": best_epoch,
                "best_pcc": best_pcc,
            }
        )

        pred_csv_path = os.path.join(actual_vis_dir, "predictions.csv")
        pred_df.to_csv(pred_csv_path, index=False)
        print(f"[OK] Predictions saved: {pred_csv_path}")

        generate_full_report(
            model_name=f"UNI2-h DenseNet121 Fold{args.fold}",
            history_csv=str(history_path),
            predictions_csv=pred_csv_path,
            output_dir=vis_dir,
            params={
                "fold": args.fold,
                "train_patients": "+".join(train_patients),
                "test_patient": test_patient,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "learning_rate": args.learning_rate,
                "dropout": args.dropout,
                "num_targets": args.num_targets,
                "best_epoch": best_epoch,
                "best_pcc": best_pcc,
            }
        )
        print(f"[OK] Visualizations saved to {actual_vis_dir}/")
    except Exception as e:
        print(f"[WARNING] Viz generation failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Fold {args.fold} complete. Best epoch={best_epoch}, Best PCC={best_pcc:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
