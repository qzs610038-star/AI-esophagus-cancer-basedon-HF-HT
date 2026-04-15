import argparse
import os
from pathlib import Path
import pandas as pd
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from uni2h_des import ( # 注意：导入文件名改为 uni2h_des
    CachedFeaturePatchDataset,
    DEFAULT_NUM_TARGETS,
    DEFAULT_TARGET_START_COL,
    DenseNet121StyleRegressor, # Import the new model
    # DenseNetInspiredRegressor, # Remove the old model import
    # BackboneRegressor, # Remove the old model import
    calculate_max_abs_diff_per_target, # Import the new function
    ensure_zscore_csv,
    evaluate, # Modified to return full metrics dict
    extract_and_cache_features,
    load_uni2h_backbone,
    train_one_epoch,
)

# 设置环境变量以允许从 HuggingFace Hub 下载模型（如果本地没有缓存）
# os.environ['HF_HUB_LOCAL_FILES_ONLY'] = '1'
HF_TOKEN = "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"  # 替换为你的实际 HuggingFace Token


def build_argparser():
    p = argparse.ArgumentParser(description="Train a regressor on frozen UNI2-h features.")
    p.add_argument("--train_patches_dir", type=str,
                   default=r"d:\AI空间转录病理研究\PFMval_new\HYZ15040\train_patches")  # 训练集
    p.add_argument("--val_patches_dir", type=str,
                   default=r"d:\AI空间转录病理研究\PFMval_new\HYZ15040\val_patches")  # 验证集/测试集
    p.add_argument("--labels_csv_raw", type=str,
                   default=r"d:\AI空间转录病理研究\PFMval_new\HYZ15040_ssGSEA_scores.csv")  # 原始csv文件
    p.add_argument("--labels_csv_zscore", type=str,
                   default=r"d:\AI空间转录病理研究\PFMval_new\HYZ15040_ssGSEA_scores_zscore.csv")  # 标准化后的csv文件
    p.add_argument("--cache_root", type=str, default=r".\uni2h_cache\HYZ15040")  # 特征缓存路径
    p.add_argument("--checkpoint_path", type=str, default=r".\checkpoints\HYZ15040\best_model_uni2h.pth")  # 最优模型保存路径
    # p.add_argument("--hf_token", type=str, default=os.environ.get("HUGGINGFACE_HUB_TOKEN", os.environ.get("HF_TOKEN", "")))
    p.add_argument("--hf_token", type=str, default=HF_TOKEN)
    p.add_argument("--batch_size", type=int, default=256)  # batch size
    p.add_argument("--num_epochs", type=int, default=100)  # epoch
    p.add_argument("--learning_rate", type=float, default=1e-3)  # 学习率
    p.add_argument("--num_workers", type=int, default=0)  # 控制 DataLoader 读取数据时用多少个子进程
    # --- Arguments for DenseNet121-style MLP ---
    p.add_argument("--initial_dim", type=int, default=256) # Initial projection dimension (e.g., growth_rate * 4 / 2)
    # Note: num_dense_blocks and num_layers_per_block are fixed according to DenseNet-121 structure
    # p.add_argument("--num_dense_blocks", type=int, default=2) # Now fixed to 4
    # p.add_argument("--num_layers_per_block", type=int, default=8) # Now fixed to [6, 12, 24, 16]
    p.add_argument("--growth_rate", type=int, default=32) # Growth rate per layer
    p.add_argument("--bottleneck_factor", type=int, default=4) # Bottleneck factor (bottleneck_width = factor * growth_rate)
    p.add_argument("--transition_factor", type=float, default=0.5) # Factor for transition layer output dim
    # ---
    p.add_argument("--dropout", type=float, default=0.2)  # 回归头里的 Dropout 概率
    p.add_argument("--early_stop_patience", type=int, default=10) # 早停
    p.add_argument("--min_delta", type=float, default=0.0) # 至少要比当前 best 好这么多才算提升
    p.add_argument("--target_start_col", type=int, default=DEFAULT_TARGET_START_COL)  # 标签从 CSV 的第几列开始，定义在uni2h_utils.py
    p.add_argument("--num_targets", type=int, default=8)  # 要预测几个指标，定义在uni2h_utils.py (修改为 8)
    p.add_argument("--rebuild_cache", action="store_true")  # 是否重新提取 UNI2-h 特征
    return p


def main():
    args = build_argparser().parse_args()

    # 生成z-score后的CSV
    labels_csv = ensure_zscore_csv(
        args.labels_csv_raw,
        args.labels_csv_zscore,
        target_start_col=args.target_start_col,
        num_targets=args.num_targets,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    backbone, transform, feature_dim = load_uni2h_backbone(token=args.hf_token, device=device)
    print(f"Loaded UNI2-h backbone. Feature dim = {feature_dim}")

    train_cache_dir = Path(args.cache_root) / "train"
    val_cache_dir = Path(args.cache_root) / "val"
    train_cache_dir.mkdir(parents=True, exist_ok=True)
    val_cache_dir.mkdir(parents=True, exist_ok=True)

    # 提取并缓存 UNI2-h 特征
    num_train_cached = extract_and_cache_features(
        backbone=backbone,
        transform=transform,
        patches_dir=args.train_patches_dir,
        cache_dir=str(train_cache_dir),
        device=device,
        rebuild=args.rebuild_cache,
    )
    num_val_cached = extract_and_cache_features(
        backbone=backbone,
        transform=transform,
        patches_dir=args.val_patches_dir,
        cache_dir=str(val_cache_dir),
        device=device,
        rebuild=args.rebuild_cache,
    )
    print(f"Cached {num_train_cached} new train features, {num_val_cached} new val features.")

    train_dataset = CachedFeaturePatchDataset(
        patches_dir=args.train_patches_dir,
        labels_csv=labels_csv,
        feature_cache_dir=str(train_cache_dir),
        target_start_col=args.target_start_col,
        num_targets=args.num_targets,
    )
    val_dataset = CachedFeaturePatchDataset(
        patches_dir=args.val_patches_dir,
        labels_csv=labels_csv,
        feature_cache_dir=str(val_cache_dir),
        target_start_col=args.target_start_col,
        num_targets=args.num_targets,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # 构建模型 - 使用 DenseNet121-style MLP
    model = DenseNet121StyleRegressor(
        feature_dim=feature_dim, # 1536
        initial_dim=args.initial_dim, # e.g., 256
        # num_dense_blocks and num_layers_per_block are now fixed inside the model
        growth_rate=args.growth_rate, # e.g., 32
        bottleneck_factor=args.bottleneck_factor, # e.g., 4
        transition_factor=args.transition_factor, # e.g., 0.5
        output_dim=args.num_targets, # 30
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4) # Keep weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_state = None
    history = []
    patience_counter = 0

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        # Modified: train_one_epoch now returns two values
        train_metrics_overall, train_metrics_full = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics_overall, val_metrics_full = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_metrics_overall["loss"])  # Use overall loss for scheduler

        # Updated print statements to include MAPE and use overall metrics (注意变量名)
        print(
            f"Train | loss={train_metrics_overall['loss']:.6f} mae={train_metrics_overall['mae']:.6f} "
            f"mape={train_metrics_overall['mape']:.6f} r2={train_metrics_overall['r2']:.6f} pcc={train_metrics_overall['pcc']:.6f}"
        )
        print(
            f"Val   | loss={val_metrics_overall['loss']:.6f} mae={val_metrics_overall['mae']:.6f} "
            f"mape={val_metrics_overall['mape']:.6f} r2={val_metrics_overall['r2']:.6f} pcc={val_metrics_overall['pcc']:.6f}"
        )

        # Updated history to include MAPE and use overall metrics (注意变量名)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics_overall["loss"], # Use overall metrics directly
            "train_mae": train_metrics_overall["mae"],
            "train_mape": train_metrics_overall["mape"], # Add MAPE
            "train_r2": train_metrics_overall["r2"],
            "train_pcc": train_metrics_overall["pcc"],
            "val_loss": val_metrics_overall["loss"], # Use overall metrics directly
            "val_mae": val_metrics_overall["mae"],
            "val_mape": val_metrics_overall["mape"], # Add MAPE
            "val_r2": val_metrics_overall["r2"],
            "val_pcc": val_metrics_overall["pcc"],
            "lr": optimizer.param_groups[0]["lr"],
        })


        if val_metrics_overall["loss"] < best_val_loss - args.min_delta: # Use overall loss directly
            best_val_loss = val_metrics_overall["loss"] # Use overall loss directly
            best_state = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "feature_dim": feature_dim,
                "num_targets": args.num_targets,
                # Add new model-specific parameters to state dict
                "initial_dim": args.initial_dim,
                "growth_rate": args.growth_rate,
                "bottleneck_factor": args.bottleneck_factor,
                "transition_factor": args.transition_factor,
                "dropout": args.dropout,
                "target_start_col": args.target_start_col,
                "labels_csv": labels_csv,
                "train_cache_dir": str(train_cache_dir),
                "val_cache_dir": str(val_cache_dir),
                "backbone_name": "MahmoodLab/UNI2-h",
                "best_val_loss": best_val_loss,
            }
            patience_counter = 0
            print(f"*** New best val loss: {best_val_loss:.6f} ***")
        else:
            patience_counter += 1
            print(f"No improvement. Early stop patience: {patience_counter}/{args.early_stop_patience}")
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        best_state = {
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "feature_dim": feature_dim,
            "num_targets": args.num_targets,
            # Add new model-specific parameters to state dict
            "initial_dim": args.initial_dim,
            "growth_rate": args.growth_rate,
            "bottleneck_factor": args.bottleneck_factor,
            "transition_factor": args.transition_factor,
            "dropout": args.dropout,
            "target_start_col": args.target_start_col,
            "labels_csv": labels_csv,
            "train_cache_dir": str(train_cache_dir),
            "val_cache_dir": str(val_cache_dir),
            "backbone_name": "MahmoodLab/UNI2-h",
            "best_val_loss": best_val_loss,
        }

    torch.save(best_state, checkpoint_path)
    print(f"Saved best checkpoint to: {checkpoint_path}")

    history_path = checkpoint_path.with_suffix(".history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Saved history to: {history_path}")

    # --- Load best model and perform final evaluation ---
    model.load_state_dict(best_state["model_state_dict"])
    # Evaluate on the validation set one last time to get the full metrics dict for the best model
    final_val_overall, final_val_full = evaluate(model, val_loader, criterion, device)
    print("\nFinal validation on best checkpoint:")
    print(
        f"Overall - loss={final_val_overall['loss']:.6f} mae={final_val_overall['mae']:.6f} "
        f"mape={final_val_overall['mape']:.6f} r2={final_val_overall['r2']:.6f} pcc={final_val_overall['pcc']:.6f}"
    )

    # --- Print detailed metrics for each target ---
    print("\n--- Detailed Metrics for Each Target ---")
    num_targets = args.num_targets
    for i in range(num_targets):
        target_key = f'target_{i}'
        if target_key in final_val_full['per_target']:
            metrics = final_val_full['per_target'][target_key]
            print(f"Target {i}: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}, MAPE={metrics['mape']:.6f}%, R²={metrics['r2']:.6f}, PCC={metrics['pcc']:.6f}")
        else:
            print(f"Target {i}: Metrics not found in results.")

    # --- Calculate and print Max Absolute Difference for each target ---
    print("\n--- Max Absolute Difference for Each Target ---")
    # Run evaluation again to get raw predictions and targets
    _, _ = evaluate(model, val_loader, criterion, device) # This call internally calculates all_targets and all_outputs

    # Manually perform the final evaluation step to get y_true and y_pred
    model.eval()
    all_targets_eval = []
    all_outputs_eval = []
    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features)
            all_targets_eval.append(targets.detach().cpu().numpy())
            all_outputs_eval.append(outputs.detach().cpu().numpy())

    y_true_final = np.concatenate(all_targets_eval, axis=0)
    y_pred_final = np.concatenate(all_outputs_eval, axis=0)

    max_abs_diff_per_target_array = calculate_max_abs_diff_per_target(y_true=y_true_final, y_pred=y_pred_final)
    for i in range(num_targets):
        print(f"Target {i}: Max Abs Diff = {max_abs_diff_per_target_array[i]:.6f}")


if __name__ == "__main__":
    main()