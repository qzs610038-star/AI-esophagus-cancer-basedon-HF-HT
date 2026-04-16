import argparse
import os
import sys
import signal
from pathlib import Path

# 将项目根目录和当前目录加入 sys.path 以支持 import config_utils 和 uni2h_des
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import pandas as pd
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 忽略 Ctrl+C 信号，防止误触中断训练
signal.signal(signal.SIGINT, signal.SIG_IGN)

# ─── 从 config.yaml 读取默认路径和 HF token ───────────────────────────────────
try:
    from config_utils import load_config, get_data_paths, get_hf_config, get_device
    _config = load_config()
    _data_paths = get_data_paths(_config)
    _hf_config = get_hf_config(_config)
    _DEFAULT_TRAIN = _data_paths.get("train_patches_dir")
    _DEFAULT_VAL = _data_paths.get("val_patches_dir")
    _DEFAULT_LABELS_RAW = _data_paths.get("labels_csv_raw")
    _DEFAULT_LABELS_ZSCORE = _data_paths.get("labels_csv_zscore")
    _DEFAULT_HF_TOKEN = _hf_config.get("token")
    _DEFAULT_HF_LOCAL_ONLY = _hf_config.get("local_only", False)
except Exception as e:
    print(f"[WARNING] 无法加载 config.yaml: {e}，使用默认路径")
    _config = None
    _DEFAULT_TRAIN = None
    _DEFAULT_VAL = None
    _DEFAULT_LABELS_RAW = None
    _DEFAULT_LABELS_ZSCORE = None
    _DEFAULT_HF_TOKEN = None
    _DEFAULT_HF_LOCAL_ONLY = False

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
from notify_utils import notify_training_complete, notify_training_error, check_pause_signal, clear_pause_signal


def build_argparser():
    p = argparse.ArgumentParser(description="Train a regressor on frozen UNI2-h features.")
    p.add_argument("--train_patches_dir", type=str, default=_DEFAULT_TRAIN)  # 训练集
    p.add_argument("--val_patches_dir", type=str, default=_DEFAULT_VAL)  # 验证集/测试集
    p.add_argument("--labels_csv_raw", type=str, default=_DEFAULT_LABELS_RAW)  # 原始csv文件
    p.add_argument("--labels_csv_zscore", type=str, default=_DEFAULT_LABELS_ZSCORE)  # 标准化后的csv文件
    p.add_argument("--cache_root", type=str, default=r".\uni2h_cache\HYZ15040")  # 特征缓存路径
    p.add_argument("--checkpoint_path", type=str, default=r".\checkpoints\HYZ15040\best_model_uni2h.pth")  # 最优模型保存路径
    p.add_argument("--hf_token", type=str, default=_DEFAULT_HF_TOKEN)
    p.add_argument("--hf_local_only", action="store_true", default=_DEFAULT_HF_LOCAL_ONLY,
                   help="强制使用本地缓存，不从网络下载模型")
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
    p.add_argument("--resume", type=str, default=None,
                   help="从checkpoint恢复训练的路径")
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

    device = get_device(_config)
    print(f"Using device: {device}")

    backbone, transform, feature_dim = load_uni2h_backbone(token=args.hf_token, device=device, local_only=args.hf_local_only)
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

    # 检查点路径
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    resume_ckpt_path = checkpoint_path.parent / "resume_uni2h.pth"
    history_path = checkpoint_path.with_suffix(".history.csv")

    # 断点续训加载
    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = 0
    best_pcc = 0.0
    best_state = None
    history = []
    patience_counter = 0

    if args.resume:
        print(f"[INFO] 从checkpoint恢复训练: {args.resume}")
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        
        start_epoch = ckpt.get('epoch', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        patience_counter = ckpt.get('patience_counter', 0)
        best_epoch = ckpt.get('best_epoch', 0)
        best_pcc = ckpt.get('best_pcc', 0.0)
        
        if 'history' in ckpt:
            history = ckpt['history']
        
        print(f"[INFO] 从 Epoch {start_epoch + 1} 继续，best_val_loss={best_val_loss:.4f}")
        
        # 清除暂停信号
        clear_pause_signal(_PROJECT_ROOT)

    early_stopped = False
    current_epoch = 0

    try:
        for epoch in range(start_epoch, args.num_epochs):
            current_epoch = epoch + 1  # 显示时从1开始
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


        if val_metrics_overall["loss"] < best_val_loss - args.min_delta:
            best_val_loss = val_metrics_overall["loss"]
            best_epoch = epoch + 1
            best_pcc = val_metrics_overall["pcc"]
            best_state = {
                "epoch": epoch + 1,
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
                "target_start_col": args.target_start_col,
                "labels_csv": labels_csv,
                "train_cache_dir": str(train_cache_dir),
                "val_cache_dir": str(val_cache_dir),
                "backbone_name": "MahmoodLab/UNI2-h",
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "best_epoch": best_epoch,
                "best_pcc": best_pcc,
            }
            patience_counter = 0
            print(f"*** New best val loss: {best_val_loss:.6f} ***")
        else:
            patience_counter += 1
            print(f"No improvement. Early stop patience: {patience_counter}/{args.early_stop_patience}")
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                early_stopped = True
                break

        # 检查暂停信号
        if check_pause_signal(_PROJECT_ROOT):
            print("\n[INFO] 检测到暂停信号，正在保存 checkpoint 并退出...")
            pause_state = {
                "epoch": epoch + 1,
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
                "target_start_col": args.target_start_col,
                "labels_csv": labels_csv,
                "train_cache_dir": str(train_cache_dir),
                "val_cache_dir": str(val_cache_dir),
                "backbone_name": "MahmoodLab/UNI2-h",
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "best_epoch": best_epoch,
                "best_pcc": best_pcc,
                "history": history,
            }
            torch.save(pause_state, resume_ckpt_path)
            print(f"[INFO] 暂停 checkpoint 已保存: {resume_ckpt_path}")
            notify_training_complete("UNI2-h", epoch + 1, best_epoch, best_pcc, "paused")
            clear_pause_signal(_PROJECT_ROOT)
            return

    except Exception as e:
        notify_training_error("UNI2-h", current_epoch, str(e))
        raise

    # 训练完成通知
    status = "early_stop" if early_stopped else "completed"
    notify_training_complete("UNI2-h", current_epoch, best_epoch, best_pcc, status)

    if best_state is None:
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
            "target_start_col": args.target_start_col,
            "labels_csv": labels_csv,
            "train_cache_dir": str(train_cache_dir),
            "val_cache_dir": str(val_cache_dir),
            "backbone_name": "MahmoodLab/UNI2-h",
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "best_epoch": best_epoch,
            "best_pcc": best_pcc,
        }

    torch.save(best_state, checkpoint_path)
    print(f"Saved best checkpoint to: {checkpoint_path}")

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

    # ── 训练结束后：加载最佳模型，对验证集推理 ────────────────────────────────
    predictions_csv_path = None
    try:
        print("\n[INFO] 加载最佳模型进行验证集推理...")
        model.load_state_dict(best_state["model_state_dict"])
        model.eval()

        # 对验证集推理
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                outputs = model(features)
                all_preds.append(outputs.cpu())
                all_labels.append(targets.cpu())

        preds_cat = torch.cat(all_preds).numpy()
        labels_cat = torch.cat(all_labels).numpy()

        # 生成 predictions.csv（真值和预测值对比）
        # 获取目标列名（从数据集中获取）
        target_cols = val_dataset.target_cols if hasattr(val_dataset, 'target_cols') else [f"target_{i}" for i in range(args.num_targets)]
        pred_df = pd.DataFrame()
        for i, col in enumerate(target_cols):
            pred_df[f'{col}_true'] = labels_cat[:, i]
            pred_df[f'{col}_pred'] = preds_cat[:, i]
        
        # 先创建可视化目录
        from visualize_results import generate_full_report
        output_vis_dir = str(checkpoint_path.parent / "results_vis")
        actual_vis_dir = generate_full_report(
            model_name="UNI2-h DenseNet MLP",
            history_csv=str(history_path),
            predictions_csv=None,  # 先不传入，等保存后再调用
            output_dir=output_vis_dir,
            params={
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "learning_rate": args.learning_rate,
                "initial_dim": args.initial_dim,
                "growth_rate": args.growth_rate,
                "bottleneck_factor": args.bottleneck_factor,
                "transition_factor": args.transition_factor,
                "dropout": args.dropout,
                "early_stop_patience": args.early_stop_patience,
                "num_targets": args.num_targets,
            }
        )
        
        # 保存 predictions.csv 到可视化目录
        predictions_csv_path = os.path.join(actual_vis_dir, "predictions.csv")
        pred_df.to_csv(predictions_csv_path, index=False)
        print(f"[OK] 验证集预测结果已保存: {predictions_csv_path}")
        
        # 重新生成可视化报告，这次包含 predictions.csv
        generate_full_report(
            model_name="UNI2-h DenseNet MLP",
            history_csv=str(history_path),
            predictions_csv=predictions_csv_path,
            output_dir=output_vis_dir,
            params={
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "learning_rate": args.learning_rate,
                "initial_dim": args.initial_dim,
                "growth_rate": args.growth_rate,
                "bottleneck_factor": args.bottleneck_factor,
                "transition_factor": args.transition_factor,
                "dropout": args.dropout,
                "early_stop_patience": args.early_stop_patience,
                "num_targets": args.num_targets,
            }
        )
        print(f"[OK] 完整可视化结果（含逐通路指标）已生成到 {actual_vis_dir}/")
    except Exception as e:
        print(f"[WARNING] 验证集推理或可视化生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()