import argparse
import os
from pathlib import Path
import pandas as pd
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from uni2h_utils import (
    CachedFeaturePatchDataset,
    DEFAULT_NUM_TARGETS,
    DEFAULT_TARGET_START_COL,
    BackboneRegressor,
    evaluate,
    extract_and_cache_features,
    load_uni2h_backbone,
    train_one_epoch,
)

HF_TOKEN = ""  # huggingface的token


def build_argparser():
    p = argparse.ArgumentParser(description="Train a regressor on frozen UNI2-h features.")
    p.add_argument("--train_patches_dir", type=str,
                   default=r"D:\PycharmProjects\AIPath-data\patch\HYZ15040\train_patches")  # 训练集
    p.add_argument("--val_patches_dir", type=str,
                   default=r"D:\PycharmProjects\AIPath-data\patch\HYZ15040\val_patches")  # 验证集/测试集
    p.add_argument("--labels_csv", type=str,
                   default=r"D:\PycharmProjects\AIPath-data\HYZ15040_ssGSEA_scores_zscore.csv")  # zscore后的csv文件
    p.add_argument("--cache_root", type=str, default=r".\uni2h_cache\HYZ15040")  # 特征缓存路径
    p.add_argument("--checkpoint_path", type=str, default=r".\checkpoints\HYZ15040\best_model_uni2h.pth")  # 最优模型保存路径
    p.add_argument("--hf_token", type=str, default=HF_TOKEN)
    p.add_argument("--batch_size", type=int, default=256)  # batch size
    p.add_argument("--num_epochs", type=int, default=100)  # epoch
    p.add_argument("--learning_rate", type=float, default=1e-3)  # 学习率
    p.add_argument("--num_workers", type=int, default=0)  # 控制 DataLoader 读取数据时用多少个子进程
    p.add_argument("--hidden_dim", type=int, default=256)  # 回归头 MLP 的隐藏层维度
    p.add_argument("--dropout", type=float, default=0.2)  # 回归头里的 Dropout 概率
    p.add_argument("--early_stop_patience", type=int, default=10) # 早停
    p.add_argument("--min_delta", type=float, default=0.0) # 至少要比当前 best 好这么多才算提升
    p.add_argument("--target_start_col", type=int, default=DEFAULT_TARGET_START_COL)  # 标签从 CSV 的第几列开始，定义在uni2h_utils.py
    p.add_argument("--num_targets", type=int, default=DEFAULT_NUM_TARGETS)  # 要预测几个指标，定义在uni2h_utils.py
    p.add_argument("--rebuild_cache", action="store_true")  # 是否重新提取 UNI2-h 特征

    return p


def main():
    args = build_argparser().parse_args()

    labels_csv = args.labels_csv

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

    # 构建模型
    model = BackboneRegressor(
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.num_targets,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_state = None
    history = []
    patience_counter = 0

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_metrics["loss"])

        print(
            f"Train | loss={train_metrics['loss']:.6f} mae={train_metrics['mae']:.6f} "
            f"r2={train_metrics['r2']:.6f} pcc={train_metrics['pcc']:.6f}"
        )
        print(
            f"Val   | loss={val_metrics['loss']:.6f} mae={val_metrics['mae']:.6f} "
            f"r2={val_metrics['r2']:.6f} pcc={val_metrics['pcc']:.6f}"
        )

        # 保存历史
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "train_pcc": train_metrics["pcc"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_r2": val_metrics["r2"],
            "val_pcc": val_metrics["pcc"],
            "lr": optimizer.param_groups[0]["lr"],
        })


        if val_metrics["loss"] < best_val_loss - args.min_delta:
            best_val_loss = val_metrics["loss"]
            best_state = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "feature_dim": feature_dim,
                "num_targets": args.num_targets,
                "hidden_dim": args.hidden_dim,
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
            "hidden_dim": args.hidden_dim,
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

    model.load_state_dict(best_state["model_state_dict"])
    final_val = evaluate(model, val_loader, criterion, device)
    print("\nFinal validation on best checkpoint:")
    print(
        f"loss={final_val['loss']:.6f} mae={final_val['mae']:.6f} "
        f"r2={final_val['r2']:.6f} pcc={final_val['pcc']:.6f}"
    )


if __name__ == "__main__":
    main()
