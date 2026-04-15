"""
HisToGene 训练脚本 - PFMval 项目
完整训练流程：数据加载、模型训练、验证、早停、保存
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 将项目根目录加入 sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from histogene.model import HisToGeneModel
from histogene.dataset import HisToGeneDataset
from histogene.utils import compute_metrics

# ─── 自动搜索数据目录 ─────────────────────────────────────────────────────────
_SEARCH_ROOTS = [
    r"d:\AI空间转录病理研究\PFMval_new\HYZ15040",
    r"d:\AI空间转录病理研究\PFMval_new",
    r"D:\PycharmProjects\AIPath-data\patch\HYZ15040",
]

def _find_patches_dirs():
    for root in _SEARCH_ROOTS:
        train_dir = os.path.join(root, "train_patches")
        val_dir = os.path.join(root, "val_patches")
        if os.path.isdir(train_dir) and os.path.isdir(val_dir):
            return train_dir, val_dir
    return None, None

_DEFAULT_TRAIN, _DEFAULT_VAL = _find_patches_dirs()
_DEFAULT_LABELS = r"d:\AI空间转录病理研究\PFMval_new\HYZ15040_ssGSEA_scores_zscore.csv"


def build_argparser():
    p = argparse.ArgumentParser(description="Train HisToGene on ssGSEA pathway scores.")
    p.add_argument("--train_patches_dir", type=str, default=_DEFAULT_TRAIN)
    p.add_argument("--val_patches_dir",   type=str, default=_DEFAULT_VAL)
    p.add_argument("--labels_csv",        type=str, default=_DEFAULT_LABELS)
    p.add_argument("--checkpoint_dir",    type=str,
                   default=str(_SCRIPT_DIR / "checkpoints"))
    p.add_argument("--history_csv",       type=str,
                   default=str(_SCRIPT_DIR / "training_history.csv"))

    # 训练超参
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--num_epochs",   type=int,   default=150)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int,   default=0)   # Windows 下用 0

    # 模型超参
    p.add_argument("--img_size",     type=int,   default=224)
    p.add_argument("--patch_size",   type=int,   default=16)
    p.add_argument("--model_dim",    type=int,   default=1024)
    p.add_argument("--model_depth",  type=int,   default=8)
    p.add_argument("--heads",        type=int,   default=16)
    p.add_argument("--mlp_dim",      type=int,   default=2048)
    p.add_argument("--n_pos",        type=int,   default=128)
    p.add_argument("--n_targets",    type=int,   default=8)
    p.add_argument("--dropout",      type=float, default=0.3)

    # 早停
    p.add_argument("--early_stop_patience", type=int, default=15)

    # 混合精度
    p.add_argument("--amp", action="store_true", default=True,
                   help="使用混合精度训练（仅 CUDA 生效）")
    return p


def get_transforms(img_size, train=True):
    """构建图像变换流水线"""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    base = []
    # 若图像非 img_size，则 resize
    base.append(transforms.Resize((img_size, img_size)))

    if train:
        base += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
        ]

    base += [
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
    return transforms.Compose(base)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for images, pos_x, pos_y, targets in loader:
        images  = images.to(device, non_blocking=True)
        pos_x   = pos_x.to(device, non_blocking=True)
        pos_y   = pos_y.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                preds = model(images, pos_x, pos_y)
                loss  = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(images, pos_x, pos_y)
            loss  = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        all_preds.append(preds.detach().cpu())
        all_labels.append(targets.detach().cpu())

    n = len(loader.dataset)
    avg_loss = total_loss / n
    all_preds  = torch.cat(all_preds,  dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_labels.numpy(), all_preds.numpy())
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for images, pos_x, pos_y, targets in loader:
        images  = images.to(device, non_blocking=True)
        pos_x   = pos_x.to(device, non_blocking=True)
        pos_y   = pos_y.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        preds = model(images, pos_x, pos_y)
        loss  = criterion(preds, targets)

        total_loss += loss.item() * images.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(targets.cpu())

    n = len(loader.dataset)
    avg_loss = total_loss / n
    all_preds  = torch.cat(all_preds,  dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_labels.numpy(), all_preds.numpy())
    return avg_loss, metrics


def main():
    args = build_argparser().parse_args()

    # ── 路径检查 ──────────────────────────────────────────────────────────────
    if args.train_patches_dir is None or not os.path.isdir(args.train_patches_dir):
        print(f"[ERROR] train_patches_dir 不存在: {args.train_patches_dir}")
        print("  请先运行 split.py 生成数据划分，或用 --train_patches_dir 指定路径")
        sys.exit(1)
    if args.val_patches_dir is None or not os.path.isdir(args.val_patches_dir):
        print(f"[ERROR] val_patches_dir 不存在: {args.val_patches_dir}")
        sys.exit(1)
    if not os.path.isfile(args.labels_csv):
        print(f"[ERROR] labels_csv 不存在: {args.labels_csv}")
        sys.exit(1)

    print(f"[INFO] Train dir : {args.train_patches_dir}")
    print(f"[INFO] Val dir   : {args.val_patches_dir}")
    print(f"[INFO] Labels CSV: {args.labels_csv}")

    # ── 设备 ──────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("[INFO] 混合精度训练已启用")

    # ── 数据集 ────────────────────────────────────────────────────────────────
    train_transform = get_transforms(args.img_size, train=True)
    val_transform   = get_transforms(args.img_size, train=False)

    train_dataset = HisToGeneDataset(
        patches_dir=args.train_patches_dir,
        labels_csv=args.labels_csv,
        n_pos=args.n_pos,
        transform=train_transform,
    )
    coord_stats = train_dataset.get_coord_stats()

    val_dataset = HisToGeneDataset(
        patches_dir=args.val_patches_dir,
        labels_csv=args.labels_csv,
        n_pos=args.n_pos,
        transform=val_transform,
        coord_stats=coord_stats,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ── 模型 ──────────────────────────────────────────────────────────────────
    model = HisToGeneModel(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=3,
        dim=args.model_dim,
        depth=args.model_depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        n_pos=args.n_pos,
        n_targets=args.n_targets,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] 模型参数量: {n_params:,}")

    # ── 损失 / 优化器 / 调度器 ────────────────────────────────────────────────
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )

    # ── 检查点目录 ────────────────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best_histogene.pth"

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    print("\n" + "=" * 90)
    print(f"开始训练 HisToGene | Epochs={args.num_epochs} | BS={args.batch_size} | LR={args.lr}")
    print("=" * 90)

    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()

        train_loss, train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_m = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:3d}/{args.num_epochs}] "
            f"Train Loss: {train_loss:.4f} MAE: {train_m['mae']:.4f} "
            f"R²: {train_m['r2']:.4f} PCC: {train_m['pcc']:.4f} | "
            f"Val Loss: {val_loss:.4f} MAE: {val_m['mae']:.4f} "
            f"R²: {val_m['r2']:.4f} PCC: {val_m['pcc']:.4f} | "
            f"LR: {current_lr:.2e} | {elapsed:.1f}s"
        )

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_mae': train_m['mae'],
            'train_r2': train_m['r2'],
            'train_pcc': train_m['pcc'],
            'val_loss': val_loss,
            'val_mae': val_m['mae'],
            'val_r2': val_m['r2'],
            'val_pcc': val_m['pcc'],
            'lr': current_lr,
        })

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_m,
                'args': vars(args),
                'coord_stats': coord_stats,
                'target_cols': train_dataset.target_cols,
            }, best_ckpt)
            print(f"  ✓ 最佳模型已保存 (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"\n早停触发！连续 {args.early_stop_patience} 个 epoch val_loss 未改善。")
                break

        # 每 10 个 epoch 保存一次历史
        if epoch % 10 == 0:
            pd.DataFrame(history).to_csv(args.history_csv, index=False)

    # 最终保存历史
    pd.DataFrame(history).to_csv(args.history_csv, index=False)
    print(f"\n[DONE] 训练结束。最佳 val_loss={best_val_loss:.4f}")
    print(f"  最佳模型: {best_ckpt}")
    print(f"  训练历史: {args.history_csv}")


if __name__ == "__main__":
    main()
