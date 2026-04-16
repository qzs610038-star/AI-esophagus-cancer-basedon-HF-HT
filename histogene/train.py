"""
HisToGene 训练脚本 - PFMval 项目
完整训练流程：数据加载、模型训练、验证、早停、保存
支持：断点续训、暂停信号检测
"""
import argparse
import os
import sys
import time
import signal
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
from notify_utils import notify_training_complete, notify_training_error, check_pause_signal, clear_pause_signal

# 忽略 Ctrl+C 信号，防止误触中断训练
signal.signal(signal.SIGINT, signal.SIG_IGN)

# ─── 从 config.yaml 读取默认路径 ───────────────────────────────────────────────
try:
    from config_utils import load_config, get_data_paths, get_device
    _config = load_config()
    _data_paths = get_data_paths(_config)
    _DEFAULT_TRAIN = _data_paths.get("train_patches_dir")
    _DEFAULT_VAL = _data_paths.get("val_patches_dir")
    _DEFAULT_LABELS = _data_paths.get("labels_csv_zscore")
except Exception as e:
    print(f"[WARNING] 无法加载 config.yaml: {e}，使用默认路径")
    _config = None
    _DEFAULT_TRAIN = None
    _DEFAULT_VAL = None
    _DEFAULT_LABELS = None


def build_argparser():
    p = argparse.ArgumentParser(description="Train HisToGene on ssGSEA pathway scores.")
    p.add_argument("--dataset_name", type=str, default="HYZ15040",
                   help="数据集名称，用于区分不同数据集的训练结果")
    p.add_argument("--train_patches_dir", type=str, default=_DEFAULT_TRAIN)
    p.add_argument("--val_patches_dir",   type=str, default=_DEFAULT_VAL)
    p.add_argument("--labels_csv",        type=str, default=_DEFAULT_LABELS)
    p.add_argument("--checkpoint_dir",    type=str, default=None,
                   help="checkpoint 保存目录，默认为 histogene/checkpoints/{dataset_name}")
    p.add_argument("--history_csv",       type=str, default=None,
                   help="训练历史 CSV 路径，默认为 histogene/training_history_{dataset_name}.csv")

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
    p.add_argument("--n_targets",    type=int,   default=30)
    p.add_argument("--dropout",      type=float, default=0.3)

    # 早停
    p.add_argument("--early_stop_patience", type=int, default=15)

    # 混合精度
    p.add_argument("--amp", action="store_true", default=True,
                   help="使用混合精度训练（仅 CUDA 生效）")

    # 断点续训
    p.add_argument("--resume", type=str, default=None,
                   help="从checkpoint恢复训练的路径")
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

    # ── 根据数据集名称设置默认路径 ───────────────────────────────────────────
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(_SCRIPT_DIR / "checkpoints" / args.dataset_name)
    if args.history_csv is None:
        args.history_csv = str(_SCRIPT_DIR / f"training_history_{args.dataset_name}.csv")

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
    device = get_device(_config)
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
    resume_ckpt = ckpt_dir / "resume_histogene.pth"

    # ── 断点续训加载 ───────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_loss = float('inf')
    best_epoch = 0
    best_pcc = 0.0
    patience_counter = 0
    history = []

    if args.resume:
        print(f"[INFO] 从checkpoint恢复训练: {args.resume}")
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        patience_counter = ckpt.get('patience_counter', 0)
        best_epoch = ckpt.get('best_epoch', 0)
        best_pcc = ckpt.get('best_pcc', 0.0)
        
        if 'scaler_state_dict' in ckpt and ckpt['scaler_state_dict'] and scaler:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        
        # 恢复训练历史
        if 'history' in ckpt:
            history = ckpt['history']
        
        print(f"[INFO] 从 Epoch {start_epoch} 继续，best_val_loss={best_val_loss:.4f}")
        
        # 清除暂停信号
        clear_pause_signal(_PROJECT_ROOT)

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    early_stopped = False

    print("\n" + "=" * 90)
    print(f"开始训练 HisToGene | Epochs={args.num_epochs} | BS={args.batch_size} | LR={args.lr}")
    print("=" * 90)

    current_epoch = 0
    try:
        for epoch in range(start_epoch, args.num_epochs + 1):
            current_epoch = epoch
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
                best_epoch = epoch
                best_pcc = val_m['pcc']
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                    'val_loss': val_loss,
                    'val_metrics': val_m,
                    'args': vars(args),
                    'coord_stats': coord_stats,
                    'target_cols': train_dataset.target_cols,
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'best_epoch': best_epoch,
                    'best_pcc': best_pcc,
                }, best_ckpt)
                print(f"  ✓ 最佳模型已保存 (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop_patience:
                    print(f"\n早停触发！连续 {args.early_stop_patience} 个 epoch val_loss 未改善。")
                    early_stopped = True
                    break

            # 每 10 个 epoch 保存一次历史
            if epoch % 10 == 0:
                pd.DataFrame(history).to_csv(args.history_csv, index=False)

            # 检查暂停信号
            if check_pause_signal(_PROJECT_ROOT):
                print("\n[INFO] 检测到暂停信号，正在保存 checkpoint 并退出...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                    'val_loss': val_loss,
                    'val_metrics': val_m,
                    'args': vars(args),
                    'coord_stats': coord_stats,
                    'target_cols': train_dataset.target_cols,
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'history': history,
                    'best_epoch': best_epoch,
                    'best_pcc': best_pcc,
                }, resume_ckpt)
                print(f"[INFO] 暂停 checkpoint 已保存: {resume_ckpt}")
                notify_training_complete("HisToGene", epoch, best_epoch, best_pcc, "paused")
                clear_pause_signal(_PROJECT_ROOT)
                return

    except Exception as e:
        notify_training_error("HisToGene", current_epoch, str(e))
        raise

    # 训练完成通知
    status = "early_stop" if early_stopped else "completed"
    notify_training_complete("HisToGene", current_epoch, best_epoch, best_pcc, status)

    # 最终保存历史
    pd.DataFrame(history).to_csv(args.history_csv, index=False)
    print(f"\n[DONE] 训练结束。最佳 val_loss={best_val_loss:.4f}")
    print(f"  最佳模型: {best_ckpt}")
    print(f"  训练历史: {args.history_csv}")

    # ── 训练结束后：加载最佳模型，对验证集推理 ────────────────────────────────
    predictions_csv_path = None
    try:
        print("\n[INFO] 加载最佳模型进行验证集推理...")
        best_ckpt_data = torch.load(best_ckpt, weights_only=False, map_location=device)
        model.load_state_dict(best_ckpt_data['model_state_dict'])
        model.eval()

        # 对验证集推理
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, pos_x, pos_y, targets in val_loader:
                images = images.to(device, non_blocking=True)
                pos_x = pos_x.to(device, non_blocking=True)
                pos_y = pos_y.to(device, non_blocking=True)
                preds = model(images, pos_x, pos_y)
                all_preds.append(preds.cpu())
                all_labels.append(targets.cpu())

        preds_cat = torch.cat(all_preds).numpy()
        labels_cat = torch.cat(all_labels).numpy()

        # 生成 predictions.csv（真值和预测值对比）
        # 注意：列名格式必须是 true_{通路名} 和 pred_{通路名}，与 visualize_results.py 期望的格式一致
        target_cols = train_dataset.target_cols
        pred_df = pd.DataFrame()
        for i, col in enumerate(target_cols):
            pred_df[f'true_{col}'] = labels_cat[:, i]
            pred_df[f'pred_{col}'] = preds_cat[:, i]
        
        # 创建可视化输出目录并保存 predictions.csv
        from visualize_results import generate_full_report
        output_vis_dir = str(ckpt_dir.parent / "results_vis")
        model_name_with_dataset = f"HisToGene_{args.dataset_name}"
        
        # 创建时间戳目录
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        actual_vis_dir = os.path.join(output_vis_dir, f"{args.dataset_name}_{timestamp}")
        os.makedirs(actual_vis_dir, exist_ok=True)
        
        # 保存 predictions.csv 到可视化目录
        predictions_csv_path = os.path.join(actual_vis_dir, "predictions.csv")
        pred_df.to_csv(predictions_csv_path, index=False)
        print(f"[OK] 验证集预测结果已保存: {predictions_csv_path}")
        
        # 生成完整可视化报告（只调用一次，传入已有的 actual_vis_dir）
        generate_full_report(
            model_name=model_name_with_dataset,
            history_csv=args.history_csv,
            predictions_csv=predictions_csv_path,
            output_dir=output_vis_dir,
            prefix=args.dataset_name,
            actual_output_dir=actual_vis_dir,
            params={
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "lr": args.lr,
                "img_size": args.img_size,
                "patch_size": args.patch_size,
                "model_dim": args.model_dim,
                "model_depth": args.model_depth,
                "heads": args.heads,
                "mlp_dim": args.mlp_dim,
                "n_pos": args.n_pos,
                "n_targets": args.n_targets,
                "dropout": args.dropout,
                "early_stop_patience": args.early_stop_patience,
                "dataset_name": args.dataset_name,
            }
        )
        print(f"[OK] 完整可视化结果（含逐通路指标）已生成到 {actual_vis_dir}/")
    except Exception as e:
        print(f"[WARNING] 验证集推理或可视化生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
