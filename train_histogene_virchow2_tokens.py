"""
HisToGene-Virchow2 Token序列训练脚本
=====================================
基于 Virchow2 backbone (1280-dim tokens)，复用 HisToGeneUNITokens 下游模型。

与 UNI 版本的关键差异:
  - feature_dim=1280 (UNI=1536)
  - 缓存目录: virchow2_cache_tokens/
  - 不包含 AugMix (后续按需添加)

支持模式:
  1. 单患者: --patient HYZ15040
  2. 跨患者: --cross_patient (默认 fold 1)
  3. 三折CV: --cross_patient --fold {1,2,3}
"""

import argparse
import os
import random
import sys
import signal
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_uni_tokens import HisToGeneUNITokensDataset
from model_uni_tokens import HisToGeneUNITokens
from histogene.utils import compute_metrics
from config_utils import load_config, get_device, get_patient_paths, get_fold_config, get_histogene_dir

signal.signal(signal.SIGINT, signal.SIG_IGN)

# ═══════════════════════════════════════════════════════════════════════════
#  数据路径配置 — 统一由 config_utils.get_patient_paths() 管理
#  本地使用默认路径，服务器通过 config.yaml 覆盖
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_DIM = 1280
BACKBONE_NAME = "Virchow2"


# ═══════════════════════════════════════════════════════════════════════════
#  训练核心
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(device):
    model = HisToGeneUNITokens(
        feature_dim=FEATURE_DIM,
        dim=1024,
        n_pos=128,
        n_targets=30,
        mlp_dim=2048,
        dropout=0.5,
        encoder_hidden_dim=512,
        n_encoder_layers=1,
        n_encoder_heads=8,
    )
    model.to(device)
    print(f"[Model] HisToGeneUNITokens(feature_dim={FEATURE_DIM}), "
          f"params={model.count_parameters()/1e6:.2f}M")
    return model


def build_datasets(train_patients, eval_patient, use_augmented=False):
    """构建训练集（多患者合并）和验证集"""
    train_configs = []
    for p in train_patients:
        cfg = get_patient_paths(p, backbone='virchow2')
        train_configs.append({
            'patches_dir': cfg['train_patches'],
            'labels_csv': cfg['labels_csv'],
            'feature_cache_dir': cfg['token_cache_train'],
            'patient_name': p,
        })

    train_dataset, coord_stats_dict, target_cols = HisToGeneUNITokensDataset.from_multiple_patients(
        train_configs,
        n_pos=128,
        n_targets=30,
        verbose=True,
        feature_dim=FEATURE_DIM,
        backbone_name=BACKBONE_NAME,
    )

    eval_cfg = get_patient_paths(eval_patient, backbone='virchow2')
    eval_dataset = HisToGeneUNITokensDataset(
        patches_dir=eval_cfg['val_patches'],
        feature_cache_dir=eval_cfg['token_cache_val'],
        labels_csv=eval_cfg['labels_csv'],
        target_cols=target_cols,
        n_pos=128,
        n_targets=30,
        coord_stats=None,
        feature_dim=FEATURE_DIM,
        backbone_name=BACKBONE_NAME,
    )

    return train_dataset, eval_dataset, target_cols


def train_epoch(model, dataloader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    loss_fn = nn.HuberLoss(delta=1.0)

    for batch in dataloader:
        tokens, pos_x, pos_y, targets = [b.to(device, non_blocking=True) for b in batch]

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            pred = model(tokens, pos_x, pos_y)
            loss = loss_fn(pred, targets)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.inference_mode()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_targets = [], []

    for batch in dataloader:
        tokens, pos_x, pos_y, targets = [b.to(device) for b in batch]
        pred = model(tokens, pos_x, pos_y)
        all_preds.append(pred.cpu())
        all_targets.append(targets.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()
    return compute_metrics(y_true, y_pred)


def save_predictions(model, dataloader, target_cols, device, save_path):
    model.eval()
    all_preds, all_targets = [], []
    with torch.inference_mode():
        for batch in dataloader:
            tokens, pos_x, pos_y, targets = [b.to(device) for b in batch]
            pred = model(tokens, pos_x, pos_y)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    truths = np.concatenate(all_targets, axis=0)

    rows = []
    for i, col in enumerate(target_cols):
        rows.append({f'true_{col}': truths[:, i], f'pred_{col}': preds[:, i]})
    pd.DataFrame(rows).to_csv(save_path, index=False)


# ═══════════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="HisToGene-Virchow2 训练")
    parser.add_argument("--patient", type=str, default=None, help="单患者名称")
    parser.add_argument("--cross_patient", action="store_true", help="跨患者模式")
    parser.add_argument("--fold", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    if not args.patient and not args.cross_patient:
        parser.error("Must specify --patient or --cross_patient")

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB)")

    # ── 确定训练/评估患者 ──
    if args.cross_patient:
        fold_cfg = get_fold_config(args.fold)
        train_patients = fold_cfg["train"]
        eval_patient = fold_cfg["test"]
        mode_str = f"CrossPatient_Fold{args.fold}"
    else:
        train_patients = [args.patient]
        eval_patient = args.patient
        mode_str = args.patient

    print(f"模式: {mode_str}")
    print(f"训练患者: {train_patients}")
    print(f"评估患者: {eval_patient}")

    # ── 數據集 ──
    train_dataset, eval_dataset, target_cols = build_datasets(
        train_patients, eval_patient)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    # ── 模型 ──
    model = build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # ── 断点续训 ──
    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = 0
    history = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        best_epoch = ckpt.get('best_epoch', start_epoch - 1)
        history = ckpt.get('history', [])
        print(f"从 epoch {start_epoch} 续训, best_val_loss={best_val_loss:.6f}")

    # ── 保存路径 ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"{mode_str}_Virchow2_Tokens"
    checkpoint_dir = Path(args.checkpoint_dir) / f"HisToGene_Virchow2_{dataset_name}_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_vis_dir = Path("histogene/checkpoints/results_vis") / f"{dataset_name}_{timestamp}"
    results_vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCheckpoint 目录: {checkpoint_dir}")
    print(f"Results 目录: {results_vis_dir}")
    print(f"{'='*60}")

    # ── 训练循环 ──
    for epoch in range(start_epoch, args.num_epochs):
        t0 = datetime.now()

        train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
        metrics = evaluate(model, eval_loader, device)
        scheduler.step(metrics['mse'])

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': metrics['mse'],
            'val_pcc': metrics['pcc'],
            'val_mae': metrics['mae'],
            'val_r2': metrics['r2'],
        })

        elapsed = (datetime.now() - t0).total_seconds()
        is_best = metrics['mse'] < best_val_loss
        if is_best:
            best_val_loss = metrics['mse']
            best_epoch = epoch

        best_marker = " *BEST*" if is_best else ""
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
              f"val_loss={metrics['mse']:.4f} | PCC={metrics['pcc']:.4f} | "
              f"MAE={metrics['mae']:.4f} | R²={metrics['r2']:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.0f}s{best_marker}")

        # 保存 checkpoint
        if is_best or epoch % 10 == 0 or epoch == args.num_epochs - 1:
            ckpt_path = checkpoint_dir / f"checkpoint_epoch{epoch:03d}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'history': history,
                'args': vars(args),
            }, ckpt_path)

        # 早停
        if epoch - best_epoch > 20:
            print(f"早停: {epoch - best_epoch} epochs 无改善")
            break

    # ── 保存最终结果 ──
    history_df = pd.DataFrame(history)
    history_df.to_csv(checkpoint_dir / "training_history.csv", index=False)

    # 用最佳 checkpoint 生成预测
    best_ckpt = checkpoint_dir / f"checkpoint_epoch{best_epoch:03d}.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
    save_predictions(model, eval_loader, target_cols, device,
                     checkpoint_dir / "predictions.csv")

    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"  最佳 epoch: {best_epoch}")
    print(f"  最佳 val_loss: {best_val_loss:.6f}")
    best_metrics = history[best_epoch]
    print(f"  PCC: {best_metrics['val_pcc']:.4f}")
    print(f"  MAE: {best_metrics['val_mae']:.4f}")
    print(f"  R²:  {best_metrics['val_r2']:.4f}")
    print(f"  Checkpoint: {checkpoint_dir}")


if __name__ == "__main__":
    main()
