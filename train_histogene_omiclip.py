"""
HisToGene-OmiCLIP Token序列训练脚本
====================================
使用 OmiCLIP vision encoder 提取的 token 特征 [255, 768] 训练通路评分预测模型。
复用 model_uni_tokens.py 中的 HisToGeneUNITokens 模型（feature_dim=768, num_tokens=255）。

运行环境: C:\\Program Files\\Python313\\python.exe
缓存目录: omiclip_cache/{patient}/{split}/  (每个 patch 一个 .pt, shape [255, 768])

支持模式:
  1. 单患者模式:  python train_histogene_omiclip.py --patient HYZ15040
  2. 跨患者模式:  python train_histogene_omiclip.py --cross_patient --fold 1
  3. 三折交叉验证: --cross_patient --fold {1,2,3}

运行命令示例:
  "C:\\Program Files\\Python313\\python.exe" train_histogene_omiclip.py --patient HYZ15040
  "C:\\Program Files\\Python313\\python.exe" train_histogene_omiclip.py --cross_patient --fold 1
  "C:\\Program Files\\Python313\\python.exe" train_histogene_omiclip.py --cross_patient --fold 2 --lr 5e-5
"""

import argparse
import os
import sys
import time
import signal
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

# ── 项目根目录 ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model_uni_tokens import HisToGeneUNITokens
from histogene.utils import compute_metrics
from notify_utils import (
    notify_training_complete, notify_training_error,
    check_pause_signal, clear_pause_signal,
)
from config_utils import load_config, get_device, get_patient_paths, get_fold_config, get_histogene_dir, get_output_dir

# 忽略 Ctrl+C 信号，防止误触中断训练
signal.signal(signal.SIGINT, signal.SIG_IGN)

# ═══════════════════════════════════════════════════════════════════════════
#  OmiCLIP 参数常量
# ═══════════════════════════════════════════════════════════════════════════
OMICLIP_FEATURE_DIM = 768   # OmiCLIP vision encoder embedding 维度
OMICLIP_NUM_TOKENS = 255    # OmiCLIP 输出 token 数量

# ═══════════════════════════════════════════════════════════════════════════
#  数据路径配置 — 统一由 config_utils.get_patient_paths() 管理
#  本地使用默认路径，服务器通过 config.yaml 覆盖
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
#  OmiCLIP Dataset（基于 dataset_uni_tokens.py 逻辑，适配 768 维）
# ═══════════════════════════════════════════════════════════════════════════
import re

def parse_coordinates(filename):
    """从文件名 patch_x4641_y16969.png 解析坐标"""
    match = re.search(r'x(\d+)_y(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


class OmiCLIPTokensDataset(torch.utils.data.Dataset):
    """OmiCLIP token 序列数据集

    与 HisToGeneUNITokensDataset 逻辑一致，但适配 OmiCLIP 的
    特征维度 (768) 和固定 token 数量 (255)。
    """

    def __init__(self, patches_dir, feature_cache_dir, labels_csv,
                 target_cols=None, n_pos=128, n_targets=30, coord_stats=None):
        self.feature_cache_dir = feature_cache_dir
        self.patches_dir = patches_dir
        self.n_pos = n_pos

        # 加载标签
        df = pd.read_csv(labels_csv)
        id_col = df.columns[0]
        if target_cols is None:
            target_cols = list(df.columns[1:])
        self.target_cols = target_cols

        # 构建标签映射: patch_stem -> target_values
        self.label_map = {}
        for _, row in df.iterrows():
            stem = str(row[id_col]).replace('.png', '')
            self.label_map[stem] = row[target_cols].values.astype(np.float32)

        # 扫描缓存目录中已有的 .pt 文件
        cached_stems = set()
        if os.path.isdir(feature_cache_dir):
            for fname in os.listdir(feature_cache_dir):
                if fname.lower().endswith('.pt'):
                    cached_stems.add(fname[:-3])

        # 三层交集过滤：缓存 .pt ∩ patches .png ∩ CSV标签
        self.samples = []
        all_x, all_y = [], []

        for fname in sorted(os.listdir(patches_dir)):
            if not fname.lower().endswith('.png'):
                continue
            stem = fname.replace('.png', '')
            if stem not in self.label_map:
                continue
            if stem not in cached_stems:
                continue
            x, y = parse_coordinates(fname)
            if x is None:
                continue
            targets = self.label_map[stem]
            self.samples.append((stem, x, y, targets))
            all_x.append(x)
            all_y.append(y)

        # 坐标统计（归一化到 [0, n_pos-1]）
        if coord_stats is not None:
            self.x_min, self.x_max = coord_stats['x_min'], coord_stats['x_max']
            self.y_min, self.y_max = coord_stats['y_min'], coord_stats['y_max']
        else:
            self.x_min = min(all_x) if all_x else 0
            self.x_max = max(all_x) if all_x else 1
            self.y_min = min(all_y) if all_y else 0
            self.y_max = max(all_y) if all_y else 1

        print(f"[OmiCLIPTokensDataset] 加载 {len(self.samples)} 个样本 from {patches_dir}")
        print(f"  特征缓存: {feature_cache_dir}")
        print(f"  坐标范围: x=[{self.x_min}, {self.x_max}], y=[{self.y_min}, {self.y_max}]")

    def get_coord_stats(self):
        return {'x_min': self.x_min, 'x_max': self.x_max,
                'y_min': self.y_min, 'y_max': self.y_max}

    def _coord_to_index(self, val, vmin, vmax):
        if vmax == vmin:
            return 0
        normalized = (val - vmin) / (vmax - vmin)
        return int(np.clip(normalized * (self.n_pos - 1), 0, self.n_pos - 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem, x, y, targets = self.samples[idx]

        # 加载 OmiCLIP token 特征
        pt_path = os.path.join(self.feature_cache_dir, f"{stem}.pt")
        tokens = torch.load(pt_path, map_location='cpu', weights_only=True)
        if isinstance(tokens, dict):
            tokens = tokens.get("tokens", tokens.get("feature", list(tokens.values())[0]))
        tokens = tokens.float()

        # 确保形状为 [255, 768]
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        assert tokens.dim() == 2 and tokens.shape[1] == OMICLIP_FEATURE_DIM, (
            f"OmiCLIP token维度不匹配: 期望 [*, {OMICLIP_FEATURE_DIM}], 实际 {tokens.shape}, stem={stem}"
        )

        # 坐标映射
        pos_x = self._coord_to_index(x, self.x_min, self.x_max)
        pos_y = self._coord_to_index(y, self.y_min, self.y_max)

        targets_t = torch.tensor(targets, dtype=torch.float32)
        return (tokens,
                torch.tensor(pos_x, dtype=torch.long),
                torch.tensor(pos_y, dtype=torch.long),
                targets_t)

    @classmethod
    def from_multiple_patients(cls, patient_configs, n_pos=128, n_targets=30):
        """多患者联合加载"""
        datasets = []
        coord_stats_dict = {}
        target_cols = None

        for i, config in enumerate(patient_configs):
            patient_name = config.get('patient_name', f'patient_{i}')
            print(f"\n[MultiPatient-OmiCLIP] 加载患者 {patient_name}...")
            dataset = cls(
                patches_dir=config['patches_dir'],
                feature_cache_dir=config['feature_cache_dir'],
                labels_csv=config['labels_csv'],
                target_cols=target_cols,
                n_pos=n_pos,
                n_targets=n_targets,
                coord_stats=None,
            )
            coord_stats_dict[patient_name] = dataset.get_coord_stats()
            if target_cols is None:
                target_cols = dataset.target_cols
            datasets.append(dataset)

        merged = ConcatDataset(datasets)
        total = sum(len(d) for d in datasets)
        print(f"\n[MultiPatient-OmiCLIP] 合并完成: {len(datasets)} 个患者, 共 {total} 个样本")
        return merged, coord_stats_dict, target_cols


# ═══════════════════════════════════════════════════════════════════════════
#  训练与评估
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, gradient_clip=1.0):
    """单 epoch 训练"""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for tokens, pos_x, pos_y, targets in loader:
        tokens  = tokens.to(device, non_blocking=True)
        pos_x   = pos_x.to(device, non_blocking=True)
        pos_y   = pos_y.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                preds = model(tokens, pos_x, pos_y)
                loss  = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(tokens, pos_x, pos_y)
            loss  = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()

        total_loss += loss.item() * tokens.size(0)
        all_preds.append(preds.detach().cpu())
        all_labels.append(targets.detach().cpu())

    n = len(loader.dataset)
    avg_loss = total_loss / n
    all_preds  = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_labels.numpy(), all_preds.numpy())
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """评估函数"""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for tokens, pos_x, pos_y, targets in loader:
        tokens  = tokens.to(device, non_blocking=True)
        pos_x   = pos_x.to(device, non_blocking=True)
        pos_y   = pos_y.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        preds = model(tokens, pos_x, pos_y)
        loss  = criterion(preds, targets)

        total_loss += loss.item() * tokens.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(targets.cpu())

    n = len(loader.dataset)
    avg_loss = total_loss / n
    all_preds  = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_labels.numpy(), all_preds.numpy())
    return avg_loss, metrics


# ═══════════════════════════════════════════════════════════════════════════
#  参数解析
# ═══════════════════════════════════════════════════════════════════════════

def build_argparser():
    p = argparse.ArgumentParser(description="HisToGene-OmiCLIP Token序列训练")

    # 模式
    p.add_argument("--patient", type=str, default=None,
                   help="单患者模式：患者名称（如 HYZ15040）")
    p.add_argument("--cross_patient", action="store_true", default=False,
                   help="跨患者模式")
    p.add_argument("--fold", type=int, choices=[1, 2, 3], default=1,
                   help="三折交叉验证: 1=JFX+LMZ→HYZ, 2=HYZ+LMZ→JFX, 3=HYZ+JFX→LMZ")
    p.add_argument("--dataset_name", type=str, default=None,
                   help="数据集名称（自动推导或手动指定）")

    # 模型超参（OmiCLIP 默认值）
    p.add_argument("--feature_dim",  type=int,   default=OMICLIP_FEATURE_DIM)
    p.add_argument("--num_tokens",   type=int,   default=OMICLIP_NUM_TOKENS)
    p.add_argument("--model_dim",    type=int,   default=1024)
    p.add_argument("--n_pos",        type=int,   default=128)
    p.add_argument("--n_targets",    type=int,   default=30)
    p.add_argument("--mlp_dim",      type=int,   default=2048)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--encoder_hidden_dim", type=int, default=512)
    p.add_argument("--n_encoder_layers", type=int, default=1)
    p.add_argument("--n_encoder_heads",  type=int, default=8)

    # 训练超参
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--num_epochs",   type=int,   default=150)
    p.add_argument("--early_stop_patience", type=int, default=20)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--gradient_clip", type=float, default=1.0)
    p.add_argument("--scheduler_patience", type=int, default=5)
    p.add_argument("--scheduler_factor",   type=float, default=0.5)
    p.add_argument("--num_workers",  type=int,   default=0)

    # 混合精度
    p.add_argument("--amp", action="store_true", default=True,
                   help="使用混合精度训练（仅 CUDA 生效）")

    # 断点续训
    p.add_argument("--resume", type=str, default=None,
                   help="从checkpoint恢复训练的路径")

    return p


# ═══════════════════════════════════════════════════════════════════════════
#  主训练流程
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = build_argparser().parse_args()

    # ── 模式校验与 dataset_name 推导 ─────────────────────────────────────
    if not args.patient and not args.cross_patient:
        print("[ERROR] 请指定 --patient <name> 或 --cross_patient")
        sys.exit(1)
    if args.patient and args.cross_patient:
        print("[ERROR] --patient 和 --cross_patient 不可同时使用")
        sys.exit(1)

    if args.dataset_name is None:
        if args.patient:
            args.dataset_name = f"{args.patient}_OmiCLIP"
        else:
            test_patient = get_fold_config(args.fold)["test"]
            train_patients = "+".join(get_fold_config(args.fold)["train"])
            args.dataset_name = f"CrossPatient_{train_patients}_to_{test_patient}_OmiCLIP"

    # ── 输出路径设置 ──────────────────────────────────────────────────────
    if args.patient:
        ckpt_dir = Path(get_output_dir(f"omiclip_{args.patient}"))
    else:
        ckpt_dir = Path(get_output_dir(f"omiclip_{args.dataset_name}"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history_csv = str(ckpt_dir / f"training_history_{args.dataset_name}.csv")
    best_ckpt = ckpt_dir / "best_histogene_omiclip.pth"
    resume_ckpt = ckpt_dir / "resume_histogene_omiclip.pth"

    # ── 打印信息 ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("HisToGene-OmiCLIP Token序列训练")
    print(f"  OmiCLIP: feature_dim={args.feature_dim}, num_tokens={args.num_tokens}")
    if args.patient:
        print(f"  模式: 单患者 ({args.patient})")
    else:
        fold_cfg = get_fold_config(args.fold)
        print(f"  模式: 跨患者 Fold {args.fold} ({'+'.join(fold_cfg['train'])} → {fold_cfg['test']})")
    print(f"  dataset_name: {args.dataset_name}")
    print(f"  输出目录: {ckpt_dir}")
    print("=" * 70)

    # ── 构建数据集 ────────────────────────────────────────────────────────
    if args.patient:
        pc = get_patient_paths(args.patient, backbone='omiclip')
        if pc is None or not os.path.isdir(pc.get('train_patches', '')):
            print(f"[ERROR] 未知患者: {args.patient}，可选: ['HYZ15040', 'JFX0729', 'LMZ12939']")
            sys.exit(1)

        # 检查路径
        for label, path in [
            ("train_patches", pc['train_patches']),
            ("val_patches", pc['val_patches']),
            ("cache_train", pc['cache_train']),
            ("cache_val", pc['cache_val']),
        ]:
            if not os.path.isdir(path):
                print(f"[ERROR] {label} 不存在: {path}")
                sys.exit(1)
        if not os.path.isfile(pc['labels_csv']):
            print(f"[ERROR] labels_csv 不存在: {pc['labels_csv']}")
            sys.exit(1)

        train_dataset = OmiCLIPTokensDataset(
            patches_dir=pc['train_patches'],
            feature_cache_dir=pc['cache_train'],
            labels_csv=pc['labels_csv'],
            n_pos=args.n_pos,
            n_targets=args.n_targets,
        )
        target_cols = train_dataset.target_cols
        train_coord_stats = train_dataset.get_coord_stats()

        val_dataset = OmiCLIPTokensDataset(
            patches_dir=pc['val_patches'],
            feature_cache_dir=pc['cache_val'],
            labels_csv=pc['labels_csv'],
            target_cols=target_cols,
            n_pos=args.n_pos,
            n_targets=args.n_targets,
            coord_stats=train_coord_stats,
        )
        coord_stats_dict = {f"{args.patient}_train": train_coord_stats}

    else:
        # 跨患者模式
        fold_cfg = get_fold_config(args.fold)
        train_patient_names = fold_cfg["train"]
        test_patient_name = fold_cfg["test"]

        # 训练集：训练患者的 train + val 全部数据
        train_configs = []
        for pname in train_patient_names:
            pc = get_patient_paths(pname, backbone='omiclip')
            for split, patches_key, cache_key in [
                ('train', 'train_patches', 'cache_train'),
                ('val', 'val_patches', 'cache_val'),
            ]:
                train_configs.append({
                    'patches_dir': pc[patches_key],
                    'labels_csv': pc['labels_csv'],
                    'feature_cache_dir': pc[cache_key],
                    'patient_name': f'{pname}_{split}',
                })

        train_dataset, coord_stats_dict, target_cols = \
            OmiCLIPTokensDataset.from_multiple_patients(
                patient_configs=train_configs,
                n_pos=args.n_pos,
                n_targets=args.n_targets,
            )

        # 测试集：测试患者的全部数据
        test_configs = []
        pc = get_patient_paths(test_patient_name, backbone='omiclip')
        for split, patches_key, cache_key in [
            ('train', 'train_patches', 'cache_train'),
            ('val', 'val_patches', 'cache_val'),
        ]:
            test_configs.append({
                'patches_dir': pc[patches_key],
                'labels_csv': pc['labels_csv'],
                'feature_cache_dir': pc[cache_key],
                'patient_name': f'{test_patient_name}_{split}',
            })

        val_dataset, test_coord_stats, _ = \
            OmiCLIPTokensDataset.from_multiple_patients(
                patient_configs=test_configs,
                n_pos=args.n_pos,
                n_targets=args.n_targets,
            )
        coord_stats_dict.update(test_coord_stats)

    print(f"\n[INFO] 训练集: {len(train_dataset)} 样本, 验证/测试集: {len(val_dataset)} 样本")

    # ── 设备 ──────────────────────────────────────────────────────────────
    _config = load_config()
    device = get_device(_config)
    print(f"[INFO] Using device: {device}")

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("[INFO] 混合精度训练已启用")

    # ── DataLoader ─────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ── 模型 ──────────────────────────────────────────────────────────────
    model = HisToGeneUNITokens(
        feature_dim=args.feature_dim,
        dim=args.model_dim,
        n_pos=args.n_pos,
        n_targets=args.n_targets,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        encoder_hidden_dim=args.encoder_hidden_dim,
        n_encoder_layers=args.n_encoder_layers,
        n_encoder_heads=args.n_encoder_heads,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] 模型参数量: {n_params:,} ({n_params/1e6:.2f}M)")

    # ── 损失 / 优化器 / 调度器 ────────────────────────────────────────────
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.scheduler_factor,
        patience=args.scheduler_patience, verbose=False
    )

    # ── 断点续训加载 ──────────────────────────────────────────────────────
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
        if 'history' in ckpt:
            history = ckpt['history']
        print(f"[INFO] 从 Epoch {start_epoch} 继续，best_val_loss={best_val_loss:.4f}")
        clear_pause_signal(_PROJECT_ROOT)

    # ── 训练循环 ──────────────────────────────────────────────────────────
    early_stopped = False
    val_label = "Test" if args.cross_patient else "Val"
    task_name = f"HisToGene-OmiCLIP_{args.dataset_name}"

    print("\n" + "=" * 90)
    print(f"开始训练 OmiCLIP | Epochs={args.num_epochs} | BS={args.batch_size} | LR={args.lr}")
    print(f"  训练集: {len(train_dataset)} 样本 | {val_label}集: {len(val_dataset)} 样本")
    print("=" * 90)

    current_epoch = 0
    try:
        for epoch in range(start_epoch, args.num_epochs + 1):
            current_epoch = epoch
            t0 = time.time()

            train_loss, train_m = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler,
                gradient_clip=args.gradient_clip)
            val_loss, val_m = evaluate(model, val_loader, criterion, device)

            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            elapsed = time.time() - t0

            print(
                f"Epoch [{epoch:3d}/{args.num_epochs}] "
                f"Train Loss: {train_loss:.4f} PCC: {train_m['pcc']:.4f} | "
                f"{val_label} Loss: {val_loss:.4f} PCC: {val_m['pcc']:.4f} R²: {val_m['r2']:.4f} | "
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
                    'coord_stats_dict': coord_stats_dict,
                    'target_cols': target_cols,
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'best_epoch': best_epoch,
                    'best_pcc': best_pcc,
                }, best_ckpt)
                print(f"  -> 最佳模型已保存 (val_pcc={val_m['pcc']:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop_patience:
                    print(f"\n早停触发！连续 {args.early_stop_patience} 个 epoch val_loss 未改善。")
                    early_stopped = True
                    break

            # 每 10 个 epoch 保存一次历史
            if epoch % 10 == 0:
                pd.DataFrame(history).to_csv(history_csv, index=False)

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
                    'coord_stats_dict': coord_stats_dict,
                    'target_cols': target_cols,
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'history': history,
                    'best_epoch': best_epoch,
                    'best_pcc': best_pcc,
                }, resume_ckpt)
                print(f"[INFO] 暂停 checkpoint 已保存: {resume_ckpt}")
                notify_training_complete(task_name, epoch, best_epoch, best_pcc, "paused")
                clear_pause_signal(_PROJECT_ROOT)
                return

    except Exception as e:
        notify_training_error(task_name, current_epoch, str(e))
        raise

    # ── 训练完成 ──────────────────────────────────────────────────────────
    status = "early_stop" if early_stopped else "completed"
    notify_training_complete(task_name, current_epoch, best_epoch, best_pcc, status)

    # 最终保存历史
    pd.DataFrame(history).to_csv(history_csv, index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("训练完成 Summary")
    print("=" * 70)
    print(f"  模型: HisToGene-OmiCLIP (feature_dim={args.feature_dim}, num_tokens={args.num_tokens})")
    print(f"  数据集: {args.dataset_name}")
    print(f"  总 Epoch: {current_epoch} ({'早停' if early_stopped else '完整训练'})")
    print(f"  最佳 Epoch: {best_epoch}")
    print(f"  Best {val_label} PCC: {best_pcc:.4f}")
    print(f"  Best {val_label} Loss: {best_val_loss:.4f}")
    print(f"  最佳模型: {best_ckpt}")
    print(f"  训练历史: {history_csv}")
    print("=" * 70)

    # ── 加载最佳模型推理 + 保存预测结果 ───────────────────────────────────
    try:
        print("\n[INFO] 加载最佳模型进行推理...")
        best_ckpt_data = torch.load(best_ckpt, weights_only=False, map_location=device)
        model.load_state_dict(best_ckpt_data['model_state_dict'])
        model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for tokens, pos_x, pos_y, targets in val_loader:
                tokens = tokens.to(device, non_blocking=True)
                pos_x = pos_x.to(device, non_blocking=True)
                pos_y = pos_y.to(device, non_blocking=True)
                preds = model(tokens, pos_x, pos_y)
                all_preds.append(preds.cpu())
                all_labels.append(targets.cpu())

        preds_cat = torch.cat(all_preds).numpy()
        labels_cat = torch.cat(all_labels).numpy()

        # 生成 predictions.csv
        pred_df = pd.DataFrame()
        for i, col in enumerate(target_cols):
            pred_df[f'true_{col}'] = labels_cat[:, i]
            pred_df[f'pred_{col}'] = preds_cat[:, i]

        predictions_csv_path = str(ckpt_dir / "predictions.csv")
        pred_df.to_csv(predictions_csv_path, index=False)
        print(f"[OK] 预测结果已保存: {predictions_csv_path}")

        # 生成可视化报告
        try:
            from visualize_results import generate_full_report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            vis_dir = str(ckpt_dir / f"results_vis_{timestamp}")
            os.makedirs(vis_dir, exist_ok=True)

            # 复制 predictions.csv 到可视化目录
            shutil.copy2(predictions_csv_path, os.path.join(vis_dir, "predictions.csv"))

            model_name = f"HisToGene-OmiCLIP_{args.dataset_name}"
            generate_full_report(
                model_name=model_name,
                history_csv=history_csv,
                predictions_csv=predictions_csv_path,
                output_dir=vis_dir,
                prefix=args.dataset_name,
                actual_output_dir=vis_dir,
                params={
                    "方案": "OmiCLIP Token序列",
                    "feature_dim": args.feature_dim,
                    "num_tokens": args.num_tokens,
                    "batch_size": args.batch_size,
                    "num_epochs": args.num_epochs,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "model_dim": args.model_dim,
                    "n_pos": args.n_pos,
                    "n_targets": args.n_targets,
                    "mlp_dim": args.mlp_dim,
                    "dropout": args.dropout,
                    "encoder_hidden_dim": args.encoder_hidden_dim,
                    "n_encoder_layers": args.n_encoder_layers,
                    "n_encoder_heads": args.n_encoder_heads,
                    "early_stop_patience": args.early_stop_patience,
                    "dataset_name": args.dataset_name,
                }
            )
            # 复制训练历史
            shutil.copy2(history_csv, os.path.join(vis_dir, os.path.basename(history_csv)))
            print(f"[OK] 可视化结果已生成: {vis_dir}/")
        except Exception as e:
            print(f"[WARNING] 可视化报告生成失败: {e}")

    except Exception as e:
        print(f"[WARNING] 推理或结果保存失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
