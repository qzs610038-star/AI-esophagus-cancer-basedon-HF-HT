"""
HisToGene-UNI Token序列训练脚本（方案B）
========================================
使用 UNI2-h 完整 token 序列代替池化特征，通过轻量 Transformer 编码器处理。

支持模式：
  1. 单患者模式：--patient HYZ15040
  2. 跨患者模式：--cross_patient（默认 fold 1: JFX+LMZ训练→HYZ测试）
  3. 三折交叉验证：--cross_patient --fold {1,2,3}

约束：
  - 不修改 histogene/ 目录下的任何文件
  - 复用 dataset_uni_tokens / model_uni_tokens
  - 可视化输出与现有训练脚本一致
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

from dataset_uni_tokens import HisToGeneUNITokensDataset
from model_uni_tokens import HisToGeneUNITokens
from histogene.utils import compute_metrics
from notify_utils import (
    notify_training_complete, notify_training_error,
    check_pause_signal, clear_pause_signal,
)
from config_utils import load_config, get_device

# 忽略 Ctrl+C 信号，防止误触中断训练
signal.signal(signal.SIGINT, signal.SIG_IGN)

# ═══════════════════════════════════════════════════════════════════════════
#  数据路径配置
# ═══════════════════════════════════════════════════════════════════════════

_PATCH_BASE = str(_PROJECT_ROOT / "data_new_3ST" / "patch_noov_spilt")
_SSGSEA_BASE = str(_PROJECT_ROOT / "data_new_3ST" / "ssGSEA_zscore")
_TOKEN_CACHE_BASE = str(_PROJECT_ROOT / "uni2h_cache_tokens")

PATIENT_CONFIG = {
    'HYZ15040': {
        'train_patches': os.path.join(_PATCH_BASE, "HYZ15040_noov_split", "train_patches"),
        'val_patches':   os.path.join(_PATCH_BASE, "HYZ15040_noov_split", "val_patches"),
        'labels_csv':    os.path.join(_SSGSEA_BASE, "HYZ15040_ssGSEA_zscore.csv"),
        'token_cache_train': os.path.join(_TOKEN_CACHE_BASE, "HYZ15040", "train"),
        'token_cache_val':   os.path.join(_TOKEN_CACHE_BASE, "HYZ15040", "val"),
    },
    'JFX0729': {
        'train_patches': os.path.join(_PATCH_BASE, "JFX0729_noov_split", "train_patches"),
        'val_patches':   os.path.join(_PATCH_BASE, "JFX0729_noov_split", "val_patches"),
        'labels_csv':    os.path.join(_SSGSEA_BASE, "JFX0729_ssGSEA_zscore.csv"),
        'token_cache_train': os.path.join(_TOKEN_CACHE_BASE, "JFX0729", "train"),
        'token_cache_val':   os.path.join(_TOKEN_CACHE_BASE, "JFX0729", "val"),
    },
    'LMZ12939': {
        'train_patches': os.path.join(_PATCH_BASE, "LMZ12939_noov_split", "train_patches"),
        'val_patches':   os.path.join(_PATCH_BASE, "LMZ12939_noov_split", "val_patches"),
        'labels_csv':    os.path.join(_SSGSEA_BASE, "LMZ12939_ssGSEA_zscore.csv"),
        'token_cache_train': os.path.join(_TOKEN_CACHE_BASE, "LMZ12939", "train"),
        'token_cache_val':   os.path.join(_TOKEN_CACHE_BASE, "LMZ12939", "val"),
    },
}

# 三折交叉验证配置
FOLD_CONFIGS = {
    1: {"train": ["JFX0729", "LMZ12939"], "test": "HYZ15040"},
    2: {"train": ["HYZ15040", "LMZ12939"], "test": "JFX0729"},
    3: {"train": ["HYZ15040", "JFX0729"], "test": "LMZ12939"},
}

# ═══════════════════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════════════════

def save_per_pathway_pcc_table(predictions_csv_path, output_dir):
    """从 predictions.csv 计算逐通路PCC并保存为CSV表格"""
    if not os.path.isfile(predictions_csv_path):
        print(f"[WARNING] predictions.csv 不存在: {predictions_csv_path}")
        return

    pred_df = pd.read_csv(predictions_csv_path)
    true_cols = [c for c in pred_df.columns if c.startswith("true_")]
    pathways = [c[5:] for c in true_cols]

    if not pathways:
        print("[WARNING] predictions.csv 中未找到 true_* 列，跳过逐通路PCC表格生成")
        return

    rows = []
    for pw in pathways:
        tc, pc = f"true_{pw}", f"pred_{pw}"
        if tc not in pred_df.columns or pc not in pred_df.columns:
            continue
        y_true = pred_df[tc].values
        y_pred = pred_df[pc].values

        if np.std(y_true) > 0 and np.std(y_pred) > 0:
            pcc = float(np.corrcoef(y_true, y_pred)[0, 1])
        else:
            pcc = float("nan")

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rows.append({"pathway": pw, "pcc": pcc, "r2": r2, "mae": mae})

    if not rows:
        print("[WARNING] 无有效通路数据，跳过逐通路PCC表格生成")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values("pcc", ascending=False, na_position="last").reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    output_path = os.path.join(output_dir, "per_pathway_pcc.csv")
    df.to_csv(output_path, index=False)
    print(f"[OK] 逐通路PCC表格已保存: {output_path}")


def generate_model_params_txt(args, n_params, history_df, output_path,
                              train_samples=None, val_samples=None):
    """训练结束后生成模型参数与结果摘要文本文件"""
    best_row = history_df.loc[history_df['val_loss'].idxmin()]
    best_epoch = int(best_row['epoch'])
    best_val_loss = best_row['val_loss']
    best_val_pcc = best_row['val_pcc']
    best_val_r2 = best_row['val_r2']

    last_row = history_df.iloc[-1]
    final_train_pcc = last_row['train_pcc']
    total_epochs = int(last_row['epoch'])

    overfit_gap = final_train_pcc - best_val_pcc
    train_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    is_cross = getattr(args, 'cross_patient', False)
    if is_cross:
        fold = getattr(args, 'fold', 1)
        fc = FOLD_CONFIGS.get(fold, FOLD_CONFIGS[1])
        mode_str = f"跨患者泛化训练 Fold {fold} ({'+'.join(fc['train'])}→{fc['test']})"
    else:
        mode_str = f"单患者训练 ({getattr(args, 'patient', 'N/A')})"

    lines = []
    lines.append("=" * 50)
    lines.append("HisToGene-UNI Token序列训练参数（方案B）")
    lines.append("=" * 50)
    lines.append(f"训练时间: {train_time}")
    lines.append(f"数据集: {args.dataset_name}")
    lines.append(f"训练模式: {mode_str}")
    if train_samples is not None:
        lines.append(f"训练样本: {train_samples}")
    if val_samples is not None:
        lines.append(f"验证/测试样本: {val_samples}")
    lines.append("")

    lines.append("=== 模型参数 ===")
    lines.append(f"方案: B (UNI Token序列)")
    lines.append(f"Token数量: 65 (lite模式)")
    lines.append(f"编码器隐藏层: {args.encoder_hidden_dim}")
    lines.append(f"编码器层数: {args.n_encoder_layers}")
    lines.append(f"编码器头数: {args.n_encoder_heads}")
    lines.append(f"特征维度: {args.feature_dim}")
    lines.append(f"模型维度: {args.model_dim}")
    lines.append(f"位置编码: {args.n_pos}")
    lines.append(f"目标通路数: {args.n_targets}")
    lines.append(f"MLP隐藏层: {args.mlp_dim}")
    lines.append(f"Dropout: {args.dropout}")
    if n_params >= 1e6:
        params_str = f"≈ {n_params / 1e6:.1f}M"
    elif n_params >= 1e3:
        params_str = f"≈ {n_params / 1e3:.1f}K"
    else:
        params_str = str(n_params)
    lines.append(f"总参数量: {params_str}")
    lines.append("")

    lines.append("=== 训练超参数 ===")
    train_param_defs = [
        ('epochs',        args.num_epochs,          '最大训练轮数（配合早停）'),
        ('batch_size',    args.batch_size,          '批大小'),
        ('learning_rate', args.lr,                  'AdamW 初始学习率'),
        ('weight_decay',  args.weight_decay,        'AdamW 权重衰减（L2正则化）'),
        ('optimizer',     'AdamW',                  '解耦正则化'),
        ('loss',          'HuberLoss',              'δ=1.0，对异常值鲁棒'),
        ('scheduler',     'ReduceLROnPlateau',      f'factor={args.scheduler_factor}, patience={args.scheduler_patience}'),
        ('gradient_clip', args.gradient_clip,        '梯度裁剪最大范数'),
        ('label_noise',   args.label_noise,          '标签高斯噪声（回归版label smoothing）'),
        ('early_stop',    f'patience {args.early_stop_patience}', '基于 val_loss'),
        ('AMP',           '启用' if args.amp else '未启用', '混合精度训练'),
    ]
    for name, val, desc in train_param_defs:
        lines.append(f"{name:<14} = {str(val):<12} # {desc}")
    lines.append("")

    lines.append("=== 训练结果 ===")
    lines.append(f"总 Epoch: {total_epochs}")
    lines.append(f"最佳 Epoch: {best_epoch}")
    lines.append(f"Best Val PCC: {best_val_pcc:.4f}")
    lines.append(f"Best Val R²: {best_val_r2:.4f}")
    lines.append(f"Best Val Loss: {best_val_loss:.4f}")
    lines.append(f"最终 Train PCC: {final_train_pcc:.4f}")
    lines.append(f"过拟合 Gap (PCC): {overfit_gap:.4f}")
    lines.append("")
    lines.append("=== 基线对比 ===")
    lines.append(f"方案A (HisToGene-UNI) Val PCC 基线: 0.577")
    lines.append(f"方案B (Token序列) Best Val PCC: {best_val_pcc:.4f}")
    diff = best_val_pcc - 0.577
    lines.append(f"差异: {diff:+.4f} ({'超过基线' if diff > 0 else '低于基线'})")

    content = "\n".join(lines) + "\n"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"[OK] 模型参数摘要已保存: {output_path}")


def train_one_epoch(model, loader, optimizer, criterion, device,
                    scaler=None, label_noise=0.0, gradient_clip=1.0):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for tokens, pos_x, pos_y, targets in loader:
        tokens  = tokens.to(device, non_blocking=True)
        pos_x   = pos_x.to(device, non_blocking=True)
        pos_y   = pos_y.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if label_noise > 0:
            targets = targets + torch.randn_like(targets) * label_noise

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
    all_preds  = torch.cat(all_preds,  dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_labels.numpy(), all_preds.numpy())
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
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
    all_preds  = torch.cat(all_preds,  dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_labels.numpy(), all_preds.numpy())
    return avg_loss, metrics


# ═══════════════════════════════════════════════════════════════════════════
#  参数解析
# ═══════════════════════════════════════════════════════════════════════════

def build_argparser():
    p = argparse.ArgumentParser(
        description="HisToGene-UNI Token序列训练（方案B）"
    )

    # 模式
    p.add_argument("--patient", type=str, default=None,
                   help="单患者模式：患者名称（如 HYZ15040）")
    p.add_argument("--cross_patient", action="store_true", default=False,
                   help="跨患者模式：JFX+LMZ训练→HYZ测试")
    p.add_argument("--fold", type=int, choices=[1, 2, 3], default=1,
                   help="三折交叉验证编号: 1=JFX+LMZ→HYZ, 2=HYZ+LMZ→JFX, 3=HYZ+JFX→LMZ")
    p.add_argument("--dataset_name", type=str, default=None,
                   help="数据集名称（自动推导或手动指定）")

    # Token缓存
    p.add_argument("--token_cache_dir", type=str, default="uni2h_cache_tokens",
                   help="Token缓存根目录")

    # 模型超参
    p.add_argument("--feature_dim",  type=int,   default=1536)
    p.add_argument("--model_dim",    type=int,   default=1024)
    p.add_argument("--n_pos",        type=int,   default=128)
    p.add_argument("--n_targets",    type=int,   default=30)
    p.add_argument("--mlp_dim",      type=int,   default=2048)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--encoder_hidden_dim", type=int, default=512,
                   help="Token编码器隐藏层维度")
    p.add_argument("--n_encoder_layers", type=int, default=1,
                   help="Token编码器Transformer层数")
    p.add_argument("--n_encoder_heads",  type=int, default=8,
                   help="Token编码器注意力头数")

    # 训练超参
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--num_epochs",   type=int,   default=150)
    p.add_argument("--early_stop_patience", type=int, default=20)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_noise",  type=float, default=0.0)
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
            args.dataset_name = f"{args.patient}_UNI_tokens"
        else:
            # --fold 1 时保持后向兼容
            if args.fold == 1:
                args.dataset_name = "CrossPatient_JFX_LMZ_to_HYZ_UNI_tokens"
            else:
                test_patient = FOLD_CONFIGS[args.fold]["test"]
                args.dataset_name = f"CrossPatient_Fold{args.fold}_to_{test_patient}_UNI_tokens"

    # ── 路径设置 ──────────────────────────────────────────────────────────
    _histogene_dir = str(_PROJECT_ROOT / "histogene")
    ckpt_dir = Path(_histogene_dir) / "checkpoints" / args.dataset_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_csv = os.path.join(_histogene_dir, f"training_history_{args.dataset_name}.csv")
    best_ckpt = ckpt_dir / "best_histogene_uni_tokens.pth"
    resume_ckpt = ckpt_dir / "resume_histogene_uni_tokens.pth"

    # ── 打印信息 ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("HisToGene-UNI Token序列训练（方案B）")
    if args.patient:
        print(f"  模式: 单患者 ({args.patient})")
    else:
        fold_cfg = FOLD_CONFIGS[args.fold]
        train_desc = "+".join(fold_cfg["train"])
        test_desc = fold_cfg["test"]
        print(f"  模式: 跨患者 Fold {args.fold} ({train_desc} → {test_desc})")
    print(f"  dataset_name: {args.dataset_name}")
    print(f"  Token编码器: hidden={args.encoder_hidden_dim}, layers={args.n_encoder_layers}, heads={args.n_encoder_heads}")
    print("=" * 70)

    # ── 构建数据集 ────────────────────────────────────────────────────────
    if args.patient:
        # 单患者模式
        pc = PATIENT_CONFIG.get(args.patient)
        if pc is None:
            print(f"[ERROR] 未知患者: {args.patient}，可选: {list(PATIENT_CONFIG.keys())}")
            sys.exit(1)

        # 检查路径
        for label, path in [
            ("train_patches", pc['train_patches']),
            ("val_patches", pc['val_patches']),
            ("token_cache_train", pc['token_cache_train']),
            ("token_cache_val", pc['token_cache_val']),
        ]:
            if not os.path.isdir(path):
                print(f"[ERROR] {label} 不存在: {path}")
                sys.exit(1)
        if not os.path.isfile(pc['labels_csv']):
            print(f"[ERROR] labels_csv 不存在: {pc['labels_csv']}")
            sys.exit(1)

        train_dataset = HisToGeneUNITokensDataset(
            patches_dir=pc['train_patches'],
            feature_cache_dir=pc['token_cache_train'],
            labels_csv=pc['labels_csv'],
            n_pos=args.n_pos,
            n_targets=args.n_targets,
        )
        target_cols = train_dataset.target_cols
        train_coord_stats = train_dataset.get_coord_stats()

        val_dataset = HisToGeneUNITokensDataset(
            patches_dir=pc['val_patches'],
            feature_cache_dir=pc['token_cache_val'],
            labels_csv=pc['labels_csv'],
            target_cols=target_cols,
            n_pos=args.n_pos,
            n_targets=args.n_targets,
            coord_stats=train_coord_stats,
        )
        coord_stats_dict = {f"{args.patient}_train": train_coord_stats}

    else:
        # 跨患者模式：基于 fold 配置动态加载
        fold_cfg = FOLD_CONFIGS[args.fold]
        train_patient_names = fold_cfg["train"]
        test_patient_name = fold_cfg["test"]

        train_patient_configs = []
        for pname in train_patient_names:
            pc = PATIENT_CONFIG[pname]
            for split, patches_key, cache_key in [
                ('train', 'train_patches', 'token_cache_train'),
                ('val', 'val_patches', 'token_cache_val'),
            ]:
                train_patient_configs.append({
                    'patches_dir': pc[patches_key],
                    'labels_csv': pc['labels_csv'],
                    'feature_cache_dir': pc[cache_key],
                    'patient_name': f'{pname}_{split}',
                })

        train_dataset, coord_stats_dict, target_cols = \
            HisToGeneUNITokensDataset.from_multiple_patients(
                patient_configs=train_patient_configs,
                n_pos=args.n_pos,
                n_targets=args.n_targets,
            )

        # 测试集：测试患者全部数据
        test_patient_configs = []
        pc = PATIENT_CONFIG[test_patient_name]
        for split, patches_key, cache_key in [
            ('train', 'train_patches', 'token_cache_train'),
            ('val', 'val_patches', 'token_cache_val'),
        ]:
            test_patient_configs.append({
                'patches_dir': pc[patches_key],
                'labels_csv': pc['labels_csv'],
                'feature_cache_dir': pc[cache_key],
                'patient_name': f'{test_patient_name}_{split}',
            })

        val_dataset, test_coord_stats, _ = \
            HisToGeneUNITokensDataset.from_multiple_patients(
                patient_configs=test_patient_configs,
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
    criterion = nn.HuberLoss(delta=1.0)
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
    task_name = f"HisToGene-UNI-Tokens_{args.dataset_name}"

    print("\n" + "=" * 90)
    print(f"开始训练 方案B (UNI Token序列) | Epochs={args.num_epochs} | BS={args.batch_size} | LR={args.lr}")
    print(f"  训练集: {len(train_dataset)} 样本 | {val_label}集: {len(val_dataset)} 样本")
    print(f"  方案A基线 Val PCC = 0.577")
    print("=" * 90)

    current_epoch = 0
    try:
        for epoch in range(start_epoch, args.num_epochs + 1):
            current_epoch = epoch
            t0 = time.time()

            train_loss, train_m = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler,
                label_noise=args.label_noise, gradient_clip=args.gradient_clip)
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
                baseline_diff = val_m['pcc'] - 0.577
                print(f"  ✓ 最佳模型已保存 (val_pcc={val_m['pcc']:.4f}, vs基线 {baseline_diff:+.4f})")
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

    # 训练完成通知
    status = "early_stop" if early_stopped else "completed"
    notify_training_complete(task_name, current_epoch, best_epoch, best_pcc, status)

    # 最终保存历史
    pd.DataFrame(history).to_csv(history_csv, index=False)
    print(f"\n[DONE] 训练结束。最佳 val_loss={best_val_loss:.4f}, best_pcc={best_pcc:.4f}")
    print(f"  最佳模型: {best_ckpt}")
    print(f"  训练历史: {history_csv}")
    print(f"  方案A基线 Val PCC = 0.577, 方案B Best Val PCC = {best_pcc:.4f}")

    # ── 训练结束后：加载最佳模型推理 + 可视化 ────────────────────────────
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

        # 可视化输出目录
        from visualize_results import generate_full_report
        output_vis_dir = str(Path(_histogene_dir) / "checkpoints" / "results_vis")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        actual_vis_dir = os.path.join(output_vis_dir, f"{args.dataset_name}_{timestamp}")
        os.makedirs(actual_vis_dir, exist_ok=True)

        # 保存 predictions.csv
        predictions_csv_path = os.path.join(actual_vis_dir, "predictions.csv")
        pred_df.to_csv(predictions_csv_path, index=False)
        print(f"[OK] 预测结果已保存: {predictions_csv_path}")

        # 生成完整可视化报告
        model_name_with_dataset = f"HisToGene-UNI-Tokens_{args.dataset_name}"
        generate_full_report(
            model_name=model_name_with_dataset,
            history_csv=history_csv,
            predictions_csv=predictions_csv_path,
            output_dir=output_vis_dir,
            prefix=args.dataset_name,
            actual_output_dir=actual_vis_dir,
            params={
                "方案": "B (UNI Token序列)",
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "label_noise": args.label_noise,
                "gradient_clip": args.gradient_clip,
                "scheduler_patience": args.scheduler_patience,
                "scheduler_factor": args.scheduler_factor,
                "feature_dim": args.feature_dim,
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

        # 逐通路PCC表格
        try:
            save_per_pathway_pcc_table(predictions_csv_path, actual_vis_dir)
        except Exception as e:
            print(f"[WARNING] 生成逐通路PCC表格失败: {e}")

        # 模型参数摘要
        try:
            history_df = pd.read_csv(history_csv)
            model_params_path = os.path.join(actual_vis_dir, "model_params.txt")
            generate_model_params_txt(
                args=args, n_params=n_params,
                history_df=history_df, output_path=model_params_path,
                train_samples=len(train_dataset), val_samples=len(val_dataset),
            )
        except Exception as e:
            print(f"[WARNING] 生成 model_params.txt 失败: {e}")

        # 复制训练历史到可视化目录
        try:
            history_dst = os.path.join(actual_vis_dir, os.path.basename(history_csv))
            if os.path.isfile(history_csv) and history_csv != history_dst:
                shutil.copy2(history_csv, history_dst)
                print(f"[OK] 训练历史已复制: {history_dst}")
        except Exception as e:
            print(f"[WARNING] 复制训练历史 CSV 失败: {e}")

        print(f"\n[OK] 完整可视化结果已生成到 {actual_vis_dir}/")

    except Exception as e:
        print(f"[WARNING] 推理或可视化生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
