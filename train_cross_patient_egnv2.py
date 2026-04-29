"""
EGNv2 跨患者泛化训练脚本
========================
训练集：JFX0729（train+val合并）+ LMZ12939（train+val合并）
测试集：HYZ15040（train+val合并）
dataset_name: CrossPatient_JFX_LMZ_to_HYZ

约束：
  - 不修改 egnv2/ 目录下的任何文件
  - 复用 egnv2 核心组件（EGNv2Dataset, EGNv2Model, preprocess_and_cache 等）
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
import torchvision.transforms as transforms
from torch_geometric.data import Data

# ── 项目根目录 ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 导入 egnv2 核心组件（不修改 egnv2 代码）
from egnv2.model import EGNv2Model, ResNetFeatureExtractor, ExemplarLibrary
from egnv2.dataset import EGNv2Dataset
from egnv2.utils import compute_metrics
from egnv2.exemplar_builder import (
    preprocess_and_cache, compute_exemplar_agg_features,
)
from egnv2.graph_builder import build_spatial_graph
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

# 基础路径
_PATCH_BASE = str(_PROJECT_ROOT / "data_new_3ST" / "patch_noov_spilt")
_SSGSEA_BASE = str(_PROJECT_ROOT / "data_new_3ST" / "ssGSEA_zscore")

# JFX0729
JFX_TRAIN_DIR = os.path.join(_PATCH_BASE, "JFX0729_noov_split", "train_patches")
JFX_VAL_DIR   = os.path.join(_PATCH_BASE, "JFX0729_noov_split", "val_patches")
JFX_CSV       = os.path.join(_SSGSEA_BASE, "JFX0729_ssGSEA_zscore.csv")

# LMZ12939
LMZ_TRAIN_DIR = os.path.join(_PATCH_BASE, "LMZ12939_noov_split", "train_patches")
LMZ_VAL_DIR   = os.path.join(_PATCH_BASE, "LMZ12939_noov_split", "val_patches")
LMZ_CSV       = os.path.join(_SSGSEA_BASE, "LMZ12939_ssGSEA_zscore.csv")

# HYZ15040（测试集）
HYZ_TRAIN_DIR = os.path.join(_PATCH_BASE, "HYZ15040_noov_split", "train_patches")
HYZ_VAL_DIR   = os.path.join(_PATCH_BASE, "HYZ15040_noov_split", "val_patches")
HYZ_CSV       = os.path.join(_SSGSEA_BASE, "HYZ15040_ssGSEA_zscore.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════════════════

def get_transforms(img_size=224, train=True):
    """构建图像变换流水线"""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    base = []
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


def save_per_pathway_pcc_table(predictions_csv_path, output_dir):
    """
    从 predictions.csv 计算逐通路PCC并保存为CSV表格

    输出文件：output_dir/per_pathway_pcc.csv
    表格格式：
    | pathway | pcc | r2 | mae | rank |
    |---------|-----|-----|-----|------|
    | tls     | 0.8 | 0.6 | 0.1 | 1    |
    | tgfb    | 0.7 | 0.5 | 0.2 | 2    |

    按PCC降序排列，便于挑选预测效果好的通路
    """
    if not os.path.isfile(predictions_csv_path):
        print(f"[WARNING] predictions.csv 不存在: {predictions_csv_path}")
        return

    pred_df = pd.read_csv(predictions_csv_path)

    # 推断通路名：从 true_xxx 列名中提取 xxx
    true_cols = [c for c in pred_df.columns if c.startswith("true_")]
    pathways = [c[5:] for c in true_cols]

    if not pathways:
        print("[WARNING] predictions.csv 中未找到 true_* 列，跳过逐通路PCC表格生成")
        return

    rows = []
    for pw in pathways:
        tc = f"true_{pw}"
        pc = f"pred_{pw}"
        if tc not in pred_df.columns or pc not in pred_df.columns:
            continue
        y_true = pred_df[tc].values
        y_pred = pred_df[pc].values

        # PCC
        if np.std(y_true) > 0 and np.std(y_pred) > 0:
            pcc = float(np.corrcoef(y_true, y_pred)[0, 1])
        else:
            pcc = float("nan")

        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        # MAE
        mae = float(np.mean(np.abs(y_true - y_pred)))

        rows.append({"pathway": pw, "pcc": pcc, "r2": r2, "mae": mae})

    if not rows:
        print("[WARNING] 无有效通路数据，跳过逐通路PCC表格生成")
        return

    df = pd.DataFrame(rows)
    # 按PCC降序排列
    df = df.sort_values("pcc", ascending=False, na_position="last").reset_index(drop=True)
    # 添加排名列
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

    lines = []
    lines.append("=" * 50)
    lines.append("EGNv2 跨患者泛化训练参数")
    lines.append("=" * 50)
    lines.append(f"训练时间: {train_time}")
    lines.append(f"数据集: {args.dataset_name}")
    lines.append(f"训练策略: JFX0729+LMZ12939 → HYZ15040")
    if train_samples is not None:
        lines.append(f"训练样本: {train_samples}")
    if val_samples is not None:
        lines.append(f"测试样本: {val_samples}")
    lines.append("")

    lines.append("--- 模型架构参数 ---")
    n_exemplars_str = str(args.n_exemplars) if args.n_exemplars > 0 else "全量"
    model_param_defs = [
        ('backbone',     'ResNet-50',         'ImageNet 预训练，前3层冻结'),
        ('graph_layers', args.graph_layers,   'GraphSAGE 层数'),
        ('hidden_dim',   args.hidden_dim,     '图卷积隐藏维度'),
        ('k_neighbors',  args.k_neighbors,    'Exemplar KNN k值'),
        ('n_exemplars',  n_exemplars_str,     '代表库大小'),
        ('radius',       args.radius,         '空间图构建半径'),
        ('n_targets',    args.n_targets,      '预测通路数'),
        ('dropout',      args.dropout,        'Dropout 比率'),
        ('freeze_layers', args.freeze_layers, 'ResNet 冻结层数'),
    ]
    for name, val, desc in model_param_defs:
        lines.append(f"{name:<16} = {str(val):<12} # {desc}")
    if n_params >= 1e6:
        params_str = f"≈ {n_params / 1e6:.1f}M"
    elif n_params >= 1e3:
        params_str = f"≈ {n_params / 1e3:.1f}K"
    else:
        params_str = str(n_params)
    lines.append(f"总参数量          {params_str}")
    lines.append("")

    lines.append("--- 训练超参数 ---")
    train_param_defs = [
        ('epochs',        args.num_epochs,          '最大训练轮数（配合早停）'),
        ('batch_size',    args.batch_size,          '特征提取阶段批大小'),
        ('learning_rate', args.lr,                  'AdamW 初始学习率'),
        ('optimizer',     'AdamW',                  'weight_decay=1e-4，解耦正则化'),
        ('loss',          'HuberLoss',              'δ=1.0，对异常值鲁棒'),
        ('scheduler',     'ReduceLROnPlateau',      'factor=0.5, patience=5'),
        ('early_stop',    f'patience {args.early_stop_patience}', '基于 val_loss'),
    ]
    for name, val, desc in train_param_defs:
        lines.append(f"{name:<16} = {str(val):<12} # {desc}")
    lines.append("")

    lines.append("--- 训练结果 ---")
    lines.append(f"总 Epoch: {total_epochs}")
    lines.append(f"最佳 Epoch: {best_epoch}")
    lines.append(f"Best Test PCC: {best_val_pcc:.4f}")
    lines.append(f"Best Test R²: {best_val_r2:.4f}")
    lines.append(f"Best Test Loss: {best_val_loss:.4f}")
    lines.append(f"最终 Train PCC: {final_train_pcc:.4f}")
    lines.append(f"过拟合 Gap (PCC): {overfit_gap:.4f}")

    content = "\n".join(lines) + "\n"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"[OK] 模型参数摘要已保存: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  参数解析
# ═══════════════════════════════════════════════════════════════════════════

def build_argparser():
    p = argparse.ArgumentParser(
        description="EGNv2 跨患者泛化训练: JFX0729+LMZ12939 → HYZ15040"
    )

    # 数据集名称
    p.add_argument("--dataset_name", type=str,
                   default="CrossPatient_JFX_LMZ_to_HYZ",
                   help="数据集名称，用于区分训练结果")

    # 输出路径
    _egnv2_dir = str(_PROJECT_ROOT / "egnv2")
    p.add_argument("--checkpoint_dir", type=str,
                   default=os.path.join(_egnv2_dir, "checkpoints", "CrossPatient_JFX_LMZ_to_HYZ"),
                   help="checkpoint 保存目录")
    p.add_argument("--history_csv", type=str,
                   default=os.path.join(_egnv2_dir, "training_history_CrossPatient_JFX_LMZ_to_HYZ.csv"),
                   help="训练历史 CSV 路径")
    p.add_argument("--cache_dir", type=str,
                   default=os.path.join(_egnv2_dir, "cache", "CrossPatient_JFX_LMZ_to_HYZ"),
                   help="特征/图缓存目录")

    # 训练超参
    p.add_argument("--batch_size",   type=int,   default=64,
                   help="特征提取阶段批大小")
    p.add_argument("--num_epochs",   type=int,   default=150)
    p.add_argument("--lr",           type=float, default=1e-4)

    # EGNv2 专属参数
    p.add_argument("--n_targets",      type=int,   default=30)
    p.add_argument("--graph_layers",   type=int,   default=2,
                   help="GraphSAGE 层数")
    p.add_argument("--hidden_dim",     type=int,   default=512,
                   help="图卷积隐藏维度")
    p.add_argument("--k_neighbors",    type=int,   default=10,
                   help="Exemplar KNN 的 k 值")
    p.add_argument("--n_exemplars",    type=int,   default=0,
                   help="代表库大小，0=全量")
    p.add_argument("--radius",         type=float, default=300,
                   help="空间图构建半径")
    p.add_argument("--freeze_layers",  type=int,   default=3,
                   help="冻结 ResNet 前 N 层")
    p.add_argument("--dropout",        type=float, default=0.3)

    # 早停
    p.add_argument("--early_stop_patience", type=int, default=15)

    # 断点续训
    p.add_argument("--resume", type=str, default='',
                   help="resume checkpoint 路径")

    return p


# ═══════════════════════════════════════════════════════════════════════════
#  主训练流程
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = build_argparser().parse_args()

    # ── 路径检查 ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("EGNv2 跨患者泛化训练")
    print("  训练集: JFX0729(train+val) + LMZ12939(train+val)")
    print("  测试集: HYZ15040(train+val)")
    print("  dataset_name: " + args.dataset_name)
    print("=" * 70)

    required_dirs = [
        ("JFX train", JFX_TRAIN_DIR), ("JFX val", JFX_VAL_DIR),
        ("LMZ train", LMZ_TRAIN_DIR), ("LMZ val", LMZ_VAL_DIR),
        ("HYZ train", HYZ_TRAIN_DIR), ("HYZ val", HYZ_VAL_DIR),
    ]
    required_files = [
        ("JFX CSV", JFX_CSV), ("LMZ CSV", LMZ_CSV), ("HYZ CSV", HYZ_CSV),
    ]

    for label, path in required_dirs:
        if not os.path.isdir(path):
            print(f"[ERROR] {label} 不存在: {path}")
            sys.exit(1)
        # 统计文件数
        n_files = len([f for f in os.listdir(path) if f.lower().endswith('.png')])
        print(f"  {label}: {n_files} patches")

    for label, path in required_files:
        if not os.path.isfile(path):
            print(f"[ERROR] {label} 不存在: {path}")
            sys.exit(1)
    print(f"  JFX CSV: {JFX_CSV}")
    print(f"  LMZ CSV: {LMZ_CSV}")
    print(f"  HYZ CSV: {HYZ_CSV}")

    # ── 设备 ──────────────────────────────────────────────────────────────
    _config = load_config()
    device = get_device(_config)
    print(f"\n[INFO] Using device: {device}")

    # ── 数据集 ────────────────────────────────────────────────────────────
    train_transform = get_transforms(train=True)
    test_transform  = get_transforms(train=False)

    print("\n" + "=" * 60)
    print("[INFO] 加载训练集: JFX0729 + LMZ12939 全部数据")
    print("=" * 60)

    # 训练集：4个目录合并（JFX train + JFX val + LMZ train + LMZ val）
    train_patient_configs = [
        {'patches_dir': JFX_TRAIN_DIR, 'labels_csv': JFX_CSV, 'patient_name': 'JFX0729_train'},
        {'patches_dir': JFX_VAL_DIR,   'labels_csv': JFX_CSV, 'patient_name': 'JFX0729_val'},
        {'patches_dir': LMZ_TRAIN_DIR, 'labels_csv': LMZ_CSV, 'patient_name': 'LMZ12939_train'},
        {'patches_dir': LMZ_VAL_DIR,   'labels_csv': LMZ_CSV, 'patient_name': 'LMZ12939_val'},
    ]

    train_dataset, target_cols = EGNv2Dataset.from_multiple_patients(
        patient_configs=train_patient_configs,
        transform=train_transform,
    )

    print(f"\n[INFO] 训练集合并完成: {len(train_dataset)} 样本")

    print("\n" + "=" * 60)
    print("[INFO] 加载测试集: HYZ15040 全部数据")
    print("=" * 60)

    # 测试集：2个目录合并（HYZ train + HYZ val）
    test_patient_configs = [
        {'patches_dir': HYZ_TRAIN_DIR, 'labels_csv': HYZ_CSV, 'patient_name': 'HYZ15040_train'},
        {'patches_dir': HYZ_VAL_DIR,   'labels_csv': HYZ_CSV, 'patient_name': 'HYZ15040_val'},
    ]

    test_dataset, _ = EGNv2Dataset.from_multiple_patients(
        patient_configs=test_patient_configs,
        transform=test_transform,
    )

    print(f"\n[INFO] 测试集合并完成: {len(test_dataset)} 样本")
    print(f"\n[INFO] 最终: 训练集 {len(train_dataset)} 样本, 测试集 {len(test_dataset)} 样本")

    # ── 特征提取 + 图构建 + 代表库 ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[INFO] EGNv2 预处理: 特征提取 → 图构建 → 代表库构建")
    print("=" * 60)

    feature_extractor = ResNetFeatureExtractor(freeze_layers=args.freeze_layers).to(device)

    cached = preprocess_and_cache(
        dataset_name=args.dataset_name,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        feature_extractor=feature_extractor,
        device=device,
        cache_dir=args.cache_dir,
        n_exemplars=args.n_exemplars if args.n_exemplars > 0 else None,
        radius=args.radius,
    )

    train_features = cached['train_features']
    train_targets = cached['train_targets']
    train_edge_index = cached['train_edge_index']
    val_features = cached['val_features']
    val_targets = cached['val_targets']
    val_edge_index = cached['val_edge_index']
    exemplar_lib = cached['exemplar_lib']

    # ── 计算 Exemplar 聚合特征 ────────────────────────────────────────────
    print("\n[INFO] 计算 Exemplar 聚合特征...")
    train_exemplar_agg, _ = compute_exemplar_agg_features(
        train_features, exemplar_lib, args.hidden_dim,
        k=args.k_neighbors, device=device,
    )
    val_exemplar_agg, _ = compute_exemplar_agg_features(
        val_features, exemplar_lib, args.hidden_dim,
        k=args.k_neighbors, device=device,
    )

    # ── 构建 PyG Data 对象 ────────────────────────────────────────────────
    train_data = Data(
        x=train_features.to(device),
        edge_index=train_edge_index.to(device),
        y=train_targets.to(device),
    ).to(device)

    val_data = Data(
        x=val_features.to(device),
        edge_index=val_edge_index.to(device),
        y=val_targets.to(device),
    ).to(device)

    print(f"\n[INFO] Train graph: {train_data.num_nodes} 节点, {train_data.num_edges} 边")
    print(f"[INFO] Test  graph: {val_data.num_nodes} 节点, {val_data.num_edges} 边")

    # ── 模型 ──────────────────────────────────────────────────────────────
    model = EGNv2Model(
        in_dim=2048,
        hidden_dim=args.hidden_dim,
        n_targets=args.n_targets,
        graph_layers=args.graph_layers,
        dropout=args.dropout,
        k_exemplars=args.k_neighbors,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] 模型参数量: {n_params:,}")

    # ── 损失 / 优化器 / 调度器 ────────────────────────────────────────────
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # ── 检查点目录 ────────────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best_egnv2.pth"
    resume_ckpt = ckpt_dir / "resume_egnv2.pth"

    # ── 断点续训加载 ──────────────────────────────────────────────────────
    start_epoch = 1
    best_val_loss = float('inf')
    best_epoch = 0
    best_pcc = 0.0
    patience_counter = 0
    history = []
    _train_exemplar_agg_state = train_exemplar_agg
    _val_exemplar_agg_state = val_exemplar_agg

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

        if 'history' in ckpt:
            history = ckpt['history']

        print(f"[INFO] 从 Epoch {start_epoch} 继续，best_val_loss={best_val_loss:.4f}")
        clear_pause_signal(_PROJECT_ROOT)

    # ── 训练循环 ──────────────────────────────────────────────────────────
    early_stopped = False

    print("\n" + "=" * 90)
    print(f"开始训练 EGNv2 跨患者泛化 | Epochs={args.num_epochs} | LR={args.lr} | "
          f"GraphLayers={args.graph_layers} | Hidden={args.hidden_dim} | K={args.k_neighbors}")
    print(f"  训练集: JFX0729+LMZ12939 ({len(train_dataset)} 样本)")
    print(f"  测试集: HYZ15040 ({len(test_dataset)} 样本)")
    print("=" * 90)

    current_epoch = 0
    try:
        for epoch in range(start_epoch, args.num_epochs + 1):
            current_epoch = epoch
            t0 = time.time()

            # --- 训练 ---
            model.train()
            optimizer.zero_grad()
            preds = model(train_data.x, train_data.edge_index, _train_exemplar_agg_state)
            train_loss = criterion(preds, train_data.y)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_m = compute_metrics(train_data.y.detach().cpu(), preds.detach().cpu())

            # --- 测试（验证） ---
            model.eval()
            with torch.no_grad():
                val_preds = model(val_data.x, val_data.edge_index, _val_exemplar_agg_state)
                val_loss = criterion(val_preds, val_data.y)

            val_m = compute_metrics(val_data.y.cpu(), val_preds.cpu())

            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss.item())

            elapsed = time.time() - t0

            print(
                f"Epoch [{epoch:3d}/{args.num_epochs}] "
                f"Train Loss: {train_loss.item():.4f} MAE: {train_m['mae']:.4f} "
                f"R²: {train_m['r2']:.4f} PCC: {train_m['pcc']:.4f} | "
                f"Test Loss: {val_loss.item():.4f} MAE: {val_m['mae']:.4f} "
                f"R²: {val_m['r2']:.4f} PCC: {val_m['pcc']:.4f} | "
                f"LR: {current_lr:.2e} | {elapsed:.1f}s"
            )

            history.append({
                'epoch': epoch,
                'train_loss': train_loss.item(),
                'train_mae': train_m['mae'],
                'train_r2': train_m['r2'],
                'train_pcc': train_m['pcc'],
                'val_loss': val_loss.item(),
                'val_mae': val_m['mae'],
                'val_r2': val_m['r2'],
                'val_pcc': val_m['pcc'],
                'lr': current_lr,
            })

            # 保存最佳模型
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
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
                    'val_loss': val_loss.item(),
                    'val_metrics': val_m,
                    'args': vars(args),
                    'target_cols': target_cols,
                    'best_epoch': best_epoch,
                    'best_pcc': best_pcc,
                }, best_ckpt)
                print(f"  ✓ 最佳模型已保存 (test_loss={val_loss.item():.4f}, test_pcc={val_m['pcc']:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop_patience:
                    print(f"\n早停触发！连续 {args.early_stop_patience} 个 epoch test_loss 未改善。")
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
                    'val_loss': val_loss.item(),
                    'val_metrics': val_m,
                    'args': vars(args),
                    'target_cols': target_cols,
                    'history': history,
                    'best_epoch': best_epoch,
                    'best_pcc': best_pcc,
                }, resume_ckpt)
                print(f"[INFO] 暂停 checkpoint 已保存: {resume_ckpt}")
                notify_training_complete("EGNv2_CrossPatient", epoch, best_epoch, best_pcc, "paused")
                clear_pause_signal(_PROJECT_ROOT)
                return

    except Exception as e:
        notify_training_error("EGNv2_CrossPatient", current_epoch, str(e))
        raise

    # 训练完成通知
    status = "early_stop" if early_stopped else "completed"
    notify_training_complete("EGNv2_CrossPatient", current_epoch, best_epoch, best_pcc, status)

    # 最终保存历史
    pd.DataFrame(history).to_csv(args.history_csv, index=False)
    print(f"\n[DONE] 训练结束。最佳 test_loss={best_val_loss:.4f}")
    print(f"  最佳模型: {best_ckpt}")
    print(f"  训练历史: {args.history_csv}")

    # ── 训练结束后：加载最佳模型，对测试集推理 ────────────────────────────
    try:
        print("\n[INFO] 加载最佳模型进行测试集推理...")
        best_ckpt_data = torch.load(best_ckpt, weights_only=False, map_location=device)
        model.load_state_dict(best_ckpt_data['model_state_dict'])
        model.eval()

        with torch.no_grad():
            val_preds = model(val_data.x, val_data.edge_index, _val_exemplar_agg_state)

        preds_cat = val_preds.cpu().numpy()
        labels_cat = val_data.y.cpu().numpy()

        # 生成 predictions.csv（列名: true_{pathway}, pred_{pathway}）
        pred_df = pd.DataFrame()
        for i, col in enumerate(target_cols):
            pred_df[f'true_{col}'] = labels_cat[:, i]
            pred_df[f'pred_{col}'] = preds_cat[:, i]

        # 创建可视化输出目录
        from visualize_results import generate_full_report
        output_vis_dir = str(ckpt_dir.parent / "results_vis")
        model_name_with_dataset = f"EGNv2_{args.dataset_name}"

        # 创建时间戳目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        actual_vis_dir = os.path.join(output_vis_dir, f"{args.dataset_name}_{timestamp}")
        os.makedirs(actual_vis_dir, exist_ok=True)

        # 保存 predictions.csv 到可视化目录
        predictions_csv_path = os.path.join(actual_vis_dir, "predictions.csv")
        pred_df.to_csv(predictions_csv_path, index=False)
        print(f"[OK] 测试集预测结果已保存: {predictions_csv_path}")

        # 生成完整可视化报告
        n_exemplars_str = str(args.n_exemplars) if args.n_exemplars > 0 else "全量"
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
                "graph_layers": args.graph_layers,
                "hidden_dim": args.hidden_dim,
                "k_neighbors": args.k_neighbors,
                "n_exemplars": n_exemplars_str,
                "radius": args.radius,
                "n_targets": args.n_targets,
                "dropout": args.dropout,
                "freeze_layers": args.freeze_layers,
                "early_stop_patience": args.early_stop_patience,
                "dataset_name": args.dataset_name,
                "train_patients": "JFX0729+LMZ12939",
                "test_patient": "HYZ15040",
            }
        )

        # 生成逐通路PCC表格
        try:
            save_per_pathway_pcc_table(predictions_csv_path, actual_vis_dir)
        except Exception as e:
            print(f"[WARNING] 生成逐通路PCC表格失败: {e}")

        # 生成模型参数摘要文件
        try:
            history_df = pd.read_csv(args.history_csv)
            model_params_path = os.path.join(actual_vis_dir, "model_params.txt")
            generate_model_params_txt(
                args=args,
                n_params=n_params,
                history_df=history_df,
                output_path=model_params_path,
                train_samples=len(train_dataset),
                val_samples=len(test_dataset),
            )
        except Exception as e:
            print(f"[WARNING] 生成 model_params.txt 失败: {e}")

        # 复制 training_history CSV 到可视化目录
        try:
            history_src = args.history_csv
            history_dst = os.path.join(actual_vis_dir, os.path.basename(history_src))
            if os.path.isfile(history_src) and history_src != history_dst:
                shutil.copy2(history_src, history_dst)
                print(f"[OK] 训练历史已复制: {history_dst}")
        except Exception as e:
            print(f"[WARNING] 复制训练历史 CSV 失败: {e}")

        print(f"[OK] 完整可视化结果（含逐通路指标）已生成到 {actual_vis_dir}/")
    except Exception as e:
        print(f"[WARNING] 测试集推理或可视化生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
