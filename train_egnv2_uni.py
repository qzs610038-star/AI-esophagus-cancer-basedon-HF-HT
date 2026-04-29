"""
EGNv2 + UNI2-h 集成训练脚本
============================
将 EGN-v2 的 ResNet-50 特征替换为 UNI2-h 预提取特征（1536维）
支持：单患者训练、跨患者泛化训练

约束：
  - 不修改 egnv2/ 目录下的任何文件
  - 复用 egnv2 核心组件（EGNv2Model, graph_builder, exemplar_builder 等）
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
from torch.utils.data import ConcatDataset
from torch_geometric.data import Data

# ── 项目根目录 ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 导入 UNI 集成组件
from egnv2_uni_dataset import EGNv2UNIDataset
from egnv2_uni_model import create_egnv2_uni_model

# 复用 egnv2 核心组件（不修改 egnv2 代码）
from egnv2.model import ExemplarLibrary
from egnv2.utils import compute_metrics
from egnv2.graph_builder import build_spatial_graph
from egnv2.exemplar_builder import build_exemplar_library, compute_exemplar_agg_features
from notify_utils import (
    notify_training_complete, notify_training_error,
    check_pause_signal, clear_pause_signal,
)
from config_utils import load_config, get_device

# 忽略 Ctrl+C 信号，防止误触中断训练
signal.signal(signal.SIGINT, signal.SIG_IGN)

_EGNV2_DIR = str(_PROJECT_ROOT / "egnv2")


# ═══════════════════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════════════════

def resolve_patient_paths(patient, base_dir, cache_dir):
    """
    自动推导患者的 patches、labels、cache 路径
    支持两种 patches 目录结构：
      1. {base_dir}/{patient}/train_patches
      2. {base_dir}/patch_noov_spilt/{patient}_noov_split/train_patches
    """
    # labels CSV
    labels_csv = os.path.join(base_dir, "ssGSEA_zscore", f"{patient}_ssGSEA_zscore.csv")
    if patient == "HYZ15040":
        labels_csv_clean = os.path.join(base_dir, "ssGSEA_zscore",
                                        f"{patient}_ssGSEA_zscore_clean.csv")
        if os.path.isfile(labels_csv_clean):
            labels_csv = labels_csv_clean

    # patches: 尝试模式1
    train_patches = os.path.join(base_dir, patient, "train_patches")
    val_patches = os.path.join(base_dir, patient, "val_patches")

    if not os.path.isdir(train_patches):
        # 备选模式2
        alt_base = os.path.join(base_dir, "patch_noov_spilt", f"{patient}_noov_split")
        train_patches_alt = os.path.join(alt_base, "train_patches")
        val_patches_alt = os.path.join(alt_base, "val_patches")
        if os.path.isdir(train_patches_alt):
            train_patches = train_patches_alt
            val_patches = val_patches_alt

    # cache
    train_cache = os.path.join(cache_dir, patient, "train")
    val_cache = os.path.join(cache_dir, patient, "val")

    return {
        'train_patches': train_patches,
        'val_patches': val_patches,
        'labels_csv': labels_csv,
        'train_cache': train_cache,
        'val_cache': val_cache,
    }


def collect_from_dataset(dataset):
    """
    遍历 dataset，收集所有特征、坐标和目标

    Returns:
        features: (N, 1536) Tensor
        coords:   (N, 2) Tensor
        targets:  (N, 30) Tensor
    """
    all_features = []
    all_coords = []
    all_targets = []

    for i in range(len(dataset)):
        feat, rx, ry, tgt = dataset[i]
        all_features.append(feat)
        all_coords.append(torch.stack([rx, ry]))
        all_targets.append(tgt)

    features = torch.stack(all_features)
    coords = torch.stack(all_coords)
    targets = torch.stack(all_targets)
    return features, coords, targets


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
    lines.append("EGNv2-UNI 模型训练参数")
    lines.append("=" * 50)
    lines.append(f"训练时间: {train_time}")
    lines.append(f"数据集: {args.dataset_name}")
    if train_samples is not None:
        lines.append(f"训练样本: {train_samples}")
    if val_samples is not None:
        lines.append(f"验证/测试样本: {val_samples}")
    lines.append("")

    lines.append("--- 模型架构参数 ---")
    n_exemplars_str = str(args.n_exemplars) if args.n_exemplars > 0 else "全量"
    model_param_defs = [
        ('backbone',     'UNI2-h',            '预提取 1536 维特征'),
        ('graph_layers', args.graph_layers,   'GraphSAGE 层数'),
        ('hidden_dim',   args.hidden_dim,     '图卷积隐藏维度'),
        ('k_neighbors',  args.k_neighbors,    'Exemplar KNN k值'),
        ('n_exemplars',  n_exemplars_str,     '代表库大小'),
        ('radius',       args.radius,         '空间图构建半径'),
        ('n_targets',    args.n_targets,      '预测通路数'),
        ('dropout',      args.dropout,        'Dropout 比率'),
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
        ('batch_size',    args.batch_size,          '特征提取阶段批大小（本脚本未使用）'),
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
    lines.append(f"Best Val PCC: {best_val_pcc:.4f}")
    lines.append(f"Best Val R²: {best_val_r2:.4f}")
    lines.append(f"Best Val Loss: {best_val_loss:.4f}")
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
        description="EGNv2-UNI 训练脚本：使用 UNI2-h 预提取特征训练 EGN-v2"
    )

    # 数据集名称（默认 None → 根据 patient/cross_patient 自动推导）
    p.add_argument("--dataset_name", type=str, default=None,
                   help="数据集名称，用于区分训练结果。"
                        "默认自动推导：单患者→{patient}_UNI，跨患者→CrossPatient_{train}_to_{test}_UNI")

    # ── 单患者模式参数 ────────────────────────────────────────────────────
    p.add_argument("--patient", type=str, default=None,
                   help="单患者模式：患者名称")
    p.add_argument("--feature_cache_dir", type=str, default=None,
                   help="单患者模式：UNI2-h 特征缓存根目录（如 uni2h_cache/HYZ15040）")
    p.add_argument("--patches_dir", type=str, default=None,
                   help="单患者模式：patches 根目录（如 data_new_3ST/HYZ15040）")
    p.add_argument("--labels_csv", type=str, default=None,
                   help="单患者模式：标签 CSV 路径")

    # ── 跨患者模式参数 ────────────────────────────────────────────────────
    p.add_argument("--cross_patient", action="store_true", default=False,
                   help="启用跨患者泛化训练模式")
    p.add_argument("--train_patients", type=str, nargs="+", default=None,
                   help="跨患者模式：训练集患者列表")
    p.add_argument("--test_patient", type=str, default=None,
                   help="跨患者模式：测试集患者名称")
    p.add_argument("--base_dir", type=str, default=str(_PROJECT_ROOT / "data_new_3ST"),
                   help="跨患者模式：数据根目录")
    p.add_argument("--cache_dir", type=str, default=str(_PROJECT_ROOT / "uni2h_cache"),
                   help="跨患者模式：UNI 特征缓存根目录")

    # 输出路径
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="checkpoint 保存目录，默认为 egnv2/checkpoints/{dataset_name}")
    p.add_argument("--history_csv", type=str, default=None,
                   help="训练历史 CSV 路径，默认为 egnv2/training_history_{dataset_name}.csv")

    # 训练超参
    p.add_argument("--batch_size",   type=int,   default=64,
                   help="保留参数，本脚本直接从缓存加载特征，不经过特征提取")
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

    # ── 自动推导 dataset_name ─────────────────────────────────────────────
    if args.dataset_name is None:
        if args.cross_patient:
            if args.train_patients and args.test_patient:
                train_short = "_".join(args.train_patients)
                args.dataset_name = f"CrossPatient_{train_short}_to_{args.test_patient}_UNI"
            else:
                args.dataset_name = "CrossPatient_UNI"
        elif args.patient:
            args.dataset_name = f"{args.patient}_UNI"
        else:
            args.dataset_name = "UNKNOWN_UNI"
        print(f"[INFO] dataset_name 自动推导为: {args.dataset_name}")

    # ── 设置默认输出路径 ──────────────────────────────────────────────────
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(_EGNV2_DIR, "checkpoints", args.dataset_name)
    if args.history_csv is None:
        args.history_csv = os.path.join(_EGNV2_DIR, f"training_history_{args.dataset_name}.csv")

    # ── 模式选择与数据集加载 ──────────────────────────────────────────────
    if args.cross_patient:
        # ===== 跨患者模式 =====
        if not args.train_patients or not args.test_patient:
            print("[ERROR] 跨患者模式需要 --train_patients 和 --test_patient")
            sys.exit(1)

        print("=" * 70)
        print("EGNv2-UNI 跨患者泛化训练")
        print(f"  训练集: {', '.join(args.train_patients)}")
        print(f"  测试集: {args.test_patient}")
        print(f"  dataset_name: {args.dataset_name}")
        print("=" * 70)

        # 构建训练集 patient_configs（每个患者的 train+val 合并）
        train_patient_configs = []
        for patient in args.train_patients:
            paths = resolve_patient_paths(patient, args.base_dir, args.cache_dir)
            # train split
            train_patient_configs.append({
                'feature_cache_dir': paths['train_cache'],
                'patches_dir': paths['train_patches'],
                'labels_csv': paths['labels_csv'],
                'patient_name': f"{patient}_train",
                'split': 'train',
            })
            # val split
            train_patient_configs.append({
                'feature_cache_dir': paths['val_cache'],
                'patches_dir': paths['val_patches'],
                'labels_csv': paths['labels_csv'],
                'patient_name': f"{patient}_val",
                'split': 'val',
            })

        train_dataset, target_cols = EGNv2UNIDataset.from_multiple_patients(
            patient_configs=train_patient_configs, verbose=True)

        # 构建测试集 patient_configs
        test_paths = resolve_patient_paths(args.test_patient, args.base_dir, args.cache_dir)
        test_patient_configs = [
            {
                'feature_cache_dir': test_paths['train_cache'],
                'patches_dir': test_paths['train_patches'],
                'labels_csv': test_paths['labels_csv'],
                'patient_name': f"{args.test_patient}_train",
                'split': 'train',
            },
            {
                'feature_cache_dir': test_paths['val_cache'],
                'patches_dir': test_paths['val_patches'],
                'labels_csv': test_paths['labels_csv'],
                'patient_name': f"{args.test_patient}_val",
                'split': 'val',
            },
        ]

        test_dataset, _ = EGNv2UNIDataset.from_multiple_patients(
            patient_configs=test_patient_configs, verbose=True)

        print(f"\n[INFO] 最终: 训练集 {len(train_dataset)} 样本, "
              f"测试集 {len(test_dataset)} 样本")

    else:
        # ===== 单患者模式 =====
        if args.patient is None or args.feature_cache_dir is None or args.labels_csv is None:
            print("[ERROR] 单患者模式需要 --patient, --feature_cache_dir, --labels_csv")
            sys.exit(1)

        print("=" * 70)
        print("EGNv2-UNI 单患者训练")
        print(f"  患者: {args.patient}")
        print(f"  dataset_name: {args.dataset_name}")
        print("=" * 70)

        # 自动推导 train/val 子目录
        train_cache = os.path.join(args.feature_cache_dir, "train")
        val_cache = os.path.join(args.feature_cache_dir, "val")

        train_patches = None
        val_patches = None
        if args.patches_dir:
            train_patches = os.path.join(args.patches_dir, "train_patches")
            val_patches = os.path.join(args.patches_dir, "val_patches")
            if not os.path.isdir(train_patches):
                train_patches = None
                val_patches = None

        # 检查路径
        for label, path in [("train_cache", train_cache), ("val_cache", val_cache)]:
            if not os.path.isdir(path):
                print(f"[ERROR] {label} 不存在: {path}")
                sys.exit(1)
        if not os.path.isfile(args.labels_csv):
            print(f"[ERROR] labels_csv 不存在: {args.labels_csv}")
            sys.exit(1)

        print(f"[INFO] Train cache: {train_cache}")
        print(f"[INFO] Val   cache: {val_cache}")
        print(f"[INFO] Labels CSV:  {args.labels_csv}")

        train_dataset = EGNv2UNIDataset(
            feature_cache_dir=train_cache,
            patches_dir=train_patches,
            labels_csv=args.labels_csv,
            split='train',
        )
        target_cols = train_dataset.target_cols

        test_dataset = EGNv2UNIDataset(
            feature_cache_dir=val_cache,
            patches_dir=val_patches,
            labels_csv=args.labels_csv,
            target_cols=target_cols,
            split='val',
        )

        print(f"\n[INFO] 最终: 训练集 {len(train_dataset)} 样本, "
              f"验证集 {len(test_dataset)} 样本")

    # ── 设备 ──────────────────────────────────────────────────────────────
    _config = load_config()
    device = get_device(_config)
    print(f"\n[INFO] Using device: {device}")

    # ── 直接从缓存收集特征、坐标、目标 ────────────────────────────────────
    print("\n" + "=" * 60)
    print("[INFO] 从 UNI2-h 缓存加载特征、坐标、标签...")
    print("=" * 60)

    train_features, train_coords, train_targets = collect_from_dataset(train_dataset)
    val_features, val_coords, val_targets = collect_from_dataset(test_dataset)

    print(f"[INFO] 训练集: features={train_features.shape}, coords={train_coords.shape}")
    print(f"[INFO] 验证集: features={val_features.shape}, coords={val_coords.shape}")

    # ── 构建空间图 ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[INFO] 构建空间图...")
    print("=" * 60)

    train_edge_index = build_spatial_graph(train_coords.cpu().numpy(), radius=args.radius)
    val_edge_index = build_spatial_graph(val_coords.cpu().numpy(), radius=args.radius)

    # ── 构建 Exemplar 库 ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[INFO] 构建 Exemplar 库...")
    print("=" * 60)

    n_exemplars = args.n_exemplars if args.n_exemplars > 0 else None
    exemplar_lib = build_exemplar_library(
        train_features.cpu(), train_targets.cpu(),
        n_exemplars=n_exemplars, method='full')

    # ── 计算 Exemplar 聚合特征 ────────────────────────────────────────────
    print("\n[INFO] 计算 Exemplar 聚合特征...")
    train_exemplar_agg, _ = compute_exemplar_agg_features(
        train_features.cpu(), exemplar_lib, args.hidden_dim,
        k=args.k_neighbors, device=device,
    )
    val_exemplar_agg, _ = compute_exemplar_agg_features(
        val_features.cpu(), exemplar_lib, args.hidden_dim,
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
    print(f"[INFO] Val   graph: {val_data.num_nodes} 节点, {val_data.num_edges} 边")

    # ── 模型 ──────────────────────────────────────────────────────────────
    model = create_egnv2_uni_model(
        hidden_dim=args.hidden_dim,
        output_dim=args.n_targets,
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
    best_ckpt = ckpt_dir / "best_egnv2_uni.pth"
    resume_ckpt = ckpt_dir / "resume_egnv2_uni.pth"

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
        print(f"[INFO] 从 checkpoint 恢复训练: {args.resume}")
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
    print(f"开始训练 EGNv2-UNI | Epochs={args.num_epochs} | LR={args.lr} | "
          f"GraphLayers={args.graph_layers} | Hidden={args.hidden_dim} | K={args.k_neighbors}")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(test_dataset)} 样本")
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

            # --- 验证 ---
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
                f"Val Loss: {val_loss.item():.4f} MAE: {val_m['mae']:.4f} "
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
                print(f"  ✓ 最佳模型已保存 (val_loss={val_loss.item():.4f}, val_pcc={val_m['pcc']:.4f})")
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
                    'val_loss': val_loss.item(),
                    'val_metrics': val_m,
                    'args': vars(args),
                    'target_cols': target_cols,
                    'history': history,
                    'best_epoch': best_epoch,
                    'best_pcc': best_pcc,
                }, resume_ckpt)
                print(f"[INFO] 暂停 checkpoint 已保存: {resume_ckpt}")
                notify_training_complete("EGNv2_UNI", epoch, best_epoch, best_pcc, "paused")
                clear_pause_signal(_PROJECT_ROOT)
                return

    except Exception as e:
        notify_training_error("EGNv2_UNI", current_epoch, str(e))
        raise

    # 训练完成通知
    status = "early_stop" if early_stopped else "completed"
    notify_training_complete("EGNv2_UNI", current_epoch, best_epoch, best_pcc, status)

    # 最终保存历史
    pd.DataFrame(history).to_csv(args.history_csv, index=False)
    print(f"\n[DONE] 训练结束。最佳 val_loss={best_val_loss:.4f}")
    print(f"  最佳模型: {best_ckpt}")
    print(f"  训练历史: {args.history_csv}")

    # ── 训练结束后：加载最佳模型，对验证集推理 ────────────────────────────
    try:
        print("\n[INFO] 加载最佳模型进行验证集推理...")
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
        model_name_with_dataset = f"EGNv2-UNI_{args.dataset_name}"

        # 创建时间戳目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        actual_vis_dir = os.path.join(output_vis_dir, f"{args.dataset_name}_{timestamp}")
        os.makedirs(actual_vis_dir, exist_ok=True)

        # 保存 predictions.csv 到可视化目录
        predictions_csv_path = os.path.join(actual_vis_dir, "predictions.csv")
        pred_df.to_csv(predictions_csv_path, index=False)
        print(f"[OK] 验证集预测结果已保存: {predictions_csv_path}")

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
                "early_stop_patience": args.early_stop_patience,
                "dataset_name": args.dataset_name,
            }
        )

        # 生成逐通路PCC表格
        try:
            save_per_pathway_pcc_table(predictions_csv_path, actual_vis_dir)
        except Exception as e:
            print(f"[WARNING] 生成逐通路PCC表格失败: {e}")

        # 生成模型参数摘要文件 model_params.txt
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
        print(f"[WARNING] 验证集推理或可视化生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
