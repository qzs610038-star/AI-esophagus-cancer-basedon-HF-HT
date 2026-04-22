"""
EGN-v1 训练脚本 - PFMval 项目
完整训练流程：数据加载、ViT特征提取、图构建(KNN/spatial/hybrid)、GCN模型训练、验证、早停、保存
支持：断点续训、暂停信号检测、多患者联合训练

与 EGNv2 训练脚本的关键差异：
- 特征提取器: ViT-Large (dim=1024) vs ResNet-50 (dim=2048)
- GNN: GCN (GCNConv) vs GraphSAGE (SAGEConv)
- 图构建: KNN 图 (特征相似性) vs 空间半径图
- hidden_dim: 1024 vs 512
- batch_size: 16 (ViT 显存大) vs 64
- lr: 1e-5 vs 1e-4
- dropout: 0.5 vs 0.3
"""

import argparse
import os
import sys
import time
import signal
import shutil
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torch_geometric.data import Data

# 将项目根目录加入 sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from egnv1.model import EGNv1Model, ViTFeatureExtractor, ExemplarLibrary
from egnv1.dataset import EGNv1Dataset
from egnv1.utils import compute_metrics
from egnv1.exemplar_builder import (
    preprocess_and_cache, compute_exemplar_agg_features,
)
from notify_utils import (
    notify_training_complete, notify_training_error,
    check_pause_signal, clear_pause_signal,
)

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
    lines.append("EGN-v1 模型训练参数")
    lines.append("=" * 50)
    lines.append(f"训练时间: {train_time}")
    lines.append(f"数据集: {args.dataset_name}")
    if train_samples is not None:
        lines.append(f"训练样本: {train_samples}")
    if val_samples is not None:
        lines.append(f"验证样本: {val_samples}")
    lines.append("")

    lines.append("--- 模型架构参数 ---")
    n_exemplars_str = str(args.n_exemplars) if args.n_exemplars > 0 else "全量"
    model_param_defs = [
        ('backbone',       'ViT-Large',         'patch32, depth=8, heads=16, dim=1024'),
        ('graph_type',     args.graph_type,      '图构建方式'),
        ('graph_layers',   args.graph_layers,    'GCN 层数'),
        ('hidden_dim',     args.hidden_dim,      'GCN 隐藏维度'),
        ('k_neighbors',    args.k_neighbors,     'Exemplar KNN k值 / KNN图k值'),
        ('n_exemplars',    n_exemplars_str,      '代表库大小'),
        ('radius',         args.radius,          '空间图构建半径'),
        ('n_targets',      args.n_targets,       '预测通路数'),
        ('dropout',        args.dropout,         'Dropout 比率'),
        ('freeze_layers',  args.freeze_layers,   'ViT 冻结前N层'),
    ]
    for name, val, desc in model_param_defs:
        lines.append(f"{name:<16} = {str(val):<12} # {desc}")
    # 总参数量
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
    lines.append(f"Best Val PCC: {best_val_pcc:.4f}")
    lines.append(f"Best Val R²: {best_val_r2:.4f}")
    lines.append(f"Best Val Loss: {best_val_loss:.4f}")
    lines.append(f"最终 Train PCC: {final_train_pcc:.4f}")
    lines.append(f"过拟合 Gap (PCC): {overfit_gap:.4f}")

    content = "\n".join(lines) + "\n"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"[OK] 模型参数摘要已保存: {output_path}")


def build_argparser():
    p = argparse.ArgumentParser(description="Train EGN-v1 on ssGSEA pathway scores.")

    # 数据路径
    p.add_argument("--dataset_name", type=str, default="HYZ15040",
                   help="数据集名称，用于区分不同数据集的训练结果")
    p.add_argument("--train_patches_dir", type=str, default=_DEFAULT_TRAIN)
    p.add_argument("--val_patches_dir",   type=str, default=_DEFAULT_VAL)
    p.add_argument("--labels_csv",        type=str, default=_DEFAULT_LABELS)
    p.add_argument("--checkpoint_dir",    type=str, default=None,
                   help="checkpoint 保存目录，默认为 egnv1/checkpoints/{dataset_name}")
    p.add_argument("--history_csv",       type=str, default=None,
                   help="训练历史 CSV 路径，默认为 egnv1/training_history_{dataset_name}.csv")

    # 训练超参
    p.add_argument("--batch_size",   type=int,   default=16,
                   help="特征提取阶段批大小（ViT 显存大，默认16）")
    p.add_argument("--num_epochs",   type=int,   default=150)
    p.add_argument("--lr",           type=float, default=1e-5,
                   help="AdamW 初始学习率（ViT 需要更小的学习率）")

    # --- EGN-v1 专属参数 ---
    p.add_argument("--backbone",       type=str,   default="vit",
                   help="backbone 类型 (默认 vit)")
    p.add_argument("--n_targets",      type=int,   default=30)
    p.add_argument("--graph_type",     type=str,   default="knn",
                   choices=['knn', 'spatial', 'hybrid'],
                   help="图构建方式: knn(特征相似性)/spatial(空间距离)/hybrid(混合)")
    p.add_argument("--graph_layers",   type=int,   default=2,
                   help="GCN 层数")
    p.add_argument("--hidden_dim",     type=int,   default=1024,
                   help="GCN 隐藏维度（匹配 ViT 输出维度）")
    p.add_argument("--k_neighbors",    type=int,   default=10,
                   help="KNN 图 k 值 / Exemplar KNN k 值")
    p.add_argument("--n_exemplars",    type=int,   default=0,
                   help="代表库大小，0=全量，>0=K-means 到该数量")
    p.add_argument("--radius",         type=float, default=300,
                   help="空间图构建半径")
    p.add_argument("--freeze_layers",  type=int,   default=4,
                   help="冻结 ViT 前 N 层")
    p.add_argument("--dropout",        type=float, default=0.5)
    p.add_argument("--weight_decay",   type=float, default=1e-4)

    # 早停
    p.add_argument("--early_stop_patience", type=int, default=15)

    # 断点续训
    p.add_argument("--resume", type=str, default='',
                   help="resume checkpoint 路径")

    # 缓存目录
    p.add_argument("--cache_dir", type=str, default=None,
                   help="特征/图缓存目录，默认为 egnv1/cache/{dataset_name}")

    # ─── 多患者联合训练模式 ───────────────────────────────────────────────────
    p.add_argument("--multi_patient", action="store_true", default=False,
                   help="启用多患者联合训练模式")
    p.add_argument("--patient_dirs", type=str, nargs="+", default=None,
                   help="多患者训练目录列表（多患者模式）")
    p.add_argument("--patient_val_dirs", type=str, nargs="+", default=None,
                   help="多患者验证目录列表（多患者模式）")
    p.add_argument("--patient_csvs", type=str, nargs="+", default=None,
                   help="多患者标签 CSV 列表（多患者模式）")
    p.add_argument("--patient_names", type=str, nargs="+", default=None,
                   help="患者名称列表（可选，默认从路径推断）")
    return p


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


def main():
    args = build_argparser().parse_args()

    # ── 根据数据集名称设置默认路径 ───────────────────────────────────────────
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(_SCRIPT_DIR / "checkpoints" / args.dataset_name)
    if args.history_csv is None:
        args.history_csv = str(_SCRIPT_DIR / f"training_history_{args.dataset_name}.csv")
    if args.cache_dir is None:
        args.cache_dir = str(_SCRIPT_DIR / "cache" / args.dataset_name)

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

    # ── 数据集 ────────────────────────────────────────────────────────────────
    train_transform = get_transforms(train=True)
    val_transform   = get_transforms(train=False)

    # ── 多患者模式 vs 单患者模式 ────────────────────────────────────────────
    if args.multi_patient:
        print("\n" + "=" * 60)
        print("[INFO] 多患者联合训练模式 (EGN-v1)")
        print("=" * 60)

        if not args.patient_dirs or not args.patient_val_dirs or not args.patient_csvs:
            print("[ERROR] 多患者模式需要 --patient_dirs, --patient_val_dirs, --patient_csvs 参数")
            sys.exit(1)

        n_patients = len(args.patient_dirs)
        if len(args.patient_val_dirs) != n_patients or len(args.patient_csvs) != n_patients:
            print(f"[ERROR] 患者目录数量不一致")
            sys.exit(1)

        if args.patient_names:
            patient_names = args.patient_names
        else:
            patient_names = []
            for dir_path in args.patient_dirs:
                match = re.search(r'(HYZ\d+|JFX\d+|LMZ\d+)', dir_path)
                if match:
                    patient_names.append(match.group(1))
                else:
                    patient_names.append(os.path.basename(os.path.dirname(dir_path)))

        train_configs = []
        val_configs = []
        for i in range(n_patients):
            train_configs.append({
                'patches_dir': args.patient_dirs[i],
                'labels_csv': args.patient_csvs[i],
                'patient_name': patient_names[i],
            })
            val_configs.append({
                'patches_dir': args.patient_val_dirs[i],
                'labels_csv': args.patient_csvs[i],
                'patient_name': patient_names[i],
            })

        print(f"[INFO] 训练集患者: {patient_names}")
        for cfg in train_configs:
            print(f"  - {cfg['patient_name']}: {cfg['patches_dir']}")

        train_dataset, target_cols = EGNv1Dataset.from_multiple_patients(
            patient_configs=train_configs,
            transform=train_transform,
        )

        val_datasets = []
        for cfg in val_configs:
            ds = EGNv1Dataset(
                patches_dir=cfg['patches_dir'],
                labels_csv=cfg['labels_csv'],
                target_cols=target_cols,
                transform=val_transform,
            )
            val_datasets.append(ds)
        val_dataset = ConcatDataset(val_datasets)

        print(f"\n[INFO] 合并后: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本")

    else:
        # 单患者模式
        train_dataset = EGNv1Dataset(
            patches_dir=args.train_patches_dir,
            labels_csv=args.labels_csv,
            transform=train_transform,
        )

        val_dataset = EGNv1Dataset(
            patches_dir=args.val_patches_dir,
            labels_csv=args.labels_csv,
            target_cols=train_dataset.target_cols,
            transform=val_transform,
        )
        target_cols = train_dataset.target_cols

    # ── 特征提取 + 图构建 + 代表库（缓存机制） ────────────────────────────────
    print("\n" + "=" * 60)
    print(f"[INFO] EGN-v1 预处理: ViT 特征提取 → {args.graph_type} 图构建 → 代表库构建")
    print("=" * 60)

    feature_extractor = ViTFeatureExtractor(freeze_layers=args.freeze_layers).to(device)

    cached = preprocess_and_cache(
        dataset_name=args.dataset_name,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        feature_extractor=feature_extractor,
        device=device,
        cache_dir=args.cache_dir,
        n_exemplars=args.n_exemplars if args.n_exemplars > 0 else None,
        graph_type=args.graph_type,
        k_neighbors=args.k_neighbors,
        radius=args.radius,
    )

    train_features = cached['train_features']
    train_targets = cached['train_targets']
    train_edge_index = cached['train_edge_index']
    val_features = cached['val_features']
    val_targets = cached['val_targets']
    val_edge_index = cached['val_edge_index']
    exemplar_lib = cached['exemplar_lib']

    # ── 计算 Exemplar 聚合特征 ────────────────────────────────────────────────
    print("\n[INFO] 计算 Exemplar 聚合特征...")
    train_exemplar_agg, _ = compute_exemplar_agg_features(
        train_features, exemplar_lib, args.hidden_dim,
        k=args.k_neighbors, device=device,
    )
    val_exemplar_agg, _ = compute_exemplar_agg_features(
        val_features, exemplar_lib, args.hidden_dim,
        k=args.k_neighbors, device=device,
    )

    # ── 构建 PyG Data 对象 ────────────────────────────────────────────────────
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

    # ── 模型 ──────────────────────────────────────────────────────────────────
    model = EGNv1Model(
        in_dim=args.hidden_dim,  # ViT 输出 1024，通过 feature_proj
        hidden_dim=args.hidden_dim,
        n_targets=args.n_targets,
        graph_layers=args.graph_layers,
        dropout=args.dropout,
        k_exemplars=args.k_neighbors,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] 模型参数量: {n_params:,}")

    # ── 损失 / 优化器 / 调度器 ────────────────────────────────────────────────
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 注意：不传 verbose 参数（新版 PyTorch 不兼容）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # ── 检查点目录 ────────────────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best_egnv1.pth"
    resume_ckpt = ckpt_dir / "resume_egnv1.pth"

    # ── 断点续训加载 ───────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_loss = float('inf')
    best_epoch = 0
    best_pcc = 0.0
    patience_counter = 0
    history = []
    # 缓存路径状态
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

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    early_stopped = False

    print("\n" + "=" * 90)
    print(f"开始训练 EGN-v1 | Epochs={args.num_epochs} | LR={args.lr} | "
          f"GraphType={args.graph_type} | GraphLayers={args.graph_layers} | "
          f"Hidden={args.hidden_dim} | K={args.k_neighbors} | Backbone={args.backbone}")
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
                print(f"  ✓ 最佳模型已保存 (val_loss={val_loss.item():.4f})")
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
                notify_training_complete("EGN-v1", epoch, best_epoch, best_pcc, "paused")
                clear_pause_signal(_PROJECT_ROOT)
                return

    except Exception as e:
        notify_training_error("EGN-v1", current_epoch, str(e))
        raise

    # 训练完成通知
    status = "early_stop" if early_stopped else "completed"
    notify_training_complete("EGN-v1", current_epoch, best_epoch, best_pcc, status)

    # 最终保存历史
    pd.DataFrame(history).to_csv(args.history_csv, index=False)
    print(f"\n[DONE] 训练结束。最佳 val_loss={best_val_loss:.4f}")
    print(f"  最佳模型: {best_ckpt}")
    print(f"  训练历史: {args.history_csv}")

    # ── 训练结束后：加载最佳模型，对验证集推理 ────────────────────────────────
    try:
        print("\n[INFO] 加载最佳模型进行验证集推理...")
        best_ckpt_data = torch.load(best_ckpt, weights_only=False, map_location=device)
        model.load_state_dict(best_ckpt_data['model_state_dict'])
        model.eval()

        with torch.no_grad():
            val_preds = model(val_data.x, val_data.edge_index, _val_exemplar_agg_state)

        preds_cat = val_preds.cpu().numpy()
        labels_cat = val_data.y.cpu().numpy()

        # 生成 predictions.csv（列名: true_{pathway}, pred_{pathway}，与 visualize_results.py 约定一致）
        pred_df = pd.DataFrame()
        for i, col in enumerate(target_cols):
            pred_df[f'true_{col}'] = labels_cat[:, i]
            pred_df[f'pred_{col}'] = preds_cat[:, i]

        # 创建可视化输出目录
        from visualize_results import generate_full_report
        output_vis_dir = str(ckpt_dir.parent / "results_vis")
        model_name_with_dataset = f"EGN-v1_{args.dataset_name}"

        # 创建时间戳目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        actual_vis_dir = os.path.join(output_vis_dir, f"{args.dataset_name}_{timestamp}")
        os.makedirs(actual_vis_dir, exist_ok=True)

        # 保存 predictions.csv 到可视化目录
        predictions_csv_path = os.path.join(actual_vis_dir, "predictions.csv")
        pred_df.to_csv(predictions_csv_path, index=False)
        print(f"[OK] 验证集预测结果已保存: {predictions_csv_path}")

        # 生成完整可视化报告（全局单次调用，避免重复存储）
        n_exemplars_str = str(args.n_exemplars) if args.n_exemplars > 0 else "全量"
        generate_full_report(
            model_name=model_name_with_dataset,
            history_csv=args.history_csv,
            predictions_csv=predictions_csv_path,
            output_dir=output_vis_dir,
            prefix=args.dataset_name,
            actual_output_dir=actual_vis_dir,
            params={
                "backbone": args.backbone,
                "graph_type": args.graph_type,
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
                "weight_decay": args.weight_decay,
                "early_stop_patience": args.early_stop_patience,
                "dataset_name": args.dataset_name,
            }
        )

        # 生成模型参数摘要文件 model_params.txt（输出到 results_vis 时间戳目录）
        try:
            history_df = pd.read_csv(args.history_csv)
            model_params_path = os.path.join(actual_vis_dir, "model_params.txt")
            generate_model_params_txt(
                args=args,
                n_params=n_params,
                history_df=history_df,
                output_path=model_params_path,
                train_samples=len(train_dataset),
                val_samples=len(val_dataset),
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
