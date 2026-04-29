"""
HisToGene-UNI 训练脚本 - PFMval 项目
完整训练流程：数据加载、UNI特征缓存、模型训练、验证、早停、保存
支持：断点续训、暂停信号检测
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

# 将项目根目录加入 sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from histogene.model_uni import HisToGeneUNI
from histogene.dataset_uni import HisToGeneUNIDataset
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


def ensure_features_cached(patches_dir, cache_dir, device, rebuild=False):
    """确保UNI2-h特征已缓存"""
    from uni2h.uni2h_utils import load_uni2h_backbone, extract_and_cache_features
    from config_utils import get_hf_config

    cache_path = Path(cache_dir)
    patches_path = Path(patches_dir)

    # 检查是否需要提取
    existing = list(cache_path.glob("*.pt")) if cache_path.exists() else []
    needed = list(patches_path.glob("*.png"))

    if len(existing) >= len(needed) and not rebuild:
        print(f"特征缓存已就绪: {len(existing)} 个文件")
        return

    print(f"正在提取UNI2-h特征: {len(needed)} 个patch...")
    hf_config = get_hf_config()
    backbone, transform, feat_dim = load_uni2h_backbone(
        token=hf_config.get('token'), device=device
    )
    cache_path.mkdir(parents=True, exist_ok=True)
    n_new = extract_and_cache_features(
        backbone, transform, str(patches_dir), str(cache_dir), device, rebuild
    )
    print(f"新提取 {n_new} 个特征，总计 {len(list(cache_path.glob('*.pt')))} 个")
    del backbone  # 释放显存
    torch.cuda.empty_cache()


def generate_model_params_txt(args, n_params, history_df, output_path,
                             train_samples=None, val_samples=None):
    """训练结束后生成模型参数与结果摘要文本文件"""
    # 从训练历史中提取最佳指标
    best_row = history_df.loc[history_df['val_loss'].idxmin()]
    best_epoch = int(best_row['epoch'])
    best_val_loss = best_row['val_loss']
    best_val_pcc = best_row['val_pcc']
    best_val_r2 = best_row['val_r2']

    # 最终 epoch 的训练指标
    last_row = history_df.iloc[-1]
    final_train_pcc = last_row['train_pcc']
    total_epochs = int(last_row['epoch'])

    # 过拟合 Gap
    overfit_gap = final_train_pcc - best_val_pcc

    # 训练完成时间
    train_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("=" * 50)
    lines.append("HisToGene-UNI 模型训练参数")
    lines.append("=" * 50)
    lines.append(f"训练时间: {train_time}")
    lines.append(f"数据集: {args.dataset_name}")
    if train_samples is not None:
        lines.append(f"训练样本: {train_samples}")
    if val_samples is not None:
        lines.append(f"验证样本: {val_samples}")
    lines.append("")

    lines.append("--- 模型架构参数 ---")
    model_param_defs = [
        ('feature_dim', args.feature_dim, 'UNI2-h 输出特征维度，1536 维'),
        ('model_dim',   args.model_dim,   '嵌入维度'),
        ('n_pos',       args.n_pos,       '坐标嵌入表大小'),
        ('n_targets',   args.n_targets,   '预测通路数'),
        ('mlp_dim',     args.mlp_dim,     'FFN 隐藏层，嵌入维度的 2 倍'),
        ('dropout',     args.dropout,     'Dropout 比率'),
    ]
    for name, val, desc in model_param_defs:
        lines.append(f"{name:<14} = {str(val):<12} # {desc}")
    # 总参数量
    if n_params >= 1e6:
        params_str = f"≈ {n_params / 1e6:.1f}M"
    elif n_params >= 1e3:
        params_str = f"≈ {n_params / 1e3:.1f}K"
    else:
        params_str = str(n_params)
    lines.append(f"总参数量        {params_str}")
    lines.append("")

    lines.append("--- 训练超参数 ---")
    train_param_defs = [
        ('epochs',        args.num_epochs,          '最大训练轮数（配合早停）'),
        ('batch_size',    args.batch_size,          '批大小，兼顾显存和梯度稳定性'),
        ('learning_rate', args.lr,                  'AdamW 初始学习率'),
        ('optimizer',     'AdamW',                  'weight_decay=1e-4，解耦正则化'),
        ('loss',          'HuberLoss',              'δ=1.0，对异常值鲁棒'),
        ('scheduler',     'ReduceLROnPlateau',      'factor=0.5, patience=5'),
        ('early_stop',    f'patience {args.early_stop_patience}', '基于 val_loss'),
        ('AMP',           '启用' if args.amp else '未启用', '混合精度训练'),
    ]
    for name, val, desc in train_param_defs:
        lines.append(f"{name:<14} = {str(val):<12} # {desc}")
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
    p = argparse.ArgumentParser(description="Train HisToGene-UNI on ssGSEA pathway scores.")
    p.add_argument("--dataset_name", type=str, default="HYZ15040_UNI",
                   help="数据集名称，用于区分不同数据集的训练结果")
    p.add_argument("--train_patches_dir", type=str, default=_DEFAULT_TRAIN)
    p.add_argument("--val_patches_dir",   type=str, default=_DEFAULT_VAL)
    p.add_argument("--labels_csv",        type=str, default=_DEFAULT_LABELS)
    p.add_argument("--checkpoint_dir",    type=str, default=None,
                   help="checkpoint 保存目录，默认为 histogene/checkpoints/{dataset_name}")
    p.add_argument("--history_csv",       type=str, default=None,
                   help="训练历史 CSV 路径，默认为 histogene/training_history_{dataset_name}.csv")
    p.add_argument("--feature_cache_dir", type=str, default=None,
                   help="UNI特征缓存目录，默认自动推断为 uni2h_cache/{基础名}/train 和 val")
    p.add_argument("--rebuild_cache", action="store_true", default=False,
                   help="强制重建UNI特征缓存")

    # 训练超参
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--num_epochs",   type=int,   default=150)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int,   default=0)
    p.add_argument("--early_stop_patience", type=int, default=15)

    # 模型超参
    p.add_argument("--feature_dim",  type=int,   default=1536)
    p.add_argument("--model_dim",    type=int,   default=1024)
    p.add_argument("--n_pos",        type=int,   default=128)
    p.add_argument("--n_targets",    type=int,   default=30)
    p.add_argument("--mlp_dim",      type=int,   default=2048)
    p.add_argument("--dropout",      type=float, default=0.3)

    # 混合精度
    p.add_argument("--amp", action="store_true", default=True,
                   help="使用混合精度训练（仅 CUDA 生效）")

    # 断点续训
    p.add_argument("--resume", type=str, default=None,
                   help="从checkpoint恢复训练的路径")

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


def _infer_base_name(dataset_name):
    """从dataset_name推断基础名用于缓存目录"""
    if dataset_name.endswith("_UNI"):
        return dataset_name[:-4]
    return dataset_name


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for features, pos_x, pos_y, targets in loader:
        features = features.to(device, non_blocking=True)
        pos_x   = pos_x.to(device, non_blocking=True)
        pos_y   = pos_y.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                preds = model(features, pos_x, pos_y)
                loss  = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(features, pos_x, pos_y)
            loss  = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * features.size(0)
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

    for features, pos_x, pos_y, targets in loader:
        features = features.to(device, non_blocking=True)
        pos_x   = pos_x.to(device, non_blocking=True)
        pos_y   = pos_y.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        preds = model(features, pos_x, pos_y)
        loss  = criterion(preds, targets)

        total_loss += loss.item() * features.size(0)
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

    # ── 推断缓存目录 ─────────────────────────────────────────────────────────
    base_name = _infer_base_name(args.dataset_name)
    if args.feature_cache_dir is None:
        train_cache_dir = str(_PROJECT_ROOT / "uni2h_cache" / base_name / "train")
        val_cache_dir = str(_PROJECT_ROOT / "uni2h_cache" / base_name / "val")
    else:
        train_cache_dir = args.feature_cache_dir
        val_cache_dir = args.feature_cache_dir

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

    # ── 确保UNI特征已缓存 ─────────────────────────────────────────────────────
    if not args.multi_patient:
        print(f"[INFO] 训练集特征缓存: {train_cache_dir}")
        ensure_features_cached(args.train_patches_dir, train_cache_dir, device, args.rebuild_cache)
        print(f"[INFO] 验证集特征缓存: {val_cache_dir}")
        ensure_features_cached(args.val_patches_dir, val_cache_dir, device, args.rebuild_cache)

    # ── 数据集 ────────────────────────────────────────────────────────────────
    # ── 多患者模式 vs 单患者模式 ────────────────────────────────────────────
    if args.multi_patient:
        # 多患者联合训练模式
        print("\n" + "=" * 60)
        print("[INFO] 多患者联合训练模式")
        print("=" * 60)

        # 参数校验
        if not args.patient_dirs or not args.patient_val_dirs or not args.patient_csvs:
            print("[ERROR] 多患者模式需要 --patient_dirs, --patient_val_dirs, --patient_csvs 参数")
            sys.exit(1)

        n_patients = len(args.patient_dirs)
        if len(args.patient_val_dirs) != n_patients or len(args.patient_csvs) != n_patients:
            print(f"[ERROR] 患者目录数量不一致: dirs={n_patients}, val_dirs={len(args.patient_val_dirs)}, csvs={len(args.patient_csvs)}")
            sys.exit(1)

        # 推断或使用提供的患者名称
        if args.patient_names:
            patient_names = args.patient_names
        else:
            # 从目录路径推断患者名称
            import re
            patient_names = []
            for dir_path in args.patient_dirs:
                match = re.search(r'(HYZ\d+|JFX\d+|LMZ\d+)', dir_path)
                if match:
                    patient_names.append(match.group(1))
                else:
                    patient_names.append(os.path.basename(os.path.dirname(dir_path)))

        # 构建患者配置列表
        train_configs = []
        val_configs = []
        for i in range(n_patients):
            p_name = patient_names[i]
            p_train_cache = str(_PROJECT_ROOT / "uni2h_cache" / p_name / "train")
            p_val_cache = str(_PROJECT_ROOT / "uni2h_cache" / p_name / "val")

            # 确保特征缓存
            ensure_features_cached(args.patient_dirs[i], p_train_cache, device, args.rebuild_cache)
            ensure_features_cached(args.patient_val_dirs[i], p_val_cache, device, args.rebuild_cache)

            train_configs.append({
                'patches_dir': args.patient_dirs[i],
                'labels_csv': args.patient_csvs[i],
                'patient_name': p_name,
                'feature_cache_dir': p_train_cache,
            })
            val_configs.append({
                'patches_dir': args.patient_val_dirs[i],
                'labels_csv': args.patient_csvs[i],
                'patient_name': p_name,
                'feature_cache_dir': p_val_cache,
            })

        print(f"[INFO] 训练集患者: {patient_names}")
        for cfg in train_configs:
            print(f"  - {cfg['patient_name']}: {cfg['patches_dir']}")

        # 创建合并的训练集
        train_dataset, coord_stats_dict, target_cols = HisToGeneUNIDataset.from_multiple_patients(
            patient_configs=train_configs,
            n_pos=args.n_pos,
        )

        # 创建合并的验证集（使用训练集的 target_cols 和各患者的 coord_stats）
        val_datasets = []
        for cfg in val_configs:
            patient_name = cfg['patient_name']
            ds = HisToGeneUNIDataset(
                feature_cache_dir=cfg['feature_cache_dir'],
                patches_dir=cfg['patches_dir'],
                labels_csv=cfg['labels_csv'],
                target_cols=target_cols,
                n_pos=args.n_pos,
                coord_stats=coord_stats_dict.get(patient_name),
            )
            val_datasets.append(ds)
        val_dataset = ConcatDataset(val_datasets)

        print(f"\n[INFO] 合并后: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本")

    else:
        # 单患者模式（原有逻辑）
        train_dataset = HisToGeneUNIDataset(
            feature_cache_dir=train_cache_dir,
            patches_dir=args.train_patches_dir,
            labels_csv=args.labels_csv,
            n_pos=args.n_pos,
        )
        coord_stats = train_dataset.get_coord_stats()

        val_dataset = HisToGeneUNIDataset(
            feature_cache_dir=val_cache_dir,
            patches_dir=args.val_patches_dir,
            labels_csv=args.labels_csv,
            n_pos=args.n_pos,
            coord_stats=coord_stats,
        )
        # 单患者模式：coord_stats_dict 格式统一
        coord_stats_dict = {args.dataset_name: coord_stats}
        target_cols = train_dataset.target_cols

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
    model = HisToGeneUNI(
        feature_dim=args.feature_dim,
        dim=args.model_dim,
        n_pos=args.n_pos,
        n_targets=args.n_targets,
        mlp_dim=args.mlp_dim,
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
    best_ckpt = ckpt_dir / "best_histogene_uni.pth"
    resume_ckpt = ckpt_dir / "resume_histogene_uni.pth"

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
    print(f"开始训练 HisToGene-UNI | Epochs={args.num_epochs} | BS={args.batch_size} | LR={args.lr}")
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
                    'coord_stats_dict': coord_stats_dict,
                    'target_cols': target_cols,
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
                    'coord_stats_dict': coord_stats_dict,
                    'target_cols': target_cols,
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'history': history,
                    'best_epoch': best_epoch,
                    'best_pcc': best_pcc,
                }, resume_ckpt)
                print(f"[INFO] 暂停 checkpoint 已保存: {resume_ckpt}")
                notify_training_complete("HisToGene-UNI", epoch, best_epoch, best_pcc, "paused")
                clear_pause_signal(_PROJECT_ROOT)
                return

    except Exception as e:
        notify_training_error("HisToGene-UNI", current_epoch, str(e))
        raise

    # 训练完成通知
    status = "early_stop" if early_stopped else "completed"
    notify_training_complete("HisToGene-UNI", current_epoch, best_epoch, best_pcc, status)

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
            for features, pos_x, pos_y, targets in val_loader:
                features = features.to(device, non_blocking=True)
                pos_x = pos_x.to(device, non_blocking=True)
                pos_y = pos_y.to(device, non_blocking=True)
                preds = model(features, pos_x, pos_y)
                all_preds.append(preds.cpu())
                all_labels.append(targets.cpu())

        preds_cat = torch.cat(all_preds).numpy()
        labels_cat = torch.cat(all_labels).numpy()

        # 生成 predictions.csv（真值和预测值对比）
        pred_df = pd.DataFrame()
        for i, col in enumerate(target_cols):
            pred_df[f'true_{col}'] = labels_cat[:, i]
            pred_df[f'pred_{col}'] = preds_cat[:, i]

        # 创建可视化输出目录并保存 predictions.csv
        from visualize_results import generate_full_report
        output_vis_dir = str(ckpt_dir.parent / "results_vis")
        model_name_with_dataset = f"HisToGene-UNI_{args.dataset_name}"

        # 创建时间戳目录
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
                "feature_dim": args.feature_dim,
                "model_dim": args.model_dim,
                "n_pos": args.n_pos,
                "n_targets": args.n_targets,
                "mlp_dim": args.mlp_dim,
                "dropout": args.dropout,
                "early_stop_patience": args.early_stop_patience,
                "dataset_name": args.dataset_name,
            }
        )
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
