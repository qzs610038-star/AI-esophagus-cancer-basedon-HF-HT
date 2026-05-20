"""
HisToGene-UNI Token + GAT 训练脚本
====================================
在方案B (Token序列) 基础上引入图注意力网络，建模patch间空间邻域关系。

支持模式：
  1. 三折跨患者交叉验证：--fold {1,2,3}
  2. 全部三折：--fold all

训练策略：
  - 两阶段训练：Stage 1 冻结encoder, Stage 2 解冻全部
  - 梯度累积：per-patient graph 累积
  - AMP混合精度
  - Early Stopping

约束：
  - 不修改任何现有文件
  - 可视化输出与现有脚本兼容
"""

import argparse
import os
import sys
import time
import json
import signal
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 项目根目录 ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model_uni_tokens_gat import HisToGeneUNITokensGAT, count_parameters
from dataset_graph_tokens import MultiSplitGraphTokenDataset, graph_collate_fn
from notify_utils import (
    notify_training_complete, notify_training_error,
    check_pause_signal, clear_pause_signal,
)
from config_utils import load_config, get_device, get_patient_paths, get_fold_config, get_output_dir, get_histogene_dir

# 忽略 Ctrl+C 信号，防止误触中断训练
signal.signal(signal.SIGINT, signal.SIG_IGN)

# ═══════════════════════════════════════════════════════════════════════════
#  数据路径配置 — 统一由 config_utils.get_patient_paths() 管理
#  本地使用默认路径，服务器通过 config.yaml 覆盖
# ═══════════════════════════════════════════════════════════════════════════

def _gat_patient_config(patient):
    """将 get_patient_paths() 输出转为 GAT 特有格式"""
    pc = get_patient_paths(patient, backbone='uni_tokens')
    return {
        'patch_dirs': [pc['train_patches'], pc['val_patches']],
        'csv_path': pc['labels_csv'],
        'cache_dirs': [pc['token_cache_train'], pc['token_cache_val']],
    }

def _gat_fold_config(fold_name):
    """将 GAT 的 'fold1'/'fold2'/'fold3' 转为标准 fold 配置"""
    fold_num = int(fold_name.replace('fold', ''))
    fc = get_fold_config(fold_num)
    return {
        'train_patients': fc['train'],
        'test_patient': fc['test'],
        'description': '+'.join(fc['train']) + ' → ' + fc['test'],
    }

# ═══════════════════════════════════════════════════════════════════════════
#  训练与评估函数
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, train_dataset, optimizer, device,
                    scaler=None, accumulation_steps=4, gradient_clip=1.0,
                    stage='stage2'):
    """
    训练一个epoch（全图模式，逐患者前向）

    Args:
        model: GAT模型
        train_dataset: MultiSplitGraphTokenDataset
        optimizer: 优化器
        device: 训练设备
        scaler: AMP GradScaler
        accumulation_steps: 梯度累积步数
        gradient_clip: 梯度裁剪范数
        stage: 'stage1' (冻结encoder) 或 'stage2' (全部可训练)

    Returns:
        avg_loss: 平均训练损失
        train_metrics: dict with 'pcc', 'mae', 'r2'
    """
    model.train()
    total_loss = 0.0
    n_samples_total = 0
    optimizer.zero_grad()
    all_preds = []
    all_labels = []

    n_patients = len(train_dataset)

    for i in range(n_patients):
        data = train_dataset[i]
        tokens_list = data['tokens']
        pos_x = data['pos_x'].to(device)
        pos_y = data['pos_y'].to(device)
        labels = data['labels'].to(device)
        edge_index = data['edge_index'].to(device)

        n_patches = labels.shape[0]

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                predictions = model(tokens_list, pos_x, pos_y, edge_index)
                loss = F.mse_loss(predictions, labels)
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
        else:
            predictions = model(tokens_list, pos_x, pos_y, edge_index)
            loss = F.mse_loss(predictions, labels)
            loss = loss / accumulation_steps
            loss.backward()

        total_loss += loss.item() * accumulation_steps * n_patches
        n_samples_total += n_patches
        all_preds.append(predictions.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        if (i + 1) % accumulation_steps == 0 or (i + 1) == n_patients:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                optimizer.step()
            optimizer.zero_grad()

    avg_loss = total_loss / max(n_samples_total, 1)

    # 计算训练指标: PCC, MAE, R2
    preds_cat = np.concatenate(all_preds, axis=0)
    labels_cat = np.concatenate(all_labels, axis=0)
    train_metrics = _compute_aggregate_metrics(preds_cat, labels_cat)

    return avg_loss, train_metrics


def _compute_aggregate_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """
    计算聚合指标：平均PCC、MAE、R²

    Args:
        preds: [N, n_targets] 预测值
        labels: [N, n_targets] 真实值

    Returns:
        dict with 'pcc', 'mae', 'r2'
    """
    # MAE: 所有通路平均绝对误差
    mae = float(np.mean(np.abs(preds - labels)))

    # R²: 整体 R² (多输出)
    ss_res = np.sum((labels - preds) ** 2)
    ss_tot = np.sum((labels - np.mean(labels, axis=0, keepdims=True)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # 逐通路PCC的平均值
    n_targets = preds.shape[1]
    pccs = []
    for j in range(n_targets):
        if np.std(preds[:, j]) > 1e-8 and np.std(labels[:, j]) > 1e-8:
            pcc = float(np.corrcoef(preds[:, j], labels[:, j])[0, 1])
            pccs.append(pcc if not np.isnan(pcc) else 0.0)
        else:
            pccs.append(0.0)
    mean_pcc = float(np.mean(pccs))

    return {'pcc': mean_pcc, 'mae': mae, 'r2': r2}


@torch.no_grad()
def evaluate(model, val_dataset, device):
    """
    评估模型性能

    Args:
        model: GAT模型
        val_dataset: 验证集 (MultiSplitGraphTokenDataset)
        device: 设备

    Returns:
        avg_loss: 平均验证损失
        val_metrics: dict with 'pcc', 'mae', 'r2'
        per_pathway_pccs: 逐通路PCC列表
        all_preds: 所有预测 numpy array [N_total, 30]
        all_labels: 所有标签 numpy array [N_total, 30]
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_samples_total = 0

    for i in range(len(val_dataset)):
        data = val_dataset[i]
        tokens_list = data['tokens']
        pos_x = data['pos_x'].to(device)
        pos_y = data['pos_y'].to(device)
        labels = data['labels'].to(device)
        edge_index = data['edge_index'].to(device)

        predictions = model(tokens_list, pos_x, pos_y, edge_index)
        loss = F.mse_loss(predictions, labels)

        n_patches = labels.shape[0]
        total_loss += loss.item() * n_patches
        n_samples_total += n_patches

        all_preds.append(predictions.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / max(n_samples_total, 1)

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # 计算聚合指标
    val_metrics = _compute_aggregate_metrics(preds, labels)

    # 逐通路PCC（用于详细分析）
    n_targets = preds.shape[1]
    pccs = []
    for j in range(n_targets):
        if np.std(preds[:, j]) > 1e-8 and np.std(labels[:, j]) > 1e-8:
            pcc = float(np.corrcoef(preds[:, j], labels[:, j])[0, 1])
            pccs.append(pcc if not np.isnan(pcc) else 0.0)
        else:
            pccs.append(0.0)

    return avg_loss, val_metrics, pccs, preds, labels


# ═══════════════════════════════════════════════════════════════════════════
#  参数解析
# ═══════════════════════════════════════════════════════════════════════════

def build_argparser():
    p = argparse.ArgumentParser(
        description="HisToGene-UNI Token + GAT 训练脚本（P0-3 GAT升级）"
    )

    # 模式
    p.add_argument("--fold", type=str, default='fold1',
                   choices=['fold1', 'fold2', 'fold3', 'all'],
                   help="三折交叉验证编号或'all'运行全部")

    # GAT超参
    p.add_argument("--gat_hidden", type=int, default=256,
                   help="GAT每头输出维度")
    p.add_argument("--gat_heads", type=int, default=4,
                   help="GAT注意力头数")
    p.add_argument("--gat_layers", type=int, default=2,
                   help="GAT层数")
    p.add_argument("--k_neighbors", type=int, default=6,
                   help="KNN图的k值")

    # Token encoder超参
    p.add_argument("--encoder_hidden_dim", type=int, default=512,
                   help="Token编码器隐藏层维度")
    p.add_argument("--num_encoder_heads", type=int, default=8,
                   help="Token编码器注意力头数")
    p.add_argument("--num_encoder_layers", type=int, default=1,
                   help="Token编码器层数")

    # 模型超参
    p.add_argument("--input_dim", type=int, default=1536)
    p.add_argument("--n_pos", type=int, default=128)
    p.add_argument("--n_targets", type=int, default=30)
    p.add_argument("--mlp_dim", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.2)

    # 训练超参
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Stage 1 学习率 (GAT + head)")
    p.add_argument("--lr_encoder", type=float, default=2e-5,
                   help="Stage 2 encoder 学习率")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--accumulation_steps", type=int, default=4,
                   help="梯度累积步数")
    p.add_argument("--gradient_clip", type=float, default=1.0)
    p.add_argument("--stage1_epochs", type=int, default=10,
                   help="Stage 1 (冻结encoder) 的epoch数")
    p.add_argument("--patience", type=int, default=5,
                   help="Early stopping patience")

    # 混合精度
    p.add_argument("--amp", action="store_true", default=True,
                   help="使用混合精度训练")

    # 设备
    p.add_argument("--device", type=str, default='cuda',
                   choices=['cuda', 'cpu', 'auto'])

    # 断点续训
    p.add_argument("--resume", type=str, default=None,
                   help="从checkpoint恢复训练")
    
    # 可视化再生成
    p.add_argument("--regenerate_vis", action="store_true", default=False,
                   help="仅从已有数据重新生成可视化，不进行训练")
    
    return p


# ═══════════════════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════════════════

def save_predictions_csv(preds, labels, target_cols, output_path):
    """保存预测结果CSV（列名格式与visualize_results.py兼容）"""
    pred_df = pd.DataFrame()
    for i, col in enumerate(target_cols):
        pred_df[f'true_{col}'] = labels[:, i]
        pred_df[f'pred_{col}'] = preds[:, i]
    pred_df.to_csv(output_path, index=False)
    print(f"[OK] 预测结果已保存: {output_path}")


def save_training_history(history, output_path):
    """保存训练历史CSV"""
    df = pd.DataFrame(history)
    df.to_csv(output_path, index=False)


def save_training_params(args, fold_name, n_params, output_path):
    """保存训练参数JSON"""
    params = {
        'model': 'HisToGene-UNI-Tokens-GAT',
        'fold': fold_name,
        'fold_description': _gat_fold_config(fold_name).get('description', fold_name),
        'total_parameters': n_params,
        'gat_hidden': args.gat_hidden,
        'gat_heads': args.gat_heads,
        'gat_layers': args.gat_layers,
        'k_neighbors': args.k_neighbors,
        'encoder_hidden_dim': args.encoder_hidden_dim,
        'num_encoder_heads': args.num_encoder_heads,
        'num_encoder_layers': args.num_encoder_layers,
        'input_dim': args.input_dim,
        'n_pos': args.n_pos,
        'n_targets': args.n_targets,
        'mlp_dim': args.mlp_dim,
        'dropout': args.dropout,
        'epochs': args.epochs,
        'lr': args.lr,
        'lr_encoder': args.lr_encoder,
        'weight_decay': args.weight_decay,
        'accumulation_steps': args.accumulation_steps,
        'gradient_clip': args.gradient_clip,
        'stage1_epochs': args.stage1_epochs,
        'patience': args.patience,
        'amp': args.amp,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"[OK] 训练参数已保存: {output_path}")


def save_per_pathway_pcc_table(preds, labels, target_cols, output_path):
    """计算并保存逐通路PCC表格"""
    rows = []
    for i, col in enumerate(target_cols):
        y_true = labels[:, i]
        y_pred = preds[:, i]
        if np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
            pcc = float(np.corrcoef(y_true, y_pred)[0, 1])
        else:
            pcc = float('nan')
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rows.append({'pathway': col, 'pcc': pcc, 'r2': r2, 'mae': mae})

    df = pd.DataFrame(rows)
    df = df.sort_values('pcc', ascending=False, na_position='last').reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    df.to_csv(output_path, index=False)
    print(f"[OK] 逐通路PCC表格已保存: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  单fold训练主逻辑
# ═══════════════════════════════════════════════════════════════════════════

def train_single_fold(args, fold_name: str, device: torch.device):
    """训练单个fold"""
    fold_cfg = _gat_fold_config(fold_name)
    train_patients = fold_cfg['train_patients']
    test_patient = fold_cfg['test_patient']

    print("\n" + "=" * 80)
    print(f"  GAT训练 - {fold_name}: {fold_cfg['description']}")
    print(f"  训练患者: {train_patients}, 测试患者: {test_patient}")
    print("=" * 80)

    # ── 构建数据集 ─────────────────────────────────────────────────────
    train_configs = []
    for pname in train_patients:
        pc = _gat_patient_config(pname)
        train_configs.append({
            'patient_name': pname,
            'patch_dirs': pc['patch_dirs'],
            'csv_path': pc['csv_path'],
            'cache_dirs': pc['cache_dirs'],
        })

    train_dataset = MultiSplitGraphTokenDataset(
        patient_configs=train_configs,
        n_pos=args.n_pos,
        k_neighbors=args.k_neighbors,
        split='train'
    )

    test_pc = _gat_patient_config(test_patient)
    test_configs = [{
        'patient_name': test_patient,
        'patch_dirs': test_pc['patch_dirs'],
        'csv_path': test_pc['csv_path'],
        'cache_dirs': test_pc['cache_dirs'],
    }]

    val_dataset = MultiSplitGraphTokenDataset(
        patient_configs=test_configs,
        n_pos=args.n_pos,
        k_neighbors=args.k_neighbors,
        split='test'
    )

    target_cols = train_dataset.target_cols or val_dataset.target_cols
    if target_cols is None:
        print("[ERROR] 无法获取target_cols")
        return

    train_patches = sum(p['n_patches'] for p in train_dataset.patients)
    val_patches = sum(p['n_patches'] for p in val_dataset.patients)
    print(f"\n[INFO] 训练: {len(train_dataset)} 患者/{train_patches} patches, "
          f"测试: {len(val_dataset)} 患者/{val_patches} patches")

    # ── 模型 ──────────────────────────────────────────────────────────
    model = HisToGeneUNITokensGAT(
        input_dim=args.input_dim,
        hidden_dim=args.encoder_hidden_dim,
        gat_hidden=args.gat_hidden,
        gat_heads=args.gat_heads,
        gat_layers=args.gat_layers,
        n_pos=args.n_pos,
        n_targets=args.n_targets,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        num_encoder_heads=args.num_encoder_heads,
        num_encoder_layers=args.num_encoder_layers,
    ).to(device)

    n_params = model.count_parameters()
    count_parameters(model)

    # ── 输出目录 ──────────────────────────────────────────────────────
    dataset_name = f"GAT_{fold_name}_{'+'.join(train_patients)}_to_{test_patient}"
    ckpt_dir = Path(get_output_dir(f"HisToGene_UNI_Tokens_GAT/{fold_name}"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = ckpt_dir / f"best_model_{fold_name}.pt"
    # 续训时使用独立的history CSV文件
    if args.resume:
        history_csv_path = ckpt_dir / f"training_history_{fold_name}_resume.csv"
    else:
        history_csv_path = ckpt_dir / f"training_history_{fold_name}.csv"
    params_json_path = ckpt_dir / f"training_params_{fold_name}.json"

    # ── 两阶段优化器设置 ──────────────────────────────────────────────
    # Stage 1: 冻结 token_encoder, 只训练 GAT + head
    def freeze_encoder():
        for param in model.token_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder():
        for param in model.token_encoder.parameters():
            param.requires_grad = True

    # ── AMP ──────────────────────────────────────────────────────────
    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("[INFO] 混合精度训练已启用")

    # ── 断点续训 ──────────────────────────────────────────────────────
    start_epoch = 1
    best_val_loss = float('inf')
    best_val_pcc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    if args.resume and os.path.isfile(args.resume):
        print(f"[INFO] 从checkpoint恢复: {args.resume}")
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        best_val_pcc = ckpt.get('best_val_pcc', 0.0)
        best_epoch = ckpt.get('best_epoch', 0)
        # 续训时重置patience计数器
        patience_counter = 0
        # 续训不加载旧history，使用独立CSV记录
        history = []
        print(f"[INFO] Resuming from epoch {start_epoch - 1}, will continue from epoch {start_epoch}")
        print(f"[INFO] best_pcc={best_val_pcc:.4f}, patience_counter reset to 0")
        clear_pause_signal(_PROJECT_ROOT)

    # ── 训练循环 ──────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"开始GAT训练 | Epochs={args.epochs} | Stage1={args.stage1_epochs}ep | "
          f"LR_GAT={args.lr} | LR_Enc={args.lr_encoder}")
    print(f"  GAT: hidden={args.gat_hidden}, heads={args.gat_heads}, "
          f"layers={args.gat_layers}, k={args.k_neighbors}")
    print(f"  基线 方案B PCC = 0.3812")
    print(f"{'='*90}")

    task_name = f"HisToGene_UNI_Tokens_GAT_{fold_name}"
    early_stopped = False
    current_epoch = 0

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            t0 = time.time()

            # 阶段切换
            if epoch <= args.stage1_epochs:
                stage = 'stage1'
                freeze_encoder()
                # Stage 1 优化器：只优化GAT和head
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(
                    trainable_params, lr=args.lr, weight_decay=args.weight_decay
                )
            else:
                stage = 'stage2'
                unfreeze_encoder()
                # Stage 2 优化器：差分学习率
                # 当首次进入Stage2 或 从Stage2断点续训时创建optimizer
                if epoch == args.stage1_epochs + 1 or (start_epoch > args.stage1_epochs and epoch == start_epoch):
                    optimizer = torch.optim.AdamW([
                        {'params': model.token_encoder.parameters(), 'lr': args.lr_encoder},
                        {'params': model.gat_convs.parameters(), 'lr': args.lr * 0.5},
                        {'params': model.gat_norms.parameters(), 'lr': args.lr * 0.5},
                        {'params': model.input_proj.parameters(), 'lr': args.lr * 0.5},
                        {'params': model.x_embed.parameters(), 'lr': args.lr * 0.5},
                        {'params': model.y_embed.parameters(), 'lr': args.lr * 0.5},
                        {'params': model.head.parameters(), 'lr': args.lr * 0.5},
                    ], weight_decay=args.weight_decay)

            # 训练
            train_loss, train_m = train_one_epoch(
                model, train_dataset, optimizer, device,
                scaler=scaler,
                accumulation_steps=args.accumulation_steps,
                gradient_clip=args.gradient_clip,
                stage=stage
            )

            # 评估
            val_loss, val_m, val_pccs, val_preds, val_labels = evaluate(
                model, val_dataset, device
            )

            val_pcc = val_m['pcc']
            elapsed = time.time() - t0

            print(
                f"Epoch [{epoch:3d}/{args.epochs}] [{stage}] "
                f"Train Loss: {train_loss:.4f} PCC: {train_m['pcc']:.4f} | "
                f"Val Loss: {val_loss:.4f} PCC: {val_pcc:.4f} "
                f"MAE: {val_m['mae']:.4f} R²: {val_m['r2']:.4f} | "
                f"{elapsed:.1f}s"
            )

            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_pcc': train_m['pcc'],
                'train_mae': train_m['mae'],
                'train_r2': train_m['r2'],
                'val_loss': val_loss,
                'val_pcc': val_pcc,
                'val_mae': val_m['mae'],
                'val_r2': val_m['r2'],
                'stage': stage,
                'lr': optimizer.param_groups[0]['lr'],
            })

            # 保存最佳模型（基于val_pcc）
            if val_pcc > best_val_pcc:
                best_val_pcc = val_pcc
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_pcc': best_val_pcc,
                    'best_epoch': best_epoch,
                    'val_pccs': val_pccs,
                    'val_metrics': val_m,
                    'args': vars(args),
                    'fold_name': fold_name,
                    'target_cols': target_cols,
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                }, best_ckpt_path)

                baseline_diff = val_pcc - 0.3812
                print(f"  ✓ 新最佳! PCC={val_pcc:.4f} (vs基线B {baseline_diff:+.4f})")
            else:
                patience_counter += 1
                # Stage 2 才计算early stopping
                if stage == 'stage2' and patience_counter >= args.patience:
                    print(f"\n早停触发！连续 {args.patience} epoch PCC未改善。")
                    early_stopped = True
                    break

            # 定期保存历史
            if epoch % 5 == 0:
                save_training_history(history, str(history_csv_path))

            # 暂停信号检测
            if check_pause_signal(_PROJECT_ROOT):
                print("\n[INFO] 检测到暂停信号，保存checkpoint...")
                resume_path = ckpt_dir / f"resume_{fold_name}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_pcc': best_val_pcc,
                    'best_epoch': best_epoch,
                    'patience_counter': patience_counter,
                    'history': history,
                    'args': vars(args),
                    'fold_name': fold_name,
                    'target_cols': target_cols,
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                }, resume_path)
                print(f"[INFO] 暂停checkpoint已保存: {resume_path}")
                notify_training_complete(task_name, epoch, best_epoch, best_val_pcc, "paused")
                clear_pause_signal(_PROJECT_ROOT)
                return

    except Exception as e:
        notify_training_error(task_name, current_epoch, str(e))
        raise

    # ── 训练结束 ──────────────────────────────────────────────────────
    status = "early_stop" if early_stopped else "completed"
    notify_training_complete(task_name, current_epoch, best_epoch, best_val_pcc, status)

    # 保存最终历史
    save_training_history(history, str(history_csv_path))

    print(f"\n[DONE] {fold_name} 训练结束。")
    print(f"  最佳 Epoch: {best_epoch}, Val PCC: {best_val_pcc:.4f}")
    print(f"  方案B基线 PCC = 0.3812, GAT Best PCC = {best_val_pcc:.4f}")
    print(f"  改进: {best_val_pcc - 0.3812:+.4f}")

    # ── 加载最佳模型，生成预测和可视化 ────────────────────────────────
    try:
        print("\n[INFO] 加载最佳模型进行最终评估...")
        best_ckpt_data = torch.load(best_ckpt_path, weights_only=False, map_location=device)
        model.load_state_dict(best_ckpt_data['model_state_dict'])

        _, final_metrics, final_pccs, final_preds, final_labels = evaluate(
            model, val_dataset, device
        )
        final_pcc = final_metrics['pcc']

        # 可视化输出目录（时间戳隔离）
        vis_base = Path(get_histogene_dir()) / "checkpoints" / "results_vis"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        actual_vis_dir = vis_base / f"{dataset_name}_{timestamp}"
        actual_vis_dir.mkdir(parents=True, exist_ok=True)

        # 保存predictions.csv
        predictions_csv_path = actual_vis_dir / "predictions.csv"
        save_predictions_csv(final_preds, final_labels, target_cols, str(predictions_csv_path))

        # 保存逐通路PCC表格
        pcc_table_path = actual_vis_dir / "per_pathway_pcc.csv"
        save_per_pathway_pcc_table(final_preds, final_labels, target_cols, str(pcc_table_path))

        # 保存训练参数
        save_training_params(args, fold_name, n_params, str(params_json_path))
        # 复制到可视化目录
        shutil.copy2(str(params_json_path), str(actual_vis_dir / f"training_params_{fold_name}.json"))

        # 复制训练历史到可视化目录
        if history_csv_path.is_file():
            shutil.copy2(str(history_csv_path), str(actual_vis_dir / f"training_history_{fold_name}.csv"))

        # 尝试生成完整可视化报告
        try:
            from visualize_results import generate_full_report
            model_name_vis = f"HisToGene-UNI-Tokens-GAT_{fold_name}"
            generate_full_report(
                model_name=model_name_vis,
                history_csv=str(history_csv_path),
                predictions_csv=str(predictions_csv_path),
                output_dir=str(vis_base),
                prefix=dataset_name,
                actual_output_dir=str(actual_vis_dir),
                params={
                    "方案": "B+GAT (Token序列+图注意力)",
                    "fold": fold_name,
                    "gat_hidden": args.gat_hidden,
                    "gat_heads": args.gat_heads,
                    "gat_layers": args.gat_layers,
                    "k_neighbors": args.k_neighbors,
                    "epochs": args.epochs,
                    "stage1_epochs": args.stage1_epochs,
                    "lr": args.lr,
                    "lr_encoder": args.lr_encoder,
                    "weight_decay": args.weight_decay,
                    "accumulation_steps": args.accumulation_steps,
                    "dropout": args.dropout,
                }
            )
        except Exception as e:
            print(f"[WARNING] 可视化报告生成失败: {e}")

        print(f"\n[OK] 完整结果已保存到: {actual_vis_dir}")

    except Exception as e:
        print(f"[WARNING] 最终评估或可视化生成失败: {e}")
        import traceback
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════
#  可视化再生成（从已有数据）
# ═══════════════════════════════════════════════════════════════════════════

def regenerate_visualization(fold_name: str):
    """
    从已有训练数据重新生成可视化图表。
    读取 training_history CSV 和 predictions CSV（如存在），
    生成训练曲线图并保存到 checkpoints 目录。
    """
    ckpt_dir = Path(get_output_dir(f"HisToGene_UNI_Tokens_GAT/{fold_name}"))

    # 查找 history CSV
    history_csv_path = ckpt_dir / f"training_history_{fold_name}.csv"
    if not history_csv_path.is_file():
        # 尝试 resume 版本
        history_csv_path = ckpt_dir / f"training_history_{fold_name}_resume.csv"
        if not history_csv_path.is_file():
            print(f"[ERROR] 找不到训练历史文件: {ckpt_dir}")
            return

    print(f"[INFO] 使用训练历史: {history_csv_path}")

    # 查找 predictions CSV（可能在 vis 目录下）
    predictions_csv_path = None
    vis_base = Path(get_histogene_dir()) / "checkpoints" / "results_vis"
    if vis_base.is_dir():
        for d in sorted(vis_base.iterdir(), reverse=True):
            if d.is_dir() and "GAT" in d.name and fold_name in d.name:
                candidate = d / "predictions.csv"
                if candidate.is_file():
                    predictions_csv_path = str(candidate)
                    break

    # 也检查 checkpoint 目录本身
    if predictions_csv_path is None:
        candidate = ckpt_dir / "predictions.csv"
        if candidate.is_file():
            predictions_csv_path = str(candidate)

    # 生成可视化
    try:
        from visualize_results import generate_full_report, plot_training_curves, _load_history

        # 生成训练曲线图到 checkpoints 目录
        df_hist = _load_history(str(history_csv_path))
        if df_hist is not None:
            output_curves = str(ckpt_dir / f"training_curves_{fold_name}.png")
            plot_training_curves(df_hist, output_curves, f"HisToGene-UNI-Tokens-GAT ({fold_name})")
            print(f"[OK] 训练曲线图已保存: {output_curves}")
        else:
            print("[WARNING] 无法加载训练历史数据")

        # 生成完整报告
        params_json_path = ckpt_dir / f"training_params_{fold_name}.json"
        params = {}
        if params_json_path.is_file():
            with open(params_json_path, 'r', encoding='utf-8') as f:
                params = json.load(f)

        actual_vis_dir = str(ckpt_dir / "vis_report")
        os.makedirs(actual_vis_dir, exist_ok=True)

        generate_full_report(
            model_name=f"HisToGene-UNI-Tokens-GAT ({fold_name})",
            history_csv=str(history_csv_path),
            predictions_csv=predictions_csv_path,
            output_dir=str(ckpt_dir),
            prefix=f"GAT_{fold_name}",
            actual_output_dir=actual_vis_dir,
            params=params,
        )
        print(f"\n[OK] 可视化报告已生成: {actual_vis_dir}")

    except Exception as e:
        print(f"[ERROR] 可视化生成失败: {e}")
        import traceback
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = build_argparser().parse_args()

    # 可视化再生成模式
    if args.regenerate_vis:
        if args.fold == 'all':
            for fold_name in ['fold1', 'fold2', 'fold3']:
                regenerate_visualization(fold_name)
        else:
            regenerate_visualization(args.fold)
        return

    # 设备选择
    if args.device == 'auto':
        _config = load_config()
        device = get_device(_config)
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            print("[ERROR] CUDA不可用，请使用 --device cpu")
            sys.exit(1)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"[INFO] Using device: {device}")
    if device.type == 'cuda':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 打印配置信息
    print("\n" + "=" * 70)
    print("HisToGene-UNI Token + GAT 训练 (P0-3 GAT升级)")
    print("=" * 70)
    print(f"  Fold: {args.fold}")
    print(f"  GAT: hidden={args.gat_hidden}, heads={args.gat_heads}, "
          f"layers={args.gat_layers}")
    print(f"  KNN k={args.k_neighbors}")
    print(f"  两阶段: Stage1={args.stage1_epochs}ep (冻结encoder), "
          f"Stage2 (全部, lr_enc={args.lr_encoder})")
    print(f"  Epochs={args.epochs}, Patience={args.patience}")
    print("=" * 70)

    # 运行训练
    if args.fold == 'all':
        for fold_name in ['fold1', 'fold2', 'fold3']:
            train_single_fold(args, fold_name, device)
            # 清理GPU缓存
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    else:
        train_single_fold(args, args.fold, device)

    print("\n[ALL DONE] GAT训练完成！")


if __name__ == "__main__":
    main()
