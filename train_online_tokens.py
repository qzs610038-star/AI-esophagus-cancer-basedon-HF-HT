"""
train_online_tokens.py — UNI2-H 在线 Token 序列模式训练脚本
=============================================================

与 train_online_tokens.py 相同的渐进式三阶段训练，但使用完整 token 序列
（而非 CLS token）作为下游输入。包含 LightweightTokenEncoder。

用法:
  python train_online_tokens.py --mode frozen --fold 1
  python train_online_tokens.py --mode lora --lora_rank 8 --fold 1
  python train_online_tokens.py --mode stage2 --lora_rank 8 --resume <stage1_ckpt> --fold 1
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

# ── 项目根目录 ────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_online import OnlinePatchDataset, from_multiple_patients as merge_online_patients
from model_online_tokens import OnlineTokenModel
from lora_utils import (
    print_trainable_summary,
    configure_stage1_lora,
    configure_stage2_unfreeze_last2,
    configure_stage3_unfreeze_last4,
    merge_lora_before_unfreeze,
    remove_lora_from_blocks,
    unfreeze_blocks,
    transfer_ckpt_plain_to_lora_original,
    apply_stage_structure,
)
from histogene.utils import compute_metrics
from notify_utils import (
    notify_training_complete, notify_training_error,
    check_pause_signal, clear_pause_signal,
)
from config_utils import load_config, get_device, get_patient_paths, get_fold_config
from uni2h.uni2h_utils import load_uni2h_backbone

# 忽略 Ctrl+C 信号
signal.signal(signal.SIGINT, signal.SIG_IGN)


# ═════════════════════════════════════════════════════════════════════
#  训练 / 评估循环
# ═════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    gradient_clip: float = 1.0,
    grad_accum_steps: int = 1,
) -> Tuple[float, dict]:
    """训练一个 epoch。

    Args:
        grad_accum_steps: 梯度累积步数（batch_size=1 时用于模拟更大 batch）
    """
    model.train()
    total_loss = 0.0
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    optimizer.zero_grad()
    n_samples = len(loader.dataset)

    for step, (images, pos_x, pos_y, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        pos_x = pos_x.to(device, non_blocking=True)
        pos_y = pos_y.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                preds = model(images, pos_x, pos_y)
                loss = criterion(preds, targets) / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            preds = model(images, pos_x, pos_y)
            loss = criterion(preds, targets) / grad_accum_steps
            loss.backward()

        total_loss += loss.item() * grad_accum_steps * images.size(0)
        all_preds.append(preds.detach().cpu())
        all_labels.append(targets.detach().cpu())

        # 梯度累积：每 grad_accum_steps 步更新一次
        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                optimizer.step()
            optimizer.zero_grad()

    # 最后不完整的累积步
    if (step + 1) % grad_accum_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / n_samples
    all_preds_t = torch.cat(all_preds, dim=0)
    all_labels_t = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_labels_t.numpy(), all_preds_t.numpy())
    return avg_loss, metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, dict]:
    """评估模型。"""
    model.eval()
    total_loss = 0.0
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for images, pos_x, pos_y, targets in loader:
        images = images.to(device, non_blocking=True)
        pos_x = pos_x.to(device, non_blocking=True)
        pos_y = pos_y.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        preds = model(images, pos_x, pos_y)
        loss = criterion(preds, targets)

        total_loss += loss.item() * images.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(targets.cpu())

    n = len(loader.dataset)
    avg_loss = total_loss / n
    all_preds_t = torch.cat(all_preds, dim=0)
    all_labels_t = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_labels_t.numpy(), all_preds_t.numpy())
    return avg_loss, metrics


# ═════════════════════════════════════════════════════════════════════
#  辅助函数
# ═════════════════════════════════════════════════════════════════════

def build_patient_configs(
    patient_names: List[str],
    backbone: str = 'uni_tokens',
) -> List[Dict]:
    """从患者名列表构建 OnlinePatchDataset 所需的 configs。

    每个患者的 train 和 val split 分别作为独立 config，
    便于合并多个患者的数据。
    """
    configs = []
    for pname in patient_names:
        pc = get_patient_paths(pname, backbone=backbone)
        for split, patches_key in [('train', 'train_patches'), ('val', 'val_patches')]:
            configs.append({
                'patches_dir': pc[patches_key],
                'labels_csv': pc['labels_csv'],
                'patient_name': f'{pname}_{split}',
            })
    return configs


def save_per_pathway_results(predictions_csv_path: str, output_dir: str) -> None:
    """计算逐通路 PCC/R²/MAE 并保存 CSV。"""
    if not os.path.isfile(predictions_csv_path):
        print(f"[WARN] predictions.csv 不存在: {predictions_csv_path}")
        return

    pred_df = pd.read_csv(predictions_csv_path)
    true_cols = [c for c in pred_df.columns if c.startswith("true_")]
    pathways = [c[5:] for c in true_cols]

    if not pathways:
        return

    rows = []
    for pw in pathways:
        tc, pc = f"true_{pw}", f"pred_{pw}"
        if tc not in pred_df.columns or pc not in pred_df.columns:
            continue
        yt, yp = pred_df[tc].values, pred_df[pc].values
        if np.std(yt) > 0 and np.std(yp) > 0:
            pcc = float(np.corrcoef(yt, yp)[0, 1])
        else:
            pcc = float("nan")
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        mae = float(np.mean(np.abs(yt - yp)))
        rows.append({"pathway": pw, "pcc": pcc, "r2": r2, "mae": mae})

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("pcc", ascending=False, na_position="last").reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df.to_csv(os.path.join(output_dir, "per_pathway_pcc.csv"), index=False)
    print(f"[OK] 逐通路结果已保存: {output_dir}/per_pathway_pcc.csv")


# ═════════════════════════════════════════════════════════════════════
#  参数解析
# ═════════════════════════════════════════════════════════════════════

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="UNI2-H 在线 CLS 训练（渐进式解冻）")

    # ── 模式 ──
    p.add_argument("--mode", type=str, required=True,
                   choices=["frozen", "lora", "stage2", "stage3"],
                   help="训练模式: frozen=冻结基线, lora=LoRA全24层, "
                        "stage2=LoRA+解冻末2层, stage3=LoRA+解冻末4层")
    p.add_argument("--patient", type=str, default=None,
                   help="单患者模式（如 HYZ15040）")
    p.add_argument("--cross_patient", action="store_true", default=False,
                   help="跨患者模式")
    p.add_argument("--fold", type=int, choices=[1, 2, 3], default=1,
                   help="三折CV编号: 1=JFX+LMZ→HYZ, 2=HYZ+LMZ→JFX, 3=HYZ+JFX→LMZ")
    p.add_argument("--dataset_name", type=str, default=None,
                   help="数据集名称（默认自动推导）")

    # ── LoRA 参数 ──
    p.add_argument("--lora_rank", type=int, default=8,
                   help="LoRA 秩 (default: 8)")
    p.add_argument("--lora_alpha", type=float, default=16.0,
                   help="LoRA 缩放系数 (default: 16.0)")
    p.add_argument("--lora_dropout", type=float, default=0.0,
                   help="LoRA dropout (default: 0.0)")
    p.add_argument("--lora_blocks", type=str, default="all",
                   choices=["last_4", "last_8", "all"],
                   help="LoRA 注入层范围: last_4 (blocks 20-23), last_8 (16-23), all (0-23)")

    # ── 模型参数 ──
    p.add_argument("--feature_dim", type=int, default=1536,
                   help="Backbone 特征维度")
    p.add_argument("--model_dim", type=int, default=1024,
                   help="内部模型维度")
    p.add_argument("--n_pos", type=int, default=128,
                   help="坐标编码表大小")
    p.add_argument("--n_targets", type=int, default=30,
                   help="输出通路数")
    p.add_argument("--mlp_dim", type=int, default=2048,
                   help="MLP 隐藏层维度")
    p.add_argument("--dropout", type=float, default=0.3,
                   help="Dropout 概率")

    # ── 训练超参数 ──
    p.add_argument("--lr", type=float, default=1e-4,
                   help="下游模型学习率")
    p.add_argument("--lora_lr", type=float, default=1e-4,
                   help="LoRA 参数学习率")
    p.add_argument("--unfrozen_lr", type=float, default=1e-5,
                   help="解冻层学习率 (stage2/stage3)")
    p.add_argument("--batch_size", type=int, default=4,
                   help="批大小 (Token 模式显存更大，默认4)")
    p.add_argument("--grad_accum_steps", type=int, default=2,
                   help="梯度累积步数 (本地8GB时 bs=1 + accum=8)")

    # ── TokenEncoder 参数 ──
    p.add_argument("--encoder_type", type=str, default="transformer",
                   choices=["transformer", "gfnet"],
                   help="TokenEncoder 类型")
    p.add_argument("--encoder_hidden_dim", type=int, default=512,
                   help="TokenEncoder 隐藏维度")
    p.add_argument("--n_encoder_layers", type=int, default=1,
                   help="TokenEncoder Transformer 层数")
    p.add_argument("--n_encoder_heads", type=int, default=8,
                   help="TokenEncoder 注意力头数")
    p.add_argument("--token_drop_rate", type=float, default=0.0,
                   help="训练时随机丢弃 token 的概率")
    p.add_argument("--num_tokens", type=int, default=65,
                   help="保留的 token 数量 (lite=65, full=265)")
    p.add_argument("--num_epochs", type=int, default=150,
                   help="最大训练轮数")
    p.add_argument("--early_stop_patience", type=int, default=20,
                   help="早停耐心值")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="AdamW 权重衰减")
    p.add_argument("--gradient_clip", type=float, default=1.0,
                   help="梯度裁剪范数")
    p.add_argument("--num_threads", type=int, default=8,
                   help="CPU 线程数限制 (default: 8, 0=不限制)")
    p.add_argument("--scheduler_patience", type=int, default=5,
                   help="LR 调度器耐心值")
    p.add_argument("--scheduler_factor", type=float, default=0.5,
                   help="LR 衰减因子")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader 工作进程数")
    p.add_argument("--amp", action="store_true", default=True,
                   help="混合精度训练 (默认启用)")
    p.add_argument("--grad_checkpointing", action="store_true", default=False,
                   help="启用 gradient checkpointing (节省 ~30% 显存, 慢 ~20%)")

    # ── 断点续训 ──
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint 路径 (用于续训或 stage 迁移)")

    return p


# ═════════════════════════════════════════════════════════════════════
#  main
# ═════════════════════════════════════════════════════════════════════

def main():
    args = build_argparser().parse_args()

    # ── CPU 线程限制（2026-06-04：避免 NUMA0 节点过载）──
    # 默认 8 线程，GPU 训练瓶颈在显卡，CPU 线程过多无益且影响系统服务
    cpu_threads = getattr(args, 'num_threads', 8)
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        # 设置 OpenMP/MKL 线程数（BLAS 操作会用到）
        os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_threads))
        print(f"[INFO] CPU 线程数已限制: {cpu_threads} (PyTorch/OMP/MKL/OpenBLAS)")

    # ── 数据集名称 ──
    if args.dataset_name is None:
        if args.cross_patient:
            args.dataset_name = f"online_tokens_cross_fold{args.fold}"
        elif args.patient:
            args.dataset_name = f"online_tokens_{args.patient}"
        else:
            args.dataset_name = "online_tokens"

    # 输出目录
    output_base = _PROJECT_ROOT / "checkpoints" / "online_tokens"
    run_name = f"{args.mode}_r{args.lora_rank}_{args.dataset_name}"
    run_dir = output_base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = str(run_dir / "best_model.pth")
    resume_ckpt = str(run_dir / "resume_checkpoint.pth")
    history_csv = str(run_dir / "training_history.csv")

    print("=" * 80)
    print(f"UNI2-H 在线 CLS 训练 | 模式: {args.mode} | LoRA rank: {args.lora_rank}")
    print(f"输出目录: {run_dir}")
    print("=" * 80)

    # ── 设备 ──
    _config = load_config()
    device = get_device(_config)
    print(f"[INFO] 设备: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── 加载 backbone + transform ──
    print("\n[INFO] 加载 UNI2-H backbone ...")
    backbone, transform, feat_dim = load_uni2h_backbone(device=device)
    print(f"[INFO] Backbone 加载完成, 特征维度: {feat_dim}")

    # ── Gradient checkpointing（显存优化）──
    if args.grad_checkpointing:
        if hasattr(backbone, 'set_grad_checkpointing'):
            backbone.set_grad_checkpointing(True)
            print("[INFO] Gradient checkpointing 已启用（节省 ~30% 显存）")
        else:
            print("[WARN] Backbone 不支持 set_grad_checkpointing，跳过")

    # ── LoRA 层范围解析 ──
    _LORA_BLOCK_MAP = {
        "last_4": list(range(20, 24)),
        "last_8": list(range(16, 24)),
        "all": list(range(24)),
    }
    lora_block_indices = _LORA_BLOCK_MAP[args.lora_blocks]
    print(f"[INFO] LoRA 层范围: {args.lora_blocks} → blocks {lora_block_indices[0]}-{lora_block_indices[-1]} "
          f"({len(lora_block_indices)} 层)")

    # ── Stage 迁移合并标记 ──
    # 跨 stage 迁移时，先注入全 24 层 LoRA → 加载 checkpoint →
    # merge 待解冻层 LoRA → 移除 LoRA → 解冻原始参数
    _ckpt_mode = None  # checkpoint 中的 mode
    _is_cross_stage = False

    # ── LoRA 注入（为加载 checkpoint 匹配结构）──
    # 非 frozen 模式都需要先注入 LoRA，让模型结构与 checkpoint 一致
    if args.mode != "frozen":
        configure_stage1_lora(
            backbone, rank=args.lora_rank, alpha=args.lora_alpha,
            target_blocks=lora_block_indices, dropout=args.lora_dropout,
        )

    # ── 模型 ──
    model = OnlineTokenModel(
        backbone=backbone,
        feature_dim=args.feature_dim,
        dim=args.model_dim,
        n_pos=args.n_pos,
        n_targets=args.n_targets,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        encoder_type=args.encoder_type,
        encoder_hidden_dim=args.encoder_hidden_dim,
        n_encoder_layers=args.n_encoder_layers,
        n_encoder_heads=args.n_encoder_heads,
        token_drop_rate=args.token_drop_rate,
        num_tokens=args.num_tokens,
    ).to(device)

    print_trainable_summary(model, prefix="[Model] ")

    # ── 优化器（先创建占位，resume 后可能重建）──
    def _build_optimizer(model, args):
        """按参数类型分组构建优化器。"""
        lora_p, unfrozen_p, downstream_p = [], [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_A" in name or "lora_B" in name:
                lora_p.append(param)
            elif name.startswith("backbone."):
                unfrozen_p.append(param)
            else:
                downstream_p.append(param)

        groups = []
        if downstream_p:
            groups.append({"params": downstream_p, "lr": args.lr, "name": "downstream"})
        if lora_p:
            groups.append({"params": lora_p, "lr": args.lora_lr, "name": "lora"})
        if unfrozen_p:
            groups.append({"params": unfrozen_p, "lr": args.unfrozen_lr, "name": "unfrozen_backbone"})

        optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        for pg in groups:
            print(f"[INFO] 优化器分组 '{pg['name']}': {len(pg['params'])} 参数, lr={pg['lr']}")
        return optimizer

    optimizer = _build_optimizer(model, args)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.scheduler_factor,
        patience=args.scheduler_patience, verbose=False)

    criterion = nn.HuberLoss(delta=1.0)

    # ── AMP ──
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("[INFO] 混合精度训练已启用")

    # ── 断点续训 / Stage 迁移 ──
    start_epoch = 1
    best_val_loss = float('inf')
    best_epoch = 0
    best_pcc = 0.0
    patience_counter = 0
    history: List[Dict] = []

    if args.resume:
        print(f"\n[INFO] 从 checkpoint 恢复: {args.resume}")
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        model_state = ckpt.get('model_state_dict', ckpt)
        _ckpt_mode = ckpt.get('args', {}).get('mode', None)
        _is_cross_stage = (_ckpt_mode is not None and _ckpt_mode != args.mode)

        if _is_cross_stage:
            # ════════════════════════════════════════════════════════
            # 跨 Stage 迁移：先加载完整权重 → merge LoRA → 移除 → 解冻
            # ════════════════════════════════════════════════════════
            print(f"[INFO] 跨阶段迁移: {_ckpt_mode} → {args.mode}")

            # Step 1: 加载 checkpoint（strict=False 容忍结构差异）
            missing, unexpected = model.load_state_dict(model_state, strict=False)
            if missing:
                print(f"  [WARN] Missing keys: {len(missing)} (可能是 stage 结构差异)")
            if unexpected:
                print(f"  [WARN] Unexpected keys: {len(unexpected)}")
            print("  ✓ Step 1/5: Checkpoint 权重已加载")

            # Step 1.5: 转移 ckpt 中 plain Linear 权重 → LoRALinear.original
            # （Stage 2→3 迁移时，ckpt 的 blocks 22-23 已移除 LoRA 变为 plain Linear，
            #   strict=False 无法自动映射 key 名不同的参数，需手动转移）
            n_transferred = transfer_ckpt_plain_to_lora_original(model, model_state)
            if n_transferred > 0:
                print(f"  ✓ Step 1.5/5: {n_transferred} 个 plain→LoRA.original 权重已转移")
            else:
                print(f"  ✓ Step 1.5/5: 无需要转移的 plain→LoRA 权重（结构与 ckpt 一致）")

            # Step 2: 确定本 Stage 要解冻的层 + merge LoRA
            if args.mode == "stage2":
                unfreeze_indices = [22, 23]
            elif args.mode == "stage3":
                unfreeze_indices = [20, 21, 22, 23]
            else:
                unfreeze_indices = []

            if unfreeze_indices:
                merged = merge_lora_before_unfreeze(backbone, unfreeze_indices)
                print(f"  ✓ Step 2/5: {merged} 个 LoRA 模块已 merge 到原始权重 (blocks {unfreeze_indices})")

            # Step 3: 移除 LoRA 包装
            if unfreeze_indices:
                remove_lora_from_blocks(backbone, unfreeze_indices)
                print(f"  ✓ Step 3/5: LoRA 已移除 (blocks {unfreeze_indices})")

            # Step 4: 解冻原始参数
            if unfreeze_indices:
                n_unfrozen = unfreeze_blocks(backbone, unfreeze_indices)
                print(f"  ✓ Step 4/5: {n_unfrozen:,} 参数已解冻 (blocks {unfreeze_indices})")

            # Step 5: 重建优化器（参数分组变了）
            optimizer = _build_optimizer(model, args)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=args.scheduler_factor,
                patience=args.scheduler_patience, verbose=False)
            print("  ✓ Step 5/5: 优化器已重建（适配新的参数分组）")

            print_trainable_summary(model, prefix="[Model-迁移后] ")
        else:
            # ════════════════════════════════════════════════════════
            # 同 Stage 续训：重映射权重 + 恢复结构 + 恢复训练状态
            # ════════════════════════════════════════════════════════
            missing, unexpected = model.load_state_dict(model_state, strict=False)
            if missing:
                print(f"  [WARN] Missing keys: {len(missing)}")
            if unexpected:
                print(f"  [WARN] Unexpected keys: {len(unexpected)}")
            print("[INFO] 模型权重已加载（同阶段续训）")

            # 检测 ckpt 结构是否与当前模型一致
            # （Stage 2/3 的 ckpt 中部分层已移除 LoRA 变为 plain Linear，
            #  但当前模型刚注入全量 LoRA，需要转移权重 + 重建结构）
            n_transferred = transfer_ckpt_plain_to_lora_original(model, model_state)
            if n_transferred > 0:
                print(f"[INFO] 检测到结构差异，{n_transferred} 个 plain→LoRA 权重已转移")
                structure_msg = apply_stage_structure(backbone, args.mode)
                print(f"  {structure_msg}")
                optimizer = _build_optimizer(model, args)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=args.scheduler_factor,
                    patience=args.scheduler_patience, verbose=False)
                print("  ✓ 优化器已重建（适配恢复后的参数分组）")
                print_trainable_summary(model, prefix="[Model-恢复结构后] ")
            else:
                print("[INFO] 模型结构与 ckpt 一致，无需重建阶段结构")

            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception:
                print("[WARN] 优化器状态加载失败，使用新优化器")

            try:
                scheduler.load_state_dict(ckpt.get('scheduler_state_dict', {}))
            except Exception:
                pass

            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            patience_counter = ckpt.get('patience_counter', 0)
            best_epoch = ckpt.get('best_epoch', 0)
            best_pcc = ckpt.get('best_pcc', 0.0)
            if 'history' in ckpt:
                history = ckpt['history']
            print(f"[INFO] 从 Epoch {start_epoch} 继续, best_val_loss={best_val_loss:.4f}")

        if 'scaler_state_dict' in ckpt and ckpt['scaler_state_dict'] and scaler:
            try:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            except Exception:
                pass

        clear_pause_signal(_PROJECT_ROOT)

    else:
        # ── 非 resume：Stage 2/3 直接创建（无需 merge，因为无前一 Stage 权重）──
        if args.mode == "stage2":
            print("[INFO] Stage 2 (无 resume): 移除 blocks 22-23 的 LoRA，解冻...")
            remove_lora_from_blocks(backbone, [22, 23])
            unfreeze_blocks(backbone, [22, 23])
            optimizer = _build_optimizer(model, args)
            print_trainable_summary(model, prefix="[Model-Stage2] ")
        elif args.mode == "stage3":
            print("[INFO] Stage 3 (无 resume): 移除 blocks 20-23 的 LoRA，解冻...")
            remove_lora_from_blocks(backbone, [20, 21, 22, 23])
            unfreeze_blocks(backbone, [20, 21, 22, 23])
            optimizer = _build_optimizer(model, args)
            print_trainable_summary(model, prefix="[Model-Stage3] ")

    # ── 构建数据集 ──
    if args.cross_patient:
        fold_cfg = get_fold_config(args.fold)
        train_names = fold_cfg["train"]
        test_name = fold_cfg["test"]

        train_configs = build_patient_configs(train_names)
        val_configs = build_patient_configs([test_name])

        train_dataset, coord_stats_dict, target_cols = merge_online_patients(
            train_configs, transform=transform, n_pos=args.n_pos, n_targets=args.n_targets)
        val_dataset, test_coord_stats, _ = merge_online_patients(
            val_configs, transform=transform, n_pos=args.n_pos, n_targets=args.n_targets)
        coord_stats_dict.update(test_coord_stats)

        mode_label = f"跨患者 Fold {args.fold} ({'+'.join(train_names)}→{test_name})"
    elif args.patient:
        pc = get_patient_paths(args.patient, backbone='uni_tokens')
        train_dataset = OnlinePatchDataset(
            patches_dir=pc['train_patches'], labels_csv=pc['labels_csv'],
            transform=transform, n_pos=args.n_pos, n_targets=args.n_targets)
        train_coord = train_dataset.get_coord_stats()
        target_cols = train_dataset.target_cols
        val_dataset = OnlinePatchDataset(
            patches_dir=pc['val_patches'], labels_csv=pc['labels_csv'],
            transform=transform, n_pos=args.n_pos, n_targets=args.n_targets,
            coord_stats=train_coord, target_cols=target_cols)
        coord_stats_dict = {f"{args.patient}_train": train_coord}
        mode_label = f"单患者 {args.patient}"
    else:
        raise ValueError("必须指定 --patient 或 --cross_patient")

    print(f"\n[INFO] 训练模式: {mode_label}")
    print(f"[INFO] 训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")

    # ── DataLoader ──
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin, drop_last=False)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin)

    # ── 训练循环 ──
    early_stopped = False
    val_label = "Test" if args.cross_patient else "Val"
    task_name = f"OnlineToken-{args.mode}-r{args.lora_rank}_{args.dataset_name}"
    eff_batch = args.batch_size * args.grad_accum_steps

    print(f"\n{'=' * 80}")
    print(f"开始训练 | 模式: {args.mode} | Epochs: {args.num_epochs} | "
          f"BS: {args.batch_size}×{args.grad_accum_steps}={eff_batch} | LR: {args.lr}")
    print(f"  训练集: {len(train_dataset)} | {val_label}集: {len(val_dataset)}")
    print(f"{'=' * 80}")

    current_epoch = 0
    try:
        for epoch in range(start_epoch, args.num_epochs + 1):
            current_epoch = epoch
            t0 = time.time()

            train_loss, train_m = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler,
                gradient_clip=args.gradient_clip, grad_accum_steps=args.grad_accum_steps)
            val_loss, val_m = evaluate(model, val_loader, criterion, device)

            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)

            elapsed = time.time() - t0

            # 定期清理显存碎片（低显存环境）
            if device.type == "cuda" and epoch % 5 == 0:
                torch.cuda.empty_cache()

            print(
                f"Epoch [{epoch:3d}/{args.num_epochs}] "
                f"Train Loss: {train_loss:.4f} PCC: {train_m['pcc']:.4f} | "
                f"{val_label} Loss: {val_loss:.4f} PCC: {val_m['pcc']:.4f} R²: {val_m['r2']:.4f} | "
                f"LR: {current_lr:.2e} | {elapsed:.1f}s"
            )

            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_pcc': train_m['pcc'],
                'train_r2': train_m['r2'],
                'train_mae': train_m['mae'],
                'val_loss': val_loss,
                'val_pcc': val_m['pcc'],
                'val_r2': val_m['r2'],
                'val_mae': val_m['mae'],
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
                print(f"  ✓ 最佳模型已保存 (val_pcc={val_m['pcc']:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop_patience:
                    print(f"\n早停触发！连续 {args.early_stop_patience} 个 epoch val_loss 未改善。")
                    early_stopped = True
                    break

            # 每 10 epoch 存历史
            if epoch % 10 == 0:
                pd.DataFrame(history).to_csv(history_csv, index=False)

            # 检查暂停信号
            if check_pause_signal(_PROJECT_ROOT):
                print("\n[INFO] 检测到暂停信号，保存 checkpoint ...")
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
                print(f"[INFO] 暂停 checkpoint: {resume_ckpt}")
                notify_training_complete(task_name, epoch, best_epoch, best_pcc, "paused")
                clear_pause_signal(_PROJECT_ROOT)
                return

    except Exception as e:
        notify_training_error(task_name, current_epoch, str(e))
        raise

    # ── 训练完成 ──
    status = "early_stop" if early_stopped else "completed"
    notify_training_complete(task_name, current_epoch, best_epoch, best_pcc, status)

    pd.DataFrame(history).to_csv(history_csv, index=False)
    print(f"\n[DONE] 训练结束。Best val_loss={best_val_loss:.4f}, best_pcc={best_pcc:.4f}")
    print(f"  最佳模型: {best_ckpt}")

    # ── 推理 + 可视化 ──
    try:
        print("\n[INFO] 加载最佳模型进行推理 ...")
        best_data = torch.load(best_ckpt, weights_only=False, map_location=device)
        model.load_state_dict(best_data['model_state_dict'], strict=False)
        model.eval()

        all_preds, all_labels = [], []
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

        # 保存 predictions.csv
        pred_df = pd.DataFrame()
        for i, col in enumerate(target_cols):
            pred_df[f'true_{col}'] = labels_cat[:, i]
            pred_df[f'pred_{col}'] = preds_cat[:, i]

        pred_csv = str(run_dir / "predictions.csv")
        pred_df.to_csv(pred_csv, index=False)
        print(f"[OK] predictions.csv 已保存: {pred_csv}")

        # 逐通路结果
        save_per_pathway_results(pred_csv, str(run_dir))

        # 模型参数摘要
        summary_path = str(run_dir / "training_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"训练模式: {args.mode}\n")
            f.write(f"Encoder type: {args.encoder_type}\n")
            f.write(f"Num tokens: {args.num_tokens}\n")
            f.write(f"LoRA rank: {args.lora_rank}\n")
            f.write(f"数据集: {args.dataset_name}\n")
            f.write(f"训练模式: {mode_label}\n")
            f.write(f"Best Epoch: {best_epoch}\n")
            f.write(f"Best Val PCC: {best_pcc:.4f}\n")
            f.write(f"Best Val Loss: {best_val_loss:.4f}\n")
            f.write(f"训练样本: {len(train_dataset)}\n")
            f.write(f"验证样本: {len(val_dataset)}\n")
        print(f"[OK] 训练摘要: {summary_path}")

    except Exception as e:
        print(f"[WARN] 推理/可视化阶段出错: {e}")

    print(f"\n{'=' * 80}")
    print(f"训练完成: {task_name}")
    print(f"  Best Val PCC: {best_pcc:.4f}")
    print(f"  输出目录: {run_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
