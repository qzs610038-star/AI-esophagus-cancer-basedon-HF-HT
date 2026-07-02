"""
train_mpp_uni2h_mlp.py — MPP 实验训练入口：UNI2-h 冻结特征 + 两层 MLP 回归头

Strategy A：不划分内部验证集（--val_strategy none），固定 epoch 预算，
训练完成后一次性评估外部患者（MPP-2/XZY）。

固定口径（对照框架文档 §1）：
  - 禁止 LoRA / Token / GFNet / 频域分支 / 渐进解冻 / 旧 3 患者 fold 逻辑
  - XZY 不参与训练、z-score 拟合、epoch 选择或调参

用法:
    python train_mpp_uni2h_mlp.py --train_mpp_id 3 --train_patients HYZ15040,JFX,LMZ12939,TGC,XSL,ZHZ --external_mpp_id 2 --external_patient XZY --val_strategy none --num_epochs 50
"""

import argparse
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from histogene.utils import compute_metrics

# ── 项目根目录 ──
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_mpp import MPPFeatureDataset, merge_mpp_patients

# ── 忽略 Ctrl+C 信号 ──
signal.signal(signal.SIGINT, signal.SIG_IGN)

# ── 环境变量 ──
os.environ.setdefault("HF_HUB_OFFLINE", "1")


# ═══════════════════════════════════════════════════════════════
# 模型：两层 MLP 回归头
# ═══════════════════════════════════════════════════════════════

class MPPMLPHead(nn.Module):
    """UNI2-h CLS [1536] → 2 层 MLP → 30 通路预测。

    Codex 审核：去掉 LayerNorm（压缩回归动态范围，不利极端 z-score），
    保持 Linear→GELU→Dropout→Linear 两层 MLP 结构。
    """

    def __init__(self, in_dim: int = 1536, hidden: int = 1024,
                 out_dim: int = 30, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ═══════════════════════════════════════════════════════════════
# 训练循环
# ═══════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, dict]:
    """训练一个 epoch，返回 (avg_loss, metrics_dict)。"""
    model.train()
    total_loss = 0.0
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    n_samples = len(loader.dataset)

    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        preds = model(features)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        all_preds.append(preds.detach().cpu())
        all_labels.append(targets.detach().cpu())

    avg_loss = total_loss / n_samples
    all_preds_t = torch.cat(all_preds, dim=0)
    all_labels_t = torch.cat(all_labels, dim=0)

    # NaN safety: clip extreme values after prediction
    preds_np = np.clip(np.nan_to_num(all_preds_t.numpy(), nan=0.0, posinf=10.0, neginf=-10.0), -100.0, 100.0)
    labels_np = np.clip(np.nan_to_num(all_labels_t.numpy(), nan=0.0, posinf=10.0, neginf=-10.0), -100.0, 100.0)

    with np.errstate(over="warn", invalid="warn"):
        metrics = compute_metrics(labels_np, preds_np)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pathway_cols: Optional[List[str]] = None,
) -> Tuple[float, dict, np.ndarray, np.ndarray]:
    """评估模型，返回 (avg_loss, metrics_dict, preds_np, labels_np)。"""
    model.eval()
    total_loss = 0.0
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    n_samples = len(loader.dataset)

    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        preds = model(features)
        loss = criterion(preds, targets)

        total_loss += loss.item() * features.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(targets.cpu())

    avg_loss = total_loss / n_samples
    preds_np = np.clip(np.nan_to_num(torch.cat(all_preds, dim=0).numpy(), nan=0.0), -100.0, 100.0)
    labels_np = np.clip(np.nan_to_num(torch.cat(all_labels, dim=0).numpy(), nan=0.0), -100.0, 100.0)

    with np.errstate(over="warn", invalid="warn"):
        metrics = compute_metrics(labels_np, preds_np)
    return avg_loss, metrics, preds_np, labels_np


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MPP UNI2-h frozen + 2-layer MLP 训练")
    parser.add_argument("--mpp_root", default=r"D:\AIPatho\Patch\visiumhd_patch",
                        help="MPP 数据根目录")
    parser.add_argument("--train_mpp_id", type=int, required=True,
                        help="训练集 MPP 编号")
    parser.add_argument("--train_patients", required=True,
                        help="训练集患者列表，逗号分隔")
    parser.add_argument("--external_mpp_id", type=int, default=2,
                        help="外部测试集 MPP 编号（默认 2）")
    parser.add_argument("--external_patient", default="XZY",
                        help="外部测试患者（默认 XZY）")
    parser.add_argument("--cache_root", default="mpp_uni2h_cache",
                        help="特征缓存根目录")
    parser.add_argument("--labels_root", default="mpp_uni2h_cache/labels",
                        help="标准化标签目录")
    parser.add_argument("--val_strategy", default="none", choices=["none"],
                        help="验证策略（仅支持 none）")
    parser.add_argument("--num_epochs", type=int, required=True,
                        help="训练 epoch 数")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_threads", type=int, default=8,
                        help="CPU 线程数限制")
    parser.add_argument("--dataset_name", default="mpp_v3_xzy_external_uni2h_mlp_20260701",
                        help="数据集名称（用于输出目录，防碰撞）")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="MLP dropout 概率")
    parser.add_argument("--hidden_dim", type=int, default=1024,
                        help="MLP 隐藏层维度")
    parser.add_argument("--output_root", default="checkpoints/mpp_uni2h_mlp",
                        help="输出根目录")
    parser.add_argument("--allow-missing", action="store_true",
                        help="允许缺失患者数据（smoke 测试用，默认报错）")
    args = parser.parse_args()

    # ── CPU 线程限制 ──
    torch.set_num_threads(args.num_threads)
    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.num_threads)

    # ── overwrite 防护 ──
    if args.dataset_name != "mpp_v3_smoke":
        history_csv = Path(args.output_root) / args.dataset_name / "training_history.csv"
        if history_csv.exists():
            print(f"[ERROR] 输出目录已存在训练结果: {history_csv}")
            print(f"[ERROR] 使用 --dataset_name 指定新名称，或手动清理旧目录")
            sys.exit(1)

    # ── 设备 ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ── 输出目录 ──
    out_dir = Path(args.output_root) / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out_dir}")

    # ── 随机种子 ──
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── 加载数据 ──
    train_patients = [p.strip() for p in args.train_patients.split(",")]
    print(f"\n训练集患者 ({len(train_patients)}): {train_patients}")

    print("\n加载训练集 ...")
    train_dataset = merge_mpp_patients(
        cache_root=args.cache_root,
        mpp_id=args.train_mpp_id,
        patients=train_patients,
        labels_root=args.labels_root,
        allow_missing=args.allow_missing,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    print(f"训练集总样本: {len(train_dataset)}")

    print(f"\n加载外部测试集 MPP-{args.external_mpp_id}/{args.external_patient} ...")
    external_cache = f"{args.cache_root}/{args.external_mpp_id}/{args.external_patient}"
    external_labels = f"{args.labels_root}/mpp{args.external_mpp_id}_{args.external_patient}_zscored.csv"
    external_ds = MPPFeatureDataset(
        cache_dir=external_cache,
        labels_csv=external_labels,
        allow_missing=args.allow_missing,
    )
    external_loader = DataLoader(external_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)
    print(f"外部测试集样本: {len(external_ds)}")
    pathway_cols = external_ds.target_cols
    print(f"通路数: {len(pathway_cols)}")

    # ── 模型 ──
    model = MPPMLPHead(
        in_dim=external_ds.feat_dim,
        hidden=args.hidden_dim,
        out_dim=len(pathway_cols),
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── 保存参数和配置 ──
    with open(out_dir / "model_params.txt", "w") as f:
        f.write(f"dataset_name={args.dataset_name}\n")
        f.write(f"train_mpp_id={args.train_mpp_id}\n")
        f.write(f"train_patients={args.train_patients}\n")
        f.write(f"external_mpp_id={args.external_mpp_id}\n")
        f.write(f"external_patient={args.external_patient}\n")
        f.write(f"val_strategy={args.val_strategy}\n")
        f.write(f"model=MPPMLPHead (Linear->GELU->Dropout->Linear)\n")
        f.write(f"in_dim={external_ds.feat_dim}, hidden={args.hidden_dim}, out={len(pathway_cols)}\n")
        f.write(f"dropout={args.dropout}, n_params={n_params}\n")
        f.write(f"num_epochs={args.num_epochs}, batch_size={args.batch_size}\n")
        f.write(f"lr={args.lr}, seed={args.seed}, num_threads={args.num_threads}\n")
        f.write(f"device={device}\n")
        f.write(f"created_at={datetime.now().isoformat()}\n")

    # ── 训练循环 ──
    history = []
    print(f"\n{'='*60}")
    print(f"开始训练: {args.num_epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(1, args.num_epochs + 1):
        t0_epoch = time.time()

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
        )

        elapsed = time.time() - t0_epoch
        train_pcc = train_metrics["pcc"]

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_pcc": round(train_pcc, 6),
        })

        print(f"Epoch {epoch:3d}/{args.num_epochs}  "
              f"loss={train_loss:.4f}  PCC={train_pcc:.4f}  "
              f"time={elapsed:.1f}s")

    # ── 保存历史 ──
    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "training_history.csv", index=False)
    print(f"\n训练历史已保存: {out_dir / 'training_history.csv'}")

    # ── 保存 final checkpoint ──
    ckpt_path = out_dir / "final_checkpoint.pth"
    torch.save({"model_state_dict": model.state_dict(), "epoch": args.num_epochs}, ckpt_path)
    print(f"Final checkpoint: {ckpt_path}")

    # ── 外部评估 ──
    print(f"\n{'='*60}")
    print(f"评估外部测试集 MPP-{args.external_mpp_id}/{args.external_patient} ...")
    test_loss, test_metrics, test_preds, test_labels = evaluate(
        model, external_loader, criterion, device, pathway_cols,
    )
    mean_pcc = test_metrics["pcc"]
    mean_mae = test_metrics["mae"]
    mean_r2 = test_metrics["r2"]
    print(f"External XZY: PCC={mean_pcc:.4f}  MAE={mean_mae:.4f}  R²={mean_r2:.4f}  Loss={test_loss:.4f}")

    # ── 保存预测 ──
    pred_cols = [f"pred_{c}" for c in pathway_cols]
    true_cols = [f"true_{c}" for c in pathway_cols]
    pred_df = pd.DataFrame(np.column_stack([test_labels, test_preds]),
                           columns=true_cols + pred_cols)
    pred_path = out_dir / "predictions_external_xzy.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"预测结果: {pred_path}")
    # 也保存标准化 predictions.csv（与项目铁律一致）
    std_pred_path = out_dir / "predictions.csv"
    pred_df.to_csv(std_pred_path, index=False)
    print(f"标准预测: {std_pred_path}")

    # ── 逐通路 PCC ──
    per_pathway = []
    for i, col in enumerate(pathway_cols):
        pcc, _ = pearsonr(test_labels[:, i], test_preds[:, i])
        mae = np.abs(test_labels[:, i] - test_preds[:, i]).mean()
        ss_res = ((test_labels[:, i] - test_preds[:, i]) ** 2).sum()
        ss_tot = ((test_labels[:, i] - test_labels[:, i].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        per_pathway.append({"pathway": col, "pcc": pcc, "r2": r2, "mae": mae})

    pp_df = pd.DataFrame(per_pathway).sort_values("pcc", ascending=False)
    pp_path = out_dir / "per_pathway_pcc_external_xzy.csv"
    pp_df.to_csv(pp_path, index=False)
    print(f"逐通路 PCC: {pp_path}")
    print(f"\nTop-3 通路: {pp_df.head(3).to_dict('records')}")
    print(f"Bottom-3 通路: {pp_df.tail(3).to_dict('records')}")

    # ── 训练摘要 ──
    with open(out_dir / "training_summary.txt", "w") as f:
        f.write(f"Experiment: {args.dataset_name}\n")
        f.write(f"Model: UNI2-h frozen features + 2-layer MLP\n")
        f.write(f"Training strategy: val_strategy={args.val_strategy}\n")
        f.write(f"Training budget: {args.num_epochs} epochs, final checkpoint\n")
        f.write(f"Train patients (MPP-{args.train_mpp_id}): {args.train_patients}\n")
        f.write(f"External test: MPP-{args.external_mpp_id}/{args.external_patient}\n")
        f.write(f"XZY used for: test ONLY (not training, not z-score fit, not epoch selection)\n")
        f.write(f"Report: Fixed-budget training; final checkpoint evaluated once on external XZY.\n")
        f.write(f"\n--- External XZY Results ---\n")
        f.write(f"Mean PCC: {mean_pcc:.4f}\n")
        f.write(f"Mean MAE: {mean_mae:.4f}\n")
        f.write(f"Mean R²:  {mean_r2:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Completed at: {datetime.now().isoformat()}\n")

    print(f"\n摘要: {out_dir / 'training_summary.txt'}")
    print(f"\n{'='*60}")
    print("训练完成!")


if __name__ == "__main__":
    main()
