"""
train_mpp_uni2h_mlp.py — MPP 实验训练入口：UNI2-h 冻结特征 + 两层 MLP 回归头

两种模式：
  - --val_strategy none    (V3 模式):  无内部验证，train_loss 早停，固定 epoch 预算
  - --val_strategy internal (V3bis 模式): 内部验证集，val_loss 选 best ckpt + 早停

固定口径（对照框架文档 §1）：
  - 禁止 LoRA / Token / GFNet / 频域分支 / 渐进解冻 / 旧 3 患者 fold 逻辑
  - XZY 不参与训练、z-score 拟合、epoch 选择或调参

用法:
    # V3 模式（无内部验证）
    python train_mpp_uni2h_mlp.py --train_mpp_id 3 --train_patients HYZ15040,JFX,LMZ12939,TGC,XSL,ZHZ --external_mpp_id 2 --external_patient XZY --val_strategy none --num_epochs 50

    # V3bis 模式（内部验证）
    python train_mpp_uni2h_mlp.py --train_mpp_id 3 --train_patients HYZ15040,JFX,LMZ12939,TGC,XSL --val_strategy internal --val_patient ZHZ --external_mpp_id 2 --external_patient XZY --num_epochs 50 --patience 10
"""

import argparse
import json as _json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from dataset_mpp_manifest import build_manifest_datasets, ManifestMPPDataset

# ── 忽略 Ctrl+C 信号 ──
signal.signal(signal.SIGINT, signal.SIG_IGN)

# ── 环境变量 ──
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def resolve_manifest_data_roots(
    splits_root: str | Path,
    labels_root: str | Path,
    mpp_id: int,
) -> Dict[str, Path]:
    """Resolve immutable split metadata separately from versioned repaired labels."""
    split_group = Path(splits_root) / f"group_{mpp_id}"
    label_group = Path(labels_root) / f"group_{mpp_id}"
    resolved = {
        "split_manifest": split_group / "split_manifest.csv",
        "split_info": split_group / "split_info.json",
        "zscore_manifest": label_group / "zscore_manifest.json",
        "zscore_params": label_group / "zscore_params_from_train.json",
        "labels": label_group / "labels",
    }
    missing = [f"{name}={path}" for name, path in resolved.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("manifest data roots are incomplete: " + ", ".join(missing))
    return resolved


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
# 路径工具函数
# ═══════════════════════════════════════════════════════════════

def _build_cache_dir(cache_root: str, mpp_id: int, patient: str,
                     partner: bool = False, split: str = "train") -> str:
    """构建特征缓存目录路径。

    partner 模式下，每个患者目录下还有 train/ val/ 子目录。
    split 参数指定加载 train 还是 val 缓存。
    """
    if partner:
        return f"{cache_root}/MPP{mpp_id}_UNI/{patient}/{split}"
    return f"{cache_root}/{mpp_id}/{patient}"


def _build_train_label(labels_root: str, mpp_id: int, patient: str,
                       partner: bool = False) -> str:
    """构建训练标签 CSV 路径。"""
    if partner:
        return f"{labels_root}/group_{mpp_id}/train/{patient}/{patient}_ssGSEA_zscore.csv"
    return f"{labels_root}/mpp{mpp_id}_{patient}_zscored.csv"


def _build_val_label(labels_root: str, mpp_id: int, patient: str,
                     partner: bool = False) -> str:
    """构建内部验证标签 CSV 路径。"""
    if partner:
        return f"{labels_root}/group_{mpp_id}/val/{patient}/{patient}_ssGSEA_zscore.csv"
    return f"{labels_root}/mpp{mpp_id}_{patient}_zscored.csv"


def _build_external_label(labels_root: str, train_mpp_id: int,
                          external_mpp_id: int, external_patient: str,
                          partner: bool = False) -> str:
    """构建外部测试标签 CSV 路径。

    partner 模式下：XZY 使用训练组 (group_{train_mpp_id}) 的 z-score 参数变换，
    命名格式为 XZY_ssGSEA_zscore_by_group_{train_mpp_id}_train.csv。
    标准模式下：使用 external_mpp_id 拼接。
    """
    if partner:
        return (f"{labels_root}/group_{train_mpp_id}/external/{external_patient}/"
                f"{external_patient}_ssGSEA_zscore_by_group_{train_mpp_id}_train.csv")
    return f"{labels_root}/mpp{external_mpp_id}_{external_patient}_zscored.csv"


def _build_zscore_params_path(labels_root: str, mpp_id: int,
                              partner: bool = False) -> str:
    """构建 z-score 参数文件路径。"""
    if partner:
        return f"{labels_root}/group_{mpp_id}/zscore_params_from_train.csv"
    return f"{labels_root}/zscore_params_mpp{mpp_id}.json"


def _run_preflight(cache_root: str, labels_root: str, train_mpp_id: int,
                   external_mpp_id: int, train_patients: List[str],
                   val_patient: Optional[str], external_patient: str,
                   partner: bool, allow_missing: bool = False) -> None:
    """打印并验证所有 train/val/external 的数据路径和 z-score 参数。

    Args:
        allow_missing: 若 True，训练患者缺失时仅 warn 不中止 (smoke 模式)。
                       val 和 external 永远强制检查（缺失即中止）。
    """
    print(f"\n{'='*60}")
    print(f"Preflight: 数据路径审计 (partner={partner}, allow_missing={allow_missing})")
    print(f"{'='*60}")

    # ── 1. z-score 参数文件（最先检查，失败则无法继续） ──
    zscore_path = _build_zscore_params_path(labels_root, train_mpp_id, partner)
    zscore_exists = Path(zscore_path).exists()
    pair_ok = True
    if zscore_exists:
        print(f"  [OK] zscore_params: {zscore_path}")
    else:
        print(f"  [MISSING] zscore_params: {zscore_path}")
        pair_ok = False

    # partner 模式下还检查 split_info.json
    if partner:
        split_path = Path(labels_root) / f"group_{train_mpp_id}" / "split_info.json"
        if split_path.exists():
            print(f"  [OK] split_info: {split_path}")
        else:
            print(f"  [MISSING] split_info: {split_path}")
            pair_ok = False

    if not pair_ok:
        print(f"\n[ERROR] Preflight 失败: z-score 参数文件缺失。"
              f"检查 --labels_root / --train_mpp_id / --use_partner_paths。")
        sys.exit(1)

    # ── 2. 数据路径 ──
    n_train_missing = 0

    # 训练患者
    for p in train_patients:
        cache_dir = _build_cache_dir(cache_root, train_mpp_id, p, partner, split="train")
        label_csv = _build_train_label(labels_root, train_mpp_id, p, partner)
        cache_exists = Path(cache_dir).exists()
        label_exists = Path(label_csv).exists()
        n_pt = len(list(Path(cache_dir).glob("*.pt"))) if cache_exists else 0
        n_csv = _read_csv_rows(label_csv) if label_exists else 0
        ok = cache_exists and label_exists and n_pt > 0 and n_csv > 0
        if not ok:
            n_train_missing += 1
        status = "OK" if ok else ("WARN" if allow_missing else "MISSING")
        print(f"  [{status:>7s}] [train] {p:>10s}  "
              f"cache=({n_pt:>5d} .pt)  label=({n_csv:>5d} rows)  "
              f"cache_dir={cache_dir}")

    # 内部验证
    if val_patient:
        cache_dir = _build_cache_dir(cache_root, train_mpp_id, val_patient, partner, split="val")
        label_csv = _build_val_label(labels_root, train_mpp_id, val_patient, partner)
        cache_exists = Path(cache_dir).exists()
        label_exists = Path(label_csv).exists()
        n_pt = len(list(Path(cache_dir).glob("*.pt"))) if cache_exists else 0
        n_csv = _read_csv_rows(label_csv) if label_exists else 0
        ok = cache_exists and label_exists and n_pt > 0 and n_csv > 0
        status = "OK" if ok else "MISSING"
        print(f"  [{status:>7s}] [val  ] {val_patient:>10s}  "
              f"cache=({n_pt:>5d} .pt)  label=({n_csv:>5d} rows)  "
              f"cache_dir={cache_dir}")
        if not ok:
            print(f"\n[ERROR] Preflight 失败: 内部验证集数据缺失 (val_patient={val_patient})。")
            sys.exit(1)

    # 外部测试
    ext_cache = _build_cache_dir(cache_root, train_mpp_id, external_patient, partner, split="train")
    ext_label = _build_external_label(labels_root, train_mpp_id, external_mpp_id,
                                      external_patient, partner)
    cache_exists = Path(ext_cache).exists()
    label_exists = Path(ext_label).exists()
    n_pt = len(list(Path(ext_cache).glob("*.pt"))) if cache_exists else 0
    n_csv = _read_csv_rows(ext_label) if label_exists else 0
    ok = cache_exists and label_exists and n_pt > 0 and n_csv > 0
    status = "OK" if ok else "MISSING"
    print(f"  [{status:>7s}] [ext  ] {external_patient:>10s}  "
          f"cache=({n_pt:>5d} .pt)  label=({n_csv:>5d} rows)  "
          f"cache_dir={ext_cache}")
    if not ok:
        print(f"\n[ERROR] Preflight 失败: 外部测试集数据缺失 (external_patient={external_patient})。")
        sys.exit(1)

    # ── 3. 决策 ──
    if n_train_missing > 0:
        if allow_missing:
            print(f"\n  Preflight: {n_train_missing} 个训练患者数据缺失 (allow_missing=True，继续)")
        else:
            print(f"\n[ERROR] Preflight 失败: {n_train_missing} 个训练患者数据缺失。"
                  f"使用 --allow-missing 跳过缺失患者 (仅 smoke 模式)。")
            sys.exit(1)
    else:
        print(f"\n  Preflight 通过 (所有数据路径存在且非空)")


def _read_csv_rows(csv_path: str) -> int:
    """安全读取 CSV 行数（不含表头）。失败返回 0。"""
    try:
        return len(pd.read_csv(csv_path))
    except Exception:
        return 0


def _load_zscore_params(labels_root: str, mpp_id: int,
                        partner: bool = False) -> Optional[dict]:
    """加载 z-score 参数并验证防泄漏。

    partner 模式：读 split_info.json 推断 fit_patients + 验证 zscore_params_from_train.csv
    标准模式：读 zscore_params_mpp{id}.json
    """
    if partner:
        # 从 split_info.json 推断 fit_patients
        split_path = Path(labels_root) / f"group_{mpp_id}" / "split_info.json"
        if not split_path.exists():
            print(f"[WARN] split_info.json 不存在: {split_path}，跳过 z-score 审计")
            return None
        with open(split_path, "r") as f:
            split_info = _json.load(f)
        fit_patients = split_info.get("train_patients", [])
        val_patients = split_info.get("val_patients", [])
        external_patient = split_info.get("external_patient", "")

        # 读取参数文件验证通路数
        param_path = Path(labels_root) / f"group_{mpp_id}" / "zscore_params_from_train.csv"
        params_df = pd.read_csv(param_path)
        pathways = params_df["label"].tolist()
        mean = params_df["mean"].tolist()
        std = params_df["std"].tolist()

        return {
            "fit_patients": fit_patients,
            "val_patients": val_patients,
            "external_patient": external_patient,
            "pathways": pathways,
            "mean": mean,
            "std": std,
            "n_pathways": len(pathways),
        }

    # 标准模式：读 JSON
    zscore_path = f"{labels_root}/zscore_params_mpp{mpp_id}.json"
    if not Path(zscore_path).exists():
        return None
    with open(zscore_path, "r") as f:
        zscore_params = _json.load(f)
    return zscore_params


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    from path_registry import get_registered_path

    parser = argparse.ArgumentParser(description="MPP UNI2-h frozen + 2-layer MLP 训练")
    parser.add_argument("--mpp_root", default=str(get_registered_path("mpp_data_root")),
                        help="MPP 数据根目录（由 path id mpp_data_root 解析）")
    parser.add_argument("--train_mpp_id", type=int, required=True,
                        help="训练集 MPP 编号")
    parser.add_argument("--train_patients", default="",
                        help="训练集患者列表，逗号分隔（--auto_split 时无需传，自动从 split_info.json 读取）")
    parser.add_argument("--external_mpp_id", type=int, default=2,
                        help="外部测试集 MPP 编号（默认 2）")
    parser.add_argument("--external_patient", default="XZY",
                        help="外部测试患者（默认 XZY）")
    parser.add_argument("--cache_root", default="mpp_uni2h_cache",
                        help="特征缓存根目录")
    parser.add_argument("--labels_root", default="mpp_uni2h_cache/labels",
                        help="标准化标签目录")
    parser.add_argument("--val_strategy", default="none", choices=["none", "internal", "manifest"],
                        help="验证策略：none=无内部验证(旧V3模式), internal=患者级内部验证(V3bis模式), "
                             "manifest=split_manifest 驱动的 patch 级各患者 10% val (20260706 统一标准)")
    parser.add_argument("--val_patient", default=None,
                        help="内部验证患者（val_strategy=internal 时必需，如 ZHZ）")
    parser.add_argument("--splits_root", default="mpp_standard_splits",
                        help="split_manifest + z-score 标签根目录 (val_strategy=manifest 时使用); "
                             "下含 group_{N}/split_manifest.csv + labels/")
    parser.add_argument("--manifest_labels_root", default="",
                        help="manifest 模式的版本化标签根；为空时兼容旧布局并使用 --splits_root")
    parser.add_argument("--flat_cache_root", default="mpp_uni2h_cache",
                        help="扁平缓存根 (含 /2/XZY 外部测试缓存 + /3/{patient} MPP-3 缓存); "
                             "manifest 模式下外部 XZY 从此加载 (移除 MPP1 硬编码)")
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
    parser.add_argument("--allow_missing", "--allow-missing", dest="allow_missing", action="store_true",
                        help="允许缺失患者数据（smoke 测试用，默认报错）")
    parser.add_argument("--patience", type=int, default=10,
                        help="早停 patience：训练 loss 连续 N 个 epoch 无改善则停止（默认 10）")
    parser.add_argument("--min_delta", type=float, default=1e-4,
                        help="早停最小改善阈值（默认 0.0001）")
    # ── 路径布局选择 ──
    parser.add_argument("--use_partner_paths", action="store_true",
                        help="使用队友目录布局 (MPP{N}_UNI/ + group_{N}/train|val|external/)")
    parser.add_argument("--auto_split", action="store_true",
                        help="自动从 split_info.json 覆盖 --train_patients 和 --val_patient")
    args = parser.parse_args()

    # This check is intentionally inside the training script as well as the
    # dispatcher. Manual invocation must not bypass current-state, path-index,
    # transport or result-evidence gates.
    from scripts.pfmval_state import active_mpp_repair, validate_state, verify_mpp_repair_server_assets
    project_root = Path(__file__).resolve().parent
    state_report = validate_state(
        project_root,
        strict=True,
        task="training",
        host_scope="server",
    )
    if not state_report.ok:
        state_report.emit()
        print("[BLOCKED] MPP training is blocked by the authoritative project state")
        return 2

    # ── train_patients 来源校验 ──
    # manifest 模式: train_patients 固定六例患者, 自动从 split_manifest 推断, 不需用户传
    if args.val_strategy == "manifest":
        args.train_patients = ",".join(["HYZ15040", "JFX", "LMZ12939", "TGC", "XSL", "ZHZ"])
    elif not args.train_patients and not args.auto_split:
        print("[ERROR] 必须指定 --train_patients 或 --auto_split（自动从 split_info.json 读取）")
        sys.exit(1)

    # ── partner 模式：默认路径覆盖 ──
    if args.use_partner_paths:
        if args.cache_root == "mpp_uni2h_cache":
            args.cache_root = str(get_registered_path("server_mpp_partner_cache"))
        if args.labels_root == "mpp_uni2h_cache/labels":
            args.labels_root = str(get_registered_path("legacy_partner_labels"))
        print(f"[INFO] use_partner_paths=True")
        print(f"       cache_root  -> {args.cache_root}")
        print(f"       labels_root -> {args.labels_root}")

    # ── auto_split：从 split_info.json 自动读取划分 ──
    if args.auto_split:
        split_info_path = (Path(args.labels_root) / f"group_{args.train_mpp_id}"
                           / "split_info.json")
        if not split_info_path.exists():
            print(f"[ERROR] --auto_split 但 split_info.json 不存在: {split_info_path}")
            sys.exit(1)
        with open(split_info_path, "r") as f:
            split_info = _json.load(f)
        args.train_patients = ",".join(split_info["train_patients"])
        val_list = split_info.get("val_patients", [])
        if val_list:
            args.val_patient = val_list[0]
            if args.val_strategy == "none":
                args.val_strategy = "internal"
                print(f"[INFO] auto_split: val_patient={args.val_patient}, "
                      f"自动切换 val_strategy=internal")
        print(f"[INFO] auto_split: train_patients={args.train_patients}, "
              f"val_patient={args.val_patient}")

    # ── val_strategy 参数校验 ──
    if args.val_strategy == "internal" and args.val_patient is None:
        print("[ERROR] --val_strategy internal 必须同时指定 --val_patient")
        sys.exit(1)
    if args.val_strategy == "none" and args.val_patient is not None:
        print("[WARN] --val_patient 已指定但 --val_strategy=none，将忽略 val_patient")
        args.val_patient = None

    # ── manifest 模式校验 + 路径推断 ──
    if args.val_strategy == "manifest":
        # manifest 模式: 从 split_manifest.csv 读 train/val 划分, 忽略 --train_patients/--val_patient
        manifest_path = (Path(args.splits_root) / f"group_{args.train_mpp_id}"
                         / "split_manifest.csv")
        if not manifest_path.exists():
            print(f"[ERROR] --val_strategy manifest 但 split_manifest 不存在: {manifest_path}")
            sys.exit(1)
        # manifest 模式: train_patients 固定六例患者 (方案 §二), val_patient 无意义
        args.train_patients = ",".join(["HYZ15040", "JFX", "LMZ12939", "TGC", "XSL", "ZHZ"])
        args.val_patient = None  # manifest 模式 val 是 patch 级, 无单一 val_patient
        use_manifest = True
        print(f"[INFO] val_strategy=manifest: split_manifest={manifest_path}")
        print(f"[INFO] manifest 模式: train_patients 固定六例患者, val 是 patch 级各 10%")
    else:
        use_manifest = False

    if use_manifest:
        repair = active_mpp_repair(project_root)
        if repair is None:
            print("[BLOCKED] manifest training requires an explicitly activated repaired-label data manifest")
            return 2
        expected_labels_root = Path(str(repair["server_stage_path"])).resolve()
        supplied_labels_root = Path(args.manifest_labels_root or args.splits_root).resolve()
        if os.path.normcase(str(supplied_labels_root)) != os.path.normcase(str(expected_labels_root)):
            print(
                "[BLOCKED] --manifest_labels_root does not match active repaired-label staging: "
                f"supplied={supplied_labels_root} active={expected_labels_root}"
            )
            return 2
        try:
            verified = verify_mpp_repair_server_assets(project_root, repair)
        except ValueError as exc:
            print(f"[BLOCKED] repaired-label staging verification failed: {exc}")
            return 2
        print(
            "[PASS] repaired-label staging verified before direct training: "
            f"assets={verified['verified_generated_assets']} audit={verified['audit_sha256']}"
        )

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

    # 防泄漏检查：val_patient 不应在 train_patients 中
    if args.val_patient and args.val_patient in train_patients:
        print(f"[ERROR] val_patient ({args.val_patient}) 出现在 train_patients 中，违反数据隔离规则")
        sys.exit(1)
    if args.external_patient in train_patients:
        print(f"[ERROR] external_patient ({args.external_patient}) 出现在 train_patients 中，违反数据隔离规则")
        sys.exit(1)
    if args.external_patient == args.val_patient:
        print(f"[ERROR] external_patient ({args.external_patient}) 与 val_patient 相同，违反数据隔离规则")
        sys.exit(1)

    use_internal_val = (args.val_strategy == "internal" and args.val_patient is not None)
    partner = args.use_partner_paths

    # ═══════════════════════════════════════════════════════════════
    # manifest 模式: split_manifest 驱动的 patch 级 train/val 构建
    # (20260706 统一标准重跑; 与旧 partner/internal 路径独立)
    # ═══════════════════════════════════════════════════════════════
    if use_manifest:
        # manifest 模式下: train_patients 固定六例, val 是 patch 级各 10%
        # external_patient 隔离仍校验 (XZY 不在六例 train 内)
        if args.external_patient in train_patients:
            print(f"[ERROR] external_patient ({args.external_patient}) 出现在 train_patients 中，违反数据隔离规则")
            sys.exit(1)

        print(f"\n训练集患者 ({len(train_patients)}): {train_patients}")
        print(f"内部验证: patch 级各患者 10% (split_manifest 驱动)")
        print(f"外部测试患者: MPP-{args.external_mpp_id}/{args.external_patient}")

        manifest_labels_root = args.manifest_labels_root or args.splits_root
        manifest_roots = resolve_manifest_data_roots(
            args.splits_root, manifest_labels_root, args.train_mpp_id
        )
        print(f"split metadata root: {Path(args.splits_root)}")
        print(f"repaired label root: {Path(manifest_labels_root)}")

        # 读 split_manifest
        manifest_path = manifest_roots["split_manifest"]
        manifest_df = pd.read_csv(manifest_path)
        print(f"split_manifest: {len(manifest_df)} 行, split 分布: "
              f"{manifest_df['split'].value_counts().to_dict()}")

        # manifest 模式 preflight: 检查 zscore_manifest.json + split_manifest split 完整性
        zscore_manifest_path = manifest_roots["zscore_manifest"]
        with open(zscore_manifest_path, "r", encoding="utf-8") as f:
            zscore_manifest = _json.load(f)
        fit_pats = set(zscore_manifest.get("fit_patients", []))
        if fit_pats != set(train_patients):
            print(f"[ERROR] zscore_manifest fit_patients {sorted(fit_pats)} != "
                  f"train_patients {sorted(train_patients)}")
            sys.exit(1)
        if args.external_patient in fit_pats:
            print(f"[ERROR] external_patient 出现在 z-score fit_patients 中，不应参与拟合")
            sys.exit(1)
        # 检查 manifest 内 MPP-3/5 无泄漏 (leakage_pairs 应=0)
        split_info_path = manifest_roots["split_info"]
        with open(split_info_path, "r", encoding="utf-8") as f:
            split_info = _json.load(f)
        leakage = split_info.get("leakage_pairs", 0)
        if leakage != 0:
            print(f"[ERROR] split_info.json 报告 leakage_pairs={leakage} (MPP-{args.train_mpp_id}) "
                  f"泄漏未清零, 严禁启动正式训练 (方案 §三.2)")
            sys.exit(1)
        print(f"manifest 模式 prelight 通过: z-score fit_patients={sorted(fit_pats)}, "
              f"leakage_pairs=0")

        labels_root_manifest = manifest_roots["labels"]

        # 构建三个数据集 (train + internal_val + external XZY)
        train_dataset, val_ds, external_ds = build_manifest_datasets(
            manifest_df=manifest_df,
            cache_root=args.cache_root,
            flat_cache_root=args.flat_cache_root,
            labels_root=str(labels_root_manifest),
            train_mpp_id=args.train_mpp_id,
            external_mpp_id=args.external_mpp_id,
            external_patient=args.external_patient,
            train_patients=train_patients,
            allow_missing=args.allow_missing,
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
        external_loader = DataLoader(external_ds, batch_size=args.batch_size,
                                     shuffle=False, num_workers=0)
        print(f"\n训练集总样本: {len(train_dataset)}")
        print(f"内部验证集样本: {len(val_ds)}")
        print(f"外部测试集样本: {len(external_ds)}")
        pathway_cols = external_ds.target_cols
        print(f"通路数: {len(pathway_cols)}")

        # z-score 参数 (供 raw-scale MAE/R² inverse transform)
        zscore_params_json_path = manifest_roots["zscore_params"]
        with open(zscore_params_json_path, "r", encoding="utf-8") as f:
            zscore_params_for_inverse = _json.load(f)
        # z-score 审计打印
        print(f"\nz-score 参数审计 (manifest): fit_split={zscore_params_for_inverse.get('fit_split')}, "
              f"ddof={zscore_params_for_inverse.get('ddof')}, "
              f"n_train_samples={zscore_params_for_inverse.get('n_train_samples')}, "
              f"fit_patients={zscore_params_for_inverse.get('fit_patients')}")
        print(f"  z-score 验证: mean 范围 {zscore_manifest.get('fit_verification', {}).get('train_z_mean_range')}, "
              f"std 范围 {zscore_manifest.get('fit_verification', {}).get('train_z_std_range')}")

    # ═══════════════════════════════════════════════════════════════
    # 旧模式 (none / internal / partner) 数据加载
    # ═══════════════════════════════════════════════════════════════
    if not use_manifest:
        if use_internal_val:
            print(f"\n训练集患者 ({len(train_patients)}): {train_patients}")
            print(f"内部验证患者: MPP-{args.train_mpp_id}/{args.val_patient}")
            print(f"外部测试患者: MPP-{args.external_mpp_id}/{args.external_patient}")
        else:
            print(f"\n训练集患者 ({len(train_patients)}): {train_patients}")
            print(f"外部测试患者: MPP-{args.external_mpp_id}/{args.external_patient}")

        # ── Preflight: 数据路径审计 ──
        _run_preflight(
            cache_root=args.cache_root,
            labels_root=args.labels_root,
            train_mpp_id=args.train_mpp_id,
            external_mpp_id=args.external_mpp_id,
            train_patients=train_patients,
            val_patient=args.val_patient,
            external_patient=args.external_patient,
            partner=partner,
            allow_missing=args.allow_missing,
        )

        print("\n加载训练集 ...")
        train_dataset = merge_mpp_patients(
            cache_root=args.cache_root,
            mpp_id=args.train_mpp_id,
            patients=train_patients,
            labels_root=args.labels_root,
            allow_missing=args.allow_missing,
            partner=partner,
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)
        print(f"训练集总样本: {len(train_dataset)}")

        # ── 内部验证集（val_strategy=internal） ──
        if use_internal_val:
            print(f"\n加载内部验证集 MPP-{args.train_mpp_id}/{args.val_patient} ...")
            val_cache = _build_cache_dir(args.cache_root, args.train_mpp_id,
                                         args.val_patient, partner, split="val")
            val_labels = _build_val_label(args.labels_root, args.train_mpp_id,
                                          args.val_patient, partner)
            val_ds = MPPFeatureDataset(
                cache_dir=val_cache,
                labels_csv=val_labels,
                allow_missing=args.allow_missing,
            )
            val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                    shuffle=False, num_workers=0)
            print(f"内部验证集样本: {len(val_ds)}")

        print(f"\n加载外部测试集 MPP-{args.external_mpp_id}/{args.external_patient} ...")
        # ── 外部测试缓存：partner 模式 XZY 固定从 MPP1 加载 (旧逻辑, 保留兼容) ──
        # 注: manifest 模式已在外部从 MPP-2/XZY 加载, 不走此分支
        if partner and args.external_patient == "XZY":
            _xzy_cache_mpp_id = 1
        else:
            _xzy_cache_mpp_id = args.train_mpp_id
        external_cache = _build_cache_dir(args.cache_root, _xzy_cache_mpp_id,
                                          args.external_patient, partner, split="train")
        external_labels = _build_external_label(args.labels_root, args.train_mpp_id,
                                                args.external_mpp_id,
                                                args.external_patient, partner)
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

        # ── z-score 参数验证（防泄漏） ──
        zscore_params = _load_zscore_params(args.labels_root, args.train_mpp_id, partner)
        if zscore_params is not None:
            fit_patients = set(zscore_params.get("fit_patients", []))
            val_patients_file = ", ".join(zscore_params.get("val_patients", []))
            ext_patient_file = zscore_params.get("external_patient", "")
            n_pw = zscore_params.get("n_pathways", len(zscore_params.get("pathways", [])))
            print(f"\nz-score 参数审计: fit_patients={sorted(fit_patients)}, "
                  f"val_patients=[{val_patients_file}], external_patient={ext_patient_file}, "
                  f"n_pathways={n_pw}")
            # 校验：训练患者 == fit_patients（训练集恰好由拟合参数的患者组成）
            train_set = set(train_patients)
            if train_set != fit_patients:
                print(f"[ERROR] train_patients ({sorted(train_set)}) 与 zscore fit_patients "
                      f"({sorted(fit_patients)}) 不一致")
                sys.exit(1)
            if args.val_patient and args.val_patient in fit_patients:
                print(f"[ERROR] val_patient ({args.val_patient}) 出现在 zscore fit_patients 中，"
                      f"不应参与拟合")
                sys.exit(1)
            if args.external_patient in fit_patients:
                print(f"[ERROR] external_patient ({args.external_patient}) 出现在 zscore fit_patients 中，"
                      f"不应参与拟合")
                sys.exit(1)
            print(f"  z-score 参数审计通过: fit_patients 不包含 "
                  f"{args.val_patient or 'N/A'} / {args.external_patient}")
        else:
            print(f"[WARN] z-score 参数文件不存在，跳过审计")

        # z-score 参数 (供 raw-scale MAE/R² inverse transform, 旧模式)
        # 尝试读 JSON; 若无则从 split 标记的 zscore_params 兜底
        zscore_params_for_inverse = None
        _inv_json = (Path(args.labels_root) / f"zscore_params_mpp{args.train_mpp_id}.json")
        if _inv_json.exists():
            with open(_inv_json, "r", encoding="utf-8") as _f:
                zscore_params_for_inverse = _json.load(_f)

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
    with open(out_dir / "model_params.txt", "w", encoding="utf-8") as f:
        f.write(f"dataset_name={args.dataset_name}\n")
        f.write(f"train_mpp_id={args.train_mpp_id}\n")
        f.write(f"train_patients={args.train_patients}\n")
        f.write(f"external_mpp_id={args.external_mpp_id}\n")
        f.write(f"external_patient={args.external_patient}\n")
        f.write(f"use_partner_paths={args.use_partner_paths}\n")
        f.write(f"val_strategy={args.val_strategy}\n")
        if use_manifest:
            f.write(f"split_manifest={args.splits_root}/group_{args.train_mpp_id}/split_manifest.csv\n")
            f.write(f"flat_cache_root={args.flat_cache_root}\n")
        if use_internal_val:
            f.write(f"val_patient={args.val_patient}\n")
        f.write(f"model=MPPMLPHead (Linear->GELU->Dropout->Linear)\n")
        f.write(f"in_dim={external_ds.feat_dim}, hidden={args.hidden_dim}, out={len(pathway_cols)}\n")
        f.write(f"dropout={args.dropout}, n_params={n_params}\n")
        f.write(f"num_epochs={args.num_epochs}, batch_size={args.batch_size}\n")
        f.write(f"lr={args.lr}, seed={args.seed}, num_threads={args.num_threads}\n")
        f.write(f"device={device}\n")
        f.write(f"created_at={datetime.now().isoformat()}\n")

    # ── 训练循环 ──
    history = []

    # 早停 / best checkpoint 判断指标
    # use_internal_val (V3bis 模式) 或 use_manifest (20260706 统一标准) 都用 val_loss 早停
    monitor_uses_val_loss = use_internal_val or use_manifest
    if monitor_uses_val_loss:
        early_stop_signal = "val_loss"
        best_loss = float("inf")
        print(f"\n早停信号: {early_stop_signal} (最小化, patience={args.patience}, min_delta={args.min_delta})")
    else:
        early_stop_signal = "train_loss"
        best_loss = float("inf")
        print(f"\n早停信号: {early_stop_signal} (最小化, patience={args.patience}, min_delta={args.min_delta})")

    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

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

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_pcc": round(train_pcc, 6),
            "is_best": False,
        }

        # ── 内部验证评估（val_strategy=internal 或 manifest） ──
        if monitor_uses_val_loss:
            val_loss, val_metrics, _, _ = evaluate(
                model, val_loader, criterion, device, pathway_cols,
            )
            val_pcc = val_metrics["pcc"]
            val_mae = val_metrics["mae"]
            val_r2 = val_metrics["r2"]
            row.update({
                "val_loss": round(val_loss, 6),
                "val_pcc": round(val_pcc, 6),
                "val_mae": round(val_mae, 6),
                "val_r2": round(val_r2, 6),
            })
            monitor_loss = val_loss
            monitor_label = "val_loss"
        else:
            monitor_loss = train_loss
            monitor_label = "train_loss"

        row["is_best"] = False
        history.append(row)

        # ── 日志 ──
        if monitor_uses_val_loss:
            print(f"Epoch {epoch:3d}/{args.num_epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"train_PCC={train_pcc:.4f}  val_PCC={val_pcc:.4f}  "
                  f"time={elapsed:.1f}s", end="")
        else:
            print(f"Epoch {epoch:3d}/{args.num_epochs}  "
                  f"loss={train_loss:.4f}  PCC={train_pcc:.4f}  "
                  f"time={elapsed:.1f}s", end="")

        # ── 早停检查 ──
        if np.isfinite(monitor_loss):
            if monitor_loss < best_loss - args.min_delta:
                best_loss = monitor_loss
                best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                best_epoch = epoch
                epochs_no_improve = 0
                # 回填 is_best
                for h in history:
                    h["is_best"] = (h["epoch"] == epoch)
                print(f"  ✅ best ({monitor_label}={monitor_loss:.6f})")
            else:
                epochs_no_improve += 1
                print(f"  (no improv. {epochs_no_improve}/{args.patience})")
        else:
            epochs_no_improve += 1
            print(f"  ⚠️ NaN/Inf {monitor_label}, 跳过 best 更新 (no improv. {epochs_no_improve}/{args.patience})")

        min_stop_epoch = max(args.patience, 3)
        if epochs_no_improve >= args.patience and epoch >= min_stop_epoch:
            print(f"\n[早停] {monitor_label} {args.patience} 个 epoch 未改善，"
                  f"提前停止 (best epoch={best_epoch}, best_{monitor_label}={best_loss:.6f}, "
                  f"min_stop_epoch={min_stop_epoch})")
            break

    # ── 选择最佳 checkpoint ──
    # monitor_loss 在循环内每次赋值，循环至少执行一次（num_epochs 必传），post-loop 可安全引用
    if best_state is not None and best_epoch != args.num_epochs:
        model.load_state_dict(best_state)
        final_epoch = best_epoch
        final_loss = best_loss
        print(f"\n使用最佳 checkpoint (epoch {best_epoch}, {early_stop_signal}={best_loss:.6f}) 进行评估")
    else:
        final_epoch = epoch
        final_loss = monitor_loss  # 使用监控信号（train_loss 或 val_loss），而非总取 train_loss
        print(f"\n使用最终 checkpoint (epoch {epoch}) 进行评估")

    # ── 保存历史 ──
    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "training_history.csv", index=False)
    print(f"\n训练历史已保存: {out_dir / 'training_history.csv'}")

    # ── 保存 best epoch 标记 ──
    best_epoch_file = out_dir / "best_epoch.txt"
    with open(best_epoch_file, "w", encoding="utf-8") as f:
        f.write(f"best_epoch={best_epoch}\n")
        f.write(f"val_strategy={args.val_strategy}\n")
        f.write(f"early_stop_signal={early_stop_signal}\n")
        f.write(f"best_{early_stop_signal}={best_loss:.6f}\n")
        f.write(f"actual_final_epoch={epoch}\n")
    print(f"Best epoch 标记: {best_epoch_file} (best_epoch={best_epoch}, final_epoch={epoch})")

    # ── 保存最佳 checkpoint ──
    ckpt_path = out_dir / "best_checkpoint.pth"
    ckpt_meta = {"epoch": best_epoch, "val_strategy": args.val_strategy,
                 early_stop_signal: best_loss}
    if best_state is not None:
        ckpt_meta["model_state_dict"] = best_state
        torch.save(ckpt_meta, ckpt_path)
    else:
        ckpt_meta["model_state_dict"] = model.state_dict()
        torch.save(ckpt_meta, ckpt_path)
    print(f"Best checkpoint (epoch {best_epoch}): {ckpt_path}")

    # ── 内部验证集最终评估（val_strategy=internal 或 manifest） ──
    if monitor_uses_val_loss:
        print(f"\n{'='*60}")
        _val_label = (f"MPP-{args.train_mpp_id}/{args.val_patient}"
                      if use_internal_val else f"MPP-{args.train_mpp_id}/manifest patch级")
        print(f"最终评估内部验证集 {_val_label} (best checkpoint) ...")
        val_loss_final, val_metrics_final, val_preds_final, val_labels_final = evaluate(
            model, val_loader, criterion, device, pathway_cols,
        )
        print(f"Internal Val: PCC={val_metrics_final['pcc']:.4f}  "
              f"MAE={val_metrics_final['mae']:.4f}  R²={val_metrics_final['r2']:.4f}  "
              f"Loss={val_loss_final:.4f}")

        val_pred_cols = [f"pred_{c}" for c in pathway_cols]
        val_true_cols = [f"true_{c}" for c in pathway_cols]
        val_pred_df = pd.DataFrame(np.column_stack([val_labels_final, val_preds_final]),
                                    columns=val_true_cols + val_pred_cols)
        val_pred_path = out_dir / "predictions_internal_val.csv"
        val_pred_df.to_csv(val_pred_path, index=False)
        print(f"内部验证预测: {val_pred_path}")

    # ── 外部评估 ──
    print(f"\n{'='*60}")
    print(f"评估外部测试集 MPP-{args.external_mpp_id}/{args.external_patient} ...")
    test_loss, test_metrics, test_preds, test_labels = evaluate(
        model, external_loader, criterion, device, pathway_cols,
    )
    mean_pcc = test_metrics["pcc"]
    mean_mae = test_metrics["mae"]
    mean_r2 = test_metrics["r2"]
    print(f"External {args.external_patient}: PCC={mean_pcc:.4f}  MAE={mean_mae:.4f}  "
          f"R²={mean_r2:.4f}  Loss={test_loss:.4f}")

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

    # ── 逐通路 PCC + raw-scale MAE/R² (方案 §4.3 P1 必交付) ──
    # z-score 空间指标现存 per_pathway; raw-scale 通过 inverse transform 还原:
    #   pred_raw = pred_z * std + mean; label_raw = label_z * std + mean
    #   (mean/std 来自 train-only, 保存在 zscore_params_from_train.json)
    # 注: R² 对线性变换不变 → r2_raw == r2_z; MAE_raw = MAE_z * std
    raw_params = None
    if zscore_params_for_inverse is not None:
        raw_params = zscore_params_for_inverse.get("pathways", {})
    if raw_params is None and use_manifest:
        # manifest 模式: 已加载 zscore_params_for_inverse; 此分支兜底
        _inv_json = (Path(args.splits_root) / f"group_{args.train_mpp_id}"
                     / "zscore_params_from_train.json")
        if _inv_json.exists():
            with open(_inv_json, "r", encoding="utf-8") as _f:
                raw_params = _json.load(_f).get("pathways", {})

    per_pathway = []
    per_pathway_raw = []
    for i, col in enumerate(pathway_cols):
        z_labels = test_labels[:, i]
        z_preds = test_preds[:, i]
        # z-space 指标
        pcc, _ = pearsonr(z_labels, z_preds)
        mae_z = np.abs(z_labels - z_preds).mean()
        ss_res = ((z_labels - z_preds) ** 2).sum()
        ss_tot = ((z_labels - z_labels.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # raw-space 指标 (inverse transform)
        if raw_params is not None and col in raw_params:
            mu = raw_params[col]["mean"]
            sd = raw_params[col]["std"]
            labels_raw = z_labels * sd + mu
            preds_raw = z_preds * sd + mu
            mae_raw = float(np.abs(labels_raw - preds_raw).mean())
            r2_raw = float(r2)  # 不变
        else:
            mae_raw = None
            r2_raw = None

        per_pathway.append({"pathway": col, "pcc": pcc, "r2": r2, "mae": mae_z,
                            "mae_raw": mae_raw, "r2_raw": r2_raw})
        if mae_raw is not None:
            per_pathway_raw.append({"pathway": col, "pcc": pcc, "mae_raw": mae_raw,
                                    "r2_raw": r2_raw})

    pp_df = pd.DataFrame(per_pathway).sort_values("pcc", ascending=False)
    pp_path = out_dir / "per_pathway_pcc_external_xzy.csv"
    pp_df.to_csv(pp_path, index=False)
    print(f"逐通路 PCC (z-space + raw-scale): {pp_path}")

    # 整体 raw-scale MAE/R² (跨通路均值, 用于 registry 比对)
    if per_pathway_raw:
        mean_mae_raw = float(np.mean([p["mae_raw"] for p in per_pathway_raw]))
        mean_r2_raw = float(np.mean([p["r2_raw"] for p in per_pathway_raw]))
        pp_raw_path = out_dir / "per_pathway_pcc_external_xzy_rawscale.csv"
        pd.DataFrame(per_pathway_raw).sort_values("pcc", ascending=False).to_csv(pp_raw_path, index=False)
        print(f"逐通路 raw-scale: {pp_raw_path}")
        print(f"External Raw scale: MAE_raw={mean_mae_raw:.4f}  R²_raw={mean_r2_raw:.4f}")
    else:
        mean_mae_raw = None
        mean_r2_raw = None
        print(f"[WARN] 无 raw-scale 参数, 跳过 raw-scale MAE/R² (z-space 指标注明 z-score 空间)")

    print(f"\nTop-3 通路: {pp_df.head(3).to_dict('records')}")
    print(f"Bottom-3 通路: {pp_df.tail(3).to_dict('records')}")

    # ── 训练摘要 ──
    with open(out_dir / "training_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Experiment: {args.dataset_name}\n")
        f.write(f"Model: UNI2-h frozen features + 2-layer MLP\n")
        f.write(f"Training strategy: val_strategy={args.val_strategy}\n")
        f.write(f"Early stopping: signal={early_stop_signal}, patience={args.patience}, "
                f"min_delta={args.min_delta}\n")
        f.write(f"Training budget: {args.num_epochs} epochs max\n")
        f.write(f"  actual_epochs={epoch}, best_epoch={best_epoch}, "
                f"best_{early_stop_signal}={best_loss:.6f}\n")
        f.write(f"\nSignal separation:\n")
        f.write(f"- train: MPP-{args.train_mpp_id} / {args.train_patients}\n")
        if monitor_uses_val_loss:
            if use_manifest:
                f.write(f"- internal_val: split_manifest patch级各患者10% (manifest), "
                        f"used for {early_stop_signal} early stopping only\n")
            else:
                f.write(f"- internal_val: MPP-{args.train_mpp_id} / {args.val_patient}, "
                        f"used for {early_stop_signal} early stopping only\n")
        else:
            f.write(f"- internal_val: none (val_strategy=none)\n")
        f.write(f"- external_test: MPP-{args.external_mpp_id} / {args.external_patient}, "
                f"evaluated once after checkpoint selection\n")
        f.write(f"\nXZY used for: test ONLY (not training, not z-score fit, not epoch selection)\n")
        f.write(f"Report: best checkpoint selected by {early_stop_signal} minimum; "
                f"external XZY evaluated once.\n")
        f.write(f"\n--- External {args.external_patient} Results ---\n")
        f.write(f"Mean PCC: {mean_pcc:.4f}\n")
        f.write(f"Mean MAE: {mean_mae:.4f} (z-score space)\n")
        f.write(f"Mean R2:  {mean_r2:.4f}\n")
        if mean_mae_raw is not None:
            f.write(f"Mean MAE_raw: {mean_mae_raw:.4f} (raw ssGSEA scale, 跨MPP可比)\n")
            f.write(f"Mean R2_raw:  {mean_r2_raw:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        if monitor_uses_val_loss:
            f.write(f"\n--- Internal Val Results (best ckpt) ---\n")
            f.write(f"Mean PCC: {val_metrics_final['pcc']:.4f}\n")
            f.write(f"Mean MAE: {val_metrics_final['mae']:.4f}\n")
            f.write(f"Mean R2:  {val_metrics_final['r2']:.4f}\n")
            f.write(f"Val Loss: {val_loss_final:.4f}\n")
        f.write(f"Completed at: {datetime.now().isoformat()}\n")

    print(f"\n摘要: {out_dir / 'training_summary.txt'}")
    print(f"\n{'='*60}")
    print("训练完成!")


if __name__ == "__main__":
    sys.exit(main())
