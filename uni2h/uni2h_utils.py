
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from huggingface_hub import login
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import Dataset


DEFAULT_MODEL_ID = "MahmoodLab/UNI2-h"
DEFAULT_FEATURE_DIM = 1536 # UNI2-h输出特征维度
DEFAULT_TARGET_START_COL = 1   # 标签从 CSV 的第几列开始，第 2 列开始是基因集分数
DEFAULT_NUM_TARGETS = 8 # # 要预测几个基因集分数

# Huggingface登录
def ensure_hf_login(token: Optional[str] = None) -> None:
    token = token or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)


# 加载UNI2-h backbone
def load_uni2h_backbone(
    token: Optional[str] = None,
    device: Optional[torch.device] = None,
    local_cache_dir: Optional[str] = None,
) -> Tuple[torch.nn.Module, callable, int]:
    """
    Load frozen UNI2-h backbone and its official preprocessing transform.

    Loading priority:
    1. local_cache_dir: direct path to pytorch_model.bin or parent dir
    2. HF_HOME cache (searches automatically via huggingface_hub)
    3. HuggingFace Hub online (if not offline)

    Returns: model, transform, feature_dim
    """
    ensure_hf_login(token)

    # UNI2-h官方结构参数
    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
                "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }

    local_ckpt = _find_local_uni2h_ckpt(local_cache_dir)
    if local_ckpt is not None:
        model, transform = _load_uni2h_from_local(local_ckpt, timm_kwargs, device)
    else:
        model = _load_uni2h_from_hfhub(timm_kwargs, device)
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if device is not None:
        model.to(device)
    return model, transform, DEFAULT_FEATURE_DIM


def _load_uni2h_from_local(
    ckpt_path: str,
    timm_kwargs: dict,
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, callable]:
    """从本地 pytorch_model.bin 直接构造 UNI2-h 模型（完全离线）。"""
    from timm.models.vision_transformer import VisionTransformer

    print(f"[UNI2-h] 本地加载: {ckpt_path}")

    # 直接构造 VisionTransformer，不依赖任何 HF hub 或 pretrained_cfg
    model = VisionTransformer(
        img_size=timm_kwargs["img_size"],
        patch_size=timm_kwargs["patch_size"],
        depth=timm_kwargs["depth"],
        num_heads=timm_kwargs["num_heads"],
        init_values=timm_kwargs["init_values"],
        embed_dim=timm_kwargs["embed_dim"],
        mlp_ratio=timm_kwargs["mlp_ratio"],
        num_classes=timm_kwargs["num_classes"],
        no_embed_class=timm_kwargs["no_embed_class"],
        global_pool="",
        mlp_layer=timm_kwargs["mlp_layer"],
        act_layer=timm_kwargs["act_layer"],
        reg_tokens=timm_kwargs["reg_tokens"],
        dynamic_img_size=timm_kwargs["dynamic_img_size"],
    )

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # UNI2-h 权重有 head 相关键，去掉它们（num_classes=0）
    state = {k: v for k, v in state.items()
             if not k.startswith("head.") and not k.startswith("pre_logits.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  [INFO] unexpected keys (skipped): {len(unexpected)}")

    # UNI2-h 预处理（已知常量，不依赖 pretrained_cfg）
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    transform = create_transform(
        input_size=timm_kwargs["img_size"],
        is_training=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        interpolation="bicubic",
    )
    return model, transform


def _load_uni2h_from_hfhub(
    timm_kwargs: dict,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """通过 HuggingFace Hub 加载 UNI2-h（需网络）。"""
    print("[UNI2-h] 从 HuggingFace Hub 加载 ...")
    model = timm.create_model(f"hf-hub:{DEFAULT_MODEL_ID}", pretrained=True, **timm_kwargs)
    return model


def _find_local_uni2h_ckpt(cache_dir: Optional[str] = None) -> Optional[str]:
    """在 HF 缓存目录中查找 UNI2-h 的 pytorch_model.bin。

    查找策略（优先级）：
    1. cache_dir 参数（显式传入的路径）
    2. 通过 refs/main → snapshot hash 直接构建路径（最可靠）
    3. rglob 全局搜索（兜底）
    """
    # ── 策略 1：显式 cache_dir ──
    if cache_dir is not None:
        p = Path(cache_dir)
        if p.is_file() and p.name == "pytorch_model.bin":
            return str(p)
        candidates = list(p.rglob("pytorch_model.bin"))
        if candidates:
            return str(candidates[0])

    # ── 确定 hub 缓存根 ──
    hf_home = (os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE") or "").strip()
    if hf_home:
        hub_dir = os.path.join(hf_home, "hub")
    else:
        hub_dir = os.path.join(str(Path.home()), ".cache", "huggingface", "hub")

    model_cache = os.path.join(hub_dir, f"models--{DEFAULT_MODEL_ID.replace('/', '--')}")

    # ── 策略 2：通过 refs/main → snapshot hash 直接构建路径 ──
    refs_main = os.path.join(model_cache, "refs", "main")
    if os.path.isfile(refs_main):
        try:
            with open(refs_main, "r") as f:
                snapshot_hash = f.read().strip()
            if snapshot_hash:
                ckpt = os.path.join(model_cache, "snapshots", snapshot_hash, "pytorch_model.bin")
                if os.path.isfile(ckpt):
                    return ckpt
        except Exception as e:
            print(f"[WARN] _find_local_uni2h_ckpt: 读取 refs/main 失败: {e}")

    # ── 策略 3：rglob 兜底（用 Path，因为 rglob 是 Path 的方法） ──
    root = Path(model_cache)
    if root.exists():
        candidates = list(root.rglob("pytorch_model.bin"))
        real = [c for c in candidates if ".incomplete" not in str(c)]
        if real:
            return str(real[0])

    print(f"[WARN] _find_local_uni2h_ckpt: 在 {model_cache} 中未找到 pytorch_model.bin")
    return None



# PCC 指标
def pearson_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Pearson PCC for one target (1D arrays) or already-selected vectors.
    """
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)

    if yt.size == 0 or yp.size == 0:
        return float("nan")
    if np.std(yt) == 0 or np.std(yp) == 0:
        return float("nan")

    return float(np.corrcoef(yt, yp)[0, 1])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError(
            f"Expected y_true and y_pred to be 2D arrays of shape [N, num_targets], "
            f"but got {y_true.shape} and {y_pred.shape}."
        )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    num_targets = y_true.shape[1]

    per_target_mse = []
    per_target_mae = []
    per_target_r2 = []
    per_target_pcc = []

    for j in range(num_targets):
        yt = y_true[:, j]
        yp = y_pred[:, j]

        per_target_mse.append(float(mean_squared_error(yt, yp)))
        per_target_mae.append(float(mean_absolute_error(yt, yp)))

        if np.std(yt) == 0:
            per_target_r2.append(np.nan)
        else:
            per_target_r2.append(float(r2_score(yt, yp)))

        per_target_pcc.append(float(pearson_corrcoef(yt, yp)))

    metrics = {
        "mse": float(np.nanmean(per_target_mse)),
        "mae": float(np.nanmean(per_target_mae)),
        "r2": float(np.nanmean(per_target_r2)),
        "pcc": float(np.nanmean(per_target_pcc)),
    }
    return metrics


# 提取UNI2-h特征并缓存
def extract_and_cache_features(
    backbone: torch.nn.Module,
    transform,
    patches_dir: str,
    cache_dir: str,
    device: torch.device,
    rebuild: bool = False,
) -> int:
    """
    Extract UNI2-h features and save each patch embedding as a .pt file.
    """
    patches_dir = Path(patches_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in patches_dir.iterdir() if p.suffix.lower() == ".png"])
    num_written = 0

    backbone.eval()
    with torch.inference_mode():
        for img_path in image_files:
            cache_path = cache_dir / f"{img_path.stem}.pt"
            if cache_path.exists() and not rebuild:
                continue

            image = Image.open(img_path).convert("RGB")
            x = transform(image).unsqueeze(0).to(device, non_blocking=True)
            feat = backbone(x).squeeze(0).detach().cpu().float()
            torch.save(feat, cache_path)
            num_written += 1

    return num_written


# Dataset
class CachedFeaturePatchDataset(Dataset):
    def __init__(
        self,
        patches_dir: str,
        labels_csv: str,
        feature_cache_dir: str,
        target_start_col: int = DEFAULT_TARGET_START_COL,
        num_targets: int = DEFAULT_NUM_TARGETS,
    ):
        self.patches_dir = Path(patches_dir)
        self.feature_cache_dir = Path(feature_cache_dir)
        self.labels_df = pd.read_csv(labels_csv)

        patch_keys = self.labels_df.iloc[:, 0].astype(str).map(lambda x: Path(x).stem).tolist()
        self.patch_to_idx = {k: i for i, k in enumerate(patch_keys)}

        self.target_cols = list(self.labels_df.columns[target_start_col:target_start_col + num_targets])
        if len(self.target_cols) != num_targets:
            raise ValueError(
                f"Expected {num_targets} target columns starting from index {target_start_col}, "
                f"but found {len(self.target_cols)}."
            )

        self.targets = self.labels_df[self.target_cols].values.astype(np.float32)

        self.patch_files = []
        for p in sorted(self.patches_dir.iterdir()):
            if p.suffix.lower() != ".png":
                continue
            if p.stem in self.patch_to_idx:
                self.patch_files.append(p)

        if len(self.patch_files) == 0:
            raise RuntimeError(f"No PNG patches in {self.patches_dir} matched labels in {labels_csv}")

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        img_path = self.patch_files[idx]
        csv_idx = self.patch_to_idx[img_path.stem]
        target = torch.tensor(self.targets[csv_idx], dtype=torch.float32)

        feat_path = self.feature_cache_dir / f"{img_path.stem}.pt"
        if not feat_path.exists():
            raise FileNotFoundError(f"Missing feature cache: {feat_path}")
        feature = torch.load(feat_path, map_location="cpu")
        if isinstance(feature, dict) and "feature" in feature:
            feature = feature["feature"]
        feature = feature.float()
        # feature = torch.nn.functional.normalize(feature, dim=-1)   #=======================

        return feature, target


class BackboneRegressor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# 训练
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_outputs = []

    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        outputs = model(features)
        loss = criterion(outputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_targets.append(targets.detach().cpu().numpy())
        all_outputs.append(outputs.detach().cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_outputs, axis=0)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = running_loss / max(len(dataloader), 1)
    return metrics


# 验证
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)

            outputs = model(features)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            all_targets.append(targets.detach().cpu().numpy())
            all_outputs.append(outputs.detach().cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_outputs, axis=0)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = running_loss / max(len(dataloader), 1)
    return metrics
