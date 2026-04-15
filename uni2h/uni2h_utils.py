
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
) -> Tuple[torch.nn.Module, callable, int]:
    """
    Load frozen UNI2-h backbone and its official preprocessing transform.
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

    # 从HF加载
    model = timm.create_model(f"hf-hub:{DEFAULT_MODEL_ID}", pretrained=True, **timm_kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if device is not None:
        model.to(device)

    # 官方预处理
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform, DEFAULT_FEATURE_DIM



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
