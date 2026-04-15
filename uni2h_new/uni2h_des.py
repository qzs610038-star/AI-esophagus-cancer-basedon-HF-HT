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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error # Import MAPE

DEFAULT_MODEL_ID = "MahmoodLab/UNI2-h"
DEFAULT_FEATURE_DIM = 1536 # UNI2-h输出特征维度
DEFAULT_TARGET_START_COL = 1   # 标签从 CSV 的第几列开始，第 2 列开始是 N 个指标
# DEFAULT_NUM_TARGETS = 8 # # 要预测几个指标 (修改为 30)
DEFAULT_NUM_TARGETS = 30 # # 要预测几个指标 (修改为 30)

# Huggingface登录
def ensure_hf_login(token: Optional[str] = None) -> None:
    token = token or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        # 尝试注释掉 login 调用，因为已设置 HF_HUB_LOCAL_FILES_ONLY=1
        # login(token=token)
        pass # 假设 token 已通过环境变量等方式设置
    else:
        print("Warning: Hugging Face token not provided. Loading public model might fail or be rate-limited.")
        # 即使没有_HUB_LOCAL_FILES_ONLY=1 且文件已缓存，加载仍可能成功


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


# Z-score标准化
def zscore_dataframe(
    df: pd.DataFrame,
    target_start_col: int = DEFAULT_TARGET_START_COL,
    num_targets: int = DEFAULT_NUM_TARGETS,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Z-score normalize target columns only.
    """
    out_df = df.copy()

    target_cols = list(out_df.columns[target_start_col:target_start_col + num_targets])
    if len(target_cols) != num_targets:
        raise ValueError(
            f"Expected {num_targets} target columns starting from index {target_start_col}, "
            f"but found {len(target_cols)}."
        )

    values = out_df[target_cols].astype(np.float64)
    means = values.mean(axis=0)
    stds = values.std(axis=0, ddof=0).replace(0, 1.0)
    out_df[target_cols] = (values - means) / stds
    return out_df, means, stds


# 保证存在zscore CSV
def ensure_zscore_csv(
    raw_csv_path: str,
    zscore_csv_path: str,
    target_start_col: int = DEFAULT_TARGET_START_COL,
    num_targets: int = DEFAULT_NUM_TARGETS,
) -> str:
    """
    Create a z-scored CSV if it does not already exist.
    """
    raw_csv_path = str(raw_csv_path)
    zscore_csv_path = str(zscore_csv_path)
    if os.path.exists(zscore_csv_path):
        return zscore_csv_path

    df = pd.read_csv(raw_csv_path)
    z_df, means, stds = zscore_dataframe(df, target_start_col=target_start_col, num_targets=num_targets)
    Path(zscore_csv_path).parent.mkdir(parents=True, exist_ok=True)
    z_df.to_csv(zscore_csv_path, index=False)

    stats_path = os.path.splitext(zscore_csv_path)[0] + "_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_start_col": target_start_col,
                "num_targets": num_targets,
                "columns": list(df.columns[target_start_col:target_start_col + num_targets]),
                "means": means.to_dict(),
                "stds": stds.to_dict(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return zscore_csv_path


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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Calculate overall and per-target metrics (MSE, MAE, MAPE, R², PCC).
    Args:
        y_true (np.ndarray): Ground truth labels, shape (N, num_targets)
        y_pred (np.ndarray): Predictions, shape (N, num_targets)
    Returns:
        dict: Dictionary containing overall and per-target metrics.
    """
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
    per_target_mape = [] # Add list for MAPE
    per_target_r2 = []
    per_target_pcc = []

    for j in range(num_targets):
        yt = y_true[:, j]
        yp = y_pred[:, j]

        per_target_mse.append(float(mean_squared_error(yt, yp)))
        per_target_mae.append(float(mean_absolute_error(yt, yp)))

        # MAPE calculation
        # Handle potential division by zero and infinite values
        mask = (yt != 0) & (~np.isinf(1/yt)) # Mask where true value is not zero and 1/y_true is finite
        if np.any(mask):
            mape_vals = np.abs((yt[mask] - yp[mask]) / yt[mask])
            # Check for inf values in the calculated MAPE parts
            mape_vals = mape_vals[np.isfinite(mape_vals)]
            if len(mape_vals) > 0:
                 per_target_mape.append(float(np.mean(mape_vals) * 100)) # Convert to percentage
            else:
                 per_target_mape.append(float('nan')) # All masked values led to inf MAPE
        else:
             per_target_mape.append(float('nan')) # No valid values for MAPE calculation for this target

        if np.std(yt) == 0:
            per_target_r2.append(np.nan)
        else:
            per_target_r2.append(float(r2_score(yt, yp)))

        per_target_pcc.append(float(pearson_corrcoef(yt, yp)))

    # Calculate overall metrics
    overall_mse = float(np.nanmean(per_target_mse))
    overall_mae = float(np.nanmean(per_target_mae))
    overall_mape = float(np.nanmean(per_target_mape)) # Calculate overall MAPE
    overall_r2 = float(np.nanmean(per_target_r2))
    overall_pcc = float(np.nanmean(per_target_pcc))

    # Prepare return dictionary
    metrics = {
        'overall': {
            "mse": overall_mse,
            "mae": overall_mae,
            "mape": overall_mape, # Add overall MAPE
            "r2": overall_r2,
            "pcc": overall_pcc,
        },
        'per_target': {}
    }

    # Add per-target metrics to the dictionary
    for j in range(num_targets):
         metrics['per_target'][f'target_{j}'] = {
            "mse": per_target_mse[j],
            "mae": per_target_mae[j],
            "mape": per_target_mape[j], # Add per-target MAPE
            "r2": per_target_r2[j],
            "pcc": per_target_pcc[j]
        }

    return metrics # Return the comprehensive dictionary


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


# --- DenseNet121 Style MLP Regressor (参照 DenseNet-121 结构) ---
class DenseNet121StyleRegressor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        initial_dim: int,
        growth_rate: int,
        bottleneck_factor: int, # Factor for bottleneck width: bottleneck_width = bottleneck_factor * growth_rate
        transition_factor: float, # Factor for transition layer output dim: transition_dim = current_block_output_dim * transition_factor
        output_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.initial_layer = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, initial_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Define the number of layers per block according to DenseNet-121
        num_layers_per_block_list = [6, 12, 24, 16] # DenseNet-121 structure
        current_dim = initial_dim
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(num_layers_per_block_list):
            block = self._make_dense_block(
                input_dim=current_dim,
                num_layers=num_layers,
                growth_rate=growth_rate,
                bottleneck_factor=bottleneck_factor,
                dropout=dropout
            )
            self.dense_blocks.append(block)

            block_output_dim = current_dim + num_layers * growth_rate
            # Add transition layer except after the last block
            if i < len(num_layers_per_block_list) - 1:
                transition_dim = int(block_output_dim * transition_factor)
                transition_layer = nn.Sequential(
                    nn.Linear(block_output_dim, transition_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
                self.transition_layers.append(transition_layer)
                current_dim = transition_dim # Update dim for next block
            else:
                current_dim = block_output_dim # Update dim for final head

        # Final regression head
        self.final_head = nn.Sequential(
            nn.Linear(current_dim, output_dim),
            # Optionally add more layers here if needed
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            # nn.Linear(intermediate_dim, output_dim),
        )

    def _make_dense_block(self, input_dim, num_layers, growth_rate, bottleneck_factor, dropout):
        layers = nn.ModuleList()
        current_dim = input_dim

        for _ in range(num_layers):
            bottleneck_width = bottleneck_factor * growth_rate
            layer = nn.Sequential(
                nn.Linear(current_dim, bottleneck_width),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_width, growth_rate),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            layers.append(layer)
            current_dim += growth_rate # Update for next layer's input
        return layers

    def forward(self, x):
        x = self.initial_layer(x)

        for i, block in enumerate(self.dense_blocks):
            block_features = [x] # Start with the input to the block

            # Iterate through each layer (dense unit) in the block
            for j, layer in enumerate(block):
                # Concatenate all previous features within the *same* block as input to the current layer
                layer_input = torch.cat(block_features, dim=1)

                # Pass the concatenated features through the current layer (dense unit)
                new_features = layer(layer_input)

                # Append the output of the current layer to the list for future layers in the same block
                block_features.append(new_features)

            # After processing all layers in the block, the final output of the block
            # is the concatenation of the initial input and all the layer outputs within the block.
            x = torch.cat(block_features, dim=1) # x is now the output of the entire block

            # Apply transition layer if it's not the last block
            if i < len(self.transition_layers):
                x = self.transition_layers[i](x)

        x = self.final_head(x)
        return x

# --- End of DenseNet121 Style MLP Regressor ---

# --- Original MLP Regressor (Commented out) ---
# class BackboneRegressor(nn.Module):
#     def __init__(
#         self,
#         feature_dim: int,
#         hidden_dim: int, # 第一层隐藏层维度
#         output_dim: int,
#         dropout: float,
#     ):
#         super().__init__()
#
#         # 定义第二层隐藏层维度 (可以根据需要调整，例如设为 hidden_dim // 2)
#         hidden_dim_2 = hidden_dim // 2 # 例如，第二层维度是第一层的一半
#
#         self.net = nn.Sequential(
#             nn.LayerNorm(feature_dim),           # 对输入特征进行 LayerNorm
#             nn.Linear(feature_dim, hidden_dim), # 第一层: feature_dim -> hidden_dim
#             nn.GELU(),                          # 激活函数
#             nn.Dropout(dropout),                # Dropout 正则化
#             nn.Linear(hidden_dim, hidden_dim_2), # 第二层: hidden_dim -> hidden_dim_2
#             nn.GELU(),                          # 激活函数
#             nn.Dropout(dropout),                # Dropout 正则化
#             nn.Linear(hidden_dim_2, output_dim), # 输出层: hidden_dim_2 -> output_dim
#         )
#
#     def forward(self, x):
#         return self.net(x)
# --- End of Original MLP Regressor ---


# --- Function to calculate Max Absolute Difference per target ---
def calculate_max_abs_diff_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate the maximum absolute difference between predictions and ground truth for each target.
    Args:
        y_true (np.ndarray): Ground truth labels, shape (N, num_targets)
        y_pred (np.ndarray): Predictions, shape (N, num_targets)
    Returns:
        np.ndarray: Array of shape (num_targets,) containing the maximum absolute difference for each target.
    """
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

    # Calculate absolute differences
    abs_diff = np.abs(y_true - y_pred) # Shape: (N, num_targets)
    # Find the maximum along the sample axis (axis=0)
    max_abs_diff_per_target = np.max(abs_diff, axis=0) # Shape: (num_targets,)

    return max_abs_diff_per_target
# --- End of function ---


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
    metrics_dict = compute_metrics(y_true, y_pred) # Get the full dictionary
    # Extract overall metrics for loss calculation and return
    overall_metrics = metrics_dict['overall']
    overall_metrics["loss"] = running_loss / max(len(dataloader), 1)
    return overall_metrics, metrics_dict # <--- Return both for consistency with evaluate


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
    metrics_dict = compute_metrics(y_true, y_pred) # Get the full dictionary
    # Extract overall metrics for loss calculation and return
    overall_metrics = metrics_dict['overall']
    overall_metrics["loss"] = running_loss / max(len(dataloader), 1)
    return overall_metrics, metrics_dict # Return both overall and the full metrics dict