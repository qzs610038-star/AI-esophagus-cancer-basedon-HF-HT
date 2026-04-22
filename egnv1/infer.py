"""
EGN-v1 独立推理脚本
加载 checkpoint + 缓存特征 → 生成 predictions.csv
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data

# 将项目根目录加入 sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from egnv1.model import EGNv1Model
from egnv1.exemplar_builder import compute_exemplar_agg_features
from egnv1.dataset import EGNv1Dataset
import torchvision.transforms as transforms

# 从 config 读取设备
try:
    from config_utils import load_config, get_device
    _config = load_config()
except Exception:
    _config = None


def get_transforms():
    """构建验证用图像变换"""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])


def main():
    parser = argparse.ArgumentParser(description="EGN-v1 Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="best_egnv1.pth 路径")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="缓存目录（含特征/图/代表库文件）")
    parser.add_argument("--dataset_name", type=str, default="HYZ15040")
    parser.add_argument("--val_patches_dir", type=str, default=None)
    parser.add_argument("--labels_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录，默认为 checkpoint 同级 results_vis")
    args = parser.parse_args()

    # 设备
    if _config is not None:
        device = get_device(_config)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 加载 checkpoint
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    ckpt_args = argparse.Namespace(**ckpt['args'])
    target_cols = ckpt['target_cols']

    # 恢复模型
    model = EGNv1Model(
        in_dim=ckpt_args.hidden_dim,  # ViT 输出 1024，已通过 feature_proj
        hidden_dim=ckpt_args.hidden_dim,
        n_targets=ckpt_args.n_targets,
        graph_layers=ckpt_args.graph_layers,
        dropout=ckpt_args.dropout,
        k_exemplars=ckpt_args.k_neighbors,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[INFO] 模型加载完成，参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 加载缓存特征
    cache_dir = args.cache_dir or str(Path(args.checkpoint).parent.parent / "cache" / args.dataset_name)
    val_feat_path = os.path.join(cache_dir, f"{args.dataset_name}_val_features.pth")
    val_graph_path = os.path.join(cache_dir, f"{args.dataset_name}_val_graph.pth")
    exemplar_path = os.path.join(cache_dir, f"{args.dataset_name}_exemplars.pth")

    from egnv1.model import ExemplarLibrary

    val_data_cache = torch.load(val_feat_path, weights_only=False)
    val_features = val_data_cache['features']
    val_targets = val_data_cache['targets']

    val_edge_index = torch.load(val_graph_path, weights_only=False)['edge_index']
    exemplar_lib = ExemplarLibrary.load(exemplar_path)

    # 计算 exemplar 聚合特征
    val_exemplar_agg, _ = compute_exemplar_agg_features(
        val_features, exemplar_lib, ckpt_args.hidden_dim,
        k=ckpt_args.k_neighbors, device=device,
    )

    # 构建 PyG Data
    val_data = Data(
        x=val_features.to(device),
        edge_index=val_edge_index.to(device),
        y=val_targets.to(device),
    ).to(device)

    # 推理
    with torch.no_grad():
        preds = model(val_data.x, val_data.edge_index, val_exemplar_agg)

    preds_np = preds.cpu().numpy()
    labels_np = val_data.y.cpu().numpy()

    # 生成 predictions.csv（列名: true_{pathway}/pred_{pathway}，与 visualize_results.py 约定一致）
    pred_df = pd.DataFrame()
    for i, col in enumerate(target_cols):
        pred_df[f'true_{col}'] = labels_np[:, i]
        pred_df[f'pred_{col}'] = preds_np[:, i]

    # 输出路径
    if args.output_dir is None:
        output_dir = str(Path(args.checkpoint).parent / "infer_results")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"predictions_{timestamp}.csv")
    pred_df.to_csv(csv_path, index=False)
    print(f"[OK] 推理结果已保存: {csv_path}")

    # 打印汇总指标
    from egnv1.utils import compute_metrics
    metrics = compute_metrics(labels_np, preds_np)
    print(f"\n[Metrics] PCC: {metrics['pcc']:.4f} | R²: {metrics['r2']:.4f} | "
          f"MAE: {metrics['mae']:.4f} | MSE: {metrics['mse']:.4f}")


if __name__ == "__main__":
    main()
