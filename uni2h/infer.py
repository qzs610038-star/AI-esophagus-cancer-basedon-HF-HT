import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from uni2h_utils import (
    CachedFeaturePatchDataset,
    DEFAULT_NUM_TARGETS,
    DEFAULT_TARGET_START_COL,
    BackboneRegressor,
    evaluate,
    extract_and_cache_features,
    load_uni2h_backbone,
    pearson_corrcoef,
)

HF_TOKEN = ""  # huggingface的token


def build_argparser():
    p = argparse.ArgumentParser(description="Inference/evaluation with frozen UNI2-h features.")
    p.add_argument("--split_patches_dir", type=str,
                   default=r"D:\PycharmProjects\AIPath-data\patch\HYZ15040\val_patches")  # 要推理的数据集
    p.add_argument("--labels_csv", type=str,
                   default=r"D:\PycharmProjects\AIPath-data\HYZ15040_ssGSEA_scores_zscore.csv")  # 标准化后的csv文件
    p.add_argument("--cache_root", type=str, default=r".\uni2h_cache\HYZ15040")  # 特征缓存路径
    p.add_argument("--checkpoint_path", type=str, default=r".\checkpoints\HYZ15040\best_model_uni2h.pth")  # 最优模型保存路径
    p.add_argument("--output_csv", type=str, default=r".\res\HYZ15040\val_predictions_uni2h.csv")  # 推理结果
    p.add_argument("--hf_token", type=str, default=HF_TOKEN)
    p.add_argument("--batch_size", type=int, default=256)  # batch size
    p.add_argument("--num_workers", type=int, default=0)  # 控制 DataLoader 读取数据时用多少个子进程
    p.add_argument("--target_start_col", type=int, default=DEFAULT_TARGET_START_COL)  # 标签从 CSV 的第几列开始，定义在uni2h_utils.py
    p.add_argument("--num_targets", type=int, default=DEFAULT_NUM_TARGETS)  # 要预测几个指标，定义在uni2h_utils.py
    p.add_argument("--rebuild_cache", action="store_true")  # 是否重新提取 UNI2-h 特征
    
    return p


def main():
    args = build_argparser().parse_args()

    labels_csv = args.labels_csv

    # 加载 checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    feature_dim = int(checkpoint["feature_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])
    dropout = float(checkpoint["dropout"])
    num_targets = int(checkpoint["num_targets"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载 UNI2-h backbone
    backbone, transform, _ = load_uni2h_backbone(token=args.hf_token, device=device)
    backbone.eval()

    split_name = Path(args.split_patches_dir).name
    split_cache_dir = Path(args.cache_root) / split_name
    split_cache_dir.mkdir(parents=True, exist_ok=True)

    # 提取特征并缓存
    num_cached = extract_and_cache_features(
        backbone=backbone,
        transform=transform,
        patches_dir=args.split_patches_dir,
        cache_dir=str(split_cache_dir),
        device=device,
        rebuild=args.rebuild_cache,
    )
    print(f"Cached {num_cached} new features for split: {split_name}")

    dataset = CachedFeaturePatchDataset(
        patches_dir=args.split_patches_dir,
        labels_csv=labels_csv,
        feature_cache_dir=str(split_cache_dir),
        target_start_col=args.target_start_col,
        num_targets=num_targets,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = BackboneRegressor(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=num_targets,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 计算每个 target 单独的指标
    target_cols = list(pd.read_csv(labels_csv).columns[args.target_start_col:args.target_start_col + num_targets])

    all_true = []
    all_pred = []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            preds = model(features).detach().cpu().numpy()
            all_true.append(targets.numpy())
            all_pred.append(preds)

    y_true_all = np.concatenate(all_true, axis=0)
    y_pred_all = np.concatenate(all_pred, axis=0)

    # 每个分数单独的指标
    per_target_rows = []
    for j, col in enumerate(target_cols):
        yt = y_true_all[:, j]
        yp = y_pred_all[:, j]

        # 单目标 R2 如果该列常数，sklearn 会报警/不稳定，这里做个保护
        if np.std(yt) == 0:
            r2 = np.nan
        else:
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            r2 = float(r2_score(yt, yp))

        from sklearn.metrics import mean_absolute_error, mean_squared_error
        per_target_rows.append({
            "scope": "target",
            "target": col,
            "mse": float(mean_squared_error(yt, yp)),
            "mae": float(mean_absolute_error(yt, yp)),
            "r2": r2,
            "pcc": float(pearson_corrcoef(yt, yp)),
            "n_samples": int(len(yt)),
        })

    # 8个 target 的宏平均（macro-average）
    macro_avg_row = {
        "scope": "macro_avg",
        "target": f"ALL_{num_targets}_TARGETS",
        "mse": float(np.nanmean([row["mse"] for row in per_target_rows])),
        "mae": float(np.nanmean([row["mae"] for row in per_target_rows])),
        "r2": float(np.nanmean([row["r2"] for row in per_target_rows])),
        "pcc": float(np.nanmean([row["pcc"] for row in per_target_rows])),
        "n_samples": int(y_true_all.shape[0]),
    }

    metrics_df = pd.DataFrame([macro_avg_row] + per_target_rows)

    metrics_path = Path(args.output_csv).with_name(Path(args.output_csv).stem + "_metrics.csv")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"Saved metrics to: {metrics_path}")

    rows = []
    for i, img_path in enumerate(dataset.patch_files):
        row = {"patch_name": img_path.stem}
        for j, col in enumerate(target_cols):
            row[f"true_{col}"] = float(y_true_all[i, j])
            row[f"pred_{col}"] = float(y_pred_all[i, j])
        rows.append(row)

    out_df = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
