"""
HisToGene 推理脚本 - PFMval 项目
加载训练好的 checkpoint，对指定目录的 patch 进行推理，输出逐通路指标和预测 CSV
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from histogene.model import HisToGeneModel
from histogene.dataset import HisToGeneDataset
from histogene.utils import compute_metrics, pearson_corrcoef

_DEFAULT_CKPT   = str(_SCRIPT_DIR / "checkpoints" / "best_histogene.pth")
_DEFAULT_LABELS = r"d:\AI空间转录病理研究\PFMval_new\HYZ15040_ssGSEA_scores_zscore.csv"
_DEFAULT_OUT    = str(_SCRIPT_DIR / "infer_results")


def build_argparser():
    p = argparse.ArgumentParser(description="HisToGene inference")
    p.add_argument("--patches_dir",  type=str, required=True,
                   help="待推理的 patch 目录")
    p.add_argument("--labels_csv",   type=str, default=_DEFAULT_LABELS,
                   help="Z-score 标签 CSV（有标签时计算指标）")
    p.add_argument("--checkpoint",   type=str, default=_DEFAULT_CKPT)
    p.add_argument("--output_dir",   type=str, default=_DEFAULT_OUT)
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--num_workers",  type=int, default=0)
    return p


def get_val_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, pos_x, pos_y, targets in loader:
        images  = images.to(device, non_blocking=True)
        pos_x   = pos_x.to(device, non_blocking=True)
        pos_y   = pos_y.to(device, non_blocking=True)
        preds = model(images, pos_x, pos_y)
        all_preds.append(preds.cpu())
        all_labels.append(targets)
    return torch.cat(all_preds, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy()


def main():
    args = build_argparser().parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"[ERROR] checkpoint 不存在: {args.checkpoint}")
        sys.exit(1)
    if not os.path.isdir(args.patches_dir):
        print(f"[ERROR] patches_dir 不存在: {args.patches_dir}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 加载 checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args    = ckpt.get('args', {})
    coord_stats   = ckpt.get('coord_stats', None)
    target_cols   = ckpt.get('target_cols', None)

    img_size    = saved_args.get('img_size',    224)
    patch_size  = saved_args.get('patch_size',  16)
    model_dim   = saved_args.get('model_dim',   1024)
    model_depth = saved_args.get('model_depth', 8)
    heads       = saved_args.get('heads',       16)
    mlp_dim     = saved_args.get('mlp_dim',     2048)
    n_pos       = saved_args.get('n_pos',       128)
    n_targets   = saved_args.get('n_targets',   8)
    dropout     = saved_args.get('dropout',     0.3)

    # 构建模型
    model = HisToGeneModel(
        img_size=img_size, patch_size=patch_size, in_channels=3,
        dim=model_dim, depth=model_depth, heads=heads, mlp_dim=mlp_dim,
        n_pos=n_pos, n_targets=n_targets, dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[INFO] 已加载 checkpoint (epoch={ckpt.get('epoch', '?')})")

    # 数据集
    transform = get_val_transform(img_size)
    dataset = HisToGeneDataset(
        patches_dir=args.patches_dir,
        labels_csv=args.labels_csv,
        target_cols=target_cols,
        n_pos=n_pos,
        transform=transform,
        coord_stats=coord_stats,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # 推理
    preds, labels = run_inference(model, loader, device)

    # 输出目录
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存预测 CSV
    target_cols_used = dataset.target_cols
    pred_df = pd.DataFrame(preds, columns=[f"pred_{c}" for c in target_cols_used])
    label_df = pd.DataFrame(labels, columns=[f"true_{c}" for c in target_cols_used])
    result_df = pd.concat([label_df, pred_df], axis=1)

    # 添加 patch_id
    patch_ids = [os.path.basename(s[0]).replace('.png', '') for s in dataset.samples]
    result_df.insert(0, 'patch_id', patch_ids)
    result_df.to_csv(out_dir / "predictions.csv", index=False)
    print(f"[INFO] 预测结果已保存: {out_dir / 'predictions.csv'}")

    # 逐通路指标
    print("\n" + "=" * 70)
    print(f"{'Pathway':<12} {'MSE':>8} {'MAE':>8} {'R²':>8} {'PCC':>8}")
    print("-" * 70)
    per_pathway = []
    for i, col in enumerate(target_cols_used):
        y_t = labels[:, i]
        y_p = preds[:, i]
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse = float(mean_squared_error(y_t, y_p))
        mae = float(mean_absolute_error(y_t, y_p))
        r2  = float(r2_score(y_t, y_p))
        pcc = pearson_corrcoef(y_t, y_p)
        print(f"{col:<12} {mse:8.4f} {mae:8.4f} {r2:8.4f} {pcc:8.4f}")
        per_pathway.append({'pathway': col, 'mse': mse, 'mae': mae, 'r2': r2, 'pcc': pcc})

    # 全局指标
    overall = compute_metrics(labels, preds)
    print("-" * 70)
    print(f"{'Overall':<12} {overall['mse']:8.4f} {overall['mae']:8.4f} "
          f"{overall['r2']:8.4f} {overall['pcc']:8.4f}")
    print("=" * 70)

    # 保存逐通路指标
    metrics_df = pd.DataFrame(per_pathway)
    metrics_df.to_csv(out_dir / "per_pathway_metrics.csv", index=False)
    print(f"[INFO] 逐通路指标已保存: {out_dir / 'per_pathway_metrics.csv'}")


if __name__ == "__main__":
    main()
