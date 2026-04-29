"""
HisToGene-UNI 推理脚本 - PFMval 项目
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

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from histogene.model_uni import HisToGeneUNI
from histogene.dataset_uni import HisToGeneUNIDataset
from histogene.utils import compute_metrics, pearson_corrcoef

_DEFAULT_OUT = str(_SCRIPT_DIR / "res")

# ─── 从 config.yaml 读取默认路径 ───────────────────────────────────────────────
try:
    from config_utils import load_config, get_data_paths
    _config = load_config()
    _data_paths = get_data_paths(_config)
    _DEFAULT_LABELS = _data_paths.get("labels_csv_zscore")
except Exception as e:
    print(f"[WARNING] 无法加载 config.yaml: {e}，使用默认路径")
    _DEFAULT_LABELS = None


def ensure_features_cached(patches_dir, cache_dir, device, rebuild=False):
    """确保UNI2-h特征已缓存"""
    from uni2h.uni2h_utils import load_uni2h_backbone, extract_and_cache_features
    from config_utils import get_hf_config

    cache_path = Path(cache_dir)
    patches_path = Path(patches_dir)

    existing = list(cache_path.glob("*.pt")) if cache_path.exists() else []
    needed = list(patches_path.glob("*.png"))

    if len(existing) >= len(needed) and not rebuild:
        print(f"特征缓存已就绪: {len(existing)} 个文件")
        return

    print(f"正在提取UNI2-h特征: {len(needed)} 个patch...")
    hf_config = get_hf_config()
    backbone, transform, feat_dim = load_uni2h_backbone(
        token=hf_config.get('token'), device=device
    )
    cache_path.mkdir(parents=True, exist_ok=True)
    n_new = extract_and_cache_features(
        backbone, transform, str(patches_dir), str(cache_dir), device, rebuild
    )
    print(f"新提取 {n_new} 个特征，总计 {len(list(cache_path.glob('*.pt')))} 个")
    del backbone
    torch.cuda.empty_cache()


def build_argparser():
    p = argparse.ArgumentParser(description="HisToGene-UNI inference")
    p.add_argument("--checkpoint",   type=str, required=True,
                   help="训练好的 checkpoint 路径")
    p.add_argument("--patches_dir",  type=str, required=True,
                   help="待推理的 patch 目录（用于文件名枚举和坐标解析）")
    p.add_argument("--labels_csv",   type=str, default=_DEFAULT_LABELS,
                   help="Z-score 标签 CSV（有标签时计算指标）")
    p.add_argument("--feature_cache_dir", type=str, required=True,
                   help="UNI特征缓存目录")
    p.add_argument("--output_dir",   type=str, default=_DEFAULT_OUT)
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--num_workers",  type=int, default=0)
    p.add_argument("--rebuild_cache", action="store_true", default=False,
                   help="强制重建UNI特征缓存")
    return p


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for features, pos_x, pos_y, targets in loader:
        features = features.to(device, non_blocking=True)
        pos_x   = pos_x.to(device, non_blocking=True)
        pos_y   = pos_y.to(device, non_blocking=True)
        preds = model(features, pos_x, pos_y)
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
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args    = ckpt.get('args', {})
    coord_stats_dict = ckpt.get('coord_stats_dict', {})
    target_cols   = ckpt.get('target_cols', None)

    feature_dim = saved_args.get('feature_dim', 1536)
    model_dim   = saved_args.get('model_dim',   1024)
    n_pos       = saved_args.get('n_pos',       128)
    n_targets   = saved_args.get('n_targets',   30)
    mlp_dim     = saved_args.get('mlp_dim',     2048)
    dropout     = saved_args.get('dropout',     0.3)

    # 构建模型
    model = HisToGeneUNI(
        feature_dim=feature_dim, dim=model_dim,
        n_pos=n_pos, n_targets=n_targets,
        mlp_dim=mlp_dim, dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[INFO] 已加载 checkpoint (epoch={ckpt.get('epoch', '?')})")

    # 确定coord_stats
    coord_stats = None
    if coord_stats_dict:
        if len(coord_stats_dict) == 1:
            coord_stats = list(coord_stats_dict.values())[0]
        else:
            # 尝试从patches_dir匹配患者名
            import re
            match = re.search(r'(HYZ\d+|JFX\d+|LMZ\d+)', args.patches_dir)
            if match:
                patient_name = match.group(1)
                coord_stats = coord_stats_dict.get(patient_name)
            if coord_stats is None:
                # 回退到第一个
                coord_stats = list(coord_stats_dict.values())[0]
                print(f"[WARNING] 无法从路径推断患者，使用第一个患者的 coord_stats")

    # 确保特征缓存
    ensure_features_cached(args.patches_dir, args.feature_cache_dir, device, args.rebuild_cache)

    # 数据集
    dataset = HisToGeneUNIDataset(
        feature_cache_dir=args.feature_cache_dir,
        patches_dir=args.patches_dir,
        labels_csv=args.labels_csv,
        target_cols=target_cols,
        n_pos=n_pos,
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
    pred_df = pd.DataFrame()
    for i, col in enumerate(target_cols_used):
        pred_df[f'true_{col}'] = labels[:, i]
        pred_df[f'pred_{col}'] = preds[:, i]

    # 添加 patch_id
    patch_ids = [os.path.basename(s[0]).replace('.png', '') for s in dataset.samples]
    pred_df.insert(0, 'patch_id', patch_ids)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
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
