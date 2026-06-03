"""
AttnPool 跨患者失败根因深度分析
================================
分析 token 注意力模式的患者特异性，为 per-pathway attention 设计提供依据。

分析维度:
  1. 注意力权重分布 — 集中 vs 均匀 (entropy)
  2. 跨患者注意力模式差异 — KL 散度 / 相关性
  3. 注意力权重空间自相关性
  4. 注意力权重与预测误差的关系
  5. 患者特异性 token 偏好

用法:
  python analyze_attnpool_failure.py
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config_utils import get_patient_paths, get_fold_config
from model_uni_tokens import HisToGeneUNITokens, LightweightTokenEncoder
from dataset_uni_tokens import HisToGeneUNITokensDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = _PROJECT_ROOT / "histogene/checkpoints/HYZ15040_UNI_tokens_attn_pool_only/best_histogene_uni_tokens.pth"
OUTPUT_DIR = _PROJECT_ROOT / "attnpool_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 模型加载 ──────────────────────────────────────────────────────────────

def load_attnpool_model(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
    a = ckpt['args']

    model = HisToGeneUNITokens(
        feature_dim=a['feature_dim'],
        dim=a['model_dim'],
        n_pos=a['n_pos'],
        n_targets=a['n_targets'],
        mlp_dim=a['mlp_dim'],
        dropout=a['dropout'],
        encoder_hidden_dim=a['encoder_hidden_dim'],
        n_encoder_layers=a['n_encoder_layers'],
        n_encoder_heads=a['n_encoder_heads'],
        use_attn_pool=True,
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


# ── 数据加载 ──────────────────────────────────────────────────────────────

def load_patient_data(patient, batch_size=64):
    """加载患者 val 数据并返回 DataLoader + coord_stats"""
    pc = get_patient_paths(patient, backbone='uni_tokens')
    ds = HisToGeneUNITokensDataset(
        patches_dir=pc['val_patches'],
        feature_cache_dir=pc['token_cache_val'],
        labels_csv=pc['labels_csv'],
        n_pos=128, n_targets=30,
        target_cols=None,
        coord_stats=None,
    )
    coord_stats = ds.get_coord_stats()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, coord_stats, len(ds)


# ── 注意力提取 ────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_attention_weights(model, loader):
    """
    逐 batch 推理，提取:
      - attn_weights: [N, num_tokens]  每个样本的 token 注意力权重
      - predictions:  [N, 30]          模型预测
      - targets:      [N, 30]          真实标签
      - pos_x, pos_y: [N]              空间坐标 (原始索引)
      - errors:       [N, 30]          每通路预测误差
    """
    all_attn = []
    all_preds = []
    all_targets = []
    all_posx = []
    all_posy = []

    for tokens, pos_x, pos_y, targets in loader:
        tokens = tokens.to(DEVICE)
        pos_x = pos_x.to(DEVICE)
        pos_y = pos_y.to(DEVICE)

        # 直接调用 token_encoder 获取注意力权重
        encoded, attn_w = model.token_encoder(tokens)  # attn_w: [B, N_tok, 1]
        x = model.proj(encoded)
        x = x + model.x_embed(pos_x) + model.y_embed(pos_y)
        preds = model.head(x)

        all_attn.append(attn_w.squeeze(-1).cpu())       # [B, N_tok]
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        all_posx.append(pos_x.cpu())
        all_posy.append(pos_y.cpu())

    return {
        'attn_weights': torch.cat(all_attn, dim=0).numpy(),     # [N, 65]
        'predictions': torch.cat(all_preds, dim=0).numpy(),      # [N, 30]
        'targets': torch.cat(all_targets, dim=0).numpy(),        # [N, 30]
        'pos_x': torch.cat(all_posx, dim=0).numpy(),             # [N]
        'pos_y': torch.cat(all_posy, dim=0).numpy(),             # [N]
    }


# ── 分析函数 ──────────────────────────────────────────────────────────────

def compute_attention_entropy(attn_weights):
    """计算每个样本注意力权重的熵 (nats)。均匀分布 → 高熵, 集中分布 → 低熵。"""
    eps = 1e-8
    entropy = -np.sum(attn_weights * np.log(attn_weights + eps), axis=1)
    max_entropy = np.log(attn_weights.shape[1])
    normalized_entropy = entropy / max_entropy  # 0=完全集中, 1=完全均匀
    return entropy, normalized_entropy


def compute_per_token_stats(attn_weights):
    """每个 token 位置的平均注意力和标准差"""
    mean_per_token = attn_weights.mean(axis=0)   # [65]
    std_per_token = attn_weights.std(axis=0)      # [65]
    cv_per_token = std_per_token / (mean_per_token + 1e-8)  # 变异系数
    return mean_per_token, std_per_token, cv_per_token


def compute_token_concentration(attn_weights, top_k=5):
    """前 k 个 token 占据的注意力质量比例"""
    sorted_weights = np.sort(attn_weights, axis=1)[:, ::-1]  # 降序
    topk_concentration = sorted_weights[:, :top_k].sum(axis=1)
    return topk_concentration.mean(), sorted_weights


def compute_cross_patient_attention_similarity(stats_a, stats_b):
    """比较两个患者的注意力统计量"""
    mean_a, std_a, cv_a = stats_a
    mean_b, std_b, cv_b = stats_b

    # 均值平均值的相关性
    corr_mean = np.corrcoef(mean_a, mean_b)[0, 1]

    # 排名相关性
    from scipy.stats import spearmanr
    rank_corr, rank_pval = spearmanr(mean_a, mean_b)

    # L2 距离
    l2_dist = np.sqrt(np.sum((mean_a - mean_b) ** 2))

    return {
        'pearson_corr': corr_mean,
        'spearman_rank_corr': rank_corr,
        'spearman_pval': rank_pval,
        'l2_distance': l2_dist,
    }


def compute_spatial_autocorr(attn_weights, pos_x, pos_y, token_idx=None):
    """
    检查注意力权重是否具有空间自相关性。
    如果位置相近的 patch 有相似的注意力模式 → 高自相关 → 模式是空间结构化的。
    """
    from scipy.spatial import KDTree
    coords = np.stack([pos_x, pos_y], axis=1)
    tree = KDTree(coords)

    if token_idx is not None:
        values = attn_weights[:, token_idx]
    else:
        values = attn_weights.mean(axis=1)  # 全局平均注意力

    # 对每个点找 k=8 个最近邻，计算值差异
    k = 8
    distances, indices = tree.query(coords, k=k+1)  # k+1 因为包含自身

    # 邻居值差异
    neighbor_diffs = []
    for i in range(len(values)):
        for j in range(1, k+1):
            neighbor_idx = indices[i, j]
            neighbor_diffs.append(abs(values[i] - values[neighbor_idx]))

    mean_diff = np.mean(neighbor_diffs)
    # Moran's I 简化版: 值越相似意味着空间自相关越强
    value_var = np.var(values)
    if value_var > 0:
        normalized_diff = mean_diff / np.sqrt(value_var)
    else:
        normalized_diff = float('inf')

    return mean_diff, normalized_diff


def analyze_per_pathway_error_pattern(predictions, targets, attn_weights):
    """分析每条通路预测误差与注意力模式的关系"""
    errors = predictions - targets  # [N, 30]
    abs_errors = np.abs(errors)

    # 每条通路的平均绝对误差
    per_pathway_mae = abs_errors.mean(axis=0)  # [30]

    # 注意力权重与每条通路误差的相关性
    # 对每个 token，计算其注意力权重与某通路误差的相关性
    n_tokens = attn_weights.shape[1]
    pathway_token_corr = np.zeros((30, n_tokens))
    for p in range(30):
        for t in range(n_tokens):
            c = np.corrcoef(attn_weights[:, t], abs_errors[:, p])[0, 1]
            pathway_token_corr[p, t] = 0.0 if np.isnan(c) else c

    return per_pathway_mae, pathway_token_corr


# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("AttnPool 跨患者失败根因深度分析")
    print("=" * 70)

    # 1. 加载模型
    print("\n[1/5] 加载 AttnPool 模型...")
    model, ckpt = load_attnpool_model(CHECKPOINT_PATH)
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  训练患者: {ckpt['args']['patient']}")
    print(f"  Best Val PCC: {ckpt['best_pcc']:.4f}")

    # 2. 加载三患者数据
    print("\n[2/5] 加载三患者数据...")
    patients = ['HYZ15040', 'JFX0729', 'LMZ12939']
    patient_data = {}
    for p in patients:
        loader, coord_stats, n_samples = load_patient_data(p)
        patient_data[p] = {'loader': loader, 'coord_stats': coord_stats, 'n': n_samples}
        print(f"  {p}: {n_samples} 样本")

    # 3. 提取注意力权重
    print("\n[3/5] 提取注意力权重...")
    results = {}
    for p in patients:
        print(f"  推理 {p}...")
        results[p] = extract_attention_weights(model, patient_data[p]['loader'])
        print(f"    注意力权重 shape: {results[p]['attn_weights'].shape}")

    # 4. 分析
    print("\n[4/5] 分析注意力模式...")

    # 4.1 注意力熵分析
    print("\n── 4.1 注意力熵分析 (0=集中, 1=均匀) ──")
    entropy_stats = {}
    for p in patients:
        ent, norm_ent = compute_attention_entropy(results[p]['attn_weights'])
        entropy_stats[p] = (ent, norm_ent)
        topk_conc, _ = compute_token_concentration(results[p]['attn_weights'], top_k=5)
        print(f"  {p}: 归一化熵={norm_ent.mean():.4f} (±{norm_ent.std():.4f}), "
              f"top-5 注意力集中度={topk_conc:.4f}")

    # 4.2 各 token 平均注意力
    print("\n── 4.2 各 Token 平均注意力权重 (前10) ──")
    token_stats = {}
    for p in patients:
        mean_tok, std_tok, cv_tok = compute_per_token_stats(results[p]['attn_weights'])
        token_stats[p] = (mean_tok, std_tok, cv_tok)
        top_indices = np.argsort(mean_tok)[::-1][:10]
        print(f"  {p} top-10 tokens (idx: weight):")
        for i, idx in enumerate(top_indices):
            print(f"    token_{idx:02d}: {mean_tok[idx]:.4f} (±{std_tok[idx]:.4f})")

    # 4.3 跨患者注意力相似度
    print("\n── 4.3 跨患者注意力模式相似度 ──")
    pairs = [('HYZ15040', 'JFX0729'), ('HYZ15040', 'LMZ12939'), ('JFX0729', 'LMZ12939')]
    for pa, pb in pairs:
        sim = compute_cross_patient_attention_similarity(token_stats[pa], token_stats[pb])
        print(f"  {pa} vs {pb}:")
        print(f"    Pearson r={sim['pearson_corr']:.4f}, "
              f"Spearman ρ={sim['spearman_rank_corr']:.4f} (p={sim['spearman_pval']:.4f}), "
              f"L2={sim['l2_distance']:.4f}")

    # 4.4 空间自相关分析
    print("\n── 4.4 注意力权重的空间自相关性 ──")
    for p in patients:
        mean_diff, norm_diff = compute_spatial_autocorr(
            results[p]['attn_weights'],
            results[p]['pos_x'],
            results[p]['pos_y']
        )
        print(f"  {p}: 邻居注意力差异={mean_diff:.6f}, 归一化={norm_diff:.4f}")

    # 4.5 通路误差与注意力模式
    print("\n── 4.5 通路预测误差与注意力模式 ──")
    for p in patients:
        mae, tok_corr = analyze_per_pathway_error_pattern(
            results[p]['predictions'],
            results[p]['targets'],
            results[p]['attn_weights']
        )
        print(f"  {p}: 平均 MAE={mae.mean():.4f}, 最难5通路: {np.argsort(mae)[::-1][:5]}")

    # 5. 保存结果
    print("\n[5/5] 保存分析结果...")
    np.savez(OUTPUT_DIR / "attnpool_analysis.npz",
             **{f"{p}_attn": results[p]['attn_weights'] for p in patients},
             **{f"{p}_preds": results[p]['predictions'] for p in patients},
             **{f"{p}_targets": results[p]['targets'] for p in patients},
             **{f"{p}_posx": results[p]['pos_x'] for p in patients},
             **{f"{p}_posy": results[p]['pos_y'] for p in patients},
    )

    # 生成文本报告
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("AttnPool 跨患者失败根因分析报告")
    report_lines.append(f"模型: {CHECKPOINT_PATH}")
    report_lines.append(f"训练患者: HYZ15040 (单患者)")
    report_lines.append("=" * 60)

    report_lines.append("\n## 1. 注意力熵分析\n")
    for p in patients:
        ent, norm_ent = entropy_stats[p]
        report_lines.append(f"- {p}: 归一化熵 = {norm_ent.mean():.4f} (±{norm_ent.std():.4f})")

    report_lines.append("\n## 2. 跨患者注意力模式相似度\n")
    for pa, pb in pairs:
        sim = compute_cross_patient_attention_similarity(token_stats[pa], token_stats[pb])
        report_lines.append(f"- {pa} vs {pb}: r={sim['pearson_corr']:.4f}, ρ={sim['spearman_rank_corr']:.4f}")

    report_lines.append("\n## 3. 失败根因判断\n")
    # 基于分析结果自动判断
    hyz_ent = entropy_stats['HYZ15040'][1].mean()
    jfx_ent = entropy_stats['JFX0729'][1].mean()
    lmz_ent = entropy_stats['LMZ12939'][1].mean()

    hyz_jfx_sim = compute_cross_patient_attention_similarity(token_stats['HYZ15040'], token_stats['JFX0729'])
    hyz_lmz_sim = compute_cross_patient_attention_similarity(token_stats['HYZ15040'], token_stats['LMZ12939'])

    avg_cross_corr = (hyz_jfx_sim['spearman_rank_corr'] + hyz_lmz_sim['spearman_rank_corr']) / 2

    report_lines.append(f"- HYZ 注意力熵: {hyz_ent:.3f}")
    report_lines.append(f"- JFX 注意力熵: {jfx_ent:.3f}")
    report_lines.append(f"- LMZ 注意力熵: {lmz_ent:.3f}")
    report_lines.append(f"- 平均跨患者 Spearman ρ: {avg_cross_corr:.3f}")

    if avg_cross_corr < 0.3:
        report_lines.append("\n**结论: 注意力模式高度患者特异 (ρ < 0.3)，HYZ 上训练的 AttnPool "
                           "在新患者上产生不相关甚至错误的 token 加权，导致跨患者性能退化。**")
    elif avg_cross_corr < 0.6:
        report_lines.append("\n**结论: 注意力模式部分可迁移但相关性中等 ({0.3} < ρ < 0.6)，"
                           "AttnPool 学到部分通用模式但患者特异性仍占主导。**")
    else:
        report_lines.append("\n**结论: 注意力模式高度一致 (ρ > 0.6)，AttnPool 失败可能由其他原因导致。**")

    report_lines.append("\n## 4. Per-Pathway Attention 设计建议\n")
    report_lines.append("1. 当前 AttnPool 对所有 30 条通路使用同一组 token 权重，无法捕捉通路特异性")
    report_lines.append("2. 建议将 attn_pool 输出维度从 1 改为 30 (每通路独立注意力)")
    report_lines.append("3. 参数量增量: Linear(128, 30) ≈ 3.9K, 仍然极小")
    report_lines.append("4. 配合通路级正则化 (如 group sparsity) 防止 30 组注意力共线性")

    report_text = "\n".join(report_lines)
    with open(OUTPUT_DIR / "attnpool_failure_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\n完整结果保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
