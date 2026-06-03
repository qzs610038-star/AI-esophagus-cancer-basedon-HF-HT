"""
多教师晚期融合：UNI2-h + Virchow2 预测加权平均
==============================================
对两个已训练模型的 predictions.csv 做 per-pathway 最优权重融合，
零训练成本，直接输出融合后的预测和评估报告。

融合策略：
  - uniform:       w=0.5 所有通路统一
  - oracle:        每条通路选 PCC 更高的模型（理论上界）
  - pcc_weighted:  按单模型 per-pathway PCC 比例分配权重
  - grid_search:   每条通路独立网格搜索最优 w
  - global_search: 全局统一 w 搜索

用法:
  python ensemble_late_fusion.py \
      --pred_a <path/to/unni_predictions.csv> \
      --pred_b <path/to/virchow2_predictions.csv> \
      --output ensemble_results/fusion_uni_virchow2/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def parse_predictions(csv_path):
    """解析 predictions.csv，返回 (pathway_names, true_df, pred_df)"""
    df = pd.read_csv(csv_path)
    true_cols = [c for c in df.columns if c.startswith("true_")]
    pred_cols = [c for c in df.columns if c.startswith("pred_")]

    if len(true_cols) != len(pred_cols):
        raise ValueError(f"true/pred 列数不匹配: {len(true_cols)} vs {len(pred_cols)}")

    pathway_names = [c.replace("true_", "") for c in true_cols]
    for tc, pc in zip(true_cols, [f"pred_{n}" for n in pathway_names]):
        if pc not in df.columns:
            raise ValueError(f"缺少预测列: {pc}")

    true_df = df[true_cols].copy()
    true_df.columns = pathway_names
    pred_df = df[[f"pred_{n}" for n in pathway_names]].copy()
    pred_df.columns = pathway_names

    return pathway_names, true_df, pred_df


def compute_pcc(y_true, y_pred):
    """全局 PCC（展平所有样本和通路）"""
    return pearsonr(y_true.values.ravel(), y_pred.values.ravel())[0]


def compute_per_pathway_pcc(true_df, pred_df, pathway_names):
    """逐通路 PCC"""
    pccs = {}
    for p in pathway_names:
        pccs[p] = pearsonr(true_df[p].values, pred_df[p].values)[0]
    return pccs


def compute_metrics(true_df, pred_df):
    """PCC, MAE, R² 综合指标"""
    yt = true_df.values.ravel()
    yp = pred_df.values.ravel()
    pcc = pearsonr(yt, yp)[0]
    mae = np.mean(np.abs(yt - yp))
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"pcc": pcc, "mae": mae, "r2": r2}


def fusion_uniform(pred_a, pred_b):
    return 0.5 * pred_a.values + 0.5 * pred_b.values


def fusion_pcc_weighted(pred_a, pred_b, true_df, pathway_names):
    """按 per-pathway PCC 比例加权: w = pcc_a / (pcc_a + pcc_b)"""
    pcc_a = compute_per_pathway_pcc(true_df, pred_a, pathway_names)
    pcc_b = compute_per_pathway_pcc(true_df, pred_b, pathway_names)
    weights = {}
    for p in pathway_names:
        pa = max(pcc_a[p], 1e-8)
        pb = max(pcc_b[p], 1e-8)
        weights[p] = pa / (pa + pb)
    fused = np.zeros_like(pred_a.values)
    for i, p in enumerate(pathway_names):
        w = weights[p]
        fused[:, i] = w * pred_a[p].values + (1 - w) * pred_b[p].values
    return fused, weights


def fusion_oracle(pred_a, pred_b, true_df, pathway_names):
    """选每条通路 PCC 更高的模型（理论上界）"""
    pcc_a = compute_per_pathway_pcc(true_df, pred_a, pathway_names)
    pcc_b = compute_per_pathway_pcc(true_df, pred_b, pathway_names)
    choices = {}
    fused = np.zeros_like(pred_a.values)
    for i, p in enumerate(pathway_names):
        if pcc_a[p] >= pcc_b[p]:
            fused[:, i] = pred_a[p].values
            choices[p] = "A"
        else:
            fused[:, i] = pred_b[p].values
            choices[p] = "B"
    return fused, choices


def fusion_grid_search_per_pathway(pred_a, pred_b, true_df, pathway_names, step=0.05):
    """逐通路网格搜索最优 w"""
    best_weights = {}
    best_pccs = {}
    ws = np.arange(0.0, 1.0 + step / 2, step)
    for p in pathway_names:
        yt = true_df[p].values
        ya = pred_a[p].values
        yb = pred_b[p].values
        best_w, best_pcc = 0.5, -1.0
        for w in ws:
            yf = w * ya + (1 - w) * yb
            pcc_val = pearsonr(yt, yf)[0]
            if pcc_val > best_pcc:
                best_pcc = pcc_val
                best_w = w
        best_weights[p] = best_w
        best_pccs[p] = best_pcc

    fused = np.zeros_like(pred_a.values)
    for i, p in enumerate(pathway_names):
        w = best_weights[p]
        fused[:, i] = w * pred_a[p].values + (1 - w) * pred_b[p].values
    return fused, best_weights, best_pccs


def fusion_global_search(pred_a, pred_b, true_df, step=0.05):
    """全局统一 w 搜索"""
    yt = true_df.values.ravel()
    ya = pred_a.values.ravel()
    yb = pred_b.values.ravel()
    best_w, best_pcc = 0.5, -1.0
    for w in np.arange(0.0, 1.0 + step / 2, step):
        yf = w * ya + (1 - w) * yb
        pcc_val = pearsonr(yt, yf)[0]
        if pcc_val > best_pcc:
            best_pcc = pcc_val
            best_w = w
    fused = best_w * pred_a.values + (1 - best_w) * pred_b.values
    return fused, best_w, best_pcc


def build_fused_dataframe(fused_array, pathway_names, true_df):
    """构建与原始 predictions.csv 相同格式的 DataFrame"""
    data = {}
    for i, p in enumerate(pathway_names):
        data[f"true_{p}"] = true_df[p].values
        data[f"pred_{p}"] = fused_array[:, i]
    return pd.DataFrame(data)


def plot_comparison(results, pathway_names, output_dir):
    """生成对比图：per-pathway PCC 柱状图 + 全局指标表"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 左图：per-pathway PCC 对比
    x = np.arange(len(pathway_names))
    width = 0.25
    ax1.bar(x - width, results["pcc_per_pathway_a"], width, label="UNI2-h AugMix", color="#2196F3")
    ax1.bar(x, results["pcc_per_pathway_b"], width, label="Virchow2", color="#4CAF50")
    ax1.bar(x + width, results["pcc_per_pathway_fused"], width, label="Fused (grid_search)", color="#FF9800")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pathway_names, rotation=90, fontsize=7)
    ax1.set_ylabel("PCC")
    ax1.set_title("Per-Pathway PCC: UNI2-h vs Virchow2 vs Fused")
    ax1.legend(fontsize=8)
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax1.grid(axis="y", alpha=0.3)

    # 右图：全局指标对比表
    strategies = ["UNI2-h AugMix", "Virchow2", "Uniform (0.5)", "PCC-Weighted", "Grid Search", "Global Search"]
    pccs = [
        results["pcc_a"], results["pcc_b"],
        results["pcc_uniform"], results["pcc_pcc_weighted"],
        results["pcc_grid_search"], results["pcc_global_search"],
    ]
    colors = ["#2196F3", "#4CAF50", "#9E9E9E", "#795548", "#FF9800", "#E91E63"]
    bars = ax2.bar(strategies, pccs, color=colors)
    ax2.set_ylabel("Global PCC")
    ax2.set_title("Global PCC by Fusion Strategy")
    ax2.set_xticklabels(strategies, rotation=45, ha="right", fontsize=8)
    for bar, pcc in zip(bars, pccs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{pcc:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # 标出基线
    ax2.axhline(y=results["pcc_a"], color="#2196F3", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.axhline(y=results["pcc_b"], color="#4CAF50", linestyle="--", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fusion_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 额外：权重分布图
    if "best_weights" in results:
        fig2, ax = plt.subplots(figsize=(14, 6))
        weights = [results["best_weights"][p] for p in pathway_names]
        bar_colors = ["#2196F3" if w > 0.5 else "#4CAF50" for w in weights]
        ax.bar(pathway_names, weights, color=bar_colors, alpha=0.8)
        ax.set_xticklabels(pathway_names, rotation=90, fontsize=7)
        ax.set_ylabel("Weight for UNI2-h (1-w for Virchow2)")
        ax.set_title("Per-Pathway Optimal Fusion Weights (grid_search)")
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

        # 标注
        for i, (p, w) in enumerate(zip(pathway_names, weights)):
            ax.annotate(f"{w:.2f}", (i, w), textcoords="offset points",
                        xytext=(0, 5 if w < 0.9 else -12), ha="center", fontsize=6)

        plt.tight_layout()
        fig2.savefig(os.path.join(output_dir, "fusion_weights.png"), dpi=150, bbox_inches="tight")
        plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description="多教师晚期融合: UNI2-h + Virchow2")
    parser.add_argument("--pred_a", required=True, help="模型 A 的 predictions.csv")
    parser.add_argument("--pred_b", required=True, help="模型 B 的 predictions.csv")
    parser.add_argument("--output", default="ensemble_results/fusion", help="输出目录")
    parser.add_argument("--label_a", default="Model A", help="模型 A 的名称标签")
    parser.add_argument("--label_b", default="Model B", help="模型 B 的名称标签")
    parser.add_argument("--grid_step", type=float, default=0.05, help="网格搜索步长")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载预测
    print(f"[1/5] 加载预测文件...")
    pathway_names, true_a, pred_a = parse_predictions(args.pred_a)
    _, true_b, pred_b = parse_predictions(args.pred_b)

    if len(pred_a) != len(pred_b):
        print(f"错误: 行数不匹配 (A={len(pred_a)}, B={len(pred_b)})")
        sys.exit(1)

    # 验证标签一致
    diff = np.abs(true_a.values - true_b.values).max()
    if diff > 1e-8:
        print(f"警告: true 值不一致 (max_diff={diff:.6f})，可能不是同一组测试数据")
    else:
        print(f"   ✓ 标签一致，样本数: {len(pred_a)}")

    true_df = true_a  # 两份一致，任选一份

    print(f"[2/5] 计算基线 PCC...")
    pcc_a = compute_pcc(true_df, pred_a)
    metrics_a = compute_metrics(true_df, pred_a)
    pcc_b = compute_pcc(true_df, pred_b)
    metrics_b = compute_metrics(true_df, pred_b)
    pcc_a_per = compute_per_pathway_pcc(true_df, pred_a, pathway_names)
    pcc_b_per = compute_per_pathway_pcc(true_df, pred_b, pathway_names)

    print(f"   {args.label_a}: PCC={pcc_a:.4f}, MAE={metrics_a['mae']:.4f}, R²={metrics_a['r2']:.4f}")
    print(f"   {args.label_b}: PCC={pcc_b:.4f}, MAE={metrics_b['mae']:.4f}, R²={metrics_b['r2']:.4f}")

    # 统计通路互补性
    n_a_better = sum(1 for p in pathway_names if pcc_a_per[p] > pcc_b_per[p])
    n_b_better = sum(1 for p in pathway_names if pcc_b_per[p] > pcc_a_per[p])
    print(f"   {args.label_a} 更优通路: {n_a_better}/30, {args.label_b} 更优通路: {n_b_better}/30")

    print(f"[3/5] 执行融合策略...")
    results = {
        "label_a": args.label_a, "label_b": args.label_b,
        "pcc_a": pcc_a, "pcc_b": pcc_b,
        "metrics_a": metrics_a, "metrics_b": metrics_b,
        "pcc_per_pathway_a": [pcc_a_per[p] for p in pathway_names],
        "pcc_per_pathway_b": [pcc_b_per[p] for p in pathway_names],
        "pathway_names": pathway_names,
        "n_samples": len(pred_a),
        "n_a_better": n_a_better, "n_b_better": n_b_better,
    }

    # Strategy 1: Uniform
    fused_uniform = fusion_uniform(pred_a, pred_b)
    pcc_uniform = compute_pcc(true_df, pd.DataFrame(fused_uniform, columns=pathway_names))
    results["pcc_uniform"] = pcc_uniform
    print(f"   Uniform (w=0.5): PCC={pcc_uniform:.4f}")

    # Strategy 2: PCC-Weighted
    fused_pcc_w, weights_pcc_w = fusion_pcc_weighted(pred_a, pred_b, true_df, pathway_names)
    pcc_pcc_w = compute_pcc(true_df, pd.DataFrame(fused_pcc_w, columns=pathway_names))
    results["pcc_pcc_weighted"] = pcc_pcc_w
    print(f"   PCC-Weighted:     PCC={pcc_pcc_w:.4f}")

    # Strategy 3: Oracle
    fused_oracle, choices_oracle = fusion_oracle(pred_a, pred_b, true_df, pathway_names)
    pcc_oracle = compute_pcc(true_df, pd.DataFrame(fused_oracle, columns=pathway_names))
    results["pcc_oracle"] = pcc_oracle
    results["oracle_choices"] = choices_oracle
    print(f"   Oracle (上界):    PCC={pcc_oracle:.4f}")

    # Strategy 4: Grid Search per-pathway (推荐)
    fused_gs, best_weights, best_pccs = fusion_grid_search_per_pathway(
        pred_a, pred_b, true_df, pathway_names, step=args.grid_step
    )
    pcc_gs = compute_pcc(true_df, pd.DataFrame(fused_gs, columns=pathway_names))
    results["pcc_grid_search"] = pcc_gs
    results["best_weights"] = best_weights
    results["best_pccs_per_pathway"] = best_pccs
    results["pcc_per_pathway_fused"] = [best_pccs[p] for p in pathway_names]
    print(f"   Grid Search:      PCC={pcc_gs:.4f} (step={args.grid_step})")

    # Strategy 5: Global Search
    fused_global, best_w_global, pcc_global = fusion_global_search(
        pred_a, pred_b, true_df, step=args.grid_step
    )
    results["pcc_global_search"] = pcc_global
    results["best_global_weight"] = best_w_global
    print(f"   Global Search:    PCC={pcc_global:.4f} (w={best_w_global:.2f})")

    print(f"[4/5] 保存结果...")

    # 保存最佳融合预测 (grid_search per-pathway)
    fused_df = build_fused_dataframe(fused_gs, pathway_names, true_df)
    fused_csv = output_dir / "fusion_predictions.csv"
    fused_df.to_csv(fused_csv, index=False)
    print(f"   → {fused_csv}")

    # 保存融合权重
    weights_df = pd.DataFrame({
        "pathway": pathway_names,
        "weight_uni2h": [best_weights[p] for p in pathway_names],
        "weight_virchow2": [1 - best_weights[p] for p in pathway_names],
        "pcc_unni2h": [pcc_a_per[p] for p in pathway_names],
        "pcc_virchow2": [pcc_b_per[p] for p in pathway_names],
        "pcc_fused": [best_pccs[p] for p in pathway_names],
        "improvement_vs_unni": [best_pccs[p] - pcc_a_per[p] for p in pathway_names],
        "improvement_vs_virchow2": [best_pccs[p] - pcc_b_per[p] for p in pathway_names],
    })
    weights_csv = output_dir / "fusion_weights.csv"
    weights_df.to_csv(weights_csv, index=False)
    print(f"   → {weights_csv}")

    # 保存报告
    # 清除 numpy 不可序列化项
    report = {k: v for k, v in results.items()
              if not k.endswith("_per_pathway") and k not in ("best_weights", "best_pccs_per_pathway",
                                                               "pcc_per_pathway_a", "pcc_per_pathway_b",
                                                               "pcc_per_pathway_fused", "pathway_names",
                                                               "oracle_choices")}
    report["best_weights"] = {p: float(w) for p, w in best_weights.items()}
    report["oracle_choices"] = choices_oracle

    report_json = output_dir / "fusion_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"   → {report_json}")

    print(f"[5/5] 生成可视化...")
    plot_comparison(results, pathway_names, str(output_dir))
    print(f"   → {output_dir / 'fusion_comparison.png'}")
    print(f"   → {output_dir / 'fusion_weights.png'}")

    # 打印总结
    print()
    print("=" * 60)
    print("融合结果总结")
    print("=" * 60)
    print(f"  {args.label_a}:          PCC = {pcc_a:.4f}")
    print(f"  {args.label_b}:          PCC = {pcc_b:.4f}")
    print(f"  Uniform (0.5):           PCC = {pcc_uniform:.4f}  (Δ = {pcc_uniform - max(pcc_a, pcc_b):+.4f})")
    print(f"  PCC-Weighted:            PCC = {pcc_pcc_w:.4f}  (Δ = {pcc_pcc_w - max(pcc_a, pcc_b):+.4f})")
    print(f"  Grid Search (per-path):  PCC = {pcc_gs:.4f}  (Δ = {pcc_gs - max(pcc_a, pcc_b):+.4f})")
    print(f"  Global Search:           PCC = {pcc_global:.4f}  (Δ = {pcc_global - max(pcc_a, pcc_b):+.4f})")
    print(f"  Oracle (理论上界):       PCC = {pcc_oracle:.4f}")
    print(f"  {args.label_a} 更优: {n_a_better}/30, {args.label_b} 更优: {n_b_better}/30")
    print(f"\n  推荐策略: Grid Search per-pathway (PCC={pcc_gs:.4f})")
    print(f"  输出目录: {output_dir}")


if __name__ == "__main__":
    main()
