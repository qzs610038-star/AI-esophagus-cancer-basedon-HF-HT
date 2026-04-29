"""
extract_per_pathway_pcc.py
========================
从已有训练结果的 predictions.csv 中提取逐通路 PCC，汇总为对比表格。

功能：
  1. 扫描 histogene/checkpoints/results_vis 和 egnv2/checkpoints/results_vis 下所有 predictions.csv
  2. 对每个文件计算 30 条通路的逐通路 PCC
  3. 输出汇总表格到 histogene/checkpoints/results_vis/AllModels_comparison/per_pathway_pcc_summary.csv
  4. 表格格式：行=通路名，列=各模型×数据集组合
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 需要扫描的 results_vis 目录
SCAN_DIRS = [
    PROJECT_ROOT / "histogene" / "checkpoints" / "results_vis",
    PROJECT_ROOT / "egnv2" / "checkpoints" / "results_vis",
]

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "histogene" / "checkpoints" / "results_vis" / "AllModels_comparison"


def infer_model_label(dir_name: str) -> tuple[str, str]:
    """
    从时间戳目录名推断模型标签和 dataset_part。

    目录名格式: {dataset_name}_{YYYYMMDD}_{HHMMSS}
    返回: (dataset_part, timestamp_str)

    例如:
      HYZ15040_20260416_213453           -> ("HYZ15040", "20260416_213453")
      HYZ15040_UNI_20260422_232743       -> ("HYZ15040_UNI", "20260422_232743")
      CrossPatient_JFX_LMZ_to_HYZ_2026   -> ("CrossPatient_JFX_LMZ_to_HYZ", "20260424_221349")
      EGNv2_MultiPatient_3ST_20260420    -> ("EGNv2_MultiPatient_3ST", "20260420_201801")
    """
    # 找时间戳位置：格式 _YYYYMMDD_HHMMSS
    # 用正则匹配 _2\d{7}_\d{6} 模式（以 _2 开头的8位日期+下划线+6位时间）
    import re
    m = re.search(r'(_2\d{7}_\d{6})$', dir_name)
    if m:
        idx = m.start()
        dataset_part = dir_name[:idx]
        timestamp_str = dir_name[idx + 1:]
    else:
        dataset_part = dir_name
        timestamp_str = ""

    return dataset_part, timestamp_str


def determine_model_prefix(results_vis_path: Path) -> str:
    """
    根据 results_vis 的父目录判断模型前缀。

    histogene/checkpoints/results_vis -> HisToGene
    egnv2/checkpoints/results_vis    -> EGNv2
    """
    parent_name = results_vis_path.parent.parent.name  # histogene 或 egnv2
    if parent_name == "histogene":
        return "HisToGene"
    elif parent_name == "egnv2":
        return "EGNv2"
    else:
        return parent_name


def compute_per_pathway_pcc(predictions_csv: str) -> dict | None:
    """
    从 predictions.csv 计算逐通路 PCC。

    返回: {pathway_name: pcc_value, ...} 或 None
    """
    try:
        df = pd.read_csv(predictions_csv)
    except Exception as e:
        print(f"  [WARNING] 读取 {predictions_csv} 失败: {e}")
        return None

    # 推断通路名：从 true_xxx 列名中提取 xxx
    true_cols = [c for c in df.columns if c.startswith("true_")]
    if not true_cols:
        print(f"  [WARNING] {predictions_csv} 中未找到 true_* 列")
        return None

    pathways = [c[5:] for c in true_cols]
    result = {}

    for pw in pathways:
        tc = f"true_{pw}"
        pc = f"pred_{pw}"
        if tc not in df.columns or pc not in df.columns:
            result[pw] = float("nan")
            continue

        y_true = df[tc].values
        y_pred = df[pc].values

        # 过滤 NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        if len(y_true_clean) < 2:
            result[pw] = float("nan")
            continue

        if np.std(y_true_clean) > 0 and np.std(y_pred_clean) > 0:
            pcc = float(np.corrcoef(y_true_clean, y_pred_clean)[0, 1])
        else:
            pcc = float("nan")

        result[pw] = pcc

    return result


def main():
    print("=" * 70)
    print("提取逐通路 PCC 汇总表")
    print("=" * 70)

    all_data = {}  # {model_label: {pathway: pcc}}

    for scan_dir in SCAN_DIRS:
        if not scan_dir.is_dir():
            print(f"[WARNING] 目录不存在: {scan_dir}")
            continue

        model_prefix = determine_model_prefix(scan_dir)
        print(f"\n[INFO] 扫描: {scan_dir} (模型前缀: {model_prefix})")

        # 遍历时间戳子目录
        for sub_dir in sorted(scan_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            # 跳过 AllModels_comparison 目录
            if sub_dir.name == "AllModels_comparison":
                continue

            pred_file = sub_dir / "predictions.csv"
            if not pred_file.is_file():
                continue

            dataset_part, timestamp_str = infer_model_label(sub_dir.name)

            # 根据模型前缀和 dataset_part 构建完整列名
            label = build_label(model_prefix, dataset_part)

            # 检查是否已有同名列，如有则追加时间戳后缀区分
            if label in all_data:
                label = f"{label}_{timestamp_str}"

            print(f"  处理: {sub_dir.name} -> {label}")

            pcc_dict = compute_per_pathway_pcc(str(pred_file))
            if pcc_dict is not None:
                all_data[label] = pcc_dict
                mean_pcc = np.nanmean(list(pcc_dict.values()))
                print(f"    通路数: {len(pcc_dict)}, Mean PCC: {mean_pcc:.4f}")

    if not all_data:
        print("\n[ERROR] 未找到任何 predictions.csv 文件")
        return

    # 构建汇总 DataFrame
    # 先收集所有通路名（取第一个文件的通路顺序）
    first_key = next(iter(all_data))
    pathway_order = list(all_data[first_key].keys())

    rows = []
    for pw in pathway_order:
        row = {"Pathway": pw}
        for label in all_data:
            row[label] = all_data[label].get(pw, float("nan"))
        rows.append(row)

    # 添加 mean 行
    mean_row = {"Pathway": "Mean"}
    for label in all_data:
        vals = [all_data[label].get(pw, float("nan")) for pw in pathway_order]
        mean_row[label] = np.nanmean(vals)
    rows.append(mean_row)

    df_summary = pd.DataFrame(rows)

    # 输出
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "per_pathway_pcc_summary.csv"
    df_summary.to_csv(str(output_path), index=False, float_format="%.4f")

    print(f"\n[OK] 汇总表已保存: {output_path}")
    print(f"  通路数: {len(pathway_order)}")
    print(f"  模型组合数: {len(all_data)}")
    print(f"  列名: {list(df_summary.columns)}")

    # 打印简要结果
    print("\n" + "=" * 70)
    print("各模型 Mean PCC:")
    print("=" * 70)
    mean_row_data = df_summary[df_summary["Pathway"] == "Mean"].iloc[0]
    for col in df_summary.columns[1:]:
        print(f"  {col:<50s} PCC = {mean_row_data[col]:.4f}")


def build_label(model_prefix: str, dataset_part: str) -> str:
    """
    根据模型前缀和 dataset 部分构建清晰的列标签。
    """
    # HisToGene 系列标签映射
    if model_prefix == "HisToGene":
        if dataset_part == "HYZ15040":
            return "HisToGene_HYZ15040"
        elif dataset_part == "HYZ15040_UNI":
            return "HisToGene-UNI_HYZ15040"
        elif dataset_part == "JFX0729":
            return "HisToGene_JFX0729"
        elif dataset_part == "LMZ12939":
            return "HisToGene_LMZ12939"
        elif dataset_part == "MultiPatient_3ST":
            return "HisToGene_MultiPatient_3ST"
        elif dataset_part == "CrossPatient_JFX_LMZ_to_HYZ":
            return "HisToGene_CrossPatient"
        elif dataset_part == "CrossPatient_JFX_LMZ_to_HYZ_orig":
            return "HisToGene_CrossPatient_orig"
        else:
            return f"HisToGene_{dataset_part}"

    # EGNv2 系列标签映射
    elif model_prefix == "EGNv2":
        if dataset_part == "HYZ15040":
            return "EGNv2_HYZ15040"
        elif dataset_part == "HYZ15040_UNI":
            return "EGNv2-UNI_HYZ15040"
        elif dataset_part == "JFX0729":
            return "EGNv2_JFX0729"
        elif dataset_part == "JFX0729_UNI":
            return "EGNv2-UNI_JFX0729"
        elif dataset_part == "LMZ12939":
            return "EGNv2_LMZ12939"
        elif dataset_part == "LMZ12939_UNI":
            return "EGNv2-UNI_LMZ12939"
        elif dataset_part == "EGNv2_MultiPatient_3ST":
            return "EGNv2_MultiPatient_3ST"
        elif dataset_part == "CrossPatient_JFX_LMZ_to_HYZ":
            return "EGNv2_CrossPatient"
        elif dataset_part == "CrossPatient_JFX_LMZ_to_HYZ_UNI":
            return "EGNv2-UNI_CrossPatient"
        else:
            return f"EGNv2_{dataset_part}"

    else:
        return f"{model_prefix}_{dataset_part}"


if __name__ == "__main__":
    main()
