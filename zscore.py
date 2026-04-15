import os
import numpy as np
import pandas as pd


# =========================
# 1. 这里改成你的配置
# =========================
CSV_PATH = r".\HYZ15040_ssGSEA_scores.csv"   # 你的csv路径
NUM_TARGET_COLS = 8                        # 最后多少列做统计 / 做zscore
DO_ZSCORE = True                           # 是否做zscore
SAVE_OUTPUT = True                         # 是否保存输出文件
DDOF = 1                                   # 标准差自由度，1=样本标准差(pandas默认), 0=总体标准差


def print_basic_info(df: pd.DataFrame, name: str = "DataFrame"):
    """打印整个表的基础信息"""
    print("=" * 80)
    print(f"[{name}] 基本信息")
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    print("=" * 80)
    print()


def get_target_columns(df: pd.DataFrame, num_target_cols: int):
    """
    取最后 num_target_cols 列作为目标列
    """
    if num_target_cols <= 0:
        raise ValueError("NUM_TARGET_COLS 必须 > 0")
    if num_target_cols > df.shape[1]:
        raise ValueError(
            f"NUM_TARGET_COLS={num_target_cols} 超过总列数 {df.shape[1]}"
        )
    target_cols = df.columns[-num_target_cols:].tolist()
    return target_cols


def ensure_numeric_columns(df: pd.DataFrame, cols):
    """
    确保目标列都能转成数值；如果有不能转的，会报错并提示是哪几列
    """
    bad_cols = []
    converted = df.copy()

    for col in cols:
        converted[col] = pd.to_numeric(converted[col], errors="coerce")
        # 如果原来非空，但转换后变成NaN，说明有非数值内容
        original_non_null = df[col].notna().sum()
        converted_non_null = converted[col].notna().sum()
        if converted_non_null < original_non_null:
            bad_cols.append(col)

    if bad_cols:
        print("警告：以下列中存在无法转换为数值的内容，这些位置已被置为 NaN：")
        for c in bad_cols:
            print(f"  - {c}")
        print()

    return converted


def compute_stats(df: pd.DataFrame, cols, ddof=1):
    """
    计算每列统计量
    """
    stats_dict = {
        "count_non_null": df[cols].count(),
        "missing": df[cols].isna().sum(),
        "mean": df[cols].mean(),
        "std": df[cols].std(ddof=ddof),
        "min": df[cols].min(),
        "25%": df[cols].quantile(0.25),
        "median": df[cols].median(),
        "75%": df[cols].quantile(0.75),
        "max": df[cols].max(),
    }

    stats_df = pd.DataFrame(stats_dict)
    return stats_df


def print_stats(stats_df: pd.DataFrame, title: str):
    """
    打印统计表
    """
    print("=" * 80)
    print(title)
    print("=" * 80)
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", "{:.6f}".format
    ):
        print(stats_df)
    print()


def zscore_by_column(df: pd.DataFrame, cols, ddof=1):
    """
    对指定列按列做 z-score:
        z = (x - mean) / std

    返回:
        df_z: 标准化后的DataFrame
        means: 每列均值
        stds: 每列标准差
    """
    df_z = df.copy()
    means = df[cols].mean()
    stds = df[cols].std(ddof=ddof)

    # 防止某列标准差为0
    zero_std_cols = stds[stds == 0].index.tolist()
    if zero_std_cols:
        print("警告：以下列标准差为0，无法做z-score，这些列将保持原值不变：")
        for c in zero_std_cols:
            print(f"  - {c}")
        print()

    valid_cols = stds[stds != 0].index.tolist()
    df_z[valid_cols] = (df[valid_cols] - means[valid_cols]) / stds[valid_cols]

    return df_z, means, stds


def make_output_path(csv_path: str, suffix: str = "_zscore"):
    """
    生成输出路径:
    a.csv -> a_zscore.csv
    """
    folder = os.path.dirname(csv_path)
    base = os.path.basename(csv_path)
    stem, ext = os.path.splitext(base)
    out_name = f"{stem}{suffix}{ext}"
    return os.path.join(folder, out_name)


def main():
    # 读取
    df = pd.read_csv(CSV_PATH)

    print_basic_info(df, "原始数据")

    # 目标列
    target_cols = get_target_columns(df, NUM_TARGET_COLS)
    print(f"将最后 {NUM_TARGET_COLS} 列作为统计 / z-score 处理列：")
    for i, col in enumerate(target_cols, 1):
        print(f"  {i}. {col}")
    print()

    # 确保目标列为数值
    df_numeric = ensure_numeric_columns(df, target_cols)

    # 原始统计
    stats_before = compute_stats(df_numeric, target_cols, ddof=DDOF)
    print_stats(stats_before, "原始目标列统计信息")

    # 是否做zscore
    if DO_ZSCORE:
        df_out, means, stds = zscore_by_column(df_numeric, target_cols, ddof=DDOF)

        print("=" * 80)
        print("z-score 使用的均值和标准差")
        print("=" * 80)
        params_df = pd.DataFrame({
            "mean_used": means,
            "std_used": stds
        })
        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", 200,
            "display.float_format", "{:.6f}".format
        ):
            print(params_df)
        print()

        stats_after = compute_stats(df_out, target_cols, ddof=DDOF)
        print_stats(stats_after, "z-score 后目标列统计信息")

    else:
        print("当前设置 DO_ZSCORE = False，不进行z-score。")
        print()
        df_out = df_numeric

    # 是否保存
    if SAVE_OUTPUT:
        if DO_ZSCORE:
            out_path = make_output_path(CSV_PATH, suffix="_zscore")
        else:
            out_path = make_output_path(CSV_PATH, suffix="_processed")

        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"文件已保存到：\n{out_path}")
    else:
        print("当前设置 SAVE_OUTPUT = False，不保存文件。")


if __name__ == "__main__":
    main()