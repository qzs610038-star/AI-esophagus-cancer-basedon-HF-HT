# zscore.py 深度解读指南

> 本文档面向初学者，详细解读 ssGSEA 评分 Z-score 标准化脚本的原理与实现。

---

## 1. 文件概述

**一句话说明**：`zscore.py` 是一个用于**ssGSEA 评分数据的统计分析与 Z-score 标准化工具**，计算数据的基本统计量并进行标准化处理，使不同基因集评分具有可比性。

**在项目中的角色**：这是数据分析流程的关键环节，位于 ssGSEA 评分计算之后、下游分析（如聚类、可视化）之前，确保不同量纲的数据可以进行公平比较。

---

## 2. 背景知识

### 2.1 什么是 ssGSEA 评分？

**ssGSEA**（single-sample Gene Set Enrichment Analysis，单样本基因集富集分析）是一种评估单个样本中特定基因集（如通路、功能模块）富集程度的方法。

- **输入**：基因表达矩阵 + 预定义的基因集（如 HALLMARK 通路）
- **输出**：每个样本在每个基因集上的富集评分
- **意义**：评分越高，表示该基因集在样本中活性越强

### 2.2 为什么需要 Z-score 标准化？

不同基因集的评分范围和分布差异很大：

| 基因集 | 原始评分范围 | 分布特点 |
|--------|-------------|----------|
| HALLMARK_APOPTOSIS | -0.5 ~ 0.8 | 偏态分布 |
| HALLMARK_DNA_REPAIR | -0.2 ~ 0.3 | 范围较窄 |
| HALLMARK_INFLAMMATORY | -0.8 ~ 1.2 | 范围较宽 |

**Z-score 标准化的作用**：

1. **消除量纲影响**：将不同基因集的评分转换到统一尺度
2. **突出异常值**：Z-score > 2 或 < -2 表示显著高/低表达
3. **便于比较**：标准化后可以直接比较不同基因集的相对活性

**Z-score 计算公式**：

```
z = (x - μ) / σ

其中：
- x：原始值
- μ：该列（基因集）的均值
- σ：该列（基因集）的标准差
```

标准化后的数据：
- 均值为 0
- 标准差为 1
- 大多数值落在 [-3, 3] 范围内

---

## 3. 整体流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                         输入阶段                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  读取 CSV 文件 (HYZ15040_ssGSEA_scores.csv)              │   │
│  │  包含样本ID列 + 多个基因集评分列                          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      目标列识别阶段                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  取最后 NUM_TARGET_COLS 列作为目标列                     │   │
│  │  （通常是基因集评分列，排除前面的ID列）                    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      数据预处理阶段                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  确保目标列为数值类型                                     │   │
│  │  非数值内容转换为 NaN 并发出警告                          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      统计分析阶段                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  计算每列的统计量：                                       │   │
│  │  count, missing, mean, std, min, 25%, median, 75%, max   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Z-score 标准化阶段                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  对每列应用 z = (x - mean) / std                         │   │
│  │  处理标准差为0的特殊情况（保持原值）                       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      输出阶段                                     │
│  ┌──────────────────────────┐    ┌──────────────────────────┐  │
│  │   控制台输出统计报告       │    │   保存结果到 CSV 文件     │  │
│  │   (原始 vs 标准化后)       │    │   (_zscore.csv)          │  │
│  └──────────────────────────┘    └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 逐函数详解

### 4.1 `print_basic_info(df, name)`

**函数签名**：
```python
def print_basic_info(df: pd.DataFrame, name: str = "DataFrame")
```

**功能说明**：
打印 DataFrame 的基本信息，包括形状（行列数）和列名列表，用于快速了解数据结构。

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df` | pd.DataFrame | 必填 | 要打印信息的 DataFrame |
| `name` | str | "DataFrame" | 打印时显示的标识名称 |

**算法逻辑**：
```
1. 打印 80 个等号作为分隔线
2. 打印 [name] 基本信息 标题
3. 打印 df.shape（行数, 列数）
4. 打印 df.columns 转换为列表后的列名
5. 打印分隔线和空行
```

**输入/输出示例**：

```python
# 输入
df = pd.DataFrame({
    'SampleID': ['S1', 'S2', 'S3'],
    'GeneSet_A': [0.5, 0.3, 0.8],
    'GeneSet_B': [-0.2, 0.1, 0.4]
})
print_basic_info(df, "原始数据")

# 输出
================================================================================
[原始数据] 基本信息
shape: (3, 3)
columns: ['SampleID', 'GeneSet_A', 'GeneSet_B']
================================================================================
```

**关键代码行**：

| 行号 | 代码 | 作用 |
|------|------|------|
| 18 | `print("=" * 80)` | 打印分隔线，提高可读性 |
| 20 | `print(f"shape: {df.shape}")` | 显示数据维度（行, 列） |
| 21 | `print(f"columns: {list(df.columns)}")` | 显示所有列名 |

---

### 4.2 `get_target_columns(df, num_target_cols)`

**函数签名**：
```python
def get_target_columns(df: pd.DataFrame, num_target_cols: int)
```

**功能说明**：
从 DataFrame 中提取最后 `num_target_cols` 列作为目标列（通常是基因集评分列）。

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `df` | pd.DataFrame | 输入数据表 |
| `num_target_cols` | int | 需要提取的目标列数量（从最后往前数） |

**算法逻辑**：
```
1. 检查 num_target_cols 是否 > 0，否则报错
2. 检查 num_target_cols 是否超过总列数，否则报错
3. 使用 df.columns[-num_target_cols:] 取最后 N 列
4. 转换为列表并返回
```

**输入/输出示例**：

```python
# 输入 DataFrame 有 5 列：['ID', 'Name', 'Score_A', 'Score_B', 'Score_C']
target_cols = get_target_columns(df, 3)

# 输出
['Score_A', 'Score_B', 'Score_C']  # 最后 3 列
```

**关键代码行**：

| 行号 | 代码 | 作用 |
|------|------|------|
| 30-31 | `if num_target_cols <= 0: raise ValueError(...)` | 参数合法性检查 |
| 32-35 | `if num_target_cols > df.shape[1]: raise ValueError(...)` | 防止越界访问 |
| 36 | `target_cols = df.columns[-num_target_cols:].tolist()` | 取最后 N 列并转列表 |

---

### 4.3 `ensure_numeric_columns(df, cols)`

**函数签名**：
```python
def ensure_numeric_columns(df: pd.DataFrame, cols)
```

**功能说明**：
确保指定的列都能转换为数值类型。如果某列包含无法转换的内容（如字符串 "N/A"），这些位置会被设为 NaN，并发出警告。

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `df` | pd.DataFrame | 输入数据表 |
| `cols` | list | 需要检查的目标列名列表 |

**算法逻辑**：
```
1. 初始化 bad_cols 列表记录有问题的列
2. 复制 df 到 converted（避免修改原数据）
3. 遍历每个目标列：
   a. 使用 pd.to_numeric(col, errors="coerce") 尝试转换
   b. 比较转换前后的非空值数量
   c. 如果转换后非空值减少，说明有无法转换的内容，加入 bad_cols
4. 如果有 bad_cols，打印警告信息
5. 返回转换后的 DataFrame
```

**输入/输出示例**：

```python
# 输入
df = pd.DataFrame({
    'GeneSet_A': [0.5, 'N/A', 0.8],  # 包含非数值
    'GeneSet_B': [0.1, 0.2, 0.3]     # 正常数值
})
result = ensure_numeric_columns(df, ['GeneSet_A', 'GeneSet_B'])

# 输出（控制台）
警告：以下列中存在无法转换为数值的内容，这些位置已被置为 NaN：
  - GeneSet_A

# result['GeneSet_A'] 变为 [0.5, NaN, 0.8]
```

**关键代码行**：

| 行号 | 代码 | 作用 |
|------|------|------|
| 48 | `converted[col] = pd.to_numeric(converted[col], errors="coerce")` | 强制转数值，失败则置 NaN |
| 50-51 | `original_non_null = df[col].notna().sum()` | 统计转换前的非空值数量 |
| 52-53 | `if converted_non_null < original_non_null: bad_cols.append(col)` | 检测转换失败的列 |

---

### 4.4 `compute_stats(df, cols, ddof=1)`

**函数签名**：
```python
def compute_stats(df: pd.DataFrame, cols, ddof=1)
```

**功能说明**：
计算指定列的详细统计量，包括计数、缺失值、均值、标准差、分位数等。

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df` | pd.DataFrame | 必填 | 输入数据表 |
| `cols` | list | 必填 | 需要统计的列名列表 |
| `ddof` | int | 1 | 标准差计算的自由度，1=样本标准差，0=总体标准差 |

**算法逻辑**：
```
1. 构建统计量字典，包含 9 个统计指标：
   - count_non_null: 非空值计数
   - missing: 缺失值计数
   - mean: 均值
   - std: 标准差
   - min: 最小值
   - 25%: 第一四分位数
   - median: 中位数
   - 75%: 第三四分位数
   - max: 最大值
2. 将字典转换为 DataFrame 并返回
```

**输入/输出示例**：

```python
# 输入
df = pd.DataFrame({
    'GeneSet_A': [0.5, 0.3, 0.8, 0.6],
    'GeneSet_B': [-0.2, 0.1, 0.4, 0.0]
})
stats = compute_stats(df, ['GeneSet_A', 'GeneSet_B'], ddof=1)

# 输出（stats DataFrame）
            count_non_null  missing      mean       std  ...
GeneSet_A               4        0    0.5500    0.2154  ...
GeneSet_B               4        0    0.0750    0.2517  ...
```

**关键代码行**：

| 行号 | 代码 | 作用 |
|------|------|------|
| 68-78 | `stats_dict = {...}` | 定义 9 个统计指标的计算方式 |
| 72 | `"std": df[cols].std(ddof=ddof)` | 标准差计算，ddof 控制自由度 |
| 80 | `stats_df = pd.DataFrame(stats_dict)` | 将字典转为 DataFrame |

---

### 4.5 `print_stats(stats_df, title)`

**函数签名**：
```python
def print_stats(stats_df: pd.DataFrame, title: str)
```

**功能说明**：
美观地打印统计表，设置 pandas 显示选项以完整展示所有行和列。

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `stats_df` | pd.DataFrame | 统计结果 DataFrame |
| `title` | str | 打印时显示的标题 |

**算法逻辑**：
```
1. 打印分隔线和标题
2. 使用 pd.option_context 临时设置显示选项：
   - max_rows: None（显示所有行）
   - max_columns: None（显示所有列）
   - width: 200（行宽）
   - float_format: 保留6位小数
3. 打印统计表
4. 恢复默认设置（上下文管理器自动处理）
```

**关键代码行**：

| 行号 | 代码 | 作用 |
|------|------|------|
| 91-96 | `with pd.option_context(...)` | 临时设置 pandas 显示选项 |
| 95 | `"display.float_format", "{:.6f}".format` | 浮点数保留6位小数 |

---

### 4.6 `zscore_by_column(df, cols, ddof=1)`

**函数签名**：
```python
def zscore_by_column(df: pd.DataFrame, cols, ddof=1)
```

**功能说明**：
对指定列按列进行 Z-score 标准化，返回标准化后的数据以及每列使用的均值和标准差。

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df` | pd.DataFrame | 必填 | 输入数据表 |
| `cols` | list | 必填 | 需要标准化的列名列表 |
| `ddof` | int | 1 | 标准差计算的自由度 |

**返回值**：

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `df_z` | pd.DataFrame | 标准化后的 DataFrame |
| `means` | pd.Series | 每列的均值 |
| `stds` | pd.Series | 每列的标准差 |

**算法逻辑**：
```
1. 复制 df 到 df_z
2. 计算目标列的均值 (means) 和标准差 (stds)
3. 检查是否有标准差为 0 的列（常数列）：
   - 如果有，打印警告，这些列保持原值不变
4. 对标准差不为 0 的列应用 Z-score 公式：
   df_z[valid_cols] = (df[valid_cols] - means[valid_cols]) / stds[valid_cols]
5. 返回 df_z, means, stds
```

**输入/输出示例**：

```python
# 输入
df = pd.DataFrame({
    'GeneSet_A': [0.5, 0.3, 0.8, 0.6],  # mean=0.55, std=0.2154
    'GeneSet_B': [0.2, 0.2, 0.2, 0.2]   # std=0，常数列
})
df_z, means, stds = zscore_by_column(df, ['GeneSet_A', 'GeneSet_B'])

# 输出（控制台）
警告：以下列标准差为0，无法做z-score，这些列将保持原值不变：
  - GeneSet_B

# df_z['GeneSet_A'] = [-0.2319, -1.1595, 1.1595, 0.2319]
# df_z['GeneSet_B'] = [0.2, 0.2, 0.2, 0.2]（保持不变）
```

**关键代码行**：

| 行号 | 代码 | 作用 |
|------|------|------|
| 112-113 | `means = df[cols].mean()`<br>`stds = df[cols].std(ddof=ddof)` | 计算均值和标准差 |
| 116-121 | `zero_std_cols = stds[stds == 0].index.tolist()` | 检测常数列并警告 |
| 123-124 | `valid_cols = stds[stds != 0].index.tolist()`<br>`df_z[valid_cols] = (df[valid_cols] - means[valid_cols]) / stds[valid_cols]` | 应用 Z-score 公式 |

---

### 4.7 `make_output_path(csv_path, suffix="_zscore")`

**函数签名**：
```python
def make_output_path(csv_path: str, suffix: str = "_zscore")
```

**功能说明**：
根据输入 CSV 路径自动生成输出路径，在原文件名后添加后缀。

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `csv_path` | str | 必填 | 输入 CSV 文件路径 |
| `suffix` | str | "_zscore" | 要添加的后缀 |

**算法逻辑**：
```
1. 使用 os.path.dirname 提取文件夹路径
2. 使用 os.path.basename 提取文件名
3. 使用 os.path.splitext 分离文件名和扩展名
4. 拼接新文件名：stem + suffix + ext
5. 返回完整输出路径
```

**输入/输出示例**：

| 输入 | 输出 |
|------|------|
| `make_output_path("data/scores.csv")` | `"data/scores_zscore.csv"` |
| `make_output_path("data/scores.csv", "_processed")` | `"data/scores_processed.csv"` |

**关键代码行**：

| 行号 | 代码 | 作用 |
|------|------|------|
| 134 | `folder = os.path.dirname(csv_path)` | 提取目录部分 |
| 136 | `stem, ext = os.path.splitext(base)` | 分离文件名和扩展名 |
| 137 | `out_name = f"{stem}{suffix}{ext}"` | 拼接新文件名 |

---

## 5. 主函数流程详解

### 步骤 1：配置参数（行 9-13）

```python
CSV_PATH = r".\HYZ15040_ssGSEA_scores.csv"   # 输入文件路径
NUM_TARGET_COLS = 8                        # 处理最后 8 列
DO_ZSCORE = True                           # 是否执行 Z-score
SAVE_OUTPUT = True                         # 是否保存结果
DDOF = 1                                   # 样本标准差（ddof=1）
```

**参数说明**：

| 参数 | 说明 | 修改建议 |
|------|------|----------|
| `CSV_PATH` | 输入 CSV 文件路径 | 根据实际文件位置修改 |
| `NUM_TARGET_COLS` | 处理最后 N 列 | 根据数据列数调整，通常是基因集数量 |
| `DO_ZSCORE` | 是否执行标准化 | False 时只做统计，不标准化 |
| `SAVE_OUTPUT` | 是否保存结果 | 调试时可设为 False，避免生成文件 |
| `DDOF` | 标准差自由度 | 1=样本标准差（推荐），0=总体标准差 |

### 步骤 2：读取数据（行 143）

```python
df = pd.read_csv(CSV_PATH)
```

使用 pandas 读取 CSV 文件，自动识别表头和数据类型。

### 步骤 3：打印基本信息（行 145）

```python
print_basic_info(df, "原始数据")
```

输出数据的形状和列名，帮助确认数据读取正确。

### 步骤 4：识别目标列（行 148-152）

```python
target_cols = get_target_columns(df, NUM_TARGET_COLS)
print(f"将最后 {NUM_TARGET_COLS} 列作为统计 / z-score 处理列：")
for i, col in enumerate(target_cols, 1):
    print(f"  {i}. {col}")
```

提取最后 8 列作为目标列（基因集评分列），并列出列名供确认。

### 步骤 5：数据类型转换（行 155）

```python
df_numeric = ensure_numeric_columns(df, target_cols)
```

确保目标列都是数值类型，非数值内容会被转换为 NaN。

### 步骤 6：计算原始统计量（行 158-159）

```python
stats_before = compute_stats(df_numeric, target_cols, ddof=DDOF)
print_stats(stats_before, "原始目标列统计信息")
```

计算并显示原始数据的统计信息，作为对比基准。

### 步骤 7：Z-score 标准化（行 162-187）

```python
if DO_ZSCORE:
    df_out, means, stds = zscore_by_column(df_numeric, target_cols, ddof=DDOF)
    # 打印使用的参数
    # 计算并显示标准化后的统计信息
else:
    print("当前设置 DO_ZSCORE = False，不进行z-score。")
    df_out = df_numeric
```

根据配置决定是否执行标准化，并输出使用的均值和标准差参数。

### 步骤 8：保存结果（行 190-199）

```python
if SAVE_OUTPUT:
    if DO_ZSCORE:
        out_path = make_output_path(CSV_PATH, suffix="_zscore")
    else:
        out_path = make_output_path(CSV_PATH, suffix="_processed")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"文件已保存到：\n{out_path}")
```

根据配置保存结果文件，使用 UTF-8-SIG 编码确保 Excel 能正确识别中文。

---

## 6. 关键参数说明

| 参数名 | 位置 | 默认值 | 含义 | 调优建议 |
|--------|------|--------|------|----------|
| `CSV_PATH` | 全局配置 行 9 | `".\HYZ15040_ssGSEA_scores.csv"` | 输入 CSV 文件路径 | **必须修改**为实际文件路径 |
| `NUM_TARGET_COLS` | 全局配置 行 10 | 8 | 处理最后 N 列 | 根据实际基因集数量调整 |
| `DO_ZSCORE` | 全局配置 行 11 | True | 是否执行 Z-score | False 时只做统计 |
| `SAVE_OUTPUT` | 全局配置 行 12 | True | 是否保存输出文件 | 调试时可设为 False |
| `DDOF` | 全局配置 行 13 | 1 | 标准差自由度 | 1=样本标准差（推荐），0=总体标准差 |

### DDOF 参数详解

**DDOF**（Delta Degrees of Freedom）控制标准差计算时的自由度调整：

```
样本标准差（ddof=1）: σ = √[Σ(xi - x̄)² / (n-1)]
总体标准差（ddof=0）: σ = √[Σ(xi - x̄)² / n]
```

| 场景 | 推荐 ddof | 原因 |
|------|----------|------|
| 数据是样本，要推断总体 | 1 | 无偏估计，更准确 |
| 数据就是总体本身 | 0 | 精确计算总体标准差 |
| 与 R 语言结果对比 | 1 | R 默认使用样本标准差 |
| 与 Excel STDEV.P 对比 | 0 | Excel STDEV.P 是总体标准差 |

---

## 7. 数据流示意

```
原始 CSV 文件
    │
    ▼
┌────────────────────────────────────────────┐
│  SampleID │ GeneSet_A │ GeneSet_B │ ...    │
│  Sample1  │    0.5    │   -0.2    │ ...    │
│  Sample2  │    0.3    │    0.1    │ ...    │
│  Sample3  │    0.8    │    0.4    │ ...    │
└────────────────────────────────────────────┘
    │
    ▼ 读取为 DataFrame
┌────────────────────────────────────────────┐
│  形状: (n_samples, n_columns)              │
│  列: ['SampleID', 'GeneSet_A', 'GeneSet_B']│
└────────────────────────────────────────────┘
    │
    ▼ 提取目标列（最后 N 列）
┌────────────────────────────────────────────┐
│  GeneSet_A │ GeneSet_B │ ...               │
│    0.5     │   -0.2    │ ...               │
│    0.3     │    0.1    │ ...               │
│    0.8     │    0.4    │ ...               │
└────────────────────────────────────────────┘
    │
    ▼ 计算统计量
┌────────────────────────────────────────────┐
│  统计量    │ GeneSet_A │ GeneSet_B         │
│  mean      │   0.533   │   0.100           │
│  std       │   0.251   │   0.300           │
│  min       │   0.300   │  -0.200           │
│  max       │   0.800   │   0.400           │
└────────────────────────────────────────────┘
    │
    ▼ Z-score 标准化 (z = (x - mean) / std)
┌────────────────────────────────────────────┐
│  GeneSet_A │ GeneSet_B │ ...               │
│  -0.131    │  -1.000   │ ...               │
│  -0.928    │    0.000  │ ...               │
│   1.059    │    1.000  │ ...               │
└────────────────────────────────────────────┘
    │
    ▼ 保存结果
┌────────────────────────────────────────────┐
│  HYZ15040_ssGSEA_scores_zscore.csv         │
└────────────────────────────────────────────┘
```

---

## 8. 初学者注意事项

### 8.1 常见错误

| 错误 | 原因 | 解决方法 |
|------|------|----------|
| `FileNotFoundError` | `CSV_PATH` 路径错误 | 修改为正确的绝对路径或确保文件在当前目录 |
| `ValueError: NUM_TARGET_COLS 必须 > 0` | 配置参数错误 | 检查 `NUM_TARGET_COLS` 是否设为 0 或负数 |
| `ValueError: NUM_TARGET_COLS=...超过总列数` | 目标列数设置过大 | 减小 `NUM_TARGET_COLS`，确保不超过总列数 |
| 警告：标准差为0 | 某列所有值相同（常数列） | 正常现象，该列将保持原值 |
| 警告：存在无法转换的内容 | 数据中有非数值内容 | 检查原始数据，确认是否为有效缺失值标记 |

### 8.2 调试建议

1. **先检查文件路径**：
   ```python
   import os
   print(os.path.exists(CSV_PATH))  # 应该输出 True
   ```

2. **查看前几行数据**：
   ```python
   df = pd.read_csv(CSV_PATH)
   print(df.head())  # 查看前5行
   print(df.dtypes)  # 查看各列数据类型
   ```

3. **确认目标列正确**：
   ```python
   target_cols = get_target_columns(df, NUM_TARGET_COLS)
   print("目标列:", target_cols)
   ```

4. **检查数值转换结果**：
   ```python
   df_numeric = ensure_numeric_columns(df, target_cols)
   print(df_numeric[target_cols].isna().sum())  # 查看每列缺失值数量
   ```

5. **验证 Z-score 结果**：
   ```python
   # 标准化后的数据应该均值为0，标准差为1
   print(df_z[target_cols].mean())  # 应该接近0
   print(df_z[target_cols].std())   # 应该接近1
   ```

### 8.3 重要提醒

- ⚠️ **备份原始数据**：虽然脚本不会修改原文件，但建议保留原始数据备份
- ⚠️ **检查编码问题**：如果中文显示乱码，尝试修改 `encoding="utf-8-sig"` 为 `"gbk"`
- ⚠️ **注意 NaN 处理**：标准化时 NaN 会保持为 NaN，不影响其他值的计算
- ⚠️ **常数列处理**：标准差为 0 的列无法进行 Z-score，会保持原值并发出警告

---

## 9. 扩展思考

### 9.1 可能的改进方向

1. **支持多种标准化方法**
   - 当前：Z-score 标准化
   - 扩展：Min-Max 归一化、Robust Scaling（使用中位数和 IQR）

2. **批量处理多个文件**
   - 当前：单文件处理
   - 扩展：遍历文件夹，批量处理所有 CSV 文件

3. **可视化功能**
   - 添加箱线图、小提琴图展示数据分布
   - 绘制标准化前后的对比图

4. **交互式参数配置**
   - 当前：硬编码参数
   - 扩展：命令行参数解析（argparse）或配置文件（YAML/JSON）

5. **缺失值处理策略**
   - 当前：保留 NaN，仅做警告
   - 扩展：支持填充策略（均值填充、中位数填充、删除等）

6. **按组标准化**
   - 当前：全局标准化
   - 扩展：支持按样本分组（如按疾病类型）分别标准化

### 9.2 与其他组件的集成

```
基因表达矩阵
    │
    ▼ ssGSEA 分析工具（如 GSVA, ssGSEA2.0）
ssGSEA 评分矩阵
    │
    ▼ zscore.py（本文档）
标准化后的评分矩阵
    │
    ▼ 下游分析
    ├─ 聚类分析（如 K-means, 层次聚类）
    ├─ 可视化（热图、PCA、UMAP）
    └─ 差异分析（如 limma, DESeq2）
```

### 9.3 Z-score 的应用场景

1. **热图绘制**：标准化后的数据更适合绘制热图，颜色对比更明显
2. **聚类分析**：消除量纲影响，使不同基因集的权重相同
3. **机器学习**：大多数算法要求输入特征具有相似尺度
4. **差异表达**：Z-score > 2 或 < -2 可作为显著活性的阈值

---

## 10. 总结

`zscore.py` 是一个简洁实用的 ssGSEA 评分标准化工具，其核心贡献是：

1. **数据质量检查**：自动检测非数值内容并给出警告
2. **全面统计分析**：提供计数、均值、标准差、分位数等完整统计信息
3. **安全标准化**：处理常数列等边界情况，避免除零错误
4. **可追溯性**：输出使用的均值和标准差参数，便于复现

对于初学者，理解这个脚本有助于掌握：
- pandas DataFrame 的基本操作（读取、列选择、统计计算）
- Z-score 标准化的原理和应用场景
- 数据预处理中的边界情况处理
- Python 文件路径操作（os.path）

---

*文档生成时间：2026年4月11日*
