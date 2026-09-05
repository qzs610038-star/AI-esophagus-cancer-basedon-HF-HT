# split.py 深度解读指南

> 本文档面向初学者，详细解读病理图像空间无重叠划分脚本的原理与实现。

---

## 1. 文件概述

**一句话说明**：`split.py` 是一个用于病理图像 patch 数据集的**空间无重叠划分工具**，将图像块按空间距离规则划分为训练集和验证集，防止因空间邻近导致的数据泄漏。

**在项目中的角色**：这是数据预处理流程的关键环节，位于图像切分（生成 patch）之后、模型训练之前，确保训练集和验证集在空间上相互独立。

---

## 2. 背景知识

### 2.1 什么是"空间泄漏"（Spatial Leakage）？

在病理图像分析中，一张全切片图像（WSI, Whole Slide Image）会被切分成大量小的图像块（patch）。这些 patch 之间存在**空间相关性**：

- 相邻的 patch 往往包含相似的病理组织
- 如果训练集和验证集中的 patch 来自相邻区域，模型可能"偷看"到验证集的信息
- 这会导致验证集性能被**虚高估计**，模型泛化能力差

### 2.2 为什么需要"空间无重叠划分"？

| 划分方式 | 问题 | 后果 |
|---------|------|------|
| 随机划分 | 相邻 patch 可能分到不同集合 | 数据泄漏，验证分数虚高 |
| 空间无重叠划分 | 确保验证集 patch 之间保持最小距离 | 真实反映模型泛化能力 |

**核心思想**：验证集中的任意两个 patch 之间必须保持足够的空间距离（如 350 像素），确保它们来自不同的组织区域。

---

## 3. 整体流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                         输入阶段                                  │
│  ┌─────────────────┐                                            │
│  │  读取PNG文件列表  │ ← patches_dir 目录下的所有 .png 文件        │
│  │  (patch_x{y}_y{x}.png) │                                      │
│  └────────┬────────┘                                            │
└───────────┼─────────────────────────────────────────────────────┘
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      坐标解析阶段                                 │
│  ┌─────────────────┐                                            │
│  │ 解析文件名中的坐标 │ ← 正则提取 x, y 坐标                      │
│  │ patch_x4641_y16969 │ → (4641, 16969)                          │
│  └────────┬────────┘                                            │
└───────────┼─────────────────────────────────────────────────────┘
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     空间划分阶段                                  │
│  ┌─────────────────────────────────────────┐                    │
│  │  随机打乱 patch 顺序                     │                    │
│  │  逐个检查与已选验证集 patch 的距离        │                    │
│  │  距离 < threshold → 排除                 │                    │
│  │  距离 >= threshold → 加入验证集          │                    │
│  └─────────────────────────────────────────┘                    │
└───────────┬─────────────────────────────────────────────────────┘
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     数据集生成阶段                                │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │   训练集索引      │    │   验证集索引      │                   │
│  │  (90% patches)  │    │  (10% patches)  │                     │
│  └────────┬────────┘    └────────┬────────┘                     │
│           ▼                      ▼                              │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ train_patches/  │    │  val_patches/   │                     │
│  │   移动文件      │    │    移动文件      │                     │
│  └─────────────────┘    └─────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 逐函数详解

### 4.1 `parse_coordinates_from_filename(filename)`

**函数签名**：
```python
def parse_coordinates_from_filename(filename)
```

**功能说明**：
从 patch 文件名中提取 x、y 坐标信息。支持带扩展名或不带扩展名的文件名。

**算法逻辑**：
1. 使用正则表达式 `patch_x(\d+)_y(\d+)` 匹配文件名
2. 提取括号中的数字作为 x 和 y 坐标
3. 如果匹配失败，打印警告并返回 None

**输入/输出示例**：

| 输入 | 输出 |
|------|------|
| `'patch_x4641_y16969.png'` | `(4641, 16969)` |
| `'patch_x100_y200'` | `(100, 200)` |
| `'invalid_name.jpg'` | `(None, None)` |

**关键代码行**：

| 行号 | 代码 | 作用 |
|------|------|------|
| 12 | `match = re.search(r'patch_x(\d+)_y(\d+)', filename)` | 正则匹配坐标模式 |
| 14-15 | `x = int(match.group(1))` | 提取并转换 x 坐标 |
| 18 | `print(f"Warning: ...")` | 解析失败时给出警告 |

---

### 4.2 `find_valid_indices_to_exclude(...)`

**函数签名**：
```python
def find_valid_indices_to_exclude(
    patch_filenames,      # patch 文件名列表
    val_size_fraction,    # 验证集比例（如 0.1）
    distance_threshold,   # 距离阈值（像素）
    random_state=42       # 随机种子
)
```

**功能说明**：
核心算法函数。根据空间距离规则，找出应该被**排除在验证集之外**的所有 patch 索引。

**算法逻辑**（贪心算法）：

```
伪代码：
1. 解析所有 patch 的坐标，建立 有效索引 → 原始索引 的映射
2. 计算目标验证集大小：总有效 patch 数 × 验证集比例
3. 随机打乱 patch 顺序
4. 遍历每个 patch：
   - 计算它与所有已选验证集 patch 的欧氏距离
   - 如果任意距离 < threshold：标记为"排除"
   - 否则且验证集未满：加入验证集
   - 否则：标记为"排除"
5. 返回所有应排除的原始索引（包括无法解析坐标的）
```

**欧氏距离计算**：
```
distance = √[(x1 - x2)² + (y1 - y2)²]
```
代码中使用 `np.linalg.norm()` 实现。

**输入/输出示例**：

假设有 100 个 patch，验证集比例 0.1（目标 10 个）：
- 输入：`['patch_x0_y0.png', 'patch_x100_y100.png', ...]` 
- 输出：`[0, 2, 3, 5, ...]`（应排除的索引列表，长度约 90）

**关键代码行**：

| 行号 | 代码 | 作用 |
|------|------|------|
| 28 | `valid_indices_map = {}` | 建立有效索引到原始索引的映射 |
| 39 | `coordinates = np.array(coordinates)` | 转换为 NumPy 数组便于计算 |
| 48 | `target_val_size = int(n_valid_patches * val_size_fraction)` | 计算目标验证集大小 |
| 52 | `shuffled_indices = np.random.permutation(n_valid_patches)` | 随机打乱顺序 |
| 63 | `distance = np.linalg.norm(current_coord - selected_coord)` | 计算欧氏距离 |
| 64 | `if distance < distance_threshold:` | 距离判断核心逻辑 |
| 92-94 | `original_idx = valid_indices_map[valid_idx]` | 映射回原始索引 |

---

### 4.3 `main()`

**函数签名**：
```python
def main()
```

**功能说明**：
脚本主入口，负责配置参数、调用划分算法、创建输出目录、移动文件。

**执行流程**：

```
1. 配置参数（路径、比例、阈值、随机种子）
2. 读取目录中的所有 PNG 文件
3. 调用 find_valid_indices_to_exclude() 获取排除索引
4. 计算验证集索引 = 全部索引 - 排除索引
5. 精确控制验证集大小（如过多则随机抽样）
6. 计算训练集索引 = 全部索引 - 验证集索引
7. 创建 train_patches/ 和 val_patches/ 目录
8. 移动文件到对应目录
9. 输出验证集坐标分布统计
```

**关键代码行**：

| 行号 | 代码 | 作用 |
|------|------|------|
| 101 | `patches_dir = r"D:\..."` | **需要修改的数据路径** |
| 108 | `patch_filenames = [f for f in all_files if f.lower().endswith('.png')]` | 筛选 PNG 文件 |
| 120 | `val_indices_in_list = sorted(list(set(all_original_indices) - set(indices_to_exclude_from_val)))` | 计算验证集索引 |
| 127 | `selected_val_indices = np.random.choice(...)` | 验证集过大时随机抽样 |
| 146-147 | `os.makedirs(..., exist_ok=True)` | 创建输出目录 |
| 162 | `shutil.move(src_path, dst_path)` | 移动文件（非复制） |
| 194-196 | `print(f"Validation X range: ...")` | 输出坐标范围统计 |

---

## 5. 主函数流程详解

以下是 `main()` 函数的逐步解读：

### 步骤 1：配置参数（行 100-104）

```python
patches_dir = r"D:\PycharmProjects\AIPath-data\patch\HYZ15040"
val_size_fraction = 0.1      # 10% 作为验证集
distance_threshold_px = 350  # 350像素距离阈值
random_state = 42            # 固定随机种子，保证可复现
```

**注意**：`patches_dir` 需要根据实际情况修改！

### 步骤 2：读取文件列表（行 107-109）

```python
all_files = os.listdir(patches_dir)
patch_filenames = [f for f in all_files if f.lower().endswith('.png')]
```

- 列出目录中所有文件
- 筛选出 `.png` 结尾的文件（不区分大小写）

### 步骤 3：空间划分（行 112-114）

```python
indices_to_exclude_from_val = find_valid_indices_to_exclude(
    patch_filenames, val_size_fraction, distance_threshold_px, random_state
)
```

调用核心算法，获取应排除的索引列表。

### 步骤 4：确定数据集划分（行 117-131）

```python
# 全部索引 [0, 1, 2, ..., n-1]
all_original_indices = list(range(len(patch_filenames)))

# 验证集索引 = 全部 - 排除
val_indices_in_list = sorted(list(set(all_original_indices) - set(indices_to_exclude_from_val)))

# 如果验证集过大，随机抽样到目标大小
target_val_count = int(len(patch_filenames) * val_size_fraction)
if len(val_indices_in_list) > target_val_count:
    selected_val_indices = np.random.choice(val_indices_in_list, size=target_val_count, replace=False)
    val_indices_in_list = sorted(selected_val_indices.tolist())

# 训练集索引 = 全部 - 验证集
train_indices_in_list = sorted(list(set(all_original_indices) - set(val_indices_in_list)))
```

### 步骤 5：创建目录并移动文件（行 142-179）

```python
# 创建目录
os.makedirs(train_folder_path, exist_ok=True)
os.makedirs(val_folder_path, exist_ok=True)

# 移动文件（使用 shutil.move）
for idx in train_indices_in_list:
    shutil.move(src_path, dst_path)
```

**注意**：`shutil.move` 是**移动**而非复制，原目录中的文件会被移除。

---

## 6. 关键参数说明

| 参数名 | 位置 | 默认值 | 含义 | 调优建议 |
|--------|------|--------|------|----------|
| `patches_dir` | main() 行 101 | 需手动设置 | 输入 patch 目录路径 | 修改为实际数据路径 |
| `val_size_fraction` | main() 行 102 | 0.1 | 验证集占总数据比例 | 通常 0.1~0.2，数据量大时可减小 |
| `distance_threshold_px` | main() 行 103 | 350 | 空间距离阈值（像素） | 根据 patch 大小调整，建议 ≥ patch 尺寸 |
| `random_state` | main() 行 104 | 42 | 随机种子 | 固定值保证结果可复现 |

### 参数调优指南

**distance_threshold_px 的选择**：

- **太小**（如 100）：相邻 patch 可能进入验证集，仍有空间泄漏风险
- **合适**（如 350）：与典型 patch 尺寸（如 224×224 或 256×256）相当，确保空间独立性
- **太大**（如 1000）：可选的验证集 patch 太少，可能导致验证集过小

**建议**：threshold ≈ 1.5 × patch_size

---

## 7. 数据流示意

```
原始数据
    │
    ▼
┌────────────────────────────────────┐
│  patch_x0_y0.png                   │
│  patch_x0_y224.png                 │
│  patch_x224_y0.png                 │
│  patch_x224_y224.png               │
│  ...                               │
└────────┬───────────────────────────┘
         │
         ▼ 解析坐标
┌────────────────────────────────────┐
│  [(0, 0), (0, 224), (224, 0), ...] │
└────────┬───────────────────────────┘
         │
         ▼ 空间划分算法（threshold=350）
┌────────────────────────────────────┐
│  验证集候选: [(0, 0)]              │
│  排除（距离<350）: [(0, 224), (224, 0)] │
│  验证集候选: [(0, 0), (224, 224)]  │
└────────┬───────────────────────────┘
         │
         ▼ 文件操作
┌────────────────────┐  ┌────────────────────┐
│  train_patches/    │  │  val_patches/      │
│  patch_x0_y224.png │  │  patch_x0_y0.png   │
│  patch_x224_y0.png │  │  patch_x224_y224.png│
└────────────────────┘  └────────────────────┘
```

---

## 8. 初学者注意事项

### 8.1 常见错误

| 错误 | 原因 | 解决方法 |
|------|------|----------|
| `FileNotFoundError` | `patches_dir` 路径错误 | 修改为正确的绝对路径 |
| 验证集为空 | 所有 patch 都被排除 | 减小 `distance_threshold_px` 或增大验证集比例 |
| 验证集过大 | 空间约束太松 | 增大 `distance_threshold_px` |
| 文件名解析失败 | 文件名格式不符 | 确保文件名格式为 `patch_x{d}_y{d}.png` |

### 8.2 调试建议

1. **先检查文件名格式**：
   ```python
   print(patch_filenames[:5])  # 查看前5个文件名
   ```

2. **验证坐标解析**：
   ```python
   for name in patch_filenames[:5]:
       x, y = parse_coordinates_from_filename(name)
       print(f"{name} -> ({x}, {y})")
   ```

3. **检查划分结果**：
   ```python
   print(f"训练集: {len(train_indices_in_list)} 个")
   print(f"验证集: {len(val_indices_in_list)} 个")
   ```

4. **可视化验证集分布**（可选）：
   ```python
   import matplotlib.pyplot as plt
   plt.scatter(val_coords_x, val_coords_y, c='red', label='Validation')
   plt.legend()
   plt.show()
   ```

### 8.3 重要提醒

- ⚠️ **备份数据**：`shutil.move` 会移动文件，运行前请备份原始数据
- ⚠️ **路径格式**：Windows 路径建议使用原始字符串 `r"C:\path\to\dir"`
- ⚠️ **可复现性**：固定 `random_state` 确保每次运行结果一致

---

## 9. 扩展思考

### 9.1 可能的改进方向

1. **支持多种划分策略**
   - 当前：基于欧氏距离的贪心算法
   - 扩展：K-means 聚类划分、网格划分、基于组织区域的划分

2. **支持多层级验证**
   - 当前：单一验证集
   - 扩展：K-fold 交叉验证，每折都保证空间独立性

3. **可视化功能**
   - 添加 matplotlib 可视化，直观展示划分结果
   - 在 WSI 缩略图上标记训练/验证区域

4. **支持其他文件格式**
   - 当前：仅支持 `.png`
   - 扩展：支持 `.jpg`, `.tiff`, `.npy` 等

5. **配置文件支持**
   - 当前：硬编码参数
   - 扩展：使用 YAML/JSON 配置文件，命令行参数解析

### 9.2 与其他组件的集成

```
WSI 全切片图像
    │
    ▼ 切分工具（如 OpenSlide）
patch 图像集合
    │
    ▼ split.py（本文档）
空间无重叠的训练/验证集
    │
    ▼ train.py（模型训练）
训练好的病理图像分类模型
```

---

## 10. 总结

`split.py` 是一个简洁而实用的病理图像数据集划分工具，其核心贡献是：

1. **解决空间泄漏问题**：通过距离阈值确保验证集的独立性
2. **贪心算法实现**：简单高效，适合大规模数据集
3. **可复现性**：固定随机种子，结果稳定

对于初学者，理解这个脚本有助于掌握：
- 正则表达式在文件名解析中的应用
- NumPy 在坐标计算中的使用
- 空间数据划分的核心思想
- Python 文件操作（os, shutil）

---

*文档生成时间：2026年4月11日*
