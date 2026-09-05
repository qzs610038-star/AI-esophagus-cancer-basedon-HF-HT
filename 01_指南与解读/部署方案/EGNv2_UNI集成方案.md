# EGN-v2 + UNI2-h 特征集成方案

---

## 第一部分：为什么要做这个修改

### 1.1 背景与动机

跨患者泛化实验结果给出了强烈的信号：

| 模型 | 特征提取器 | 单患者 Val PCC | 跨患者 Test PCC | 衰减幅度 |
|------|-----------|---------------|----------------|---------|
| HisToGene-UNI | UNI2-h 预提取特征 | **0.5336** | **0.3946** | **-26%** |
| EGN-v2 | ResNet-50 (ImageNet预训练) | **0.4048** | **0.1950** | **-52%** |

- UNI2-h 预训练特征在跨患者场景下展现了显著优势：衰减仅 **-26%**，而 ResNet-50 衰减高达 **-52%**
- 自然思路：将 EGN-v2 的 ResNet-50 特征提取器替换为 UNI2-h 预提取特征

> **通俗类比**：比如 ResNet-50 是"只看过普通照片的观察员"，它在 ImageNet 的猫狗汽车图片上学会了看纹理和边缘；UNI2-h 是"在全球百万张病理切片上训练过的专业病理医生"，它见过无数肿瘤组织、细胞形态和染色模式。当你让它看一张新的食管癌切片时，病理医生能迅速识别出关键特征，而普通观察员只能说出"这里有红色的区域"。

### 1.2 预期收益

| 收益维度 | 说明 |
|---------|------|
| 跨患者泛化能力提升 | 参考 HisToGene-UNI 的经验，UNI2-h 特征的跨患者通用性是关键优势 |
| 训练速度大幅提升 | 无需实时用 ResNet-50 推理提取特征，直接从磁盘加载 .pt 缓存 |
| 参数量减少 | 去掉 ResNet-50 的 ~23M 参数（大部分冻结但仍占用显存） |
| 保留图神经网络优势 | GraphSAGE 的空间建模能力完整保留，这是 EGN-v2 相对 HisToGene 的独有优势 |

> **通俗解释**：这个改动相当于给 EGN-v2 换了一个"更专业的眼睛"（UNI2-h），同时保留了它"善于分析邻居关系"（GraphSAGE）和"善于查找参考案例"（Exemplar）的特长。

---

## 第二部分：EGN-v2 当前架构解析

### 2.1 完整数据流

```
当前 EGN-v2 数据流：
PNG图像 (224×224)
    ↓
ResNet-50 特征提取器 ← ★ 这是要替换的部分 ★
    ↓
2048维特征向量
    ↓
┌───────────────────────────────────────────────────────┐
│  以下部分完全保留，不做任何修改                         │
│                                                       │
│  投影层: Linear(2048→512) + LayerNorm + GELU + Dropout│
│      ↓                                                │
│  空间图构建 (基于物理坐标，与特征无关)                   │
│      ↓                                                │
│  GraphSAGE (2层图卷积 + 残差连接)                      │
│      ↓                                                │
│  Exemplar融合 (KNN检索+加权平均+拼接投影)              │
│      ↓                                                │
│  回归头 → 30条通路预测                                 │
└───────────────────────────────────────────────────────┘
```

> **详细注释 — 各模块的作用**：
>
> - **投影层**（`feature_proj`）：将高维特征（2048维）降维到512维。就像一个"翻译官"，把不同语言（不同维度的特征空间）翻译成统一格式，后面的 GraphSAGE 只需要处理512维的统一表示。
>
> - **空间图构建**（`graph_builder.py`）：根据每个 patch 的物理像素坐标（x, y），用 `radius_neighbors_graph` 找到距离在 `radius=300` 像素以内的邻居，构建"邻居关系网"。关键特点：**只看坐标，不看特征**——无论特征是2048维还是1536维，图的拓扑结构完全一样。
>
> - **GraphSAGE**（`SAGEConv`层）：在邻居网络上传递信息。每个节点从邻居那里"收集意见"，然后更新自己的特征表示。就像"问邻居的意见来修正自己的判断"。2层 GraphSAGE 意味着每个节点可以获取2跳邻居的信息。
>
> - **Exemplar融合**：从训练集中检索最相似的 K 个"参考案例"（Exemplar），将它们的特征加权平均后与当前节点的 GraphSAGE 输出拼接，再通过一个投影层融合。就像"查找训练集中最相似的参考案例来辅助预测"。

### 2.2 为什么可以替换（耦合度分析）

| 模块 | 与 ResNet 的耦合程度 | 需要的修改 | 原因 |
|------|---------------------|-----------|------|
| 空间图构建 | **零耦合** | 无需修改 | 只用物理坐标（x, y），不看特征 |
| GraphSAGE | **零耦合** | 无需修改 | 只看 hidden_dim=512 的输入，不关心原始特征维度 |
| 回归头 | **零耦合** | 无需修改 | 输入是 hidden_dim=512，与特征维度无关 |
| Exemplar库 | **中耦合** | 需要适配 | 特征维度变化 → KNN距离尺度可能变化，需从1536维特征构建 |
| 投影层 | **高耦合** | 需要修改 | 输入维度从 2048 → **1536**，这是唯一的结构性改动 |

> **类比**：就像换了一台更好的摄像机（UNI），照片尺寸略有不同（2048→1536像素），但后面的分析流程（图卷积、邻居参考）完全不受影响，只需要在"接口处"（投影层）做个小调整——把门框从2048mm改成1536mm，里面房间的家具布局完全不变。

---

## 第三部分：修改方案详解

### 3.1 核心原则

- **只新增文件，不修改现有 `egnv2/` 目录下的任何文件**
- 新文件放在项目根目录（与现有 `train_cross_patient_egnv2.py` 同级）
- 完整复用 `egnv2/` 下的图构建、Exemplar库、模型核心组件

### 3.2 需要新建的三个文件

---

#### 文件1：`egnv2_uni_dataset.py` — UNI特征版数据集

**设计参考**：`histogene/dataset_uni.py` 的三层交集过滤 + `egnv2/dataset.py` 的原始坐标保留

**数据来源**：`uni2h_cache/{患者名}/{train|val}/` 下的 `.pt` 特征文件（**1536维**）

**关键区别**（与 HisToGene-UNI 的 dataset_uni.py 对比）：
- HisToGene-UNI 使用**归一化坐标**（`_coord_to_index` 映射到 [0, n_pos-1]）
- EGN-v2-UNI 使用**原始像素坐标**（直接从文件名解析 x、y 整数值），因为 EGN-v2 的图构建需要原始坐标来计算物理距离

> **注释 — 三层交集过滤的概念**：
> 三个"名单"取交集，确保每个样本同时满足三个条件：
> 1. `patches_dir` 中有对应的 PNG 图像 → 说明这个 patch 存在
> 2. `labels_csv` 中有对应的标签 → 说明这个 patch 有监督信号
> 3. `uni2h_cache` 中有对应的 .pt 缓存 → 说明 UNI2-h 特征已预提取
>
> 只有三个条件同时满足的样本才会被纳入数据集，避免训练时出现"有图无标签"或"有标签无特征"的错误。

**关键代码框架**：

```python
import os, re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset

def parse_coordinates(filename):
    """从文件名 patch_x4641_y16969.pt 解析坐标"""
    match = re.search(r'x(\d+)_y(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

class EGNv2UNIDataset(Dataset):
    """
    EGN-v2 的 UNI2-h 特征数据集
    与 HisToGeneUNIDataset 的关键区别：保留原始像素坐标（不做归一化）
    """
    def __init__(self, feature_cache_dir, patches_dir, labels_csv, target_cols=None):
        """
        Args:
            feature_cache_dir: UNI2-h 缓存目录，如 uni2h_cache/HYZ15040/train/
            patches_dir: PNG 图像目录（仅用于枚举文件名，不加载图像）
            labels_csv: Z-score 标准化标签 CSV
            target_cols: 目标列名列表（默认自动检测）
        """
        # 第一步：扫描 .pt 缓存文件，构建 stem 集合
        cached_stems = set()
        for fname in os.listdir(feature_cache_dir):
            if fname.lower().endswith('.pt'):
                cached_stems.add(fname[:-3])  # 去掉 .pt 后缀

        # 第二步：读取 CSV 标签
        df = pd.read_csv(labels_csv)
        id_col = df.columns[0]
        if target_cols is None:
            target_cols = list(df.columns[1:])
        self.target_cols = target_cols
        label_map = {}
        for _, row in df.iterrows():
            stem = str(row[id_col]).replace('.png', '')
            label_map[stem] = row[target_cols].values.astype(np.float32)

        # 第三步：三层交集过滤（PNG ∩ CSV ∩ .pt 缓存）
        self.samples = []  # (stem, x, y, targets)
        self.feature_cache_dir = feature_cache_dir
        for fname in sorted(os.listdir(patches_dir)):
            if not fname.lower().endswith('.png'):
                continue
            stem = fname.replace('.png', '')
            if stem not in label_map:
                continue
            if stem not in cached_stems:
                continue
            # 第四步：提取原始像素坐标（从文件名解析，不做归一化！）
            x, y = parse_coordinates(fname)
            if x is None:
                continue
            self.samples.append((stem, x, y, label_map[stem]))

        print(f"[EGNv2UNIDataset] 加载 {len(self.samples)} 样本")
        print(f"  特征缓存: {feature_cache_dir}")
        print(f"  坐标模式: 原始像素坐标（不归一化）")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem, x, y, targets = self.samples[idx]
        # 加载 UNI2-h 预提取特征（1536维）
        pt_path = os.path.join(self.feature_cache_dir, f"{stem}.pt")
        feature = torch.load(pt_path, map_location='cpu', weights_only=True)
        if isinstance(feature, dict) and "feature" in feature:
            feature = feature["feature"]
        feature = feature.float()
        if feature.dim() > 1:
            feature = feature.flatten()

        # 注意：返回原始坐标（float），不是归一化后的索引！
        return (feature,
                torch.tensor(x, dtype=torch.float32),   # 原始像素 x
                torch.tensor(y, dtype=torch.float32),   # 原始像素 y
                torch.tensor(targets, dtype=torch.float32))

    @classmethod
    def from_multiple_patients(cls, patient_configs):
        """
        跨患者联合数据集，合并多个患者的数据
        Args:
            patient_configs: list of dicts，每个包含:
                - feature_cache_dir: str, 如 uni2h_cache/JFX0729/train/
                - patches_dir: str
                - labels_csv: str
                - patient_name: str（可选）
        Returns:
            merged_dataset: ConcatDataset
            target_cols: list
        """
        datasets = []
        target_cols = None
        for i, config in enumerate(patient_configs):
            dataset = cls(
                feature_cache_dir=config['feature_cache_dir'],
                patches_dir=config['patches_dir'],
                labels_csv=config['labels_csv'],
                target_cols=target_cols,
            )
            if target_cols is None:
                target_cols = dataset.target_cols
            datasets.append(dataset)
            name = config.get('patient_name', f'patient_{i}')
            print(f"  [{name}] {len(dataset)} 样本")
        merged = ConcatDataset(datasets)
        total = sum(len(d) for d in datasets)
        print(f"\n[MultiPatient-UNI] 合并完成: {len(datasets)} 患者, 共 {total} 样本")
        return merged, target_cols
```

---

#### 文件2：`egnv2_uni_model.py` — 调整输入维度的模型

**设计思路**：基于 `egnv2/model.py` 的 `EGNv2Model`，唯一改动是 `in_dim` 从 **2048 → 1536**

**去掉的部分**：`ResNetFeatureExtractor` 类（不再需要，UNI2-h 特征已预提取）

**完全不变的部分**：
- `feature_proj` 的结构（Linear + LayerNorm + GELU + Dropout）
- `graph_convs`（2层 SAGEConv + LayerNorm + 残差连接）
- `exemplar_fuse`（拼接投影层）
- `regressor`（三层 MLP 回归头）

> **类比**：这个修改就像把门框从2048mm调整到1536mm，里面房间的家具布局完全不变。GraphSAGE、Exemplar融合、回归头都在门框后面的"房间"里，不受影响。

**关键改动**：

```python
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class EGNv2UNIModel(nn.Module):
    """
    EGN-v2 + UNI2-h 版模型
    与原版 EGNv2Model 唯一区别：in_dim=1536（UNI2-h输出维度）
    不含 ResNetFeatureExtractor（特征已预提取）
    """
    def __init__(self, in_dim=1536, hidden_dim=512, n_targets=30,
                 graph_layers=2, dropout=0.3, k_exemplars=10):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.k_exemplars = k_exemplars

        # 1. 特征投影层 — ★ 唯一改动：2048 → 1536 ★
        self.feature_proj = nn.Sequential(
            nn.Linear(1536, 512),     # 原版: nn.Linear(2048, 512)
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2-4. 以下与原版 EGNv2Model 完全一致，不做任何修改
        # GraphSAGE层
        self.graph_convs = nn.ModuleList()
        self.graph_norms = nn.ModuleList()
        for _ in range(graph_layers):
            self.graph_convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.graph_norms.append(nn.LayerNorm(hidden_dim))
        self.graph_dropout = nn.Dropout(dropout)

        # Exemplar融合层
        self.exemplar_fuse = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_targets),
        )

    def forward(self, node_features, edge_index, exemplar_agg_features=None):
        """前向传播，与原版完全一致"""
        h = self.feature_proj(node_features)         # (N, 512)
        for conv, norm in zip(self.graph_convs, self.graph_norms):
            h_res = h
            h = conv(h, edge_index)
            h = norm(h)
            h = h + h_res   # 残差连接
            h = nn.functional.gelu(h)
            h = self.graph_dropout(h)
        if exemplar_agg_features is not None:
            h = self.exemplar_fuse(torch.cat([h, exemplar_agg_features], dim=-1))
        out = self.regressor(h)   # (N, 30)
        return out
```

---

#### 文件3：`train_egnv2_uni.py` — 完整训练脚本

**设计参考**：`train_cross_patient_egnv2.py` 的训练流程结构

**关键变化**（对比原版 `train_cross_patient_egnv2.py`）：

| 步骤 | 原版（ResNet-50） | UNI版 |
|------|------------------|-------|
| 数据集 | `EGNv2Dataset`（加载PNG图像） | `EGNv2UNIDataset`（加载.pt缓存） |
| 特征提取 | `ResNetFeatureExtractor` 实时推理 | `torch.load()` 直接加载缓存 |
| 模型 | `EGNv2Model(in_dim=2048)` | `EGNv2UNIModel(in_dim=1536)` |
| Exemplar库 | 从2048维特征构建 | 从**1536维**特征构建 |
| 图构建 | 从原始坐标构建（不变） | 从原始坐标构建（不变） |
| 图卷积/融合/回归 | 完全不变 | 完全不变 |

**训练流程伪代码**：

```python
# 1. 加载数据（从缓存直接读取，无需图像推理）
train_dataset, target_cols = EGNv2UNIDataset.from_multiple_patients(train_configs)
test_dataset, _ = EGNv2UNIDataset.from_multiple_patients(test_configs)

# 2. 批量加载 UNI 特征（替代 ResNet 推理步骤）
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
for features, raw_x, raw_y, targets in train_loader:
    all_features.append(features)   # 直接用 (B, 1536) 特征
    all_coords.append(torch.stack([raw_x, raw_y], dim=1))
    all_targets.append(targets)
train_features = torch.cat(all_features)   # (N, 1536)
train_coords = torch.cat(all_coords)       # (N, 2)
train_targets = torch.cat(all_targets)     # (N, 30)

# 3. 图构建（复用 egnv2/graph_builder.py，坐标不变，图不变）
train_edge_index = build_spatial_graph(train_coords.numpy(), radius=300)

# 4. Exemplar库构建（从1536维UNI特征构建）
exemplar_lib = ExemplarLibrary(train_features, train_targets)
train_exemplar_agg, _ = compute_exemplar_agg_features(
    train_features, exemplar_lib, hidden_dim=512, k=10, device=device)

# 5. 模型（in_dim=1536）
model = EGNv2UNIModel(in_dim=1536, hidden_dim=512, n_targets=30)

# 6. 训练循环（与原版完全一致）
for epoch in range(num_epochs):
    preds = model(train_data.x, train_data.edge_index, train_exemplar_agg)
    loss = criterion(preds, train_data.y)
    loss.backward()
    optimizer.step()
    # ... 验证、早停、保存 ...
```

> **通俗解释**：原版训练流程中，最耗时的步骤是"ResNet-50推理"——需要把每张224×224的PNG图像送入GPU计算2048维特征。UNI版完全跳过了这一步，因为特征已经预提取好存在磁盘上，只需 `torch.load()` 一行代码即可加载。训练速度可提升数倍。

### 3.3 可复用的现有模块（不需要修改）

| 模块 | 文件路径 | 说明 |
|------|---------|------|
| 空间图构建 | `egnv2/graph_builder.py` | `build_spatial_graph(coords, radius)` — 只用坐标，与特征维度无关 |
| Exemplar库 | `egnv2/model.py` → `ExemplarLibrary` | KNN检索+加权平均，特征维度透明传递 |
| Exemplar聚合 | `egnv2/exemplar_builder.py` → `compute_exemplar_agg_features` | 投影到 hidden_dim，输入维度自适应 |
| 评估指标 | `egnv2/utils.py` → `compute_metrics` | PCC/R²/MAE 计算，与模型无关 |
| 通知机制 | `notify_utils.py` | 训练通知和暂停恢复 |
| 配置管理 | `config_utils.py` | 设备选择和配置加载 |

---

## 第四部分：修改后的数据流对比

```
修改前（ResNet-50）              修改后（UNI2-h）
───────────────────             ───────────────────
PNG图像 (224×224)               .pt缓存文件 (1536维)
       ↓                               ↓
ResNet-50 [~23M参数]            torch.load() [0参数]
       ↓                               ↓
2048维特征                      1536维特征
       ↓                               ↓
投影 2048→512                   投影 1536→512  ← ★ 唯一改动 ★
       ↓                               ↓
  [以下完全相同]                  [以下完全相同]
  ──────────────                  ──────────────
  GraphSAGE图卷积                GraphSAGE图卷积
       ↓                               ↓
  Exemplar融合                   Exemplar融合
       ↓                               ↓
  30条通路预测                   30条通路预测
```

> **总结**：整个改动只涉及"入口处"——特征来源从"实时推理"变成"磁盘加载"，维度从2048变成1536。投影层之后的所有计算完全一致，GraphSAGE、Exemplar融合、回归头这些核心模块原封不动。

---

## 第五部分：实施步骤（操作指南）

### 步骤1：验证UNI特征缓存完整性

检查路径 `uni2h_cache/{HYZ15040,JFX0729,LMZ12939}/{train,val}/` 下的 `.pt` 文件：

```powershell
# 统计各患者各分区的 .pt 文件数量
$patients = @("HYZ15040", "JFX0729", "LMZ12939")
$splits = @("train", "val")
foreach ($p in $patients) {
    foreach ($s in $splits) {
        $dir = "d:\AI空间转录病理研究\PFMval_new\uni2h_cache\$p\$s"
        $count = (Get-ChildItem $dir -Filter *.pt).Count
        Write-Host "$p/$s : $count 个 .pt 文件"
    }
}
```

> **注意**：如果某个患者缺少 .pt 缓存，需要先运行 `extract_uni_features_3st.py` 提取特征。

### 步骤2：创建数据集文件

- **文件**：`d:\AI空间转录病理研究\PFMval_new\egnv2_uni_dataset.py`
- **关键**：保留原始坐标（不做 n_pos 归一化），三层交集过滤
- 参照第三部分 3.2 节的代码框架

### 步骤3：创建模型文件

- **文件**：`d:\AI空间转录病理研究\PFMval_new\egnv2_uni_model.py`
- **关键**：`in_dim=1536`，去掉 `ResNetFeatureExtractor`，其余不变
- 参照第三部分 3.2 节的代码框架

### 步骤4：创建训练脚本

- **文件**：`d:\AI空间转录病理研究\PFMval_new\train_egnv2_uni.py`
- **关键**：
  1. 使用 `EGNv2UNIDataset` 替代 `EGNv2Dataset`
  2. 特征提取步骤替换为直接从 `.pt` 缓存批量加载
  3. Exemplar 库从 **1536维** UNI 特征构建
  4. 模型使用 `EGNv2UNIModel(in_dim=1536)`
  5. 支持单患者训练和跨患者泛化训练两种模式

### 步骤5：单患者训练验证

在 HYZ15040 单患者上训练：

```powershell
cd d:\AI空间转录病理研究\PFMval_new
conda activate base
python train_egnv2_uni.py --dataset_name HYZ15040_UNI --num_epochs 150
```

**预期**：Val PCC 应高于原始 EGN-v2 的 **0.4048**（参考 HisToGene-UNI 的 **0.5336**）

### 步骤6：跨患者泛化训练

```powershell
python train_egnv2_uni.py --dataset_name CrossPatient_JFX_LMZ_to_HYZ_UNI --num_epochs 150
```

**预期**：Test PCC 应高于原始 EGN-v2 的 **0.1950**（参考 HisToGene-UNI 的 **0.3946**）

---

## 第六部分：风险与注意事项

| 风险 | 等级 | 说明 | 应对方案 |
|------|------|------|---------|
| UNI特征缓存缺失 | **高** | 某些患者/分区可能未提取UNI特征 | 步骤1先验证完整性，缺失时运行 `extract_uni_features_3st.py` |
| 坐标系统混淆 | **中** | EGN-v2 用原始像素坐标（不同于 HisToGene 的 n_pos 归一化坐标） | `EGNv2UNIDataset` 中不做坐标归一化，直接从文件名解析原始 x、y |
| Exemplar KNN 距离阈值 | **中** | 1536维 vs 2048维的距离尺度不同，KNN检索的"近邻"定义可能变化 | 可能需要重新调参（如调整 k 值）；`compute_exemplar_agg_features` 的投影层会做维度适配 |
| 图稀疏性 | **低** | radius=300 的图构建依赖坐标分布，与特征无关 | 不同患者坐标范围不同，可调整 `--radius` 参数 |
| 过拟合风险 | **低** | UNI2-h 特征更强，可能导致训练集过拟合 | 可增大 `--dropout`（0.3→0.4-0.5）或添加正则化 |

> **特别注意 — 坐标系统**：这是最容易出错的地方。HisToGene 的 `dataset_uni.py` 使用 `_coord_to_index()` 将坐标归一化到 `[0, n_pos-1]` 的离散索引，用于位置编码 Embedding。而 EGN-v2 的 `dataset.py` 保留原始像素坐标（如 x=10017, y=16969），用于 `radius_neighbors_graph` 计算物理距离。**EGN-v2-UNI 必须使用原始坐标，不能做归一化**。

---

## 第七部分：预期效果与评估标准

| 指标 | 原始 EGN-v2 | EGN-v2+UNI 预期 | 参考（HisToGene-UNI） |
|------|------------|----------------|---------------------|
| 单患者 Val PCC | **0.4048** | **0.45-0.55** | 0.5336 |
| 跨患者 Test PCC | **0.1950** | **0.30-0.40** | 0.3946 |
| 训练参数量 | ~3.0M | ~0.8M | ~4.0M |
| 训练速度 | 慢（需ResNet推理） | **快**（直接加载缓存） | 快 |

> **注释**：
> - 预期值基于 HisToGene-UNI 的经验推测，实际结果可能因 GraphSAGE 的空间建模能力而有所不同
> - EGN-v2+UNI 的参数量更低（~0.8M vs ~3.0M），因为去掉了 ResNet-50 的投影层参数（`Linear(2048,512)` 约 105 万参数 → `Linear(1536,512)` 约 79 万参数，加上 LayerNorm 等，总可训练参数约 0.8M）
> - 如果 EGN-v2+UNI 的跨患者 PCC 达到 **0.30 以上**，即可视为成功验证了"UNI2-h 特征 + GraphSAGE 空间建模"的组合优势
> - 如果同时超过 HisToGene-UNI 的 **0.3946**，则说明 GraphSAGE 在 UNI 特征基础上还能进一步提升跨患者泛化能力

---

## 附录

### A. 术语表

| 术语 | 全称 | 简要说明 |
|------|------|---------|
| **UNI2-h** | UNI2-histo-pathology | MahmoodLab 发布的病理图像基础模型，在百万级病理切片上预训练，输出 **1536维** 特征向量 |
| **ResNet-50** | Residual Network-50 | 50层残差网络，在 ImageNet 自然图像上预训练，输出 **2048维** 特征向量 |
| **GraphSAGE** | Graph SAmple and aggreGatE | 图神经网络，通过采样邻居并聚合信息来更新节点表示，支持归纳学习 |
| **Exemplar** | 代表/范例 | 训练集中与当前样本最相似的 K 个参考样本，用于辅助预测 |
| **KNN** | K-Nearest Neighbors | K最近邻算法，找到特征空间中距离最近的 K 个样本 |
| **PCC** | Pearson Correlation Coefficient | 皮尔逊相关系数，衡量预测值与真实值的线性相关程度，1.0=完美预测 |
| **R²** | R-squared / Coefficient of Determination | 决定系数，衡量模型解释目标变量方差的比例，1.0=完美解释 |
| **ssGSEA** | single-sample Gene Set Enrichment Analysis | 单样本基因集富集分析，将基因表达压缩为通路活性评分 |

### B. 相关文件清单

**现有文件（不修改）**：

| 文件 | 路径 | 用途 |
|------|------|------|
| EGNv2模型 | `egnv2/model.py` | 原版模型定义（含 `EGNv2Model`, `ExemplarLibrary`） |
| EGNv2数据集 | `egnv2/dataset.py` | 原版数据集（加载PNG图像） |
| 图构建 | `egnv2/graph_builder.py` | `build_spatial_graph()` 空间图构建 |
| Exemplar构建 | `egnv2/exemplar_builder.py` | 特征提取、代表库构建、聚合特征计算 |
| 评估指标 | `egnv2/utils.py` | `compute_metrics()` PCC/R²/MAE |
| 跨患者训练 | `train_cross_patient_egnv2.py` | 原版跨患者训练脚本（参考结构） |
| HisToGene-UNI数据集 | `histogene/dataset_uni.py` | UNI数据集参考实现 |
| UNI特征提取 | `extract_uni_features_3st.py` | 提取UNI2-h特征的脚本 |
| 可视化 | `visualize_results.py` | 生成训练报告和可视化 |

**将要新建的文件**：

| 文件 | 路径 | 用途 |
|------|------|------|
| UNI数据集 | `egnv2_uni_dataset.py` | UNI特征版数据集（保留原始坐标） |
| UNI模型 | `egnv2_uni_model.py` | 调整 in_dim=1536 的模型 |
| UNI训练脚本 | `train_egnv2_uni.py` | 完整训练脚本（单患者+跨患者） |

### C. 参考资料

1. **EGN-v2 论文**：Yang et al., "Spatial Transcriptomics Analysis of Gene Expression Prediction Using Exemplar Guided Graph Neural Network", Pattern Recognition, 2024
2. **UNI2-h 技术说明**：MahmoodLab, UNI: Towards Generalizable Pretrained Encoders for Histopathology, 2024
3. **HisToGene-UNI 集成经验**：项目内文档 `HisToGene_UNI特征集成方案.md`
4. **EGN-v1 与 v2 架构区别**：EGN-v1（WACV 2023）不包含图神经网络，核心是 ViT + Exemplar Bridging；EGN-v2（PR 2024）首次引入基于物理坐标的 KNN 图构建和 GraphSAGE 消息传递
