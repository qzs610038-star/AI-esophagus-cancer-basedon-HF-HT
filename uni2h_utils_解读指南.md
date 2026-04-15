# uni2h_utils.py 深度解读指南

> 本文档面向初学者，详细解读病理图像特征提取与回归预测工具库的原理与实现。

---

## 1. 文件概述

**一句话说明**：`uni2h_utils.py` 是项目的**核心工具库**，提供 UNI2-h 预训练模型加载、病理图像特征提取与缓存、以及基于提取特征的下游回归任务训练与评估的全套功能。

**在项目中的角色**：这是连接上游数据预处理（split.py）和下游模型训练（train.py）的核心桥梁，负责将病理图像转换为可用于机器学习模型的高维特征向量。

---

## 2. 背景知识

### 2.1 Vision Transformer (ViT)

**什么是 ViT？**

Vision Transformer 是一种将 Transformer 架构应用于图像处理的深度学习模型。与传统卷积神经网络（CNN）不同，ViT 将图像切分成多个小块（patch），然后将这些 patch 转换为序列数据进行处理。

**核心思想**：
```
图像 (224×224)
    │
    ▼ 切分成 16×16 的 patch
┌─────────────────────────────┐
│  patch1 │ patch2 │ ...      │
│  patch5 │ patch6 │ ...      │
│  ...    │ ...    │ ...      │
└─────────────────────────────┘
    │
    ▼ 展平 + 线性投影
序列数据 [patch1_embed, patch2_embed, ..., patchN_embed]
    │
    ▼ 输入 Transformer
提取全局特征表示
```

### 2.2 UNI2-h 预训练模型

**UNI2-h 是什么？**

UNI2-h 是由 Mahmood Lab 开发的大规模病理图像预训练模型，基于 Vision Transformer 架构，在超过 1 亿张病理图像 patch 上进行自监督学习训练。

**为什么使用预训练模型？**

| 方式 | 问题 | 解决方案 |
|------|------|----------|
| 从头训练 | 需要大量标注数据，训练时间长，容易过拟合 | 使用预训练模型 |
| 预训练 + 微调 | 利用大规模无标注数据学习通用特征，只需少量标注数据即可适配下游任务 | UNI2-h 方案 |

**UNI2-h 的关键参数**：

| 参数 | 值 | 含义 |
|------|-----|------|
| 输入尺寸 | 224×224 | 模型接受的图像大小 |
| Patch 大小 | 14×14 | 每个 patch 为 14×14 像素 |
| 网络深度 | 24 | Transformer 层数 |
| 注意力头数 | 24 | 多头注意力机制的头数 |
| 特征维度 | 1536 | 输出特征向量的维度 |
| 注册令牌 | 8 | 可学习的全局信息令牌 |

### 2.3 特征提取与缓存

**为什么需要特征缓存？**

1. **计算成本高**：UNI2-h 是大型模型，每次前向传播需要大量计算
2. **特征固定**：在迁移学习中，预训练 backbone 通常被冻结（不更新参数），因此同一图像的特征是固定的
3. **加速训练**：将特征预先提取并保存到磁盘，训练时直接加载，大幅提升效率

**缓存策略**：
```
原始 PNG 图像
    │
    ▼ 第一次处理：提取特征
UNI2-h backbone
    │
    ▼ 保存到磁盘
.pt 文件（PyTorch tensor 格式）
    │
    ▼ 后续训练：直接加载
快速读取，无需重复计算
```

### 2.4 迁移学习 (Transfer Learning)

**什么是迁移学习？**

将在一个任务（源任务）上学习到的知识，应用到另一个相关任务（目标任务）上的技术。

**本项目中的迁移学习架构**：
```
┌─────────────────────────────────────────────────────────┐
│                    迁移学习架构                          │
├─────────────────────────────────────────────────────────┤
│  上游任务（已预训练）                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │ UNI2-h backbone                                 │   │
│  │ • 在 1亿+ 病理图像上预训练                       │   │
│  │ • 参数冻结（requires_grad=False）               │   │
│  │ • 输出 1536 维通用病理特征                       │   │
│  └──────────────────┬──────────────────────────────┘   │
│                     │ 特征提取                          │
│                     ▼                                   │
│  下游任务（需要训练）                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │ BackboneRegressor（回归头）                      │   │
│  │ • 输入：1536 维特征                              │   │
│  │ • 输出：8 个基因集分数                           │   │
│  │ • 参数可训练                                    │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         uni2h_utils.py 组件架构                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────┐                                                │
│  │   模型加载模块        │                                                │
│  │  ┌───────────────┐  │                                                │
│  │  │ensure_hf_login│  │  ← HuggingFace 登录认证                        │
│  │  └───────┬───────┘  │                                                │
│  │          ▼          │                                                │
│  │  ┌───────────────┐  │                                                │
│  │  │load_uni2h_    │  │  ← 加载 UNI2-h backbone + 预处理 transform     │
│  │  │backbone()     │  │                                                │
│  │  └───────────────┘  │                                                │
│  └──────────┬──────────┘                                                │
│             │ model, transform, feature_dim                              │
│             ▼                                                           │
│  ┌─────────────────────┐                                                │
│  │   特征提取与缓存模块   │                                                │
│  │  ┌───────────────┐  │                                                │
│  │  │extract_and_   │  │  ← 批量提取特征并保存为 .pt 文件               │
│  │  │cache_features │  │                                                │
│  │  └───────┬───────┘  │                                                │
│  │          ▼          │                                                │
│  │  ┌───────────────┐  │                                                │
│  │  │CachedFeature  │  │  ← 从缓存加载特征的数据集类                    │
│  │  │PatchDataset   │  │                                                │
│  │  └───────────────┘  │                                                │
│  └──────────┬──────────┘                                                │
│             │ feature, target                                           │
│             ▼                                                           │
│  ┌─────────────────────┐                                                │
│  │   模型架构模块        │                                                │
│  │  ┌───────────────┐  │                                                │
│  │  │Backbone       │  │  ← 回归头网络（MLP）                           │
│  │  │Regressor      │  │                                                │
│  │  └───────────────┘  │                                                │
│  └──────────┬──────────┘                                                │
│             │ predictions                                               │
│             ▼                                                           │
│  ┌─────────────────────┐                                                │
│  │   训练与评估模块      │                                                │
│  │  ┌───────────────┐  │                                                │
│  │  │train_one_epoch│  │  ← 单轮训练循环                                │
│  │  └───────┬───────┘  │                                                │
│  │          ▼          │                                                │
│  │  ┌───────────────┐  │                                                │
│  │  │evaluate()     │  │  ← 验证/测试评估                               │
│  │  └───────┬───────┘  │                                                │
│  │          ▼          │                                                │
│  │  ┌───────────────┐  │                                                │
│  │  │compute_metrics│  │  ← 计算 MSE/MAE/R²/PCC 指标                    │
│  │  └───────┬───────┘  │                                                │
│  │          ▼          │                                                │
│  │  ┌───────────────┐  │                                                │
│  │  │pearson_corrcoef│  │  ← 皮尔逊相关系数计算                          │
│  │  └───────────────┘  │                                                │
│  └─────────────────────┘                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 逐函数/类详解

### 4.1 `ensure_hf_login()`

**函数签名**：
```python
def ensure_hf_login(token: Optional[str] = None) -> None
```

**功能说明**：
确保已登录 HuggingFace Hub，用于下载 UNI2-h 等需要授权的预训练模型。

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `token` | `Optional[str]` | `None` | HuggingFace API token，可从 https://huggingface.co/settings/tokens 获取 |

**算法逻辑**：
1. 优先使用传入的 `token` 参数
2. 若未传入，尝试从环境变量 `HUGGINGFACE_HUB_TOKEN` 或 `HF_TOKEN` 读取
3. 若存在 token，调用 `login()` 完成认证

**关键代码行**：

| 行号 | 代码 | 作用 |
|------|------|------|
| 26 | `token = token or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")` | 优先级：参数 > 环境变量1 > 环境变量2 |
| 28 | `login(token=token)` | HuggingFace Hub 登录 |

---

### 4.2 `load_uni2h_backbone()`

**函数签名**：
```python
def load_uni2h_backbone(
    token: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, callable, int]
```

**功能说明**：
加载预训练的 UNI2-h backbone 模型及其官方预处理 transform，并将模型设置为评估模式（eval mode）且冻结所有参数。

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `token` | `Optional[str]` | `None` | HuggingFace API token |
| `device` | `Optional[torch.device]` | `None` | 计算设备（CPU/GPU），若为 None 则不移动模型 |

**返回值**：

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `model` | `torch.nn.Module` | 加载好的 UNI2-h backbone |
| `transform` | `callable` | 图像预处理函数（归一化、resize 等）|
| `feature_dim` | `int` | 输出特征维度（1536）|

**算法逻辑**：
1. 调用 `ensure_hf_login()` 确保已登录
2. 定义 UNI2-h 官方结构参数（`timm_kwargs`）
3. 使用 `timm.create_model()` 从 HuggingFace Hub 加载模型
4. 设置模型为评估模式（`model.eval()`）
5. 冻结所有参数（`requires_grad = False`）
6. 若指定设备，将模型移动到对应设备
7. 创建官方预处理 transform

**关键代码行**：

| 行号 | 代码 | 作用 |
|------|------|------|
| 43-57 | `timm_kwargs = {...}` | UNI2-h 官方配置参数 |
| 60 | `model = timm.create_model(f"hf-hub:{DEFAULT_MODEL_ID}", ...)` | 从 HuggingFace 加载模型 |
| 61 | `model.eval()` | 设置评估模式（禁用 dropout、batch norm 更新等）|
| 62-63 | `for p in model.parameters(): p.requires_grad = False` | 冻结所有参数，不参与梯度更新 |
| 69 | `transform = create_transform(...)` | 创建官方预处理管道 |

**UNI2-h 配置参数详解**：

| 参数 | 值 | 含义 |
|------|-----|------|
| `img_size` | 224 | 输入图像尺寸 |
| `patch_size` | 14 | patch 大小（224/14=16，即 16×16=256 个 patch）|
| `depth` | 24 | Transformer encoder 层数 |
| `num_heads` | 24 | 多头注意力头数 |
| `embed_dim` | 1536 | 嵌入维度（特征维度）|
| `reg_tokens` | 8 | 注册令牌数量（用于聚合全局信息）|
| `dynamic_img_size` | True | 支持动态输入尺寸 |
| `num_classes` | 0 | 不加载分类头（只取 backbone 特征）|

---

### 4.3 `pearson_corrcoef()`

**函数签名**：
```python
def pearson_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float
```

**功能说明**：
计算皮尔逊相关系数（Pearson Correlation Coefficient, PCC），衡量两个变量之间的线性相关程度。

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `y_true` | `np.ndarray` | 真实值数组 |
| `y_pred` | `np.ndarray` | 预测值数组 |

**返回值**：
- 皮尔逊相关系数（范围：-1 到 1）
- 1 表示完全正相关，-1 表示完全负相关，0 表示无线性相关
- 若输入为空或标准差为 0，返回 `nan`

**算法逻辑**：
1. 将输入展平为一维数组
2. 检查输入是否为空
3. 检查标准差是否为 0（避免除零错误）
4. 使用 `np.corrcoef()` 计算相关系数矩阵，取 [0,1] 元素

**关键代码行**：

| 行号 | 代码 | 作用 |
|------|------|------|
| 79-80 | `yt = np.asarray(y_true).reshape(-1)` | 展平为一维数组 |
| 82-85 | `if yt.size == 0 or ...` | 边界条件检查 |
| 87 | `return float(np.corrcoef(yt, yp)[0, 1])` | 计算皮尔逊相关系数 |

---

### 4.4 `compute_metrics()`

**函数签名**：
```python
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]
```

**功能说明**：
计算回归任务的多个评估指标：MSE、MAE、R² 和 PCC，对每个目标单独计算后取平均。

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `y_true` | `np.ndarray` | 真实值，形状 `[N, num_targets]` |
| `y_pred` | `np.ndarray` | 预测值，形状 `[N, num_targets]` |

**返回值**：
```python
{
    "mse": float,   # 平均均方误差
    "mae": float,   # 平均绝对误差
    "r2": float,    # 平均 R² 分数
    "pcc": float,   # 平均皮尔逊相关系数
}
```

**指标说明**：

| 指标 | 全称 | 含义 | 最优值 |
|------|------|------|--------|
| MSE | Mean Squared Error | 均方误差 | 0 |
| MAE | Mean Absolute Error | 平均绝对误差 | 0 |
| R² | Coefficient of Determination | 决定系数 | 1 |
| PCC | Pearson Correlation Coefficient | 皮尔逊相关系数 | 1 或 -1 |

**算法逻辑**：
1. 验证输入为 2D 数组且形状一致
2. 对每个目标列分别计算指标
3. 对所有目标的指标取平均（忽略 nan）

**关键代码行**：

| 行号 | 代码 | 作用 |
|------|------|------|
| 96-105 | `if y_true.ndim != 2 ...` | 输入形状验证 |
| 114-126 | `for j in range(num_targets):` | 逐目标计算指标 |
| 121-124 | `if np.std(yt) == 0:` | 处理常数标签情况（R² 无意义）|
| 128-133 | `metrics = {...}` | 返回平均指标字典 |

---

### 4.5 `extract_and_cache_features()`

**函数签名**：
```python
def extract_and_cache_features(
    backbone: torch.nn.Module,
    transform,
    patches_dir: str,
    cache_dir: str,
    device: torch.device,
    rebuild: bool = False,
) -> int
```

**功能说明**：
批量提取病理图像 patch 的 UNI2-h 特征，并将每个特征保存为独立的 `.pt` 文件（PyTorch tensor 格式）。

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `backbone` | `torch.nn.Module` | - | 已加载的 UNI2-h backbone |
| `transform` | `callable` | - | 图像预处理 transform |
| `patches_dir` | `str` | - | 输入 patch 图像目录 |
| `cache_dir` | `str` | - | 特征缓存输出目录 |
| `device` | `torch.device` | - | 计算设备 |
| `rebuild` | `bool` | `False` | 是否强制重新生成缓存 |

**返回值**：
- 实际写入的缓存文件数量（int）

**算法逻辑**：
1. 确保缓存目录存在
2. 遍历 patches_dir 中的所有 PNG 文件
3. 对每个图像：
   - 检查缓存是否已存在（且 rebuild=False 则跳过）
   - 打开图像并转换为 RGB
   - 应用预处理 transform
   - 使用 backbone 提取特征
   - 保存特征到缓存目录

**关键代码行**：

| 行号 | 代码 | 作用 |
|------|------|------|
| 151 | `cache_dir.mkdir(parents=True, exist_ok=True)` | 创建缓存目录 |
| 153 | `image_files = sorted([p for p in patches_dir.iterdir() if p.suffix.lower() == ".png"])` | 获取所有 PNG 文件 |
| 157 | `with torch.inference_mode():` | 禁用梯度计算，节省内存 |
| 160-161 | `if cache_path.exists() and not rebuild:` | 缓存复用逻辑 |
| 164 | `x = transform(image).unsqueeze(0).to(device, non_blocking=True)` | 预处理并添加 batch 维度 |
| 165 | `feat = backbone(x).squeeze(0).detach().cpu().float()` | 提取特征并移至 CPU |
| 166 | `torch.save(feat, cache_path)` | 保存为 .pt 文件 |

---

### 4.6 `CachedFeaturePatchDataset` 类

**类签名**：
```python
class CachedFeaturePatchDataset(Dataset):
    def __init__(
        self,
        patches_dir: str,
        labels_csv: str,
        feature_cache_dir: str,
        target_start_col: int = DEFAULT_TARGET_START_COL,  # 1
        num_targets: int = DEFAULT_NUM_TARGETS,            # 8
    )
```

**功能说明**：
PyTorch Dataset 类，用于从缓存加载 UNI2-h 特征并与标签配对，供 DataLoader 使用。

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `patches_dir` | `str` | - | patch 图像目录（用于匹配文件名）|
| `labels_csv` | `str` | - | 标签 CSV 文件路径 |
| `feature_cache_dir` | `str` | - | 特征缓存目录 |
| `target_start_col` | `int` | 1 | 标签从第几列开始（0-based）|
| `num_targets` | `int` | 8 | 要预测的基因集分数数量 |

**核心属性**：

| 属性 | 类型 | 说明 |
|------|------|------|
| `patch_to_idx` | `Dict[str, int]` | patch 文件名（不含扩展名）到 CSV 行索引的映射 |
| `target_cols` | `List[str]` | 目标列名列表 |
| `targets` | `np.ndarray` | 标签数组，形状 `[N, num_targets]` |
| `patch_files` | `List[Path]` | 有效的 patch 文件路径列表 |

**算法逻辑**（`__init__`）：
1. 读取标签 CSV 文件
2. 建立 patch 文件名到 CSV 索引的映射
3. 提取目标列名和标签数据
4. 遍历 patches_dir，筛选存在对应标签的 PNG 文件

**算法逻辑**（`__getitem__`）：
1. 根据索引获取 patch 文件路径
2. 查找对应的 CSV 行索引，获取标签
3. 加载对应的缓存特征文件（.pt）
4. 返回 (feature, target) 元组

**关键代码行**：

| 行号 | 代码 | 作用 |
|------|------|------|
| 186 | `patch_keys = self.labels_df.iloc[:, 0].astype(str).map(lambda x: Path(x).stem).tolist()` | 提取 CSV 第一列的文件名（无扩展名）|
| 187 | `self.patch_to_idx = {k: i for i, k in enumerate(patch_keys)}` | 建立文件名到索引的映射 |
| 189 | `self.target_cols = list(self.labels_df.columns[target_start_col:target_start_col + num_targets])` | 提取目标列名 |
| 202-203 | `if p.stem in self.patch_to_idx:` | 只保留有标签的 patch |
| 216-219 | `feat_path = self.feature_cache_dir / f"{img_path.stem}.pt"` | 构造缓存文件路径并加载 |
| 220-222 | `if isinstance(feature, dict) and "feature" in feature:` | 处理可能的字典格式缓存 |

---

### 4.7 `BackboneRegressor` 类

**类签名**：
```python
class BackboneRegressor(nn.Module):
    def __init__(
        self,
        feature_dim: int,   # 输入特征维度（1536）
        hidden_dim: int,    # 隐藏层维度
        output_dim: int,    # 输出维度（8）
        dropout: float,     # Dropout 比率
    )
```

**功能说明**：
基于 MLP（多层感知机）的回归头，接收 UNI2-h 提取的 1536 维特征，输出多个基因集分数预测值。

**网络结构**：
```
输入: [batch_size, feature_dim]
    │
    ▼
┌─────────────────┐
│   LayerNorm     │  ← 层归一化，稳定训练
│  (feature_dim,) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Linear       │  ← 全连接层
│  feature_dim →  │
│   hidden_dim    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      GELU       │  ← 激活函数
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Dropout      │  ← 正则化，防止过拟合
│    (dropout)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Linear       │  ← 输出层
│  hidden_dim →   │
│   output_dim    │
└────────┬────────┘
         │
输出: [batch_size, output_dim]
```

**关键代码行**：

| 行号 | 代码 | 作用 |
|------|------|------|
| 238-244 | `self.net = nn.Sequential(...)` | 定义网络结构 |
| 239 | `nn.LayerNorm(feature_dim)` | 层归一化 |
| 240 | `nn.Linear(feature_dim, hidden_dim)` | 第一个全连接层 |
| 241 | `nn.GELU()` | GELU 激活函数 |
| 242 | `nn.Dropout(dropout)` | Dropout 正则化 |
| 243 | `nn.Linear(hidden_dim, output_dim)` | 输出层 |

---

### 4.8 `train_one_epoch()`

**函数签名**：
```python
def train_one_epoch(
    model,           # BackboneRegressor 实例
    dataloader,      # DataLoader
    criterion,       # 损失函数（如 nn.MSELoss()）
    optimizer,       # 优化器（如 Adam）
    device,          # 计算设备
) -> Dict[str, float]
```

**功能说明**：
执行一个 epoch 的训练循环，包括前向传播、损失计算、反向传播、参数更新，并返回训练指标。

**算法逻辑**：
1. 设置模型为训练模式（`model.train()`）
2. 遍历 DataLoader 中的每个 batch：
   - 将数据和标签移动到设备
   - 前向传播得到预测
   - 计算损失
   - 梯度清零 → 反向传播 → 参数更新
   - 累加损失和收集预测结果
3. 计算并返回训练指标

**关键代码行**：

| 行号 | 代码 | 作用 |
|------|------|------|
| 252 | `model.train()` | 设置训练模式（启用 dropout、batch norm 统计更新）|
| 257-266 | `for features, targets in dataloader:` | 训练循环 |
| 264 | `optimizer.zero_grad(set_to_none=True)` | 梯度清零（set_to_none=True 更高效）|
| 265 | `loss.backward()` | 反向传播计算梯度 |
| 266 | `optimizer.step()` | 更新模型参数 |
| 272-276 | 计算并返回指标 | 合并所有 batch 的预测结果 |

---

### 4.9 `evaluate()`

**函数签名**：
```python
def evaluate(
    model,
    dataloader,
    criterion,
    device,
) -> Dict[str, float]
```

**功能说明**：
在验证集或测试集上评估模型性能，不更新参数，只计算损失和指标。

**算法逻辑**：
1. 设置模型为评估模式（`model.eval()`）
2. 使用 `torch.no_grad()` 禁用梯度计算
3. 遍历 DataLoader，收集预测结果
4. 计算并返回评估指标

**与 `train_one_epoch()` 的区别**：

| 方面 | `train_one_epoch()` | `evaluate()` |
|------|---------------------|--------------|
| 模式 | `model.train()` | `model.eval()` |
| 梯度 | 启用 | 禁用（`torch.no_grad()`）|
| 反向传播 | 有 | 无 |
| 参数更新 | 有 | 无 |
| dropout | 启用 | 禁用 |

**关键代码行**：

| 行号 | 代码 | 作用 |
|------|------|------|
| 281 | `model.eval()` | 设置评估模式 |
| 286 | `with torch.no_grad():` | 禁用梯度计算，节省内存 |

---

## 5. 模型架构详解

### 5.1 UNI2-h Backbone 架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         UNI2-h Backbone                                 │
│                    (Vision Transformer 架构)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  输入图像: [batch_size, 3, 224, 224]                                    │
│      │                                                                  │
│      ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Patch Embedding                             │   │
│  │  • 将 224×224 图像切分为 16×16=256 个 14×14 的 patch            │   │
│  │  • 每个 patch 投影到 1536 维嵌入空间                            │   │
│  │  • 添加位置编码                                                  │   │
│  │  • 添加 8 个可学习的注册令牌（reg_tokens）                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│      │                                                                  │
│      ▼ [batch_size, 256+8, 1536] = [batch_size, 264, 1536]             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   Transformer Encoder × 24                        │   │
│  │                                                                  │   │
│  │  每层包含:                                                        │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │   │
│  │  │ Multi-Head  │ →  │  LayerNorm  │ →  │    MLP      │          │   │
│  │  │  Attention  │    │             │    │ (SwiGLU)    │          │   │
│  │  │  (24 heads) │    │             │    │             │          │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘          │   │
│  │                                                                  │   │
│  │  重复 24 次                                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│      │                                                                  │
│      ▼ [batch_size, 264, 1536]                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      特征聚合                                      │   │
│  │  • 使用注册令牌聚合全局信息                                       │   │
│  │  • 输出 1536 维特征向量                                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│      │                                                                  │
│      ▼                                                                  │
│  输出特征: [batch_size, 1536]                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 BackboneRegressor 架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      BackboneRegressor                                  │
│                      (回归头 / MLP)                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  输入特征: [batch_size, 1536]  ← UNI2-h backbone 输出                   │
│      │                                                                  │
│      ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  nn.LayerNorm(1536)                                             │   │
│  │  • 对输入特征进行层归一化                                        │   │
│  │  • 稳定训练，加速收敛                                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│      │ [batch_size, 1536]                                              │
│      ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  nn.Linear(1536, hidden_dim)                                    │   │
│  │  • 全连接层，将特征映射到隐藏空间                                │   │
│  │  • hidden_dim 通常为 512 或 256                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│      │ [batch_size, hidden_dim]                                        │
│      ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  nn.GELU()                                                      │   │
│  │  • GELU 激活函数（高斯误差线性单元）                             │   │
│  │  • 平滑的非线性激活，性能优于 ReLU                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│      │ [batch_size, hidden_dim]                                        │
│      ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  nn.Dropout(dropout)                                            │   │
│  │  • 随机丢弃部分神经元输出                                        │   │
│  │  • 防止过拟合，dropout 通常为 0.1~0.5                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│      │ [batch_size, hidden_dim]                                        │
│      ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  nn.Linear(hidden_dim, output_dim)                              │   │
│  │  • 输出层，将隐藏特征映射到预测目标                              │   │
│  │  • output_dim = 8（8 个基因集分数）                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│      │                                                                  │
│      ▼                                                                  │
│  输出预测: [batch_size, 8]  ← 8 个基因集分数预测值                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 关键参数说明

### 6.1 全局常量

| 参数名 | 行号 | 值 | 说明 |
|--------|------|-----|------|
| `DEFAULT_MODEL_ID` | 19 | `"MahmoodLab/UNI2-h"` | HuggingFace 模型 ID |
| `DEFAULT_FEATURE_DIM` | 20 | `1536` | UNI2-h 输出特征维度 |
| `DEFAULT_TARGET_START_COL` | 21 | `1` | 标签从 CSV 第 2 列开始（0-based 索引为 1）|
| `DEFAULT_NUM_TARGETS` | 22 | `8` | 默认预测 8 个基因集分数 |

### 6.2 UNI2-h 模型参数

| 参数名 | 行号 | 值 | 说明 |
|--------|------|-----|------|
| `img_size` | 44 | `224` | 输入图像尺寸 |
| `patch_size` | 45 | `14` | Patch 大小 |
| `depth` | 46 | `24` | Transformer 层数 |
| `num_heads` | 47 | `24` | 注意力头数 |
| `embed_dim` | 49 | `1536` | 嵌入维度 |
| `mlp_ratio` | 50 | `5.33334` | MLP 隐藏层比例 |
| `reg_tokens` | 55 | `8` | 注册令牌数量 |
| `dynamic_img_size` | 56 | `True` | 支持动态图像尺寸 |

### 6.3 BackboneRegressor 参数

| 参数名 | 类型 | 典型值 | 说明 |
|--------|------|--------|------|
| `feature_dim` | int | `1536` | 输入特征维度（与 UNI2-h 输出一致）|
| `hidden_dim` | int | `512` 或 `256` | 隐藏层维度，越大模型容量越大 |
| `output_dim` | int | `8` | 输出维度（基因集分数数量）|
| `dropout` | float | `0.1` ~ `0.3` | Dropout 比率，防止过拟合 |

---

## 7. 数据流示意

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           完整数据流转                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  阶段 1: 原始数据                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  patch_x0_y0.png                                                  │   │
│  │  patch_x0_y224.png                                                │   │
│  │  patch_x224_y0.png                                                │   │
│  │  ...                                                              │   │
│  └────────────────────────┬────────────────────────────────────────┘   │
│                           │                                            │
│                           ▼  extract_and_cache_features()              │
│                                                                         │
│  阶段 2: 特征提取与缓存                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  UNI2-h backbone                                                  │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐                       │   │
│  │  │224×224  │ →  │ ViT     │ →  │1536-dim │                       │   │
│  │  │ 图像    │    │ 编码器   │    │ 特征    │                       │   │
│  │  └─────────┘    └─────────┘    └─────────┘                       │   │
│  └────────────────────────┬────────────────────────────────────────┘   │
│                           │                                            │
│                           ▼  torch.save()                              │
│                                                                         │
│  阶段 3: 特征缓存                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  cache_dir/                                                       │   │
│  │  ├── patch_x0_y0.pt        ← [1536] 维 tensor                     │   │
│  │  ├── patch_x0_y224.pt                                             │   │
│  │  ├── patch_x224_y0.pt                                             │   │
│  │  └── ...                                                          │   │
│  └────────────────────────┬────────────────────────────────────────┘   │
│                           │                                            │
│                           ▼  CachedFeaturePatchDataset                 │
│                                                                         │
│  阶段 4: 数据集包装                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Dataset                                                          │   │
│  │  ┌─────────────────┐    ┌─────────────────┐                       │   │
│  │  │ 特征 [1536]      │    │ 标签 [8]         │                       │   │
│  │  │ (from .pt)      │ ←→ │ (from CSV)      │                       │   │
│  │  └─────────────────┘    └─────────────────┘                       │   │
│  └────────────────────────┬────────────────────────────────────────┘   │
│                           │                                            │
│                           ▼  DataLoader                                │
│                                                                         │
│  阶段 5: 模型训练                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  BackboneRegressor                                                │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐        │   │
│  │  │1536-dim │ →  │ Layer   │ →  │ Hidden  │ →  │ 8-dim   │        │   │
│  │  │ 特征    │    │ Norm    │    │ Layer   │    │ 输出    │        │   │
│  │  └─────────┘    └─────────┘    └─────────┘    └─────────┘        │   │
│  │                                                                  │   │
│  │  损失计算: MSELoss(predictions, targets)                         │   │
│  │  优化器: Adam / SGD                                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                           │                                            │
│                           ▼                                            │
│  阶段 6: 评估指标                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • MSE: 均方误差                                                   │   │
│  │  • MAE: 平均绝对误差                                               │   │
│  │  • R²: 决定系数                                                   │   │
│  │  • PCC: 皮尔逊相关系数                                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. 初学者注意事项

### 8.1 常见错误

| 错误 | 原因 | 解决方法 |
|------|------|----------|
| `FileNotFoundError: Missing feature cache` | 特征缓存文件不存在 | 先运行 `extract_and_cache_features()` 生成缓存 |
| `RuntimeError: No PNG patches matched labels` | patch 文件名与 CSV 不匹配 | 检查 CSV 第一列的文件名格式是否与 patch 文件名一致 |
| `ValueError: Expected 8 target columns` | 标签列数不足 | 检查 CSV 文件，确保有足够的标签列 |
| `HuggingFace 认证失败` | token 无效或未设置 | 设置正确的 `HF_TOKEN` 环境变量或传入 token 参数 |
| CUDA out of memory | GPU 显存不足 | 减小 batch size，或使用 CPU（`device='cpu'`）|

### 8.2 调试建议

1. **检查特征缓存是否成功生成**：
   ```python
   cache_dir = Path("path/to/cache")
   print(f"缓存文件数量: {len(list(cache_dir.glob('*.pt')))}")
   ```

2. **验证 Dataset 是否能正常加载**：
   ```python
   dataset = CachedFeaturePatchDataset(...)
   print(f"数据集大小: {len(dataset)}")
   feature, target = dataset[0]
   print(f"特征形状: {feature.shape}, 标签形状: {target.shape}")
   ```

3. **检查模型输出维度**：
   ```python
   model = BackboneRegressor(1536, 512, 8, 0.1)
   dummy_input = torch.randn(4, 1536)  # batch_size=4
   output = model(dummy_input)
   print(f"输出形状: {output.shape}")  # 应为 [4, 8]
   ```

4. **监控训练过程**：
   ```python
   # 在 train_one_epoch 中添加打印
   if batch_idx % 10 == 0:
       print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
   ```

### 8.3 重要提醒

- ⚠️ **特征缓存是一次性的**：一旦生成，除非修改 backbone 或预处理，否则无需重新生成
- ⚠️ **UNI2-h backbone 是冻结的**：训练时只更新 BackboneRegressor 的参数
- ⚠️ **CSV 格式要求**：第一列必须是 patch 文件名（可带或不带扩展名），后续列为标签
- ⚠️ **设备一致性**：确保模型和数据在同一设备上（CPU 或 GPU）

---

## 9. 扩展思考

### 9.1 可能的改进方向

1. **特征归一化**
   - 当前：注释掉的 L2 归一化（第 223 行）
   - 扩展：尝试不同的归一化策略（batch norm、instance norm）

2. **回归头架构改进**
   - 当前：简单的 2 层 MLP
   - 扩展：添加更多隐藏层、残差连接、注意力机制

3. **多任务学习优化**
   - 当前：所有目标共享同一回归头
   - 扩展：为每个基因集分数使用独立的子网络

4. **特征融合策略**
   - 当前：使用 UNI2-h 的单一输出特征
   - 扩展：融合多层特征、使用注意力加权

5. **数据增强**
   - 当前：无数据增强（特征已缓存）
   - 扩展：在特征空间添加噪声、使用 Mixup/CutMix

### 9.2 与其他组件的集成

```
WSI 全切片图像
    │
    ▼ 切分工具
patch 图像集合
    │
    ▼ split.py
空间无重叠的训练/验证集
    │
    ▼ uni2h_utils.py（本文档）
    │   ├─ extract_and_cache_features() → 特征缓存
    │   └─ CachedFeaturePatchDataset → 数据集
    │
    ▼ train.py（模型训练）
训练好的回归模型
    │
    ▼ infer.py（推理预测）
基因集分数预测结果
```

### 9.3 深入学习的建议

1. **理解 Vision Transformer**
   - 阅读论文《An Image is Worth 16x16 Words》
   - 可视化 attention map，理解模型关注哪些区域

2. **学习迁移学习**
   - 尝试不同的冻结/微调策略
   - 比较不同预训练模型的效果

3. **掌握 PyTorch 训练流程**
   - 理解 `train()` / `eval()` 模式的区别
   - 学习学习率调度、早停等训练技巧

---

## 10. 总结

`uni2h_utils.py` 是 PFMval 项目的核心技术库，其设计体现了现代深度学习项目的最佳实践：

1. **模块化设计**：清晰分离特征提取、数据加载、模型架构、训练评估
2. **预训练 + 微调**：利用 UNI2-h 强大的病理图像理解能力
3. **特征缓存策略**：避免重复计算，大幅提升训练效率
4. **标准化评估**：提供 MSE、MAE、R²、PCC 等多维度指标

对于初学者，理解这个文件有助于掌握：
- 预训练模型的加载与使用
- 迁移学习的实现方式
- PyTorch Dataset 和 DataLoader 的自定义
- 深度学习模型的训练与评估流程
- 特征缓存的性能优化技巧

---

*文档生成时间：2026年4月11日*
