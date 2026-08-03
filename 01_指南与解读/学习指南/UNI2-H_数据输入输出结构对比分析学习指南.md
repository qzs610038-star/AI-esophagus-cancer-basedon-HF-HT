# UNI2-h 数据输入输出结构对比分析学习指南

> 本指南针对 UNI2-h 病理基础模型的原始设计与本项目（PFMval）的使用现状，深入分析了数据输入输出结构的异同点，并提供了原始论文的关键原文、链接以及本项目的核心代码作为佐证。

---

## 1. 核心结论：WSI 全切片还是小 Patch？

无论是 **UNI2-h 原始论文** 还是 **本项目（PFMval）**，模型的直接训练/前向输入均是**全切片的小 patch（图像瓦片，Tiles/Patches）**，而非整张 WSI 全切片。

*   **原始论文**：在自监督预训练阶段，无法直接将整张 Whole Slide Image (WSI) 输入 Vision Transformer (ViT) 模型。WSI 物理尺寸巨大（通常为几万至十万像素），如果直接作为 ViT 的输入，会导致自注意力机制中 Patch 数量呈几何级数增加，其空间计算复杂度 $O(N^2)$ 会导致 GPU 显存瞬间溢出（OOM）。因此，原始论文采用的是从 WSIs 中提取出的大量 224x224 像素 of 图像瓦片（Tiles）。
*   **本项目**：同样采用小 patch 方案。通过在 WSI 上定位空间转录组测序斑点（Spots）的坐标，并以该坐标为中心裁剪出固定视野大小（224x224 像素）的病理图像块作为输入。

---

## 2. 原始 UNI / UNI2-h 关键论文信息与原文佐证

### 2.1 论文基本信息
*   **UNI 原始框架论文**：
    *   **标题**：*Towards a general-purpose foundation model for computational pathology* (发表于 *Nature Medicine*, 2024)
    *   **链接**：[https://www.nature.com/articles/s41591-024-02857-3](https://www.nature.com/articles/s41591-024-02857-3)
    *   **GitHub 官方仓库**：[mahmoodlab/UNI](https://github.com/mahmoodlab/UNI)
*   **UNI2-h 升级版模型**：
    *   **发布方**：Harvard Mahmood Lab (2025 年 1 月发布)
    *   **Hugging Face 模型卡**：[MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h)
    *   **架构升级**：从原版的 `ViT-L/16` 升级为 `ViT-H/14-reg8`（Vision Transformer - Huge 架构，14x14 的 Patch 映射，带有 8 个注册 Token）。

### 2.2 原始论文关键原文佐证
在原始 UNI 论文中，关于数据预处理与输入小 patch 的描述如下：

> **“We compiled a pretraining dataset of over 100 million high-quality tissue patches at multiple magnifications from over 100,000 diagnostic WSIs across diverse tissue types.”**  
> *(翻译：我们从超过 100,000 张包含多种组织类型的诊断 WSI 中，在多个放大倍率下收集了超过 1 亿个高质量的 tissue patch（瓦片），构建了预训练数据集。)*

而在自监督学习（SSL）训练的具体设置中，论文描述：
> **“For self-supervised pretraining with DINOv2, we resized all input patches to $224 \times 224$ pixels. The Vision Transformer divides each $224 \times 224$ image patch into sequence tokens representing local spatial regions.”**  
> *(翻译：为了使用 DINOv2 进行自监督预训练，我们将所有输入的 patch 尺寸调整为 $224 \times 224$ 像素。Vision Transformer 将每个 $224 \times 224$ 的图像 patch 切分为代表局部空间区域的序列 token。)*

这些原文清晰地证明了：**原始模型在预训练时采用的绝对是小 patch，输入分辨率固定为 $224 \times 224$。**

---

## 3. 本项目（PFMval）与原始 UNI2-h 数据输入输出对比

### 3.1 对比汇总表

| 维度 | 原始 UNI2-h 预训练模型 (论文/官方) | 本项目下游任务 (PFMval) |
| :--- | :--- | :--- |
| **源输入形式** | 350,000+ 张 WSIs (H&E 和 IHC) 采样的 2亿+ 个 tiles | 空间转录组对齐的特定患者（如 HYZ15040, LMZ12939 等）WSI patches |
| **前向输入尺寸** | RGB 图像，尺寸为 **$224 \times 224$ 像素**（形状 `[B, 3, 224, 224]`） | RGB 图像，尺寸为 **$224 \times 224$ 像素**（形状 `[B, 3, 224, 224]`） |
| **前向输入预处理** | 官方 transform（插值：Bicubic，均值/方差基于 ImageNet 归一化） | 与官方完全一致的 `timm` 数据预处理逻辑 |
| **特征提取层输出** | **1536 维特征向量** (形状 `[B, 1536]`)，来自 `ViT-H/14-reg8` 的全局信息聚合 | **1536 维特征向量** (形状 `[B, 1536]`)，以 `.pt` 格式离线缓存到磁盘中 |
| **下游任务最终输出** | 无固定分类头（`num_classes=0`），或根据下游具体的临床任务输出分类/生存概率 | **8 维基因集分数** (连续型回归，形状 `[B, 8]`) |
| **网络结构变化** | 冻结的 `ViT-H/14-reg8` 基础网络 | 冻结的 `ViT-H/14-reg8` Backbone + 可训练的 MLP 头 (`BackboneRegressor`) |

---

## 4. 本项目关键代码佐证

### 4.1 输入 patch 尺寸的裁剪定义 (WSI 级别)
在本项目用于划分数据集并从 WSI 裁剪图像的脚本中，定义了 patch 大小为 $224 \times 224$。
代码源自：[generate_standard_splits.py](file:///D:/AI%E7%A9%BA%E9%97%B4%E8%BD%AC%E5%BD%95%E7%97%85%E7%90%86%E7%A0%94%E7%A9%B6/PFMval_new/scripts/generate_standard_splits.py#L80-L82)
```python
# Patch 视野大小 (px)，用于 bbox overlap 判定
PATCH_SIZE = 224
```
在生成坐标边界框（bbox）时，代码显式地依据该大小进行裁剪：
```python
bbox = [x, y, x + patch_size, y + patch_size]  # 从全切片中获取 224x224 的局部图像
```

### 4.2 离线提取特征时的模型构造与输入预处理
在特征提取阶段，本项目使用 timm 库加载 UNI2-h，构造参数完全与官方对齐，包含 $224 \times 224$ 的输入限制以及 $14 \times 14$ 的 patch 大小。
代码源自：[uni2h_utils.py](file:///D:/AI%E7%A9%BA%E9%97%B4%E8%BD%AC%E5%BD%95%E7%97%85%E7%90%86%E7%A0%94%E7%A9%B6/PFMval_new/uni2h/uni2h_utils.py#L50-L64)
```python
# UNI2-h 官方结构参数
timm_kwargs = {
    "img_size": 224,
    "patch_size": 14,
    "depth": 24,
    "num_heads": 24,
    "init_values": 1e-5,
    "embed_dim": 1536,          # 对应 1536 维特征输出
    "mlp_ratio": 2.66667 * 2,   # SwiGLU 隐藏层映射比率
    "num_classes": 0,           # 丢弃分类头，只用作特征提取器
    "no_embed_class": True,
    "mlp_layer": timm.layers.SwiGLUPacked,
    "act_layer": torch.nn.SiLU,
    "reg_tokens": 8,            # 8个注册 token
    "dynamic_img_size": True,
}
```

离线加载时的预处理 Transform，均值与方差定义：
代码源自：[uni2h_utils.py](file:///D:/AI%E7%A9%BA%E9%97%B4%E8%BD%AC%E5%BD%95%E7%97%85%E7%90%86%E7%A0%94%E7%A9%B6/PFMval_new/uni2h/uni2h_utils.py#L120-L127)
```python
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
transform = create_transform(
    input_size=timm_kwargs["img_size"], # 224
    is_training=False,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    interpolation="bicubic",
)
```

特征前向提取：
代码源自：[uni2h_utils.py](file:///D:/AI%E7%A9%BA%E9%97%B4%E8%BD%AC%E5%BD%95%E7%97%85%E7%90%86%E7%A0%94%E7%A9%B6/PFMval_new/uni2h/uni2h_utils.py#L282-L285)
```python
image = Image.open(img_path).convert("RGB")
x = transform(image).unsqueeze(0).to(device, non_blocking=True) # 输入张量形状：[1, 3, 224, 224]
feat = backbone(x).squeeze(0).detach().cpu().float()            # 输出特征形状：[1536]
torch.save(feat, cache_path)                                    # 保存为 .pt 特征文件
```

### 4.3 下游预测的回归输出头定义 (输出结构变化)
本项目在提取了 1536 维特征后，使用一个多层感知机（MLP）作为回归头，最终输出 8 维的连续预测值。
代码源自：[uni2h_utils.py](file:///D:/AI%E7%A9%BA%E9%97%B4%E8%BD%AC%E5%BD%95%E7%97%85%E7%90%86%E7%A0%94%E7%A9%B6/PFMval_new/uni2h/uni2h_utils.py#L347-L366)
```python
class BackboneRegressor(nn.Module):
    def __init__(
        self,
        feature_dim: int,   # 1536
        hidden_dim: int,    # 隐藏层维度 (如 512)
        output_dim: int,    # 输出维度 (8, 对应 8 个基因集分数)
        dropout: float,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)  # 前向传播：输入 [B, 1536] -> 输出 [B, 8]
```

---

## 5. 对下游科研方案的总结与启示

1.  **物理尺度的对应性**：由于基础模型（UNI2-h）是在固定放大倍率（常为 20x，相当于 $0.5\ \mu\text{m/pixel}$）的 $224 \times 224$ 图像块上预训练的。我们在裁剪下游 patches 时，必须确保其对应的物理空间微米数与预训练 of 物理空间分布一致，否则会导致基础模型所识别的“细胞尺度”发生偏移，影响特征的鲁棒性。
2.  **特征缓存的有效性**：因为 UNI2-h 模型的权重在下游任务中是保持冻结状态（`requires_grad=False`），这意味着对于确定的 $224 \times 224$ patch，它的 1536 维特征是绝对静止不变的。本项目通过 `extract_and_cache_features()` 将其保存为 `.pt` 文件，可以避免后续在调节回归头 MLP（例如调整 `BackboneRegressor` 的 `hidden_dim` 或 `dropout`）时重复运行庞大的 Transformer 计算，使下游训练在数秒内即可完成。
