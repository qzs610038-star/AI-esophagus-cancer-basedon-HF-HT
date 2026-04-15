# UNI2-h+MLP训练流程

<cite>
**本文引用的文件**
- [README.md](file://README.md)
- [histogene/train.py](file://histogene/train.py)
- [histogene/model.py](file://histogene/model.py)
- [histogene/dataset.py](file://histogene/dataset.py)
- [histogene/utils.py](file://histogene/utils.py)
- [histogene/infer.py](file://histogene/infer.py)
- [uni2h/train.py](file://uni2h/train.py)
- [uni2h/uni2h_utils.py](file://uni2h/uni2h_utils.py)
- [uni2h/infer.py](file://uni2h/infer.py)
- [HYZ15040_ssGSEA_scores_zscore.csv](file://HYZ15040_ssGSEA_scores_zscore.csv)
</cite>

## 目录
1. [简介](#简介)
2. [项目结构](#项目结构)
3. [核心组件](#核心组件)
4. [架构总览](#架构总览)
5. [详细组件分析](#详细组件分析)
6. [依赖关系分析](#依赖关系分析)
7. [性能与内存优化](#性能与内存优化)
8. [故障排查指南](#故障排查指南)
9. [结论](#结论)
10. [附录：超参数与配置指南](#附录超参数与配置指南)

## 简介
本项目围绕“两阶段训练流程”展开：第一阶段使用冻结的 UNI2-h 特征提取器，将组织学切片 patch 转换为 1536 维特征向量，并缓存；第二阶段以这些特征作为输入，训练一个轻量级 MLP 多任务回归头，直接拟合 ssGSEA 通路评分。该流程显著降低计算成本，便于快速迭代与部署。

- 第一阶段（UNI2-h 特征提取与缓存）
  - 通过 Hugging Face 加载官方 UNI2-h 模型，冻结参数，使用官方预处理流水线。
  - 对训练/验证集 patch 逐一提取特征，按 patch 名称保存为 .pt 文件，形成持久化缓存。
- 第二阶段（MLP回归头训练）
  - 使用缓存的特征与标签（z-score 后的 ssGSEA 通路分数）训练简单 MLP 回归头。
  - 支持早停、学习率调度、混合精度、梯度裁剪等工程化优化。

## 项目结构
- histogene：基于 ViT-MLP 的端到端训练（可选，本流程更推荐 UNI2-h + MLP）。
- uni2h：两阶段流程的核心实现，包含特征提取、缓存、训练与推理。
- 数据与脚本：
  - 数据划分与 z-score 标准化脚本（split.py、zscore.py）。
  - 分析与统计输出（analyze_stats.py、data_distribution_analysis.py）。
  - 通用工具与实用函数（uni2h_utils.py、histogene/utils.py）。

```mermaid
graph TB
subgraph "数据与脚本"
S1["split.py"]
S2["zscore.py"]
A1["analyze_stats.py"]
A2["data_distribution_analysis.py"]
end
subgraph "UNI2-h阶段"
U1["uni2h/train.py"]
U2["uni2h/uni2h_utils.py"]
U3["uni2h/infer.py"]
C1[".cache/huggingface<br/>UNI2-h 权重"]
CK["checkpoints<br/>best_model_uni2h.pth"]
HC["uni2h_cache<br/>特征缓存(.pt)"]
end
subgraph "HisToGene阶段"
H1["histogene/train.py"]
H2["histogene/model.py"]
H3["histogene/dataset.py"]
H4["histogene/utils.py"]
H5["histogene/infer.py"]
CK2["checkpoints<br/>best_histogene.pth"]
end
S1 --> S2 --> H3
S2 --> U2
U1 --> U2 --> C1
U1 --> CK
U1 --> HC
H1 --> H2
H1 --> H3
H1 --> H4
H1 --> CK2
```

图表来源
- [uni2h/train.py:52-227](file://uni2h/train.py#L52-L227)
- [uni2h/uni2h_utils.py:31-71](file://uni2h/uni2h_utils.py#L31-L71)
- [histogene/train.py:174-338](file://histogene/train.py#L174-L338)
- [histogene/model.py:64-160](file://histogene/model.py#L64-L160)
- [histogene/dataset.py:23-118](file://histogene/dataset.py#L23-L118)

章节来源
- [README.md:1-44](file://README.md#L1-L44)

## 核心组件
- UNI2-h 特征提取与缓存
  - 加载官方 UNI2-h 模型与官方预处理，冻结参数，推理模式。
  - 遍历 patch 目录，对每张图提取特征并保存为 .pt 文件，支持重建缓存开关。
- CachedFeaturePatchDataset
  - 从缓存目录加载特征，按 patch 匹配标签，返回 (feature, target)。
- BackboneRegressor（MLP回归头）
  - 线性归一化 + 线性层 + GELU + Dropout + 线性输出，多任务回归。
- 训练与评估循环
  - 训练：前向、Huber/L2损失、反向、梯度裁剪、优化器步进、混合精度缩放。
  - 评估：前向、计算指标（MSE、MAE、R²、PCC）。
- 推理与指标导出
  - 加载 checkpoint，对新数据集进行推理，输出预测 CSV 与指标 CSV。

章节来源
- [uni2h/uni2h_utils.py:137-170](file://uni2h/uni2h_utils.py#L137-L170)
- [uni2h/uni2h_utils.py:173-226](file://uni2h/uni2h_utils.py#L173-L226)
- [uni2h/uni2h_utils.py:228-247](file://uni2h/uni2h_utils.py#L228-L247)
- [uni2h/train.py:120-131](file://uni2h/train.py#L120-L131)
- [uni2h/train.py:137-191](file://uni2h/train.py#L137-L191)
- [uni2h/train.py:209-223](file://uni2h/train.py#L209-L223)

## 架构总览
两阶段训练流程的端到端序列如下：

```mermaid
sequenceDiagram
participant User as "用户"
participant Train as "uni2h/train.py"
participant Utils as "uni2h/uni2h_utils.py"
participant HF as "HuggingFace Hub"
participant Backbone as "UNI2-h Backbone"
participant Cache as "特征缓存(.pt)"
participant Loader as "CachedFeaturePatchDataset"
participant Model as "BackboneRegressor(MLP)"
participant Eval as "evaluate()"
User->>Train : 设置参数并启动
Train->>Utils : load_uni2h_backbone()
Utils->>HF : 登录并加载模型
HF-->>Utils : 返回模型/transform/feature_dim
Utils-->>Train : 返回backbone, transform, feature_dim
Train->>Utils : extract_and_cache_features(train/val)
Utils->>Backbone : eval() + inference_mode()
loop 遍历每个patch
Utils->>Backbone : 预处理+前向
Backbone-->>Utils : 特征向量
Utils->>Cache : 保存feature.pt
end
Train->>Loader : 构建训练/验证Dataset
Train->>Model : 初始化MLP回归头
loop 训练循环
Train->>Model : 前向
Train->>Model : 计算损失
Train->>Model : 反向+优化
Train->>Eval : 验证集评估
end
Train-->>User : 保存最佳checkpoint与history
```

图表来源
- [uni2h/train.py:52-227](file://uni2h/train.py#L52-L227)
- [uni2h/uni2h_utils.py:31-71](file://uni2h/uni2h_utils.py#L31-L71)
- [uni2h/uni2h_utils.py:137-170](file://uni2h/uni2h_utils.py#L137-L170)
- [uni2h/uni2h_utils.py:279-303](file://uni2h/uni2h_utils.py#L279-L303)

## 详细组件分析

### 第一阶段：UNI2-h特征提取与缓存
- 功能要点
  - 官方模型加载与冻结：通过 Hugging Face 加载 UNI2-h，设置 eval() 且 requires_grad=False。
  - 官方预处理：使用 timm 的官方数据配置与 transforms，保证与训练一致。
  - 特征提取与缓存：遍历 patch 目录，对每张图执行 transform + 前向，保存为 .pt 文件，文件名为 patch 名称（不含扩展名）。
  - 缓存重建：支持 rebuild 参数，若缓存存在则跳过，否则重新提取。
- 关键接口
  - load_uni2h_backbone：返回 backbone、transform、feature_dim。
  - extract_and_cache_features：执行特征提取与缓存。
  - CachedFeaturePatchDataset：从缓存目录读取特征并匹配标签。
- 性能与可靠性
  - 推理模式 + inference_mode 减少梯度开销。
  - 缓存文件命名与 patch 匹配，避免重复计算。
  - 缓存缺失时抛出异常，便于定位问题。

```mermaid
flowchart TD
Start(["开始"]) --> Load["加载UNI2-h(backbone, transform)"]
Load --> Loop{"遍历patch目录"}
Loop --> |存在缓存且不重建| Skip["跳过"]
Loop --> |不存在缓存或需重建| Extract["预处理+前向提取特征"]
Extract --> Save["保存feature.pt"]
Save --> Loop
Skip --> Loop
Loop --> Done(["完成"])
```

图表来源
- [uni2h/uni2h_utils.py:31-71](file://uni2h/uni2h_utils.py#L31-L71)
- [uni2h/uni2h_utils.py:137-170](file://uni2h/uni2h_utils.py#L137-L170)

章节来源
- [uni2h/uni2h_utils.py:31-71](file://uni2h/uni2h_utils.py#L31-L71)
- [uni2h/uni2h_utils.py:137-170](file://uni2h/uni2h_utils.py#L137-L170)
- [uni2h/uni2h_utils.py:173-226](file://uni2h/uni2h_utils.py#L173-L226)

### 第二阶段：MLP回归头训练
- 模型结构
  - BackboneRegressor：LayerNorm → Linear → GELU → Dropout → Linear，输出多任务分数。
- 数据加载
  - CachedFeaturePatchDataset：按 patch 匹配标签，返回 (feature, target)。
- 训练流程
  - 损失函数：MSE（或可替换为 Huber）。
  - 优化器：AdamW，权重衰减。
  - 学习率调度：ReduceLROnPlateau。
  - 早停：基于验证集 loss，patience 控制。
  - 混合精度：GradScaler（CUDA）。
  - 梯度裁剪：clip_grad_norm_。
- 指标计算
  - compute_metrics：MSE、MAE、R²、PCC，支持多任务平均。

```mermaid
classDiagram
class BackboneRegressor {
+forward(x)
}
class CachedFeaturePatchDataset {
+__len__()
+__getitem__(idx)
}
class train_one_epoch {
+forward()
+backward()
}
class evaluate {
+forward()
}
CachedFeaturePatchDataset --> BackboneRegressor : "提供特征与标签"
train_one_epoch --> BackboneRegressor : "训练"
evaluate --> BackboneRegressor : "验证"
```

图表来源
- [uni2h/uni2h_utils.py:228-247](file://uni2h/uni2h_utils.py#L228-L247)
- [uni2h/uni2h_utils.py:173-226](file://uni2h/uni2h_utils.py#L173-L226)
- [uni2h/uni2h_utils.py:250-277](file://uni2h/uni2h_utils.py#L250-L277)
- [uni2h/uni2h_utils.py:279-303](file://uni2h/uni2h_utils.py#L279-L303)

章节来源
- [uni2h/uni2h_utils.py:228-247](file://uni2h/uni2h_utils.py#L228-L247)
- [uni2h/uni2h_utils.py:173-226](file://uni2h/uni2h_utils.py#L173-L226)
- [uni2h/uni2h_utils.py:250-277](file://uni2h/uni2h_utils.py#L250-L277)
- [uni2h/uni2h_utils.py:279-303](file://uni2h/uni2h_utils.py#L279-L303)

### 数据加载与批次管理
- HisToGene 数据集（可选）
  - 从 patch 文件名解析坐标，匹配 z-score 标准化的标签，归一化到 [0, n_pos-1]。
  - 支持训练/验证共享坐标统计，保证一致性。
- UNI2-h 数据集
  - 从缓存目录读取特征，按 patch 匹配标签，支持重建缓存。
- 批次管理
  - DataLoader：shuffle、pin_memory、num_workers。
  - 非阻塞移动张量至设备，减少 CPU-GPU 传输等待。

章节来源
- [histogene/dataset.py:23-118](file://histogene/dataset.py#L23-L118)
- [uni2h/uni2h_utils.py:173-226](file://uni2h/uni2h_utils.py#L173-L226)
- [histogene/train.py:222-230](file://histogene/train.py#L222-L230)
- [uni2h/train.py:102-115](file://uni2h/train.py#L102-L115)

### 推理与指标导出
- HisToGene 推理
  - 加载 checkpoint，重建模型结构与参数，对指定 patch 目录进行推理，输出预测 CSV 与逐通路指标。
- UNI2-h 推理
  - 加载 checkpoint，重建回归头，对指定 split 的 patch 进行特征提取与缓存，再进行推理，输出预测与指标 CSV。

章节来源
- [histogene/infer.py:66-169](file://histogene/infer.py#L66-L169)
- [uni2h/infer.py:43-175](file://uni2h/infer.py#L43-L175)

## 依赖关系分析
- 外部依赖
  - HuggingFace Hub：加载 UNI2-h 模型。
  - timm：官方数据配置与 transforms。
  - PyTorch：模型、优化器、混合精度、数据加载。
  - scikit-learn：指标计算（MSE、MAE、R²、PCC）。
- 内部模块
  - uni2h/train.py 依赖 uni2h/uni2h_utils.py。
  - histogene/train.py 依赖 histogene/model.py、histogene/dataset.py、histogene/utils.py。
  - 推理脚本依赖各自模块与 utils。

```mermaid
graph TB
T["uni2h/train.py"] --> U["uni2h/uni2h_utils.py"]
U --> HF["HuggingFace Hub"]
U --> TIMM["timm"]
T --> CKPT["checkpoints/*.pth"]
T --> CACHE["uni2h_cache/*.pt"]
HT["histogene/train.py"] --> HM["histogene/model.py"]
HT --> HD["histogene/dataset.py"]
HT --> HU["histogene/utils.py"]
HT --> CKPT2["checkpoints/*.pth"]
```

图表来源
- [uni2h/train.py:12-21](file://uni2h/train.py#L12-L21)
- [uni2h/uni2h_utils.py:12-16](file://uni2h/uni2h_utils.py#L12-L16)
- [histogene/train.py:24-26](file://histogene/train.py#L24-L26)

章节来源
- [uni2h/train.py:12-21](file://uni2h/train.py#L12-L21)
- [uni2h/uni2h_utils.py:12-16](file://uni2h/uni2h_utils.py#L12-L16)
- [histogene/train.py:24-26](file://histogene/train.py#L24-L26)

## 性能与内存优化
- 混合精度训练
  - 使用 GradScaler（CUDA），在不牺牲精度的前提下显著降低显存占用与加速训练。
- 梯度裁剪
  - clip_grad_norm_ 限制梯度范数，提升稳定性。
- 非阻塞数据传输
  - DataLoader 与张量移动设置 non_blocking=True，减少等待时间。
- 缓存复用
  - 特征缓存避免重复前向，显著缩短训练与推理时间。
- 数据加载策略
  - pin_memory 在 GPU 可用时启用，提高数据搬运效率。
  - num_workers 在 Windows 上设为 0（兼容性考虑），Linux 可适当增大。

章节来源
- [histogene/train.py:197-199](file://histogene/train.py#L197-L199)
- [histogene/train.py:124-127](file://histogene/train.py#L124-L127)
- [uni2h/train.py:128-131](file://uni2h/train.py#L128-L131)
- [uni2h/train.py:107-115](file://uni2h/train.py#L107-L115)

## 故障排查指南
- HuggingFace 登录失败
  - 确认 HF_TOKEN 或环境变量 HUGGINGFACE_HUB_TOKEN/HF_TOKEN 设置正确。
- 缓存文件缺失
  - CachedFeaturePatchDataset 在缓存缺失时抛出 FileNotFoundError，检查 cache_root 与 rebuild 参数。
- 标签列不匹配
  - CachedFeaturePatchDataset 与 z-score CSV 的列范围不一致会导致错误，核对 target_start_col 与 num_targets。
- 设备不匹配
  - 确保 checkpoint 与模型结构参数一致，推理时 map_location 正确。
- 数据集为空
  - 若 patch 目录中没有匹配的 PNG 文件或标签映射为空，会报错，检查路径与文件名。

章节来源
- [uni2h/uni2h_utils.py:24-29](file://uni2h/uni2h_utils.py#L24-L29)
- [uni2h/uni2h_utils.py:205-207](file://uni2h/uni2h_utils.py#L205-L207)
- [uni2h/infer.py:48-56](file://uni2h/infer.py#L48-L56)

## 结论
两阶段训练流程以 UNI2-h 作为特征提取器，结合轻量级 MLP 回归头，实现了高效、稳定且可复现的多任务回归训练。通过特征缓存与工程化优化（混合精度、早停、学习率调度、梯度裁剪），在有限资源下获得良好性能与可维护性。建议优先采用该流程，并根据下游任务进一步扩展与定制。

## 附录：超参数与配置指南
- 第一阶段（UNI2-h 特征提取）
  - cache_root：特征缓存根目录（建议与数据集划分同名子目录）。
  - rebuild_cache：是否重建缓存（首次或变更预处理时使用）。
  - hf_token：HuggingFace 访问令牌。
- 第二阶段（MLP回归头）
  - batch_size：建议 128–512，依据显存调整。
  - num_epochs：建议 50–200，配合早停。
  - learning_rate：建议 1e-3–1e-4，AdamW + weight_decay。
  - hidden_dim：建议 128–512，视数据复杂度与显存。
  - dropout：建议 0.1–0.3。
  - early_stop_patience：建议 5–20。
  - num_workers：Windows 默认 0，Linux 可设为 2–8。
  - min_delta：验证集 loss 至少下降幅度阈值。
- 数据与标签
  - labels_csv：z-score 标准化后的 ssGSEA 通路分数 CSV。
  - target_start_col、num_targets：从 CSV 中选择目标列的起始与数量。
- 模型保存与恢复
  - best_model_uni2h.pth：包含模型状态字典、超参数、缓存目录等。
  - 推理时通过 checkpoint_path 加载，自动重建回归头。

章节来源
- [uni2h/train.py:26-49](file://uni2h/train.py#L26-L49)
- [uni2h/uni2h_utils.py:173-226](file://uni2h/uni2h_utils.py#L173-L226)
- [uni2h/infer.py:24-41](file://uni2h/infer.py#L24-L41)