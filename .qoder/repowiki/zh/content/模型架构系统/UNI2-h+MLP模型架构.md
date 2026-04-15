# UNI2-h+MLP模型架构

<cite>
**本文引用的文件**
- [README.md](file://README.md)
- [uni2h/train.py](file://uni2h/train.py)
- [uni2h/infer.py](file://uni2h/infer.py)
- [uni2h/uni2h_utils.py](file://uni2h/uni2h_utils.py)
- [histogene/model.py](file://histogene/model.py)
- [histogene/dataset.py](file://histogene/dataset.py)
- [histogene/train.py](file://histogene/train.py)
- [histogene/infer.py](file://histogene/infer.py)
- [histogene/utils.py](file://histogene/utils.py)
- [HYZ15040_ssGSEA_scores_zscore.csv](file://HYZ15040_ssGSEA_scores_zscore.csv)
- [histogene/training_history.csv](file://histogene/training_history.csv)
</cite>

## 目录
1. [简介](#简介)
2. [项目结构](#项目结构)
3. [核心组件](#核心组件)
4. [架构总览](#架构总览)
5. [详细组件分析](#详细组件分析)
6. [依赖关系分析](#依赖关系分析)
7. [性能考量](#性能考量)
8. [故障排查指南](#故障排查指南)
9. [结论](#结论)
10. [附录](#附录)

## 简介
本文件面向“UNI2-h+MLP”两阶段训练架构，系统阐述以下内容：
- 第一阶段：使用UNI2-h特征提取器输出1536维特征向量，冻结特征提取器权重，实现稳定的特征表示。
- 第二阶段：使用轻量级MLP回归头对8通路ssGSEA评分进行回归预测，显著降低训练成本与提高推理效率。
- 特征缓存系统：自动提取并持久化UNI2-h特征，避免重复计算，加速训练与推理。
- 两阶段优势：稳定特征提取、高效训练、良好泛化能力。
- 模型配置、训练流程与推理过程详解，并提供流程图与性能对比分析。

## 项目结构
本仓库采用模块化组织，围绕两个核心模块展开：
- uni2h：基于HuggingFace Hub的UNI2-h特征提取与回归训练/推理管线。
- histogene：基于ViT-MLP的直接图像到评分回归模型（作为对比或替代方案）。

```mermaid
graph TB
subgraph "数据与标签"
CSV["HYZ15040_ssGSEA_scores_zscore.csv"]
TRAIN["train_patches/"]
VAL["val_patches/"]
end
subgraph "UNI2-h阶段"
UTrain["uni2h/train.py"]
UInfer["uni2h/infer.py"]
UUtils["uni2h/uni2h_utils.py"]
Cache["uni2h_cache/"]
MLP["BackboneRegressor(MLP)"]
end
subgraph "Histogene阶段"
HModel["histogene/model.py"]
HDataset["histogene/dataset.py"]
HTrain["histogene/train.py"]
HInfer["histogene/infer.py"]
HUtils["histogene/utils.py"]
end
CSV --> UTrain
CSV --> UInfer
TRAIN --> UTrain
VAL --> UInfer
UTrain --> Cache
UInfer --> Cache
Cache --> MLP
MLP --> CSV
TRAIN --> HTrain
VAL --> HInfer
CSV --> HTrain
CSV --> HInfer
```

图表来源
- [uni2h/train.py:52-227](file://uni2h/train.py#L52-L227)
- [uni2h/infer.py:43-175](file://uni2h/infer.py#L43-L175)
- [uni2h/uni2h_utils.py:138-303](file://uni2h/uni2h_utils.py#L138-L303)
- [histogene/train.py:174-338](file://histogene/train.py#L174-L338)
- [histogene/infer.py:66-169](file://histogene/infer.py#L66-L169)

章节来源
- [README.md:1-44](file://README.md#L1-L44)

## 核心组件
- UNI2-h特征提取器：从HuggingFace Hub加载预训练模型，输出1536维特征向量；训练阶段冻结参数，推理阶段可选择是否冻结。
- 特征缓存系统：按patch目录扫描PNG图像，调用官方transform进行预处理，提取特征并以.pt文件缓存至本地目录。
- MLP回归头：输入维度固定为1536，隐藏层维度可配置，输出8通路评分。
- 数据集适配：通过CSV中的id列与patch文件名匹配，确保特征与标签一一对应。
- 训练/推理脚本：统一的命令行参数接口，支持早停、学习率调度、指标评估与历史记录导出。

章节来源
- [uni2h/uni2h_utils.py:31-71](file://uni2h/uni2h_utils.py#L31-L71)
- [uni2h/uni2h_utils.py:138-170](file://uni2h/uni2h_utils.py#L138-L170)
- [uni2h/uni2h_utils.py:173-226](file://uni2h/uni2h_utils.py#L173-L226)
- [uni2h/uni2h_utils.py:228-248](file://uni2h/uni2h_utils.py#L228-L248)
- [uni2h/train.py:26-49](file://uni2h/train.py#L26-L49)
- [uni2h/infer.py:24-41](file://uni2h/infer.py#L24-L41)

## 架构总览
两阶段训练策略的核心思想是“特征冻结 + 轻量回归”。第一阶段由UNI2-h完成高质量特征提取，第二阶段由MLP回归头完成下游任务预测。

```mermaid
sequenceDiagram
participant Data as "数据/标签"
participant Loader as "CachedFeaturePatchDataset"
participant UNI2 as "UNI2-h Backbone"
participant Cache as "特征缓存(.pt)"
participant MLP as "BackboneRegressor(MLP)"
participant Train as "训练循环(train.py)"
participant Infer as "推理(infer.py)"
Data->>Loader : 读取CSV与patch目录
Loader->>UNI2 : 对每张PNG执行transform
UNI2-->>Cache : 保存1536维特征向量
Train->>Loader : DataLoader加载缓存特征
Train->>MLP : 前向传播(1536->H->8)
Train->>Train : 计算损失与指标
Infer->>Loader : 加载缓存特征
Infer->>MLP : 前向传播得到8通路预测
```

图表来源
- [uni2h/uni2h_utils.py:138-170](file://uni2h/uni2h_utils.py#L138-L170)
- [uni2h/uni2h_utils.py:173-226](file://uni2h/uni2h_utils.py#L173-L226)
- [uni2h/train.py:120-131](file://uni2h/train.py#L120-L131)
- [uni2h/infer.py:92-100](file://uni2h/infer.py#L92-L100)

## 详细组件分析

### UNI2-h特征提取器
- 模型来源：从HuggingFace Hub加载MahmoodLab/UNI2-h，官方参数与结构已封装在工具函数中。
- 冻结策略：加载后将所有参数requires_grad置False，确保特征提取阶段不更新权重。
- 预处理：使用官方resolve_data_config与create_transform生成与模型一致的预处理流水线。
- 输出维度：固定为1536维特征向量。

章节来源
- [uni2h/uni2h_utils.py:31-71](file://uni2h/uni2h_utils.py#L31-L71)

### 特征缓存系统
- 自动提取：遍历patch目录下所有PNG文件，若缓存文件不存在或强制重建，则调用backbone提取特征并保存为.pt。
- 复用机制：后续训练/推理直接从缓存加载，避免重复计算。
- 数据一致性：通过CSV首列patch_id与PNG文件名stem匹配，保证特征与标签对齐。

```mermaid
flowchart TD
Start(["开始"]) --> Scan["扫描patch目录PNG文件"]
Scan --> Exists{"缓存文件存在且无需重建？"}
Exists --> |是| Next["跳过该文件"]
Exists --> |否| Pre["应用官方transform"]
Pre --> Forward["UNI2-h前向提取特征"]
Forward --> Save["保存为.pt缓存"]
Save --> Next
Next --> End(["结束"])
```

图表来源
- [uni2h/uni2h_utils.py:138-170](file://uni2h/uni2h_utils.py#L138-L170)

章节来源
- [uni2h/uni2h_utils.py:138-170](file://uni2h/uni2h_utils.py#L138-L170)

### MLP回归头
- 输入维度：1536（UNI2-h输出）
- 隐藏层：可配置维度，激活函数为GELU，Dropout可配置
- 输出维度：8（ssGSEA通路评分）
- 结构：LayerNorm -> Linear -> GELU -> Dropout -> Linear

```mermaid
classDiagram
class BackboneRegressor {
+forward(x)
}
class LayerNorm
class Linear
class GELU
class Dropout
BackboneRegressor --> LayerNorm : "输入归一化"
BackboneRegressor --> Linear : "1536->H"
BackboneRegressor --> GELU : "激活"
BackboneRegressor --> Dropout : "正则化"
BackboneRegressor --> Linear : "H->8"
```

图表来源
- [uni2h/uni2h_utils.py:228-248](file://uni2h/uni2h_utils.py#L228-L248)

章节来源
- [uni2h/uni2h_utils.py:228-248](file://uni2h/uni2h_utils.py#L228-L248)

### 数据集适配与标签对齐
- 通过CSV首列patch_id与PNG文件名stem进行映射，确保每个patch都有对应的8通路标签。
- 支持训练/推理阶段的坐标统计传递，保证位置信息的一致性（该部分在另一套模型中使用，此处用于对比说明）。

章节来源
- [uni2h/uni2h_utils.py:173-226](file://uni2h/uni2h_utils.py#L173-L226)
- [histogene/dataset.py:23-87](file://histogene/dataset.py#L23-L87)

### 训练流程（UNI2-h+MLP）
- 参数解析：支持batch_size、num_epochs、learning_rate、hidden_dim、dropout、早停等。
- 特征缓存：训练/验证集均执行特征提取与缓存。
- 数据加载：CachedFeaturePatchDataset按批次返回特征与标签。
- 损失与优化：HuberLoss或MSE，AdamW优化器，ReduceLROnPlateau调度。
- 早停与保存：监控验证损失，达到早停阈值停止；保存最佳checkpoint与训练历史。

```mermaid
sequenceDiagram
participant CLI as "命令行参数"
participant Train as "train.py"
participant Utils as "uni2h_utils.py"
participant Cache as "特征缓存"
participant Loader as "CachedFeaturePatchDataset"
participant Model as "BackboneRegressor"
participant Opt as "优化器/调度器"
CLI->>Train : 解析参数
Train->>Utils : 加载UNI2-h(backbone, transform)
Train->>Utils : extract_and_cache_features(训练/验证)
Train->>Loader : 构建数据集
loop 每个epoch
Train->>Loader : DataLoader迭代
Train->>Model : 前向传播
Train->>Opt : 反向传播与优化
Train->>Train : 早停与指标记录
end
Train->>Train : 保存最佳checkpoint与历史
```

图表来源
- [uni2h/train.py:52-227](file://uni2h/train.py#L52-L227)
- [uni2h/uni2h_utils.py:138-170](file://uni2h/uni2h_utils.py#L138-L170)
- [uni2h/uni2h_utils.py:251-303](file://uni2h/uni2h_utils.py#L251-L303)

章节来源
- [uni2h/train.py:52-227](file://uni2h/train.py#L52-L227)

### 推理流程（UNI2-h+MLP）
- 加载checkpoint：读取保存的模型权重与超参数。
- 特征缓存：对推理数据集执行特征提取与缓存。
- 数据加载：CachedFeaturePatchDataset加载特征与标签（如需）。
- 推理：模型前向得到8通路预测，计算逐通路与整体指标，保存结果与指标CSV。

章节来源
- [uni2h/infer.py:43-175](file://uni2h/infer.py#L43-L175)

### 对比模型：HisToGene（ViT-MLP）
- 直接从图像到评分的端到端模型，包含多头自注意力与前馈网络。
- 位置信息通过坐标嵌入与CLS token融合，输出8通路评分。
- 训练/推理流程与UNI2-h+MLP类似，但特征提取阶段需要反向传播，计算开销更大。

章节来源
- [histogene/model.py:64-160](file://histogene/model.py#L64-L160)
- [histogene/dataset.py:23-118](file://histogene/dataset.py#L23-L118)
- [histogene/train.py:174-338](file://histogene/train.py#L174-L338)
- [histogene/infer.py:66-169](file://histogene/infer.py#L66-L169)

## 依赖关系分析
- UNI2-h+MLP
  - 依赖：HuggingFace Hub（模型下载）、timm（模型创建与预处理）、PIL（图像读取）、torch（深度学习框架）。
  - 关键依赖链：train.py -> uni2h_utils.py(load_uni2h_backbone/extract_and_cache_features/CachedFeaturePatchDataset/BackboneRegressor) -> 特征缓存 -> 训练/推理。
- HisToGene（ViT-MLP）
  - 依赖：torchvision.transforms（图像预处理）、sklearn.metrics（指标计算）。
  - 关键依赖链：train.py -> model.py -> dataset.py -> utils.py。

```mermaid
graph TB
UTrain["uni2h/train.py"] --> UUtils["uni2h/uni2h_utils.py"]
UInfer["uni2h/infer.py"] --> UUtils
UUtils --> HF["HuggingFace Hub"]
UUtils --> Timm["timm"]
UUtils --> PIL["PIL.Image"]
HTrain["histogene/train.py"] --> HModel["histogene/model.py"]
HTrain --> HDataset["histogene/dataset.py"]
HTrain --> HUtils["histogene/utils.py"]
HTrain --> TV["torchvision.transforms"]
```

图表来源
- [uni2h/train.py:12-21](file://uni2h/train.py#L12-L21)
- [uni2h/infer.py:10-19](file://uni2h/infer.py#L10-L19)
- [uni2h/uni2h_utils.py:12-16](file://uni2h/uni2h_utils.py#L12-L16)
- [histogene/train.py:18-26](file://histogene/train.py#L18-L26)
- [histogene/utils.py:1-4](file://histogene/utils.py#L1-L4)

章节来源
- [uni2h/train.py:12-21](file://uni2h/train.py#L12-L21)
- [uni2h/infer.py:10-19](file://uni2h/infer.py#L10-L19)
- [histogene/train.py:18-26](file://histogene/train.py#L18-L26)

## 性能考量
- 训练效率
  - UNI2-h+MLP：特征提取阶段冻结权重，仅训练MLP回归头，显著降低显存与时间开销。
  - HisToGene：端到端训练，参数量大，训练时间长，但能学习更复杂的空间关系。
- 泛化能力
  - UNI2-h+MLP：利用预训练特征，对新数据具有更强的泛化能力，尤其在小样本场景。
  - HisToGene：直接从图像学习，可能过拟合特定数据分布。
- 推理速度
  - UNI2-h+MLP：特征缓存后推理极快，适合大规模部署。
  - HisToGene：每次推理都需要完整的图像前向，速度较慢。
- 空间信息保留
  - UNI2-h+MLP：空间信息通过patch坐标嵌入在另一套模型中体现，UNI2-h本身不直接保留空间坐标。
  - HisToGene：通过位置编码与CLS token融合，显式保留空间信息。

## 故障排查指南
- HuggingFace Token缺失
  - 现象：无法下载UNI2-h模型。
  - 处理：设置环境变量HF_TOKEN或在代码中传入token。
- 缓存路径权限问题
  - 现象：无法写入特征缓存文件。
  - 处理：确认cache_root目录可写，必要时以管理员权限运行。
- 数据不匹配
  - 现象：patch_id与CSV不一致导致特征与标签无法对齐。
  - 处理：检查CSV首列与PNG文件名stem是否一致，确保无扩展名差异。
- 显存不足
  - 现象：训练/推理OOM。
  - 处理：减小batch_size，关闭pin_memory，或使用更小的hidden_dim。
- 早停过早触发
  - 现象：验证损失未改善即停止。
  - 处理：增大early_stop_patience或min_delta，调整学习率与优化器。

章节来源
- [README.md:17-39](file://README.md#L17-L39)
- [uni2h/train.py:23-47](file://uni2h/train.py#L23-L47)
- [uni2h/infer.py:21-38](file://uni2h/infer.py#L21-L38)

## 结论
UNI2-h+MLP两阶段架构通过“特征冻结 + 轻量回归”的设计，在保证预测性能的同时大幅提升了训练与推理效率。特征缓存系统有效避免重复计算，使模型在大规模数据上具备良好的可扩展性。对于需要显式空间信息的任务，可结合HisToGene模型；对于追求高效率与强泛化的任务，UNI2-h+MLP是更优选择。

## 附录

### 模型配置参数（UNI2-h+MLP）
- 训练阶段
  - 训练/验证集目录：--train_patches_dir, --val_patches_dir
  - 标签CSV：--labels_csv
  - 特征缓存根目录：--cache_root
  - 检查点保存路径：--checkpoint_path
  - 批大小：--batch_size
  - 训练轮数：--num_epochs
  - 学习率：--learning_rate
  - MLP隐藏层维度：--hidden_dim
  - Dropout概率：--dropout
  - 早停耐心：--early_stop_patience
  - 最小提升：--min_delta
  - 目标列起始：--target_start_col
  - 目标数量：--num_targets
  - 强制重建缓存：--rebuild_cache
- 推理阶段
  - 待推理目录：--split_patches_dir
  - 输出CSV：--output_csv
  - 其余参数与训练阶段一致

章节来源
- [uni2h/train.py:26-49](file://uni2h/train.py#L26-L49)
- [uni2h/infer.py:24-41](file://uni2h/infer.py#L24-L41)

### 训练流程与推理流程
- 训练流程
  - 加载UNI2-h与transform
  - 对训练/验证集执行特征提取与缓存
  - 构建CachedFeaturePatchDataset
  - 训练循环：前向->反向->优化->早停->保存
- 推理流程
  - 加载checkpoint与backbone
  - 对推理集执行特征提取与缓存
  - 构建数据集并加载模型权重
  - 推理得到8通路预测，计算指标并保存

章节来源
- [uni2h/train.py:52-227](file://uni2h/train.py#L52-L227)
- [uni2h/infer.py:43-175](file://uni2h/infer.py#L43-L175)

### 性能对比分析
- 训练时间：UNI2-h+MLP显著短于HisToGene（端到端训练）。
- 显存占用：UNI2-h+MLP更低，适合大规模训练。
- 推理速度：UNI2-h+MLP更快，缓存后几乎无特征提取开销。
- 指标表现：两者均针对8通路ssGSEA评分进行回归，具体数值取决于数据与超参数调优。

章节来源
- [histogene/training_history.csv:1-12](file://histogene/training_history.csv#L1-L12)
- [README.md:17-39](file://README.md#L17-L39)