# UNI2-H 渐进式解冻训练 — 初学者指南

> **目标读者**：有基础 ML/DL 知识（知道什么是梯度下降、Transformer、Fine-tuning），想深入理解"为什么这样设计三阶段训练"的初学者。
>
> **阅读方式**：前半部分（第 1-4 节）讲基础概念和直觉，后半部分（第 5-8 节）深入代码逻辑和设计决策。可分段阅读。

---

## 目录

1. [为什么需要渐进式解冻？—— 问题的起点](#1-为什么需要渐进式解冻-问题的起点)
2. [整体架构一览](#2-整体架构一览)
3. [Stage 1：LoRA 微调 —— 用最小的代价让大模型适应新数据](#3-stage-1lora-微调-用最小的代价让大模型适应新数据)
4. [Stage 2：解冻末 2 层 —— "松开枷锁"的第一步](#4-stage-2解冻末-2-层-松开枷锁的第一步)
5. [Stage 3：解冻末 4 层 —— 再松开一点](#5-stage-3解冻末-4-层-再松开一点)
6. [差分学习率：三组参数，三种"学习速度"](#6-差分学习率三组参数三种学习速度)
7. [跨阶段 Checkpoint 转换：为什么不能直接加载？](#7-跨阶段-checkpoint-转换为什么不能直接加载)
8. [完整训练命令示例](#8-完整训练命令示例)
9. [常见问题 FAQ](#9-常见问题-faq)
10. [关键源码索引](#10-关键源码索引)

---

## 1. 为什么需要渐进式解冻？—— 问题的起点

### 1.1 原始场景

我们有一个在数十亿病理图像上预训练的大模型 **UNI2-H**（24 层 ViT，~300M 参数），它的任务是：输入一张 H&E 染色病理切片 patch（224×224），输出一个 1536 维的特征向量。

我们的下游任务是：用这个特征向量预测 30 条基因通路的 ssGSEA 活性评分（一个回归问题）。

### 1.2 基线方案的瓶颈

最直接的方案是：**完全冻结 UNI2-H，只训练下游回归头**。

```mermaid
flowchart LR
    Image["H&E Patch<br/>224×224"] --> Frozen["UNI2-H<br/>🔒 完全冻结"] --> Feature["1536维特征<br/>固定不变"] --> Head["MLP 回归头<br/>🔓 可训练"] --> Pred["30条通路预测"]
```

这个方法的问题是：**无论下游架构怎么改（加 GAT、加坐标编码、换更大的 MLP、换 AttnPool），跨患者 PCC 始终卡在 ~0.38**。这说明瓶颈不在下游架构，而在 **UNI2-H 的冻结特征本身**——它是在通用病理图像上预训练的，没有针对"食管癌 + 基因通路预测"这个特定任务适配过。

### 1.3 怎么办？

我们有两个极端选择：

| 方案 | 优点 | 缺点 |
|------|------|------|
| **全量微调**（解冻全部 24 层） | 最大适配能力 | 300M 参数全量训练，显存爆炸（RTX 4060 8GB 根本跑不了）；小数据集（~5000 patches）极易过拟合 |
| **保持冻结**（当前基线） | 稳定，显存省 | PCC 到天花板了，无法继续提升 |

我们需要一个**中间路线**——既能让 backbone 适配任务数据，又不会过拟合或显存溢出。这就是**渐进式解冻（Progressive Unfreezing）** 的动机。

### 1.4 渐进式解冻的核心思想

> **分阶段逐步"松开"模型的约束，每一步都只增加少量可训练参数，让模型稳步适应新数据，避免剧烈震荡。**

类比：就像教一个已经会走路的人学跳舞——你不会让他从头学走路，而是先让他穿着原来的鞋跟着音乐微调步伐（LoRA），再逐渐允许他改变腿部动作（解冻末层），最后让他自由发挥（解冻更多层）。

```
Stage 1:  全24层 LoRA（只训 A/B 矩阵，backbone 冻结）
            ↓
Stage 2:  末2层解冻 + 其余层 LoRA
            ↓
Stage 3:  末4层解冻 + 其余层 LoRA
```

---

## 2. 整体架构一览

### 2.1 模型结构

```
┌─────────────────────────────────────────────────────┐
│                   OnlineCLSModel                     │
├─────────────────────────────────────────────────────┤
│  Input: H&E Patch [B, 3, 224, 224]                  │
│       ↓                                              │
│  ┌─────────────────────────────────────────────┐    │
│  │  UNI2-H Backbone (24 ViT Blocks)            │    │
│  │  Block 0  ─── Block 1  ─── ... ─── Block 23 │    │
│  │  [可能含 LoRA]  [可能含 LoRA]    [可能解冻]  │    │
│  │                                              │    │
│  │  每 Block 包含:                               │    │
│  │    - attn.qkv  (Linear: 1536→4608)           │    │
│  │    - attn.proj (Linear: 1536→1536)           │    │
│  │    - mlp.fc1, mlp.fc2                        │    │
│  │    - norm1, norm2, ls1, ls2                  │    │
│  └─────────────────────────────────────────────┘    │
│       ↓ CLS Token [B, 1536]                          │
│  ┌─────────────────────────────────────────────┐    │
│  │  MLP 回归头 (与 HisToGeneUNI 完全一致)       │    │
│  │  Linear(1536→1024) → LayerNorm               │    │
│  │  + pos_x_embed + pos_y_embed                 │    │
│  │  → LayerNorm → Linear(1024→2048) → GELU      │    │
│  │  → Dropout → Linear(2048→30)                 │    │
│  └─────────────────────────────────────────────┘    │
│       ↓                                              │
│  Output: 30条通路预测 [B, 30]                        │
└─────────────────────────────────────────────────────┘
```

### 2.2 UNI2-H Backbone 关键参数

| 属性 | 值 |
|------|-----|
| 层数 (depth) | 24 |
| 特征维度 (embed_dim) | 1536 |
| 注意力头数 (num_heads) | 24 |
| Patch 大小 | 14×14 |
| 输入图像尺寸 | 224×224 |
| 每层 Attention 参数量 | ~14.2M（qkv 7.1M + proj 2.4M + MLP 4.7M） |
| 总参数量 | ~300M |

### 2.3 三阶段参数状态速查表

| | Stage 1 (lora) | Stage 2 | Stage 3 |
|---|---|---|---|
| **Blocks 0-19** | LoRA (qkv+proj)，backbone 冻结 | LoRA，backbone 冻结 | LoRA，backbone 冻结 |
| **Blocks 20-21** | LoRA，backbone 冻结 | LoRA，backbone 冻结 | **🔓 全部解冻** |
| **Blocks 22-23** | LoRA，backbone 冻结 | **🔓 全部解冻** | **🔓 全部解冻** |
| **下游 MLP 头** | 🔓 可训练 | 🔓 可训练 | 🔓 可训练 |
| **可训练参数量** | ~3.5M（仅 LoRA + 头） | ~3.5M + 末2层 ~28M | ~3.5M + 末4层 ~56M |

---

## 3. Stage 1：LoRA 微调 —— 用最小的代价让大模型适应新数据

### 3.1 什么是 LoRA？
https://www.doubao.com/thread/wacf5e3242b01533c

**LoRA（Low-Rank Adaptation）** 是一种参数高效微调（PEFT）技术。核心思想：

> 不直接修改预训练权重 W，而是训练一个低秩增量 ΔW = B × A，最终输出 = W·x + ΔW·x。

```
原始 Linear 层:        y = W · x          (W 冻结)
LoRA 注入后:           y = W·x + (B·A)·x·(α/r)
                                 └──┬──┘
                              低秩增量 ΔW
                              仅 A, B 可训练
```

**为什么叫"低秩"？**

W 的尺寸是 1536×1536（约 2.36M 参数）。如果 rank=8，则 A 是 8×1536，B 是 1536×8，合计只有 1536×8×2 = 24,576 参数——**仅占原始权重的 1%**！

直观理解：秩（rank）代表"自由度"。rank=8 意味着我们假设任务适配只需要在 8 个独立方向上调整权重，而非全部 1536 个方向。这个假设在迁移学习中通常成立——预训练模型已经学到了通用特征，任务适配只需要小幅度调整。

### 3.2 为什么 rank=8？

在代码中默认 `--lora_rank 8`，这是 LoRA 原论文和大量实践的 sweet spot：

- **rank=1~2**：表达能力不足，适配效果差
- **rank=4~8**：性价比最优区间，参数少效果好
- **rank=16~32**：参数增多，但增益递减（边际效应）
- **rank=64+**：接近全量微调，失去 LoRA 的参数效率优势

对于我们的任务（1536 维特征，~5000 训练样本），rank=8 在"表达力"和"防过拟合"之间取得平衡。`alpha=16` 则让 LoRA 输出有足够的量级（缩放因子 = 16/8 = 2.0）。

### 3.3 为什么只注入 qkv 和 proj，不注入 MLP？

```python
# lora_utils.py: inject_lora_to_block()
target_modules: Set[str] = {"qkv", "proj"}  # 默认只注入 Attention 的 QKV 和投影层
```

| 模块 | 是否注入 LoRA | 原因 |
|------|:---:|------|
| `attn.qkv` | ✅ | Attention 的 QKV 投影是"信息检索"的核心——决定 patch 之间如何交互。任务适配最需要调整的就是"关注什么"。 |
| `attn.proj` | ✅ | Attention 输出投影，直接影响传给下一层的特征表示 |
| `mlp.fc1` / `mlp.fc2` | ❌（默认） | MLP 层负责"知识存储"（前馈网络中的事实记忆）。预训练积累的病理知识主要在这里，不需要大幅调整 |
| `norm1` / `norm2` | ❌ | LayerNorm 只有少量参数（~3072），注入 LoRA 的 overhead 不划算 |

> **经验规律**：在视觉 Transformer 上做 LoRA 微调，优先注入 Attention 层足够；注入 MLP 通常收益很小但参数翻倍。代码中通过 `target_modules` 参数支持扩展到 MLP，但目前实验表明不需要。

### 3.4 A 初始化为 Kaiming Uniform，B 初始化为零 —— 为什么？

```python
# lora_utils.py: LoRALinear.__init__()
nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
nn.init.zeros_(self.lora_B.weight)
```

这是 LoRA 的关键设计：**训练开始时，LoRA 的输出为零（ΔW=0），模型行为与原始预训练模型完全一致。** 随着训练推进，B 逐渐学到非零值，模型平滑地偏离原始行为。

如果 A 和 B 都随机初始化，训练开始时模型就有一个随机的 ΔW，等于给预训练权重加了噪声——这对于保持预训练知识非常不利。

### 3.5 Stage 1 代码路径

当你运行 `--mode lora` 时，代码的执行路径是：

```
1. load_uni2h_backbone()         → 加载原始 UNI2-H，全部参数冻结
2. configure_stage1_lora()       → 在全部 24 层的 attn.qkv 和 attn.proj 上注入 LoRALinear
3. OnlineCLSModel(backbone)      → 包装 backbone + 下游 MLP 头
4. _build_optimizer()            → 创建三组优化器（此时只有 lora 组和 downstream 组）
5. 训练循环                       → 仅 LoRA 参数 + 下游头参与梯度更新
```

**关键**：backbone 的原始权重自始至终 `requires_grad=False`，连优化器里都没有它们。

---

## 4. Stage 2：解冻末 2 层 —— "松开枷锁"的第一步

### 4.1 为什么需要解冻？

LoRA 虽然高效，但有一个根本局限：**低秩假设**。ΔW = B×A（rank=8）只有 8 个自由度，而原始权重有 1536 个自由度。当任务与预训练数据分布差异较大时，8 个方向不足以表达所需的全部调整。

Stage 1 的 LoRA 已经让模型适应了新数据的大方向（类比：调整了"关注什么"），现在需要允许更深层的结构调整（类比：调整"如何组织学到的信息"）。

### 4.2 为什么先解冻最后 2 层（blocks 22-23）？

这是渐进式解冻的核心策略问题。选择从最后几层开始有三个原因：

**① 深层 = 任务特异，浅层 = 通用特征**

Transformer 的层级结构有一个公认规律：

```
浅层 (Block 0-7)   → 边缘、纹理、颜色等低级视觉特征
中层 (Block 8-15)  → 细胞形态、组织结构等中级语义
深层 (Block 16-23) → 病理诊断相关的高级语义（肿瘤 vs 正常、分级等）
```

我们的下游任务（基因通路预测）需要的是高级语义特征，因此优先调整深层。如果反过来先解冻浅层，可能破坏通用的病理特征提取能力，得不偿失。

**② 深层梯度最大，调整收益最高**

反向传播中，梯度从 loss 出发逐层回传。离 loss 最近的层（深层）梯度信号最直接、最大。解冻它们能获得最快、最显著的性能提升。

**③ 风险控制：少量参数先试探**

末 2 层约占模型总参数的 8%（~28M / 300M）。如果解冻后效果不升反降，说明"全量微调"这条路不适合当前数据量——损失可控，可以及时回退。

### 4.3 "解冻"具体做了什么？

```python
# lora_utils.py: configure_stage2_unfreeze_last2()
def configure_stage2_unfreeze_last2(backbone):
    # 步骤 1: 移除 blocks 22-23 的 LoRA 包装
    remove_lora_from_blocks(backbone, [22, 23], target_modules={"qkv", "proj"})
    # 步骤 2: 解冻这两个 block 的全部参数
    unfrozen = unfreeze_blocks(backbone, [22, 23],
                                unfreeze_attn=True,   # qkv + proj 权重
                                unfreeze_mlp=True,    # fc1 + fc2 权重
                                unfreeze_norm=True)   # LayerNorm + LayerScale
    return unfrozen
```

解冻后，blocks 22-23 的 **所有** 参数（attn + mlp + norm）都设置 `requires_grad=True`：

| 子模块 | 包含什么 | 解冻后行为 |
|--------|---------|-----------|
| `attn.qkv` | QKV 投影权重 (1536×4608) | 从 LoRA 模式恢复原始 Linear，全部可训 |
| `attn.proj` | 输出投影权重 (1536×1536) | 从 LoRA 模式恢复原始 Linear，全部可训 |
| `mlp.fc1` | MLP 第一层 | 可训练（Stage 1 中原本冻结） |
| `mlp.fc2` | MLP 第二层 | 可训练（Stage 1 中原本冻结） |
| `norm1`, `norm2` | LayerNorm | 可训练 |
| `ls1`, `ls2` | LayerScale | 可训练 |

### 4.4 为什么解冻 MLP 和 Norm，不仅仅是 Attention？

你可能会问：Stage 1 只给 Attention 加了 LoRA，为什么 Stage 2 连 MLP 和 Norm 也一起解冻？

- **MLP 层**：虽然是"知识存储"，但深层 MLP 存储的是任务相关的高级知识。对于"食管癌基因通路预测"这种预训练中几乎没见过的新任务，深层 MLP 的知识需要调整。
- **LayerNorm/LayerScale**：参数极少（每层 ~4608 个），解冻它们几乎没有过拟合风险，但能让层间特征分布适配新数据。

---

## 5. Stage 3：解冻末 4 层 —— 再松开一点

### 5.1 从末 2 层到末 4 层的逻辑

Stage 2 解冻末 2 层后，如果验证集 loss 继续下降且没有过拟合迹象，说明模型还有进一步适配的空间。Stage 3 将解冻范围从末 2 层（22-23）扩展到末 4 层（20-23）。

```python
# lora_utils.py: configure_stage3_unfreeze_last4()
def configure_stage3_unfreeze_last4(backbone):
    # 只需要移除 blocks 20-21 的 LoRA（22-23 已经在 Stage 2 移除了）
    remove_lora_from_blocks(backbone, [20, 21], target_modules={"qkv", "proj"})
    # 解冻 blocks 20-21（22-23 已经解冻了）
    unfrozen = unfreeze_blocks(backbone, [20, 21],
                                unfreeze_attn=True, unfreeze_mlp=True, unfreeze_norm=True)
    return unfrozen
```

### 5.2 为什么解冻 4 层而不是 6 层、8 层或全部？

这是根据"边际收益递减"原则的经验选择：

| 解冻层数 | 可训练 backbone 参数 | 过拟合风险 | 预期收益 |
|----------|---------------------|:---:|:---:|
| 2 层 | ~28M | 低 | 中等（+0.01~0.02 PCC） |
| **4 层** | **~56M** | **中低** | **较高（+0.02~0.03 PCC）** |
| 8 层 | ~112M | 中高 | 边际递减 |
| 全部 24 层 | ~300M | 高 | 可能反而下降（过拟合） |

在当前数据规模（~5000 patches × 3 患者）下，4 层是安全边界。随着更多患者数据加入（9 患者，预计 2026 年 6 月底），可以尝试 6-8 层甚至全量微调。

### 5.3 三阶段的渐进逻辑回顾

```
Stage 1: LoRA (全24层 Attention)
  └→ 模型学会了"在这个任务上应该关注什么"
  
Stage 2: LoRA (0-21层) + 末2层全解冻
  └→ 模型可以调整最高级语义特征的提取方式
  
Stage 3: LoRA (0-19层) + 末4层全解冻
  └→ 模型可以调整更多层的特征表示
```

每一阶段都是在前一阶段收敛的基础上"再进一步"，而非从头开始。这种设计避免了：
- 同时优化 LoRA + 解冻参数导致的训练不稳定
- 太多参数同时变化导致的 catastrophic forgetting（灾难性遗忘）

---

## 6. 差分学习率：三组参数，三种"学习速度"

### 6.1 为什么需要不同的学习率？

在同一次训练中，三组参数处于完全不同的状态：

| 参数组 | 状态 | 需要的 LR | 原因 |
|--------|------|:---:|------|
| **LoRA 参数** (lora_A, lora_B) | 从零开始学习（A 随机，B 初始为零） | `1e-4`（较快） | 需要快速学到有效的 ΔW |
| **解冻的 backbone 参数** | 已有预训练权重，只需微调 | `1e-5`（慢 10 倍） | 预训练权重已经很好了，大 LR 会破坏它们 |
| **下游 MLP 头** | 随机初始化（或从 Stage 1 训练好的） | `1e-4`（较快） | 这是从头学的，需要正常 LR |

核心原则：**预训练权重已经在一个"优值点"附近，只需要小步微调；LoRA 和下游头是从零开始，需要大步学习。**

### 6.2 代码实现

```python
# train_online_cls.py: _build_optimizer()
def _build_optimizer(model, args):
    lora_p, unfrozen_p, downstream_p = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" in name or "lora_B" in name:
            lora_p.append(param)          # → lr = 1e-4
        elif name.startswith("backbone."):
            unfrozen_p.append(param)      # → lr = 1e-5
        else:
            downstream_p.append(param)    # → lr = 1e-4

    groups = []
    if downstream_p:
        groups.append({"params": downstream_p, "lr": args.lr, "name": "downstream"})
    if lora_p:
        groups.append({"params": lora_p, "lr": args.lora_lr, "name": "lora"})
    if unfrozen_p:
        groups.append({"params": unfrozen_p, "lr": args.unfrozen_lr, "name": "unfrozen_backbone"})
    
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)
```

### 6.3 LR 比例的选择依据

- `lora_lr : unfrozen_lr = 10 : 1` 是 LoRA 社区的常见实践（参考 HuggingFace PEFT 指南）
- 如果 `unfrozen_lr` 设为 `1e-4`（和 LoRA 一样大），预训练权重会被大幅扰动，验证 loss 会出现剧烈震荡
- 如果 `unfrozen_lr` 设为 `1e-6`（太小），解冻带来的收益几乎不可见——模型基本还在用 Stage 1 的权重

---

## 7. 跨阶段 Checkpoint 转换：为什么不能直接加载？

### 7.1 问题

假设你在 Stage 1 训练了一个 checkpoint，现在想用它启动 Stage 2：

```
Stage 1 checkpoint 中的 blocks 22-23:
  attn.qkv = LoRALinear(original=qkv_weight, lora_A=..., lora_B=...)
  attn.proj = LoRALinear(original=proj_weight, lora_A=..., lora_B=...)
  
Stage 2 需要的 blocks 22-23:
  attn.qkv = nn.Linear (原始 Linear，无 LoRA 包装)
  attn.proj = nn.Linear (原始 Linear，无 LoRA 包装)
  且所有参数 requires_grad = True
```

**直接加载会失败**：模型结构不匹配！Stage 2 的 blocks 22-23 已经没有 LoRALinear 了，而 checkpoint 里是 LoRALinear。

### 7.2 解决方案：四步转换

代码中的跨阶段迁移逻辑（[train_online_cls.py:475-516](train_online_cls.py#L475-L516)）：

```
Step 1: 先注入全 24 层 LoRA（让模型结构 match checkpoint）
        → 加载 checkpoint 权重
        
Step 2: merge_lora_before_unfreeze()
        → 将 blocks 22-23 的 LoRA 权重"烧录"进原始权重
        
Step 3: remove_lora_from_blocks()
        → 移除 LoRALinear 包装，恢复原始 nn.Linear
        
Step 4: unfreeze_blocks()
        → 设置 requires_grad = True
        → 重建优化器（参数分组变了）
```

### 7.3 为什么 Step 2（merge）是绝对必要的？

这是整个方案中最关键的细节。考虑如果不 merge 就移除 LoRA 会怎样：

```
❌ 不 merge 直接 remove:
  Stage 1 学到的 ΔW 全部丢弃！
  blocks 22-23 的 qkv 权重 = 原始预训练权重
  等价于 Stage 1 在 blocks 22-23 上什么都没学到

✅ 先 merge 再 remove:
  Stage 1 学到的 ΔW 被永久合并到原始权重中
  新的 qkv 权重 = 原始预训练权重 + (lora_B @ lora_A) × (alpha/rank)
  Stage 2 从这个"已适配"的权重开始继续训练
```

### 7.4 merge_to_original 的实现细节

```python
# lora_utils.py: LoRALinear.merge_to_original()
def merge_to_original(self) -> None:
    # 计算 ΔW = B @ A × scale
    delta_w = (self.lora_B.weight @ self.lora_A.weight) * self.scale
    # 将 ΔW 加到原始权重上
    self.original.weight.data += delta_w
    # 清零 LoRA 矩阵（保证 merge 后再 forward 输出不变）
    nn.init.zeros_(self.lora_A.weight)
    nn.init.zeros_(self.lora_B.weight)
```

注意：merge 后 A 和 B 被清零，所以 `original(x) + 0 = original_new(x)`，前向传播结果完全不变。这是经过自检测试验证的（[lora_utils.py:432-437](lora_utils.py#L432-L437)）。

### 7.5 为什么不直接在 Stage 1 就解冻？

一个自然的疑问：既然最终要解冻，为什么不在 Stage 1 一步到位？

**答**：Stage 1 的 LoRA 阶段相当于一个"预热（warmup）"。在 LoRA 阶段：
- 模型通过低秩适配器找到任务适配的"大方向"
- 训练稳定，不会破坏预训练权重
- 解冻阶段继承了这个大方向，避免从随机方向开始全量微调

如果直接在 Stage 1 就解冻末 4 层 + LoRA：
- 解冻的预训练权重和随机初始化的 LoRA 同时训练，相互干扰
- 预训练权重可能在前几个 epoch 被大梯度冲垮（灾难性遗忘）
- 收敛更慢，最终效果也可能更差

---

## 8. 完整训练命令示例

### 8.1 Stage 1：LoRA（全 24 层）

```bash
# 本地冒烟测试（1 epoch）
PYTHONIOENCODING=utf-8 "C:\Program Files\Python313\python.exe" train_online_cls.py \
  --mode lora --lora_rank 8 --lora_blocks all \
  --patient HYZ15040 --epochs 1 --batch_size 4

# 完整训练（跨患者 Fold 1，150 epochs）
PYTHONIOENCODING=utf-8 "C:\Program Files\Python313\python.exe" train_online_cls.py \
  --mode lora --lora_rank 8 --lora_blocks all \
  --cross_patient --fold 1 --epochs 150 --batch_size 8 \
  --lr 1e-4 --lora_lr 1e-4
```

### 8.2 Stage 2：从 Stage 1 checkpoint 继续

```bash
PYTHONIOENCODING=utf-8 "C:\Program Files\Python313\python.exe" train_online_cls.py \
  --mode stage2 --lora_rank 8 --lora_blocks all \
  --cross_patient --fold 1 --epochs 150 --batch_size 8 \
  --lr 1e-4 --lora_lr 1e-4 --unfrozen_lr 1e-5 \
  --resume checkpoints/online_cls/lora_r8_online_cls_cross_fold1/best_model.pth
```

### 8.3 Stage 3：从 Stage 2 checkpoint 继续

```bash
PYTHONIOENCODING=utf-8 "C:\Program Files\Python313\python.exe" train_online_cls.py \
  --mode stage3 --lora_rank 8 --lora_blocks all \
  --cross_patient --fold 1 --epochs 150 --batch_size 8 \
  --lr 1e-4 --lora_lr 1e-4 --unfrozen_lr 1e-5 \
  --resume checkpoints/online_cls/stage2_r8_online_cls_cross_fold1/best_model.pth
```

### 8.4 变体：只在最后 4 层用 LoRA 的轻量方案

```bash
# Stage 1: 只给最后 4 层加 LoRA（参数更少，适合显存紧张时）
PYTHONIOENCODING=utf-8 "C:\Program Files\Python313\python.exe" train_online_cls.py \
  --mode lora --lora_rank 8 --lora_blocks last_4 \
  --cross_patient --fold 1 --epochs 150 --batch_size 8
```

---

## 9. 常见问题 FAQ

### Q1: 为什么不直接用 PEFT 库的 LoRA，而要手写？

PEFT 库的 LoRA 默认按参数名正则匹配（如 `.*q_proj.*`），但 UNI2-H 用的是 timm 的自定义 ViT 结构（`attn.qkv` 是一个融合的 Linear 而非三个独立的 q_proj/k_proj/v_proj），与 HuggingFace 的标准命名不兼容。手写实现可以精确控制注入目标，也避免了 PEFT 库的依赖。

### Q2: Stage 2 训练时，如果 val_loss 不降反升怎么办？

可能原因：
1. `unfrozen_lr` 太大 → 降低到 `5e-6` 试试
2. 数据太少，解冻 2 层仍然过拟合 → 增加 `weight_decay` 或减少 `unfrozen_lr`
3. Stage 1 的 checkpoint 还没收敛好 → 确保 Stage 1 val_loss 已经 plateau

### Q3: 能否跳过 Stage 1，直接 Stage 2 训练？

技术上可以（代码支持不加 `--resume` 直接 `--mode stage2`），但强烈不推荐。没有 LoRA 预热的解冻层从随机方向开始微调，效果通常不如渐进式。

### Q4: Stage 1/2/3 各需要训练多少个 epoch？

- Stage 1 (LoRA)：150 epoch，通常 ~80-100 epoch 达到最佳（早停触发）
- Stage 2：100-150 epoch，从 Stage 1 最佳 checkpoint 开始，通常 50-80 epoch 内改善
- Stage 3：100 epoch，改善空间比 Stage 2 小

### Q5: 能否扩展到解冻 6 层或 8 层？

代码已支持。修改 `configure_stage3_unfreeze_last4` 的模式即可。但需要注意：
- 解冻更多层 → 更大的过拟合风险 → 需要更多训练数据
- 建议等 9 患者数据到齐后再尝试 Stage 4（末 6-8 层）

### Q6: 和 EGNv2 的 freeze_layers 有什么区别？

| | UNI2-H 渐进式解冻 | EGNv2 freeze_layers |
|---|---|---|
| **策略** | 正向渐进（先 LoRA，再逐步解冻深层） | 反向冻结（先冻结前 N 层，训练后几层） |
| **Backbone** | ViT (300M) | ResNet-50 (23M) |
| **冻结粒度** | 按 block 索引精确控制 | 按 ResNet layer 组冻结 |
| **LoRA** | ✅ 有（手工实现） | ❌ 无 |
| **阶段数** | 3 stage（lora → stage2 → stage3） | 2 阶段（freeze → unfreeze） |

两者的共同思想都是"先约束再释放"，但 UNI2-H 方案更精细（有 LoRA 中间层、差分学习率、merge 机制），适合参数量大的 ViT；EGNv2 的方案更简单直接，适合参数量小的 ResNet。

---

## 10. 关键源码索引

| 文件 | 行号 | 内容 |
|------|------|------|
| [lora_utils.py:29-79](lora_utils.py#L29-L79) | 29-79 | `LoRALinear` 类——低秩适配包装器的完整实现 |
| [lora_utils.py:85-122](lora_utils.py#L85-L122) | 85-122 | `inject_lora_to_block()`——单层 LoRA 注入逻辑 |
| [lora_utils.py:125-166](lora_utils.py#L125-L166) | 125-166 | `inject_lora_to_backbone()`——批量注入指定 block |
| [lora_utils.py:169-211](lora_utils.py#L169-L211) | 169-211 | `merge_lora_before_unfreeze()`——跨阶段权重合并 |
| [lora_utils.py:214-250](lora_utils.py#L214-L250) | 214-250 | `remove_lora_from_blocks()`——移除 LoRA 恢复原始 Linear |
| [lora_utils.py:253-287](lora_utils.py#L253-L287) | 253-287 | `unfreeze_blocks()`——解冻指定 block 的参数 |
| [lora_utils.py:346-374](lora_utils.py#L346-L374) | 346-374 | `configure_stage1_lora()`——Stage 1 构建器 |
| [lora_utils.py:377-389](lora_utils.py#L377-L389) | 377-389 | `configure_stage2_unfreeze_last2()`——Stage 2 构建器 |
| [lora_utils.py:392-404](lora_utils.py#L392-L404) | 392-404 | `configure_stage3_unfreeze_last4()`——Stage 3 构建器 |
| [train_online_cls.py:1-24](train_online_cls.py#L1-L24) | 1-24 | 训练脚本文档头——三阶段概述 |
| [train_online_cls.py:251-327](train_online_cls.py#L251-L327) | 251-327 | 命令行参数定义（mode/lora/lr 等） |
| [train_online_cls.py:420-444](train_online_cls.py#L420-L444) | 420-444 | `_build_optimizer()`——差分学习率分组 |
| [train_online_cls.py:468-518](train_online_cls.py#L468-L518) | 468-518 | 跨阶段 checkpoint 四步转换逻辑 |
| [train_online_cls.py:553-566](train_online_cls.py#L553-L566) | 553-566 | 非 resume 模式下的 Stage 2/3 直接构建 |
| [model_online_cls.py:22-96](model_online_cls.py#L22-L96) | 22-96 | `OnlineCLSModel`——完整模型架构 |
| [uni2h/uni2h_utils.py:32-70](uni2h/uni2h_utils.py#L32-L70) | 32-70 | `load_uni2h_backbone()`——UNI2-H 加载与初始冻结 |

---

## 附录 A：自检实验建议

如果你想手动验证各阶段的效果，可以按以下步骤进行：

### A.1 验证 LoRA 注入正确性

```python
# 在 Python 环境中运行
import torch
from uni2h.uni2h_utils import load_uni2h_backbone
from lora_utils import inject_lora_to_backbone, get_lora_parameters, count_lora_parameters

backbone, _, _ = load_uni2h_backbone(device=torch.device('cpu'))
print(f"注入前可训练参数: {sum(p.numel() for p in backbone.parameters() if p.requires_grad)}")  # 应为 0

inject_lora_to_backbone(backbone, list(range(24)), rank=8)
print(f"LoRA 参数: {count_lora_parameters(backbone):,}")  # 应有 ~3.5M

# 验证原始权重仍然冻结
for name, p in backbone.named_parameters():
    if "lora" not in name:
        assert not p.requires_grad, f"{name} 应该冻结！"
print("✓ 所有原始权重冻结，仅 LoRA 参数可训练")
```

### A.2 验证 merge 不改变前向传播

```python
import torch
from lora_utils import LoRALinear

# 创建 LoRA 层，随机训练几步
original = torch.nn.Linear(16, 32)
lora = LoRALinear(original, rank=4, alpha=8.0)

# 手动"训练"一下 LoRA（模拟 Stage 1 学到了东西）
x = torch.randn(4, 16)
for _ in range(10):
    out = lora(x)
    loss = out.sum()
    loss.backward()
    # 手动更新 LoRA（不碰 original）
    with torch.no_grad():
        lora.lora_A.weight -= 0.01 * lora.lora_A.weight.grad
        lora.lora_B.weight -= 0.01 * lora.lora_B.weight.grad

# 记录 merge 前的输出
x_test = torch.randn(4, 16)
out_before = lora(x_test).clone()

# Merge + 清零
lora.merge_to_original()

# 验证 merge 后输出不变
out_after = lora(x_test)
print(f"Merge 前后差异: {(out_before - out_after).abs().max():.10f}")  # 应为 0
```

### A.3 对比 frozen vs lora vs stage2 单 epoch 输出

```bash
# 分别在三种模式下跑 1 个 epoch，对比 loss 曲线
PYTHONIOENCODING=utf-8 "C:\Program Files\Python313\python.exe" train_online_cls.py \
  --mode frozen --patient HYZ15040 --epochs 1 --batch_size 4

PYTHONIOENCODING=utf-8 "C:\Program Files\Python313\python.exe" train_online_cls.py \
  --mode lora --lora_rank 8 --patient HYZ15040 --epochs 1 --batch_size 4
```

观察：LoRA 模式的初始 loss 应该和 frozen 模式接近（因为 B 初始化为零，ΔW=0），但下降更快。

---

> **编写日期**: 2026-05-30
> **相关文档**: [模型性能排名](../模型性能排名_Model_Performance_Ranking.md) | [下一步改进建议](../分析报告/下一步改进操作建议_0516.md)
