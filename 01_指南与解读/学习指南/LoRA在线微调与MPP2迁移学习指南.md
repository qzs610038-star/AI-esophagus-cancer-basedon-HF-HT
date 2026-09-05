# LoRA 在线微调方案剖析与 MPP2 迁移学习指南

本指南旨在深入剖析前三患者实验中手工实现的 LoRA 在线微调方案，结合前沿的先进 LoRA 实践以及最新的审查建议，探讨其如何规范地迁移至当前的 MPP2 主线方案中，并提供具体的迁移步骤、经过校准的伪代码实现以及推荐补充的背景知识。

---

## 目录
1. [手工 LoRA 在线微调方案深度剖析](#1-手工-lora-在线微调方案深度剖析)
2. [对抗性审查：基于前沿实践的旧方案隐患评估](#2-对抗性审查基于前沿实践的旧方案隐患评估)
3. [为什么需要迁移到 MPP2？面临的挑战是什么？](#3-为什么需要迁移到-mpp2面临的挑战是什么)
4. [MPP2 迁移的具体方案设计（修正优化版）](#4-mpp2-迁移的具体方案设计修正优化版)
5. [推荐补充的专业知识](#5-推荐补充的专业知识)

---

## 1. 手工 LoRA 在线微调方案深度剖析

在之前的历史三患者实验中，团队并没有直接使用 HuggingFace 的 PEFT 库，而是基于 PyTorch 手工编写了一套高度定制的 LoRA 微调体系（见 [lora_utils.py](../../lora_utils.py)）。其核心组件包含四个层面：

### 1.1 LoRALinear 核心实现原理

在 `lora_utils.py` 中，[LoRALinear](../../lora_utils.py#L29) 继承自 `nn.Module`，充当了 `nn.Linear` 层的低秩包装器。

$$\text{forward}(x) = W_0 x + \Delta W x = \text{original}(x) + \text{lora\_B}(\text{lora\_A}(\text{lora\_dropout}(x))) \times \frac{\alpha}{r}$$

> **💡 形象类比：**
> 我们可以把原始的预训练层（$W_0$）想象成一位**极其固执、经验丰富的顶级大厨**。他已经掌握了海量的通用厨艺（预训练知识），但你不能改变他已有的任何习惯（权重冻结）。
> 为了让他做新的病理学特色菜，我们给他配了**两名年轻学徒**（$A$ 和 $B$）：
> - **学徒 A** 负责“抓重点”（降维）：把复杂的食材需求精简成几个最核心的关键词（输入维度 $\to$ 秩 $r$）。
> - **学徒 B** 负责“还原”：根据关键词，将其展开为具体的调料比例（秩 $r \to$ 输出维度）。
> 大厨和学徒们同时处理食材（输入 $x$），最后把他们的结果加在一起。

- **权重冻结**：初始化时，原始 Linear 层（即 $W_0$）的全部参数被彻底冻结（`requires_grad = False`）。
- **旁路低秩投影**：使用 `lora_A`（Kaiming Uniform 随机初始化）和 `lora_B`（全 0 初始化）。

<details>
<summary><b>💬 答疑：LoRA_A 的随机初始化是什么意思？如何实现？有什么作用？</b></summary>
<br>
<b>是什么：</b> 随机初始化是指在网络训练开始前，给矩阵 $A$ 的参数赋予一组遵循特定概率分布（如高斯分布或均匀分布）的微小随机值。Kaiming Uniform（何恺明均匀分布）是专为带有激活函数的深层网络设计的一种初始化方法。<br>
<b>如何实现：</b> 在 PyTorch 中，通过调用 <code>nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))</code> 即可实现。<br>
<b>有什么作用与梯度传播细节：</b> <br>
1. <b>打破对称性（Symmetry Breaking）：</b> 如果 $A$ 和 $B$ 都是全零，网络就无法学习到复杂的特征。给 $A$ 随机值可以打破参数对称性。<br>
2. <b>初始步的梯度传播细节（修正）：</b> 在初始第 0 步，由于 $B$ 初始化为 0，根据链式法则，关于 $A$ 权重的损失函数梯度 $\nabla_A \mathcal{L} = B^T \cdot (\text{upstream\_grad} \cdot x^T) \times \text{scale} = 0$。因此在<b>最开始的一瞬间，只有 $B$ 矩阵会接收到非零梯度并进行更新</b>。一旦 $B$ 在第一步更新后不再为 0，$A$ 矩阵就会立即开始接收非零梯度并随之更新。这种初始化设计不仅维持了前向传播的等价性（$\Delta W=0$），也保证了反向传播能够稳定启动。<br>
3. <b>保持方差稳定：</b> Kaiming 初始化能保证信号在传播时方差维持在健康区间，防止梯度爆炸或消失。
</details>

  > **🧐 为什么 B 要全零初始化？**
  > 就像学徒第一天上班，主管告诉他们：“你们今天先别动手，只许看。” 由于 $B=0$，学徒们的输出增量 $\Delta W = 0$。这保证了在微调的第 0 步，菜品的味道完全是由大厨（冻结的主干）决定的，**绝不会因为加入了学徒而毁掉原本的招牌菜**（即不对预训练模型造成初始破坏）。
  
- **权重合并（Merge）**：推理或结构转换时，通过将 $B \cdot A \times \frac{\alpha}{r}$ 合并进原始权重。

<details>
<summary><b>💬 答疑：为什么需要乘一个 $\alpha / r$？</b></summary>
<br>
这其实是一个<b>“解耦”</b>设计，让你可以自由调整秩 $r$ 而不需要每次都重新调参。<br>
1. <b>稳定梯度：</b> 在矩阵乘法 $B \cdot A$ 中，结果的数值大小会随着秩 $r$（参与累加的维度）的增大而变大。如果不做缩放，当你把 $r$ 从 8 增加到 64 时，$\Delta W$ 的输出幅度会直接翻倍甚至更多，导致你不得不重新寻找更小的学习率。<br>
2. <b>维持音量恒定：</b> 通过除以 $r$，无论你设定多大的秩（无论学徒的记事本有多厚），他们最终对大厨输出的总“音量”（信号幅度）都是恒定、可控的。<br>
3. <b>$\alpha$ 的作用：</b> $\alpha$ 就是纯粹的音量放大器。$\alpha / r$ 的比值固定了 LoRA 对原权重的实际影响力。通常设 $\alpha = r$ 或 $2r$，这样比值就是 $1$ 或 $2$。
</details>

  > **💡 形象类比（参数再参数化）：**
  > 学徒们经过几个月的微调训练后，终于掌握了窍门。我们将他们总结的“秘方”（$\Delta W$）直接**抄写进大厨的官方食谱书里**（$W_{\text{new}} = W_0 + \Delta W$）。一旦抄写完成，学徒们就可以下班了（将 $A$、$B$ 参数从网络中丢弃）。这样一来，厨房出菜的速度依然和大厨一个人工作时一样快（**零额外推理延迟**）。

### 1.2 多阶段（Multi-Stage）渐进式微调机制

在旧版微调脚本（如 [train_online_cls.py](../../train_online_cls.py)）中设计了多阶段递进结构：
1. **Stage 1 (lora)**：全网络 Attention 层的 `qkv` 和 `proj` 注入 LoRA（给沉睡的主干穿上保暖外套）。

<details>
<summary><b>💬 答疑：qkv 与 proj 是什么？为什么在这两层注入 LoRA？</b></summary>
<br>
<b>是什么：</b> 它们是 Transformer 注意力机制（Attention）中的核心线性变换层。<br>
- <b>qkv (Query, Key, Value)：</b> 负责将输入的图像信息转换为三个向量：Query（我要找什么）、Key（我有什么特征）、Value（我的实际内容）。这是模型理解图像中各个部位（Patch）相互关系的基础。<br>
- <b>proj (Projection)：</b> 注意力计算完成后，将融合了全局信息的结果重新投影回原本的维度，传递给下一层。<br>
<b>为什么注入这两层：</b> <br>
1. <b>信息枢纽：</b> 在注意力机制中，<code>qkv</code> 是特征交互最密集的地方，控制着模型如何“看”数据。改变这里，能用最少的参数获得最大的行为改变。<br>
2. <b>参数性价比极高：</b> 早期研究发现，仅仅微调 <code>qkv</code> 和 <code>proj</code>，就能达到极其接近全参微调的效果，是性价比最高的目标。<br>
<b>⚠️ 规范约束：</b> MPP2 P0 阶段<b>只运行 Stage 1 LoRA</b>；多阶段 merge 与 unfreeze (Stage 2/3) 本轮实验不进入实验矩阵，以防将 LoRA 微调效果与主干解冻效果混淆。
</details>

2. **Stage 2 (stage2)**：将最后 2 层 Block 的 LoRA 参数 `merge_to_original` 并彻底剥离包装，随后完全解冻这些层的所有参数。
3. **Stage 3 (stage3)**：将完全解冻范围扩大到最后 4 层 Block。

### 1.3 参数分组优化与跨阶段映射

- **参数分组**：优化器被精细划分为三组：
  > **💡 形象类比：**
  > - **`lora` 与 `downstream` 分组（学习率较高，如 $1e-4$）**：就像公司里的**新员工**，需要快速学习新业务，所以步子可以迈得大一点。
  > - **`unfrozen_backbone` 分组（解冻层，学习率极低，如 $1e-5$）**：就像公司的**元老级高管**，他们的决策千一发而动全身。对他们的调整必须极其谨慎、如履薄冰。
- **结构映射**：通过自研的 `transfer_ckpt_plain_to_lora_original` 解决了因多阶段解冻导致的网络结构变化带来的 checkpoint 载入 `key` 冲突问题。

---

## 2. 对抗性审查：基于前沿实践的旧方案隐患评估

结合前沿的 Vision Transformer (ViT) 微调实践对我们旧版“三患者实验”进行审查后，整理出以下隐患及修正路径：

> [!WARNING]
> **隐患 1：注入模块的范围限制**
> - **旧方案状态**：仅针对 Attention 层的 `qkv` 和 `proj` 注入 LoRA。
> - **改进建议与 P0 边界**：前沿实践指出了 MLP 层对特定知识表征的重要性，主张扩展至全线性层。但**为了保证历史可比性并控制变量，MPP2 P0 必须固定注入 `{"qkv", "proj"}` 且 $r=8$**。MLP LoRA 仅在此 P0 通过后作为单因素消融对比。

> [!CAUTION]
> **隐患 2：医疗小样本下的正则化参数设定**
> - **概念解释 - Weight Decay（权重衰减）**：一种惩罚机制（类似于奥卡姆剃刀），强迫模型尽量保持权重值很小，防止其“死记硬背”训练集里的特例噪声。
> - **优化边界**：前沿实践常推荐较大的 weight decay（0.01~0.1）。但**为了维持与历史方案的公平比对，MPP2 P0 阶段优化器默认依旧使用 `1e-4` 的偏小权重衰减**。后续仅在 internal val 上对 `1e-4` 和 `1e-2` 进行消融，不默认上调至 0.1。

> [!TIP]
> **隐患 3：秩 (Rank) 与缩放因子 ($\alpha$) 的消融纪律**
> - **概念解释**：
>   - **秩 $r$（Rank）**：决定了 $\Delta W$ 矩阵的“容量”或“智商”。容量太大容易记下废话（过拟合），容量太小则学不会复杂的病理模式（欠拟合）。
>   - **缩放因子 $\alpha$（Alpha）**：决定了 LoRA 的“发言音量”，缩放 LoRA 旁路的梯度和影响。
> - **优化边界**：**目前门禁规定 smoke 必须只跑 $r=8$ 且最多 3 epoch**。后续的 Rank 消融（如 $r=16, 32$）需获取新实验批准，且参数选择必须完全基于 internal val 性能，绝不能在外部测试集 XZY 上调参。

> [!IMPORTANT]
> **隐患 4：显存开销与梯度检查点启用条件**
> - **概念解释 - Gradient Checkpointing（梯度检查点）**：通过“以时间换空间”，反向传播时临时重算部分前向结果以节约显存。
> - **优化边界**：旧版的 `qkv/proj` LoRA 在 8GB 显卡上的峰值显存仅为约 4.5GB。因此，**Gradient Checkpointing 在本项目中绝非必须**。Phase 1 应首先探测显存，仅在确实受限且模型 backbone 运行时探测到支持 `set_grad_checkpointing` 时才选择性启用，并记录其带来的吞吐量速度代价。

---

## 3. 为什么需要迁移到 MPP2？面临的挑战是什么？

### 3.1 现有 MPP2 的设计规范快照
根据 2026-07-09 决策：
- **禁止使用旧三患者实验结论作为性能证据**。
- **数据污染的边界隔离**：明确区分“旧三患者 JFX 染色数据坐标污染（因同名 barcode 跨患者泄露）”与“旧标签重建时的跨患者 barcode 污染（多患者拼表时坐标主键碰撞）”。MPP2 彻底废弃旧数据链，统一绑定已验收且修复的 `split_manifest.csv`。
- **统一重跑协议（V3bis/Manifest）**：
  - **固定空间 block 划分**：消费由 `group_2/split_manifest.csv` 驱动的已划分数据集。其划分依据是固定空间 block（`split_info.json` 中记录 `block_size=672`，`leakage_pairs=0`），**禁止将 Dataset 的 split 改为 Patch 级随机打乱 90/10**。
  - **无泄漏 Z-score 归一化**：均值和方差必须且仅在 `train` 的 Patch 上拟合，严禁外部测试集（XZY）前瞻泄露。

### 3.2 迁移的主要挑战
1. **数据模态转变**：现存 MPP2 ([train_mpp_uni2h_mlp.py](../../train_mpp_uni2h_mlp.py)) 使用离线缓存 `.pt`，而跑 LoRA 必须实时载入图像。
2. **复合主键匹配与 Hard Fail 门禁**：新的在线数据集必须通过 `(mpp_id, patient, patch_stem)` 复合主键匹配干净标签，防范重名冲突。

### 3.3 💡 数据规模扩增对“过拟合”问题的预期影响（待验证假设）
在旧版的 3 患者实验中，由于生物学多样性极低，模型很容易利用 LoRA 强大的拟合能力去“死记硬背”这 3 个人的专属批次效应。

**当前 MPP2 数据的有利变化（6 患者）与评估原则：**
- **待验证假设**：MPP2 引入了 **6 例患者**（HYZ, JFX, LMZ, TGC, XSL, ZHZ）作为训练基础。数据方差的增大可能对抑制过拟合起到积极作用。**但这一预测属于待验证假设**。
- **评估原则**：同患者的空间 block 验证（internal val）并不等同于跨患者泛化。模型依然需要经过短 smoke 的严格验证与外部测试集（XZY）的最终客观评估。我们不会因该假设而预先放宽 epoch 限制。
- **Batch 采样观测**：在 P0 默认 `batch_size=1`（或小物理 batch）下，单 batch 无法同时包含多患者图像，因此 shuffle/梯度累积不等于患者均衡。我们应当在训练中统计并输出患者采样占比，后续若引入患者均衡 sampler 必须作为独立消融项。

---

## 4. MPP2 迁移的具体方案设计（修正优化版）

### 4.1 构建基于 Manifest 的在线图像加载 Dataset

```python
import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torch
import sys

# 引入项目路径注册器以解析 mpp_data_root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from path_registry import mpp_data_root  # 动态解析数据根目录，而非硬编码

class OnlineManifestMPPDataset(Dataset):
    """MPP2 在线图像微调 Dataset (消费已划分的 split_manifest)"""
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        mpp_id: int,                # 当前处理的 MPP 组 ID (如 2)
        split: str,                 # "train" 或 "internal_val"
        labels_csv: str,            # 经过修复的、无污染的已验收标签路径
        transform,                  # 对应 UNI2-h 模型配置的图像预处理转换
        target_cols=None,
    ):
        self.transform = transform
        
        # 1. 严格原样消费 manifest 的划分，禁止重新进行随机切分
        self.sub_manifest = manifest_df[manifest_df["split"] == split].copy()
        if self.sub_manifest.empty:
            raise ValueError(f"指定划分 split={split} 的样本为空，请检查 manifest_df！")
            
        # 2. 读取标签数据，统一使用唯一标识符 (mpp_id, patient, patch_stem) 建立映射，防止跨患者同名坐标覆盖
        self.label_map = {}
        if not os.path.exists(labels_csv):
            raise FileNotFoundError(f"未找到指定的已验收标签文件: {labels_csv}")
            
        lw = pd.read_csv(labels_csv)
        first_col = lw.columns[0] # 通常是 barcode 或 filename
        if target_cols is None:
            skip = {first_col, "x", "y", "spot", "id", "barcode", "filename"}
            target_cols = [c for c in lw.select_dtypes(include=["number"]).columns if c not in skip]
        
        for _, row in lw.iterrows():
            stem = os.path.splitext(os.path.basename(str(row[first_col])))[0]
            # 假定标签 CSV 中包含或可推断 patient 字段。如缺失，必须严格核对
            patient = str(row.get("patient", ""))
            if not patient:
                # 若无直接 patient 列，尝试从首列路径或另行解析，若解析失败则必须 hard fail
                raise ValueError("标签数据中缺少 patient 字段，无法建立无歧义映射！")
            
            key = (int(mpp_id), patient, stem)
            if key in self.label_map:
                raise ValueError(f"检测到重复的复合主键: {key}，数据存在污染风险，直接硬拦截！")
            self.label_map[key] = row[target_cols].values.astype("float32")
        
        self.target_cols = target_cols
        
        # 3. 校验并收集样本路径，不接受任何隐式扩展名 fallback
        self.samples = []
        for _, row in self.sub_manifest.iterrows():
            stem = str(row["patch_stem"])
            patient = str(row["patient"])
            
            # 根据规范构造绝对路径：<mpp_root>/<mpp_id>/<patient>/patch_images/<stem>.png
            img_path = os.path.join(mpp_data_root, str(mpp_id), patient, "patch_images", f"{stem}.png")
            
            key = (int(mpp_id), patient, stem)
            
            # 检查图像与标签完整性，缺失或不匹配一律直接抛出异常 (Hard Fail)，杜绝静默跳过
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"未找到图像文件: {img_path}，为防止静默漏训直接 hard fail！")
            if key not in self.label_map:
                raise KeyError(f"图像存在但未在标签中找到对应复合主键 {key}，直接 hard fail！")
                
            self.samples.append((img_path, torch.tensor(self.label_map[key], dtype=torch.float32)))
            
        print(f"  [Online DS] mpp_id={mpp_id} | split={split} | 成功加载 {len(self.samples)} 个样本（严格匹配且无缺失）")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target
```

### 4.2 新训练脚本 `train_mpp_uni2h_lora.py` 优化架构

```python
# train_mpp_uni2h_lora.py — 伪代码框架

def main():
    # ── Step 1: 严格的多维度状态与资产防污染校验 ──
    from scripts.pfmval_state import validate_state, verify_mpp_repair_server_assets
    
    # 完整复用三层校验逻辑，明确 host_scope
    state_report = validate_state(
        Path("."), 
        strict=True, 
        task="training", 
        host_scope="server"
    )
    if not state_report.ok:
        raise RuntimeError(f"AUTHORITATIVE BLOCK: 数据状态或验证有泄漏风险！原因: {state_report.reason}")
        
    # 服务器数据资产与固定校验
    verify_mpp_repair_server_assets(mpp_id=2, strict=True)

    # ── Step 2: 加载模型与运行时探测显存优化 ──
    # 不在方案中硬写 ViT-Large，以实际加载模型 snapshot/config 为准（如 dim=1536, 24 blocks）
    backbone, transform, model_config = load_uni2h_backbone(device=device)
    
    # 预检显存占用，仅在支持且确有需要时才在运行时启用，并记录速度代价
    if is_vram_constrained() and hasattr(backbone, "set_grad_checkpointing"):
        backbone.set_grad_checkpointing(True)
        print("已启用 Gradient Checkpointing (会带来一定的速度开销)")
    
    # ── Step 3: 手工注入 LoRA 包装（P0 锁定 qkv+proj） ──
    from lora_utils import inject_lora_to_backbone
    # P0 固定使用 r=8 和注意力层 qkv+proj，后续 rank 或 MLP 消融需另行批准
    inject_lora_to_backbone(
        backbone,
        target_blocks=list(range(24)),
        rank=8,  # smoke/P0 锁定 r=8
        alpha=16.0,
        target_modules={"qkv", "proj"}
    )
    # MPP2 P0 仅采用 Stage 1 LoRA，Stage 2/3 多阶段机制本轮不进入实验矩阵
    model = OnlineMPPModel(backbone=backbone, out_dim=30).to(device)
    
    # ── Step 4: 带有强正则化的参数分组优化器 ──
    # P0 先使用 1e-4 weight_decay 保持与历史同等可比性
    optimizer = build_optimizer_with_groups(
        model, 
        lr=args.lr,
        lora_lr=args.lora_lr,
        weight_decay=1e-4
    )
    
    # ── Step 5: 按 MPP2 规范加载数据与训练循环 ──
    # 原样消费由空间 block 划分的 split_manifest.csv
    manifest_df = pd.read_csv("mpp_standard_splits/group_2/split_manifest.csv")
    
    train_dataset = OnlineManifestMPPDataset(
        manifest_df=manifest_df, 
        mpp_id=2,
        split="train", 
        labels_csv="mpp_standard_splits/group_2/labels/train/..._ssGSEA_zscore.csv",
        transform=transform
    )
    
    # 物理 batch=1 时输出患者采样占比，依靠 internal_val 的 val_loss 进行严格 smoke 早停控制
    ...
```

---

## 5. 推荐补充的专业知识

为了更全面地理解并在实践中修改调试此方案，建议重点补充以下几个维度的知识：

### 📘 A. 参数高效微调 (Parameter-Efficient Fine-Tuning, PEFT)
1. **数学原理**：阅读 LoRA 经典论文《LoRA: Low-Rank Adaptation of Large Language Models》。重点理解低秩矩阵乘法（$B_{d \times r} \cdot A_{r \times d}$）为何能大幅削减显存，并且在前向传播中是如何通过加法与原冻结权重合并的。
2. **初始化艺术**：学习为什么 $B$ 矩阵初始化为零，$A$ 矩阵初始化为高斯/均匀分布，以及缩放因子 $\frac{\alpha}{r}$ 对梯度和学习率的影响。
3. **参数再参数化（Reparameterization）**：理解为什么在模型推理（inference）前将 $B \cdot A \times \text{scale}$ 加回 $W_0$ 可以实现**零额外推理延迟**。

### 📙 B. Transformer 结构与 Timms 库进阶
1. **注意力机制原理**：理解 UNI2-h 中注意力层的 `qkv` 映射，以及其与 Feed-Forward (MLP) 层在视觉特征提炼和全局交互上的差异。
2. **LoRA 目标层选择**：深入学习在不同应用场景下，改变 Attention 层与 FFN 层权重对泛化能力及参数复杂度的影响。

###  green 📗 C. 跨患者迁移与数据治理防线
1. **批次效应与空间 block 划分**：在空间转录组与病理图像融合分析中，学习固定空间 block 划分相对于简单随机 Patch 划分的严谨性，以及为什么该验证方式不能直接等同于跨患者泛化。
2. **绝对防信息泄露（Data Leakage Prevention）**：理解为什么 Z-score 归一化的均值和标准差**必须且只能**在训练样本集上计算，任何测试集先频特征泄露（Look-ahead leakage）都会导致评估失真。
