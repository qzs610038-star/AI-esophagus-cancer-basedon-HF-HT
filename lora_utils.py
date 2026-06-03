"""
lora_utils.py — 手工 LoRA 实现（不依赖 PEFT 库）

为 UNI2-H ViT backbone 提供低秩适配注入/移除/参数管理工具。
UNI2-H: 24 层 ViT Block，每层 block.attn.qkv (Linear 1536→4608) 和
block.attn.proj (Linear 1536→1536) 是 LoRA 注入目标。

用法:
    from lora_utils import LoRALinear, inject_lora_to_backbone, get_lora_parameters

    backbone, transform, _ = load_uni2h_backbone(device=device)
    inject_lora_to_backbone(backbone, target_blocks=range(24), rank=8, alpha=16)
    lora_params = get_lora_parameters(backbone)  # 传给优化器
"""

from __future__ import annotations

import math
from typing import List, Optional, Set

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════
# LoRALinear — 低秩适配包装器
# ═══════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """对 nn.Linear 的低秩适配包装。

    forward = original(x) + lora_B(lora_A(x)) * (alpha / rank)

    仅 lora_A / lora_B 参与梯度更新；original.weight 保持冻结。
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        in_features = original.in_features
        out_features = original.out_features

        # ── 保留原始层为冻结子模块 ──
        self.original = original
        for p in self.original.parameters():
            p.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank if rank > 0 else 1.0

        # ── LoRA 低秩矩阵 ──
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # 初始化: Kaiming Uniform A, Zero B（保证初始状态 ΔW=0）
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scale
        return base + lora_out

    def merge_to_original(self) -> None:
        """将 LoRA 权重合并到原始权重中（推理优化，merge 后无法恢复）。"""
        delta_w = (self.lora_B.weight @ self.lora_A.weight) * self.scale
        self.original.weight.data += delta_w
        # 清零 LoRA 矩阵
        nn.init.zeros_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)


# ═══════════════════════════════════════════════════════════════════
# 注入 / 移除工具
# ═══════════════════════════════════════════════════════════════════

def inject_lora_to_block(
    block: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Set[str] = {"qkv", "proj"},
    dropout: float = 0.0,
) -> List[LoRALinear]:
    """对单个 ViT Block 的 Attention 注入 LoRA。

    Args:
        block: timm vision_transformer.Block 实例
        rank: LoRA 秩
        alpha: LoRA 缩放系数
        target_modules: 注入目标集合，可选 {"qkv", "proj", "fc1", "fc2"}
        dropout: LoRA dropout 概率

    Returns:
        新创建的 LoRALinear 实例列表
    """
    created: List[LoRALinear] = []

    if "qkv" in target_modules and hasattr(block.attn, "qkv"):
        block.attn.qkv = LoRALinear(block.attn.qkv, rank=rank, alpha=alpha, dropout=dropout)
        created.append(block.attn.qkv)

    if "proj" in target_modules and hasattr(block.attn, "proj"):
        block.attn.proj = LoRALinear(block.attn.proj, rank=rank, alpha=alpha, dropout=dropout)
        created.append(block.attn.proj)

    if "fc1" in target_modules and hasattr(block.mlp, "fc1"):
        block.mlp.fc1 = LoRALinear(block.mlp.fc1, rank=rank, alpha=alpha, dropout=dropout)
        created.append(block.mlp.fc1)

    if "fc2" in target_modules and hasattr(block.mlp, "fc2"):
        block.mlp.fc2 = LoRALinear(block.mlp.fc2, rank=rank, alpha=alpha, dropout=dropout)
        created.append(block.mlp.fc2)

    return created


def inject_lora_to_backbone(
    model: nn.Module,
    target_blocks: List[int],
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Set[str] = {"qkv", "proj"},
    dropout: float = 0.0,
) -> List[LoRALinear]:
    """对 UNI2-H backbone 的指定层批量注入 LoRA。

    Args:
        model: timm ViT 模型 (需有 model.blocks[i] 的 Sequential)
        target_blocks: 要注入的 block 索引列表，如 [20,21,22,23]
        rank: LoRA 秩
        alpha: LoRA 缩放系数
        target_modules: {"qkv", "proj"} 或加 {"fc1", "fc2"}
        dropout: LoRA dropout

    Returns:
        所有创建的 LoRALinear 实例

    Example:
        # 注入全部 24 层
        inject_lora_to_backbone(backbone, list(range(24)), rank=8)
        # 仅最后 4 层
        inject_lora_to_backbone(backbone, [20, 21, 22, 23], rank=4)
    """
    all_lora: List[LoRALinear] = []
    for idx in target_blocks:
        if idx >= len(model.blocks):
            raise IndexError(
                f"Block index {idx} out of range (model has {len(model.blocks)} blocks)"
            )
        created = inject_lora_to_block(
            model.blocks[idx],
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
            dropout=dropout,
        )
        all_lora.extend(created)
    return all_lora


def merge_lora_before_unfreeze(
    model: nn.Module,
    block_indices: List[int],
    target_modules: Set[str] = {"qkv", "proj"},
) -> int:
    """Stage 迁移关键步骤：将指定层的 LoRA 权重合并到原始权重。

    调用时机：从 Stage N checkpoint 加载权重后，移除 LoRA 解冻前。
    确保被解冻的 blocks 不会丢失前一个 Stage 学到的 LoRA 知识。

    Args:
        model: timm ViT 模型（已加载前一 Stage 权重）
        block_indices: 即将解冻的 block 索引
        target_modules: 要 merge 的 LoRA 模块

    Returns:
        成功 merge 的 LoRALinear 数量
    """
    merged_count = 0
    for idx in block_indices:
        if idx >= len(model.blocks):
            continue
        block = model.blocks[idx]

        if "qkv" in target_modules and isinstance(block.attn.qkv, LoRALinear):
            block.attn.qkv.merge_to_original()
            merged_count += 1

        if "proj" in target_modules and isinstance(block.attn.proj, LoRALinear):
            block.attn.proj.merge_to_original()
            merged_count += 1

        if "fc1" in target_modules and hasattr(block.mlp, "fc1"):
            if isinstance(block.mlp.fc1, LoRALinear):
                block.mlp.fc1.merge_to_original()
                merged_count += 1

        if "fc2" in target_modules and hasattr(block.mlp, "fc2"):
            if isinstance(block.mlp.fc2, LoRALinear):
                block.mlp.fc2.merge_to_original()
                merged_count += 1

    return merged_count


def remove_lora_from_blocks(
    model: nn.Module,
    block_indices: List[int],
    target_modules: Set[str] = {"qkv", "proj", "fc1", "fc2"},
) -> None:
    """从指定层移除 LoRA 包装，恢复原始 nn.Linear。

    调用时机：merge_lora_before_unfreeze() 之后。
    注意：直接调用（不先 merge）会丢弃 LoRA 学习成果。

    Args:
        model: timm ViT 模型
        block_indices: 要移除 LoRA 的 block 索引
        target_modules: 要恢复的模块名集合
    """
    for idx in block_indices:
        if idx >= len(model.blocks):
            continue
        block = model.blocks[idx]

        # attn.qkv
        if "qkv" in target_modules and isinstance(block.attn.qkv, LoRALinear):
            block.attn.qkv = block.attn.qkv.original

        # attn.proj
        if "proj" in target_modules and isinstance(block.attn.proj, LoRALinear):
            block.attn.proj = block.attn.proj.original

        # mlp.fc1
        if "fc1" in target_modules and hasattr(block.mlp, "fc1"):
            if isinstance(block.mlp.fc1, LoRALinear):
                block.mlp.fc1 = block.mlp.fc1.original

        # mlp.fc2
        if "fc2" in target_modules and hasattr(block.mlp, "fc2"):
            if isinstance(block.mlp.fc2, LoRALinear):
                block.mlp.fc2 = block.mlp.fc2.original


def unfreeze_blocks(
    model: nn.Module,
    block_indices: List[int],
    unfreeze_attn: bool = True,
    unfreeze_mlp: bool = True,
    unfreeze_norm: bool = True,
) -> int:
    """解冻指定 ViT Block 的所有（或部分）参数。

    Args:
        model: timm ViT 模型
        block_indices: 要解冻的 block 索引
        unfreeze_attn: 解冻 Attention 参数
        unfreeze_mlp: 解冻 MLP 参数
        unfreeze_norm: 解冻 LayerNorm + LayerScale

    Returns:
        解冻的参数总数
    """
    count = 0
    for idx in block_indices:
        if idx >= len(model.blocks):
            continue
        block = model.blocks[idx]

        for name, param in block.named_parameters():
            is_attn = name.startswith("attn.")
            is_mlp = name.startswith("mlp.")
            is_norm = name.startswith("norm") or name.startswith("ls")

            if (is_attn and unfreeze_attn) or (is_mlp and unfreeze_mlp) or (is_norm and unfreeze_norm):
                param.requires_grad = True
                count += param.numel()

    return count


# ═══════════════════════════════════════════════════════════════════
# 参数收集
# ═══════════════════════════════════════════════════════════════════

def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """从模型中收集所有 LoRA 参数（lora_A.weight, lora_B.weight）。

    用于优化器参数分组，确保仅 LoRA 参数被训练。
    """
    params: List[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend(list(module.lora_A.parameters()))
            params.extend(list(module.lora_B.parameters()))
    return params


def count_lora_parameters(model: nn.Module) -> int:
    """统计模型中 LoRA 参数总量。"""
    return sum(p.numel() for p in get_lora_parameters(model))


def get_trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    """收集所有 requires_grad=True 的参数（含 LoRA 和非 LoRA）。"""
    return [p for p in model.parameters() if p.requires_grad]


def freeze_all_parameters(model: nn.Module) -> None:
    """冻结模型所有参数（除已存在的 LoRA 外）。"""
    for name, param in model.named_parameters():
        if not name.endswith(".lora_A.weight") and not name.endswith(".lora_B.weight"):
            param.requires_grad = False


# ═══════════════════════════════════════════════════════════════════
# 辅助：模型状态报告
# ═══════════════════════════════════════════════════════════════════

def print_trainable_summary(model: nn.Module, prefix: str = "") -> None:
    """打印模型可训练参数摘要，按来源分组。"""
    lora_params = count_lora_parameters(model)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_all = sum(p.numel() for p in model.parameters())
    unfrozen_non_lora = total_trainable - lora_params

    print(f"{prefix}参数概览:")
    print(f"  总参数:        {total_all / 1e6:.2f}M")
    print(f"  可训练:        {total_trainable / 1e6:.2f}M ({total_trainable / total_all * 100:.1f}%)")
    print(f"    其中 LoRA:   {lora_params / 1e6:.2f}M")
    print(f"    其中 非LoRA: {unfrozen_non_lora / 1e6:.2f}M")


# ═══════════════════════════════════════════════════════════════════
# 阶段性训练构建器
# ═══════════════════════════════════════════════════════════════════

def configure_stage1_lora(
    backbone: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_blocks: Optional[List[int]] = None,
) -> List[LoRALinear]:
    """Stage 1: 指定层注入 LoRA (qkv + proj)，backbone 其余参数冻结。

    Args:
        backbone: UNI2-H ViT 模型
        rank: LoRA 秩
        alpha: LoRA 缩放系数
        dropout: LoRA dropout 概率
        target_blocks: 要注入的 block 索引列表，默认全部 24 层

    Returns:
        LoRALinear 列表（用于后续 stage 的 remove 操作引用）
    """
    if target_blocks is None:
        target_blocks = list(range(24))
    return inject_lora_to_backbone(
        backbone,
        target_blocks=target_blocks,
        rank=rank,
        alpha=alpha,
        target_modules={"qkv", "proj"},
        dropout=dropout,
    )


def configure_stage2_unfreeze_last2(
    backbone: nn.Module,
) -> int:
    """Stage 2: 移除 blocks 22-23 的 LoRA，解冻这些层全部参数。

    Prerequisites: 必须先运行 configure_stage1_lora()

    Returns:
        解冻的参数数量
    """
    remove_lora_from_blocks(backbone, [22, 23], target_modules={"qkv", "proj"})
    unfrozen = unfreeze_blocks(backbone, [22, 23], unfreeze_attn=True, unfreeze_mlp=True, unfreeze_norm=True)
    return unfrozen


def configure_stage3_unfreeze_last4(
    backbone: nn.Module,
) -> int:
    """Stage 3: 移除 blocks 20-23 的 LoRA，解冻这些层全部参数。

    Prerequisites: 必须先运行 configure_stage2_unfreeze_last2()

    Returns:
        解冻的参数数量
    """
    remove_lora_from_blocks(backbone, [20, 21], target_modules={"qkv", "proj"})
    unfrozen = unfreeze_blocks(backbone, [20, 21], unfreeze_attn=True, unfreeze_mlp=True, unfreeze_norm=True)
    return unfrozen


# ═══════════════════════════════════════════════════════════════════
# Checkpoint 加载辅助：跨结构权重转移 + 阶段结构重建
# ═══════════════════════════════════════════════════════════════════

import re
from typing import Dict as _Dict


def transfer_ckpt_plain_to_lora_original(
    model: nn.Module,
    ckpt_state_dict: _Dict[str, torch.Tensor],
) -> int:
    """将 checkpoint 中 plain Linear 权重手动写入模型 LoRALinear.original。

    场景：当 checkpoint 中某层已移除 LoRA（为 plain Linear），
    但当前模型因刚注入 LoRA 而使用 LoRALinear 包装时，
    strict=False 无法自动映射 key 名不同的参数。

    此函数扫描 ckpt 中形如 ``backbone.blocks.N.attn.{qkv,proj}.{weight,bias}``
    的 key，若模型中对应位置为 LoRALinear，则将权重复制到 ``.original`` 子模块。

    Args:
        model: 完整的下游模型（如 OnlineCLSModel / OnlineTokenModel），
               其 backbone 存储在 ``model.backbone``
        ckpt_state_dict: checkpoint 中的 state_dict

    Returns:
        成功转移的参数数量（每 weight/bias 计 1）
    """
    transferred = 0
    backbone = getattr(model, 'backbone', None)
    if backbone is None:
        return transferred

    # 编译正则：匹配 backbone.blocks.{idx}.attn.{module}.{param}
    _PATTERN = re.compile(
        r'^backbone\.blocks\.(\d+)\.attn\.(qkv|proj)\.(weight|bias)$'
    )

    for key, tensor in ckpt_state_dict.items():
        m = _PATTERN.match(key)
        if not m:
            continue

        block_idx = int(m.group(1))
        mod_name = m.group(2)    # "qkv" or "proj"
        param_name = m.group(3)  # "weight" or "bias"

        if block_idx >= len(backbone.blocks):
            continue

        block = backbone.blocks[block_idx]
        target_attr = getattr(block.attn, mod_name, None)

        # 仅当模型中该位置是 LoRALinear 时才需要转移
        if not isinstance(target_attr, LoRALinear):
            continue

        # 写入 .original.{weight|bias}
        original_param = getattr(target_attr.original, param_name, None)
        if original_param is not None and original_param.shape == tensor.shape:
            original_param.data.copy_(tensor)
            transferred += 1

    return transferred


def apply_stage_structure(
    backbone: nn.Module,
    mode: str,
) -> str:
    """根据 mode 重建 backbone 的阶段结构（移除 LoRA + 解冻指定层）。

    用于同阶段 resume 时，将刚注入 LoRA 的 backbone 恢复到该阶段应有的结构。

    Args:
        backbone: 已注入 LoRA 的 UNI2-H ViT 模型
        mode: 阶段名 (``"lora"`` / ``"stage2"`` / ``"stage3"``)

    Returns:
        描述已执行操作的字符串
    """
    if mode == "lora":
        # Stage 1: 全层 LoRA，无需移除
        return "Stage 1 (lora): 保持全层 LoRA，无结构变更"

    elif mode == "stage2":
        # Stage 2: blocks 0-21 保留 LoRA；blocks 22-23 移除 LoRA + 解冻
        remove_lora_from_blocks(backbone, [22, 23], target_modules={"qkv", "proj"})
        n_unfrozen = unfreeze_blocks(backbone, [22, 23],
                                     unfreeze_attn=True, unfreeze_mlp=True, unfreeze_norm=True)
        return f"Stage 2: 移除 blocks 22-23 LoRA + 解冻 {n_unfrozen:,} 参数"

    elif mode == "stage3":
        # Stage 3: blocks 0-19 保留 LoRA；blocks 20-23 移除 LoRA + 解冻
        remove_lora_from_blocks(backbone, [20, 21, 22, 23], target_modules={"qkv", "proj"})
        n_unfrozen = unfreeze_blocks(backbone, [20, 21, 22, 23],
                                     unfreeze_attn=True, unfreeze_mlp=True, unfreeze_norm=True)
        return f"Stage 3: 移除 blocks 20-23 LoRA + 解冻 {n_unfrozen:,} 参数"

    else:
        return f"未知 mode '{mode}'，未执行结构变更"


# ═══════════════════════════════════════════════════════════════════
# 自检
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("LoRA 工具模块自检")
    print("=" * 60)

    # 测试 LoRALinear 基本功能
    print("\n1. LoRALinear 单元测试 ...")
    original = nn.Linear(16, 32)
    lora = LoRALinear(original, rank=4, alpha=8.0)

    x = torch.randn(2, 16)
    out = lora(x)
    assert out.shape == (2, 32), f"Expected (2,32), got {out.shape}"

    # 验证原权重冻结
    assert not any(p.requires_grad for p in lora.original.parameters()), "原始权重应冻结"
    assert lora.lora_A.weight.requires_grad, "lora_A 应可训"
    assert lora.lora_B.weight.requires_grad, "lora_B 应可训"
    print("   ✓ 前向/反向/冻结逻辑正常")

    # 测试 merge
    out_before = lora(x)
    lora.merge_to_original()
    out_after = lora(x)
    # merge 后 A/B 权重为 0，输出应等于 original(x)
    assert torch.allclose(out_before, out_after, atol=1e-6), "merge 应保持前向不变"
    print("   ✓ merge_to_original 通过")

    print("\n   LoRA 工具模块自检全部通过 ✓")
    print("=" * 60)
