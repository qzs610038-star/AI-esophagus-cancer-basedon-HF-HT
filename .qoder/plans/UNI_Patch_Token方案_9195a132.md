# HisToGene-UNI 方案B：保留 UNI Patch Token 空间信息

## 背景与问题

当前方案A（HisToGene-UNI）使用 UNI2-h 的全局平均池化输出（1个1536维向量），**丢失了patch内部的空间细节**。UNI2-h实际输出 `[1, 265, 1536]`（1 CLS + 256 patch tokens + 8 register tokens），包含丰富的局部空间信息。

## 核心约束

- **受保护目录**：`histogene/`、`egnv1/`、`egnv2/`、`uni2h/` 不可修改
- **原则**：只新增文件，不修改现有文件
- **向后兼容**：现有方案A的缓存和训练流程不受影响

## 技术方案

### 方案选择：方案B（轻量Transformer重编码）

保留UNI2-h输出的**全部265个token**，通过1-2层轻量Transformer重新编码空间关系，最终聚合为单向量输入回归头。

**选择理由**：
- 实现复杂度适中（vs 方案C的空间注意力图构建）
- 与现有HisToGene架构风格一致（都用Transformer）
- 参数量可控（新增约50-100K参数）

### 存储成本分析

| 维度 | 方案A（当前） | 方案B（全token） | 方案B-Lite（Top-K） |
|------|------------|----------------|-------------------|
| 单文件大小 | ~7 KB | ~1.6 MB | ~100 KB |
| HYZ15040全量 | ~80 MB | ~16 GB | ~1 GB |
| I/O影响 | 无 | 严重瓶颈 | 可接受 |

**推荐方案B-Lite**：只保留 CLS token + 前64个patch token（8x8网格中心区域），存储增长约15倍（vs 200倍），兼顾信息量和存储成本。

---

## 实施计划（4个新文件）

### Task 1：新建特征提取脚本 `extract_uni_tokens.py`

在项目根目录新建，功能：
- 调用 `backbone.forward_features(x)` 获取完整 `[1, 265, 1536]` 输出
- 支持两种模式：`--mode full`（保留全部265 token）或 `--mode lite`（CLS + Top-64 patch token = 65个token）
- 保存为 `.pt` 文件到新的缓存目录 `uni2h_cache_tokens/{patient}/{split}/`
- 复用 `uni2h/uni2h_utils.py` 中的 `load_uni2h_backbone()` 加载模型
- 三个患者依次提取

**核心代码逻辑**：
```python
with torch.no_grad():
    all_tokens = backbone.forward_features(x)  # [1, 265, 1536]
    if mode == 'lite':
        # CLS(index 0) + 前64个patch token(index 1:65)
        tokens = all_tokens[:, :65, :]  # [1, 65, 1536]
    else:
        tokens = all_tokens  # [1, 265, 1536]
    torch.save(tokens.squeeze(0).cpu(), cache_path)
```

### Task 2：新建数据集类 `dataset_uni_tokens.py`

在项目根目录新建，功能：
- 继承/参考 `histogene/dataset_uni.py` 的接口设计
- 加载 `uni2h_cache_tokens/` 下的2D tensor（`[num_tokens, 1536]`）
- 保留原有的坐标处理逻辑（`pos_x`, `pos_y`）
- 支持 `from_multiple_patients()` 用于跨患者训练

**返回格式**：
```python
def __getitem__(self, idx):
    return {
        'tokens': tensor([num_tokens, 1536]),  # 2D token序列
        'pos_x': long_tensor,
        'pos_y': long_tensor,
        'targets': tensor([n_targets])
    }
```

### Task 3：新建模型 `model_uni_tokens.py`

在项目根目录新建，包含：

**3.1 LightweightTokenEncoder**：
```python
class LightweightTokenEncoder(nn.Module):
    """将 [B, num_tokens, 1536] 编码为 [B, dim]"""
    def __init__(self, embed_dim=1536, dim=1024, n_heads=8, n_layers=1, dropout=0.3):
        # 1层Transformer编码器
        # 可学习的CLS查询token用于聚合
        # 全局平均池化作为备选聚合方式
```

**3.2 HisToGeneUNITokens（完整模型）**：
```python
class HisToGeneUNITokens(nn.Module):
    """在HisToGeneUNI基础上增加token编码能力"""
    def __init__(self, feature_dim=1536, dim=1024, n_pos=128,
                 n_targets=30, mlp_dim=2048, dropout=0.3,
                 n_tokens=65, n_encoder_layers=1):
        # token_encoder: LightweightTokenEncoder
        # 其余结构与HisToGeneUNI一致：
        #   proj层、x/y坐标Embedding、回归头MLP
    
    def forward(self, tokens, pos_x, pos_y):
        # tokens: [B, num_tokens, 1536]
        encoded = self.token_encoder(tokens)  # [B, 1536]
        x = self.proj(encoded)                # [B, dim]
        x = x + self.x_embed(pos_x) + self.y_embed(pos_y)
        return self.head(x)                   # [B, n_targets]
```

**新增参数量估算**：
- 1层Transformer编码器（d_model=1536, nhead=8）：约 9.4M 参数
- 如果用更轻量的设计（d_model=512 + 投影）：约 2-3M 参数
- 总模型：约 6-7M 参数（vs 方案A的4M）

### Task 4：新建训练脚本 `train_histogene_uni_tokens.py`

在项目根目录新建，参考 `train_cross_patient_histogene_uni.py` 结构：
- 支持 `--patient` 单患者模式和 `--cross_patient` 跨患者模式
- 导入新的 `HisToGeneUNITokens` 模型和 `HisToGeneUNITokensDataset`
- 集成已有的正则化参数（dropout, lr, weight_decay, label_noise, gradient_clip等）
- 可视化输出遵循时间戳隔离 + 逐通路PCC双输出规范
- dataset_name 使用 `{patient}_UNI_tokens` 或 `CrossPatient_..._UNI_tokens`

---

## 执行策略：渐进式验证

**阶段1 - 快速可行性验证**（推荐先做）：
1. 提取 HYZ15040 的 token 缓存（lite模式，65个token）
2. 单患者训练 3-5 个 epoch
3. 对比 Val PCC 是否超过方案A基线（0.577）
4. 如果无显著提升（<2%），则暂停后续投入

**阶段2 - 全量训练**（阶段1有效后）：
1. 提取三个患者全量 token 缓存
2. 三数据集分别单患者训练
3. 跨患者泛化训练

---

## 风险与备注

1. **性能不确定**：UNI2-h 的全局平均池化可能已充分捕获信息，额外token可能收益有限
2. **存储压力**：即使lite模式，三患者缓存总量约 3GB（可接受）
3. **训练速度**：token序列输入Transformer编码器会增加约30-50%训练时间
4. **受保护目录**：所有4个新文件均在项目根目录，不触及任何受保护目录
