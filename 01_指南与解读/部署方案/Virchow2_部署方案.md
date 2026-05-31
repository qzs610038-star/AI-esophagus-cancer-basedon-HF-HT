# Virchow2 部署方案

> **模型**：Virchow2 — A Self-Supervised Mixed Magnification ViT-H/14 for Pathology  
> **论文**：Vorontsov E, et al. *Nature Medicine*, 2025  
> **HuggingFace**：[paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2)  
> **许可**：CC-BY-NC-ND 4.0（学术非商业使用）  
> **创建日期**：2026-05-19

---

## 1. 模型参数速查

| 属性 | Virchow2 | UNI2-h（当前） |
|------|----------|---------------|
| 架构 | ViT-H/14 | ViT-L/14 |
| 参数量 | 632M | 307M |
| Token 维度 | **1280** | 1536 |
| Token 数量 | 261（1 CLS + 4 reg + 256 patch） | 265（1 CLS + 8 reg + 256 patch） |
| 输入尺寸 | 224×224 | 224×224 |
| Patch 大小 | 14×14 | 14×14 |
| 训练数据量 | 3.1M WSIs | ~100K WSIs |
| 多倍率 | 5×/10×/20×/40× | 20× |
| 访问要求 | **需申请 HF 授权**（机构邮箱） | 公开 |

**重要**：Virchow2 的 per-token 维度是 **1280**（不是 2560）。2560 是 CLS(1280) + mean(patch_tokens)(1280) 拼接后的全局 embedding，不适用于 token 序列模式。

## 2. 前置要求

### 2.1 HuggingFace 授权（必须）

1. 访问 https://huggingface.co/paige-ai/Virchow2
2. 点击 "Request Access"，使用**机构/学术邮箱**（非个人邮箱）
3. 等待 Paige AI 批准（通常 1-2 个工作日）

### 2.2 HuggingFace CLI 登录

```bash
"D:\miniconda\Scripts\conda.exe" run -n base pip install huggingface_hub
huggingface-cli login
# 输入你的 HF token（从 https://huggingface.co/settings/tokens 获取）
```

---

## 3. 安装步骤

### 3.1 Python 环境

使用现有 HisToGene 环境（Python 3.13），Virchow2 仅需 `timm` 加载，不引入新依赖。

```bash
# 安装/升级 timm（需要 >= 0.9.11）
"C:\Program Files\Python313\python.exe" -m pip install timm --upgrade
# 无需额外安装 virchow 包，timm 直接从 HF Hub 加载
```

### 3.2 依赖清单

| 包 | 最低版本 | 说明 |
|----|---------|------|
| `timm` | >= 0.9.11 | ViT 模型加载 |
| `torch` | >= 2.0 | 已安装 2.6.0+cu118 |
| `huggingface_hub` | 最新 | 已安装，下载权重 |
| `Pillow` | 已安装 | 图像预处理 |

### 3.3 磁盘空间预估

| 项目 | 空间 | 位置 |
|------|------|------|
| Virchow2 权重缓存 | ~2.5 GB | `D:\AI空间转录病理研究\PFMval_new\hf_cache\` |
| Token 缓存 lite（3患者） | ~7 GB | `D:\AI空间转录病理研究\PFMval_new\virchow2_cache_tokens\` |
| Token 缓存 full（3患者） | ~28 GB | 同上 |

**所有文件均在 D 盘项目目录内，不写 C 盘。**

HF 缓存目录设置：
```bash
export HF_HOME="D:\AI空间转录病理研究\PFMval_new\hf_cache"
# 或在 Python 中：
import os; os.environ["HF_HOME"] = "D:/AI空间转录病理研究/PFMval_new/hf_cache"
```

---

## 4. 代码文件创建

### 4.1 文件清单

```
PFMval_new/
├── virchow2/
│   └── virchow2_utils.py          # Backbone 加载 + 验证
├── extract_virchow2_tokens.py     # Token 特征提取
├── train_histogene_virchow2_tokens.py  # 训练脚本
├── virchow2_cache_tokens/         # Token 缓存（自动创建）
│   ├── HYZ15040/{train,val}/
│   ├── JFX0729/{train,val}/
│   └── LMZ12939/{train,val}/
```

**不修改的文件**：
- `model_uni_tokens.py` — 直接 `import HisToGeneUNITokens` 并传 `feature_dim=1280`
- `dataset_uni_tokens.py` — 修改添加 `feature_dim` 参数（默认1536不变）
- `histogene/`、`egnv1/`、`egnv2/` — 受保护不动

### 4.2 virchow2/virchow2_utils.py（~80行）

镜像 `uni2h/uni2h_utils.py`，关键差异：

```python
import os
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

# 强制 HF 缓存到 D 盘
os.environ.setdefault("HF_HOME", "D:/AI空间转录病理研究/PFMval_new/hf_cache")

VIRCHOW2_MODEL_ID = "paige-ai/Virchow2"
VIRCHOW2_FEATURE_DIM = 1280    # per-token 维度！
VIRCHOW2_NUM_TOKENS = 261      # 1 CLS + 4 reg + 256 patch
VIRCHOW2_LITE_TOKENS = 65      # CLS + first 64 patch (skip reg)


def load_virchow2_backbone(device=None):
    """
    加载 Virchow2 ViT-H/14 backbone，返回 (model, transform, feature_dim)。

    Returns:
        model: 冻结的 eval 模式 ViT 模型，调用 model(x) 返回 [B, 261, 1280]
        transform: torchvision 预处理管线
        feature_dim: 1280
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model(
        f"hf-hub:{VIRCHOW2_MODEL_ID}",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    model.to(device)
    model.eval()

    # 冻结所有参数
    for p in model.parameters():
        p.requires_grad = False

    # 从模型配置推导预处理
    data_config = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**data_config)

    return model, transform, VIRCHOW2_FEATURE_DIM
```

### 4.3 extract_virchow2_tokens.py（~200行）

与 `extract_uni_tokens.py` 几乎一致，差异：

| 项目 | UNI2-h | Virchow2 |
|------|--------|----------|
| 导入 | `from uni2h.uni2h_utils import load_uni2h_backbone` | `from virchow2.virchow2_utils import load_virchow2_backbone` |
| 输出目录 | `uni2h_cache_tokens` | `virchow2_cache_tokens` |
| 特征维度 | 1536 | 1280 |
| Token 总数 | 265 | 261 |
| Lite 策略 | `[:, :65]` — CLS+前64patch | `[:, [0] + list(range(5, 69))]` — CLS + 跳过4个寄存器取64个patch |
| Full 模式 | `[:, :265]` | `[:, :261]` |

**关键点**：Virchow2 的 token 布局是 CLS@0, registers@1-4, patches@5-260。Lite 模式应取 CLS + patches（跳过 registers），索引为 `[0] + list(range(5, 69))`。

同样支持 `--patient`、`--mode`（lite/full）、`--rebuild`、`--output_dir` 参数。PATIENT_PATHS 复用现有映射。

### 4.4 修改 dataset_uni_tokens.py（最小改动）

在 `HisToGeneUNITokensDataset.__init__` 添加 `feature_dim=1536` 参数：

```python
def __init__(self, patches_dir, feature_cache_dir, labels_csv,
             target_cols=None, n_pos=128, n_targets=30, coord_stats=None,
             feature_dim=1536, backbone_name="UNI2-h"):  # 新增参数
    self.feature_dim = feature_dim
    ...
    # 第126行断言改为：
    assert tokens.dim() == 2 and tokens.shape[1] == self.feature_dim, (
        f"Token特征维度不匹配: 期望 [{backbone_name}] [num_tokens, {self.feature_dim}], "
        f"实际 {tokens.shape}, stem={stem}"
    )
```

默认值保持 1536，现有 UNI 训练脚本无需任何修改。

### 4.5 train_histogene_virchow2_tokens.py（~600行）

基于 `train_histogene_uni_tokens_augmix.py` 精简版（先不加 AugMix/MixUp），修改点：

```python
# 路径配置
_TOKEN_CACHE_BASE = str(_PROJECT_ROOT / "virchow2_cache_tokens")
_FEATURE_DIM = 1280

# 模型创建
from model_uni_tokens import HisToGeneUNITokens  # 复用现有模型类！
model = HisToGeneUNITokens(
    feature_dim=1280,
    dim=1024,
    n_pos=128,
    n_targets=30,
    mlp_dim=2048,
    dropout=0.5,
    encoder_hidden_dim=512,
    n_encoder_layers=1,
    n_encoder_heads=8,
)

# 数据集创建
from dataset_uni_tokens import HisToGeneUNITokensDataset
dataset = HisToGeneUNITokensDataset(
    ...,
    feature_dim=1280,
    backbone_name="Virchow2",
)
```

训练逻辑（AMP、HuberLoss、ReduceLROnPlateau、早停、断点续训）完全复用。

### 4.6 hf_cache 目录配置

在训练脚本和提取脚本开头都加上：

```python
import os
os.environ.setdefault("HF_HOME", "D:/AI空间转录病理研究/PFMval_new/hf_cache")
```

确保所有模型下载都走 D 盘。

---

## 5. 验证步骤

### 5.1 Backbone 加载测试（~5 min）

```bash
PYTHONIOENCODING=utf-8 "C:\Program Files\Python313\python.exe" virchow2/virchow2_utils.py
```

预期输出：
```
Virchow2 loaded: ViT-H/14
Feature dim: 1280
Token shape: [1, 261, 1280]
```

### 5.2 Token 提取（~30 min/患者，lite 模式）

```bash
PYTHONIOENCODING=utf-8 "C:\Program Files\Python313\python.exe" extract_virchow2_tokens.py --patient HYZ15040 --mode lite
```

验证：
```bash
"C:\Program Files\Python313\python.exe" -c "
import torch
t = torch.load('virchow2_cache_tokens/HYZ15040/train/patch_x4641_y16969.pt')
print(f'Shape: {t.shape}')  # 期望 [65, 1280]
"
```

### 5.3 单患者训练测试（~15 min）

```bash
PYTHONIOENCODING=utf-8 "C:\Program Files\Python313\python.exe" train_histogene_virchow2_tokens.py \
    --patient HYZ15040 \
    --lr 5e-05 \
    --dropout 0.5 \
    --num_epochs 3 \
    --batch_size 16
```

验证：loss 正常下降，无 CUDA OOM，checkpoint 正常保存。

### 5.4 3折跨患者（~1h）

提取全部3患者后，运行完整3折交叉验证。

---

## 6. 新病例兼容

新增患者（如 `NEW001`）的操作步骤：

1. 将 patch 数据放在 `data_new_3ST/patch_noov_spilt/NEW001_noov_split/{train_patches,val_patches}/`
2. 准备标签文件 `data_new_3ST/ssGSEA_zscore/NEW001_ssGSEA_zscore.csv`
3. 在 `extract_virchow2_tokens.py` 的 PATIENT_PATHS 中添加 `"NEW001": {...}`
4. 在 `train_histogene_virchow2_tokens.py` 的 PATIENT_CONFIG 中添加配置
5. 运行提取 + 训练

`from_multiple_patients()` 方法天然支持任意患者组合，无需额外改动。

---

## 7. 常见问题

### Q1: CUDA Out of Memory
- batch_size 提取时固定为 1，不应 OOM
- 训练时减小 `--batch_size`（默认16，可降至8或4）

### Q2: HF Hub 访问超时
```bash
export HF_ENDPOINT="https://hf-mirror.com"  # 国内镜像
```

### Q3: 模型加载报错 "no SwiGLUPacked"
```bash
pip install timm --upgrade  # timm < 0.9.11 不支持
```

### Q4: 磁盘空间不足
- 优先使用 lite 模式（65 tokens），节省 ~75% 空间
- 新增患者仅提取 lite 模式

---

## 8. 预期效果

根据文献分析和 UNI→UNI2-h 的历史提升规律：

| 指标 | 预估值 | 依据 |
|------|--------|------|
| 跨患者 PCC 提升 | +3% ~ +8% | 更多训练数据(3.1M vs 100K WSIs) + 多倍率 |
| 训练时间 | 同等（~15 min/折） | 模型参数量相近（~5M vs ~8M） |
| 推理速度 | 同等或略快 | 1280-dim < 1536-dim，计算量减少 ~17% |
