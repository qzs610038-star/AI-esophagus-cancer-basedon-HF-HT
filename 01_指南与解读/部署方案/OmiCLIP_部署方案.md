# OmiCLIP (Loki) 部署方案

> **模型**：OmiCLIP — A visual–omics foundation model bridging histopathology with spatial transcriptomics  
> **论文**：Chen W, Zhang P, et al. *Nature Methods*, 2025  
> **GitHub**：[GuangyuWangLab2021/Loki](https://github.com/GuangyuWangLab2021/Loki) ⭐139 · BSD-3-Clause  
> **HuggingFace 权重**：[WangGuangyuLab/Loki](https://huggingface.co/WangGuangyuLab/Loki)  
> **安装指南**：[Loki_OmiCLIP_安装指南.md](Loki_OmiCLIP_安装指南.md)  
> **创建日期**：2026-05-19

---

## 1. 模型概况

| 属性 | OmiCLIP | UNI2-h（当前） |
|------|---------|---------------|
| 架构 | CoCa（CLIP风格视觉编码器 + 文本解码器） | ViT-L/14 |
| 多模态 | 图像 + 基因文本 | 仅图像 |
| Embedding 维度 | **768**（CLS）/ **255×768**（tokens） | 1536 |
| 输入尺寸 | 224×224（CLIP 标准） | 224×224 |
| 核心能力 | H&E → 基因表达直接预测 | H&E → 通用视觉特征 |
| 许可 | BSD-3-Clause | 学术 |
| 权重大小 | ~7.14 GB | ~3 GB |

**与本项目的适配点**：OmiCLIP 的视觉编码器可作为替代特征提取器，提取 H&E 图像的 embedding，输入到 HisToGene 下游回归模型预测 30 条通路评分。

---

## 2. 环境搭建

### 2.1 前置检查

⚠️ OmiCLIP **严格要求 Python 3.9**（非 3.10+），需要新建独立的 conda 环境。

现有环境一览：
```
D:\conda_envs\
  ├── pfmval_py310\     # EGN-v2/GAT 训练（Python 3.10）
  └── loki_env\          # [新建] OmiCLIP 特征提取（Python 3.9）
C:\Program Files\Python313\  # HisToGene 训练（Python 3.13）
```

### 2.2 创建 Conda 环境（D 盘）

```bash
# 创建环境到 D 盘
"D:\miniconda\Scripts\conda.exe" create -n loki_env python=3.9 -y

# 激活环境
"D:\miniconda\Scripts\conda.exe" activate loki_env
# 如果 activate 不生效，直接在 PowerShell 中：
& "D:\miniconda\Scripts\conda.exe" "shell.powershell" "hook" | Out-String | Invoke-Expression
conda activate loki_env
```

### 2.3 克隆 Loki 仓库

```bash
# 克隆到 D 盘项目目录内
git clone https://github.com/GuangyuWangLab2021/Loki.git "D:\AI空间转录病理研究\PFMval_new\loki_src"
# 备选：使用国内镜像
# git clone https://gitclone.com/github.com/GuangyuWangLab2021/Loki.git "D:\AI空间转录病理研究\PFMval_new\loki_src"
```

### 2.4 安装 Loki

```bash
cd "D:\AI空间转录病理研究\PFMval_new\loki_src\src"
pip install .
```

### 2.5 安装 PyTorch（CUDA 11.8）

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### 2.6 下载预训练权重（~7.14 GB）

```bash
# 安装下载工具
pip install huggingface_hub

# 下载到 D 盘项目目录
python -c "
import os
os.environ['HF_HOME'] = 'D:/AI空间转录病理研究/PFMval_new/hf_cache'
from huggingface_hub import hf_hub_download
hf_hub_download(
    'WangGuangyuLab/Loki',
    'checkpoint.pt',
    local_dir='D:/AI空间转录病理研究/PFMval_new/pretrained_omiclip'
)
print('Download complete -> D:/AI空间转录病理研究/PFMval_new/pretrained_omiclip/checkpoint.pt')
"
```

### 2.7 验证安装

```bash
"D:\conda_envs\loki_env\python.exe" -c "
import loki
import loki.preprocess
import loki.utils
import loki.plot
import loki.align
import loki.annotate
import loki.decompose
import loki.retrieve
import loki.predex
print('Loki import: OK')
"
```

---

## 3. 架构探查（关键前置步骤）

### 3.1 探查脚本

在编写提取脚本前，需运行以下探查脚本确定 OmiCLIP 内部结构：

```python
# inspect_omiclip.py — 在 loki_env 中运行
import torch
from loki.predex import OmiCLIP_Predictor

CKPT = "D:/AI空间转录病理研究/PFMval_new/pretrained_omiclip/checkpoint.pt"
predictor = OmiCLIP_Predictor(ckpt_path=CKPT)

print("=== Predictor attributes ===")
model = predictor.model
print(f"model type: {type(model)}")
print(f"model: {model}")

# 探查子模块
for attr_name in dir(model):
    if not attr_name.startswith('_'):
        attr = getattr(model, attr_name)
        if isinstance(attr, torch.nn.Module):
            print(f"  model.{attr_name}: {type(attr).__name__}")

# 探查视觉编码器
for candidate in ['visual', 'vision_encoder', 'image_encoder', 'encode_image', 'trunk']:
    if hasattr(model, candidate):
        vis = getattr(model, candidate)
        print(f"\n=== Vision encoder: model.{candidate} ===")
        print(f"  type: {type(vis)}")
        print(f"  {vis}")

# 测试前向传播
print("\n=== Forward pass test ===")
dummy = torch.randn(1, 3, 224, 224)
model.eval()
with torch.inference_mode():
    # 尝试不同的调用方式
    for method in ['forward_features', 'forward', 'encode_image']:
        if hasattr(model, method):
            try:
                output = getattr(model, method)(dummy)
                print(f"  model.{method}(): shape={output.shape if hasattr(output,'shape') else type(output)}")
            except Exception as e:
                print(f"  model.{method}(): FAILED - {e}")

# 打印 checkpoint keys（如果以上方法都失败）
print("\n=== Checkpoint keys (fallback) ===")
ckpt = torch.load(CKPT, map_location='cpu', weights_only=True)
if isinstance(ckpt, dict):
    top_keys = [k for k in ckpt.keys() if not any(k.startswith(p) for p in ['optimizer', 'scheduler', 'epoch', 'step'])]
    print(f"  Top-level keys: {top_keys[:20]}")
    # 查找 vision 相关的 key
    vis_keys = [k for k in ckpt.get('model', {}).keys() if 'visual' in k.lower() or 'vision' in k.lower() or 'image' in k.lower()][:10]
    print(f"  Vision-related keys: {vis_keys}")
```

### 3.2 探查结果（已完成，2026-05-19）

| 问题 | 结论 |
|------|------|
| 模型架构 | `coca_ViT-L-14`（CoCa 架构），通过 `open_clip` 加载；**注意：`loki.predex.OmiCLIP_Predictor` 在已安装版本（loki==0.0.1）中不存在** |
| 加载方式 | `open_clip.create_model('coca_ViT-L-14')` + `model.load_state_dict(ckpt['state_dict'])`（checkpoint key 为 `state_dict`） |
| Vision encoder 属性 | `model.visual`（VisionTransformer，307M 参数） |
| Embedding 维度 | **768**（attention pool 输出维度） |
| Token 序列输出 | `model.visual(x)[1]` → `[B, 255, 768]`（255 tokens，256-query attention pool） |
| CLS 向量输出 | `model.encode_image(x)` → `[B, 768]` |
| 输入尺寸 | 224×224 |
| 归一化参数 | mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711) |
| 下游模型选择 | **策略 A**：复用 `HisToGeneUNITokens(feature_dim=768, num_tokens=255)`，无需新建模型 |

---

## 4. 集成策略

### 策略 A：提取 Embedding → 下游模型（首选）

```
H&E Patches → OmiCLIP Vision Encoder → [embed_dim] embedding
    → HisToGene-style downstream model → 30 pathway scores
```

- **适用条件**：能独立调用 vision encoder
- **优点**：与 UNI/Virchow2 完全可比，遵循现有离线提取→训练模式
- **下游模型**：如果输出是 token 序列 → 复用 `HisToGeneUNITokens(feature_dim=xxx)`；如果仅 CLS → 新建简化模型（Linear proj + 坐标嵌入 + MLP head）

### 策略 B：端到端基因预测 → 通路聚合（备选）

```
H&E Patches → OmiCLIP predex → gene expression → aggregate to pathway scores
```

- **适用条件**：vision encoder 不可单独访问，仅能通过 `OmiCLIP_Predictor.predict()` 接口
- **缺点**：OmiCLIP 预测的是基因表达（非通路评分），需要基因→通路映射，且与 UNI/Virchow2 不可直接对比
- **仅作为 fallback**

---

## 5. 代码文件创建

### 5.1 文件清单

```
PFMval_new/
├── pretrained_omiclip/
│   └── checkpoint.pt            # OmiCLIP 预训练权重（~7.14 GB）
├── extract_omiclip_features.py  # 特征提取脚本（loki_env Python 3.9）
├── model_omiclip.py             # 下游模型（仅在 token 序列不可用且需简化模型时创建）
├── train_histogene_omiclip.py   # 训练脚本（Python 3.13）
├── omiclip_cache/               # 特征缓存（自动创建）
│   ├── HYZ15040/{train,val}/
│   ├── JFX0729/{train,val}/
│   └── LMZ12939/{train,val}/
```

### 5.2 extract_omiclip_features.py

运行环境：`D:\conda_envs\loki_env\python.exe`（Python 3.9）

基于 `extract_uni_tokens.py` 模式，差异点：
- 模型加载：`OmiCLIP_Predictor(ckpt_path=...)` → `predictor.model.{vision_encoder}`
- 预处理：使用 OmiCLIP 自带的 transform 或 CLIP 标准预处理（224×224, mean=[0.481,0.457,0.408], std=[0.268,0.261,0.275]）
- 前向传播：调用 `vision_encoder(x)` 或 `model.encode_image(x)`（取决于探查结果）
- 输出格式：`model.visual(x)[1]` → `[255, 768]`（token 序列）
- 缓存目录：`omiclip_cache/{patient}/{split}/`

### 5.3 model_omiclip.py（按需创建）

#### ✅ 已确认：OmiCLIP 输出 token 序列
直接复用 `model_uni_tokens.py` 的 `HisToGeneUNITokens(feature_dim=768, num_tokens=255)`。不需要 `model_omiclip.py`。

#### （备选）情况 2：仅需 CLS 向量 [embed_dim]
创建简化模型（无 token_encoder，因为已经是单个向量）：

```python
class OmiCLIPModel(nn.Module):
    """CLS embedding → 投影 + 坐标嵌入 → MLP head"""
    def __init__(self, feature_dim, dim=1024, n_pos=128, n_targets=30, mlp_dim=2048, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(feature_dim, dim), nn.LayerNorm(dim))
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)
        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, mlp_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_dim, n_targets)
        )

    def forward(self, features, pos_x, pos_y):
        x = self.proj(features)
        x = x + self.x_embed(pos_x) + self.y_embed(pos_y)
        return self.head(x)
```

### 5.4 train_histogene_omiclip.py

运行环境：`C:\Program Files\Python313\python.exe`（Python 3.13）

基于 `train_histogene_uni_tokens_augmix.py` 精简版，修改点：
- 缓存目录 → `omiclip_cache`
- `feature_dim=768`, `num_tokens=255`
- 模型创建：`HisToGeneUNITokens(feature_dim=768, num_tokens=255)`
- 数据集：自定义 `OmiCLIPTokensDataset`（内置在 train 脚本中，适配 768 维；原 dataset_uni_tokens.py 有 1536 断言）

---

## 6. 验证步骤

### 6.1 环境验证（~5 min）
```bash
"D:\conda_envs\loki_env\python.exe" -c "import loki; print('OK')"
```

### 6.2 架构探查（~10 min）
```bash
"D:\conda_envs\loki_env\python.exe" inspect_omiclip.py
```
记录：embedding 维度、前向接口方法、token 是否可用。

### 6.3 特征提取（~1h，单患者）
```bash
"D:\conda_envs\loki_env\python.exe" extract_omiclip_features.py --patient HYZ15040 --mode lite
```
验证缓存 `.pt` 文件 shape 符合预期。

### 6.4 训练测试（~15 min）
```bash
PYTHONIOENCODING=utf-8 "C:\Program Files\Python313\python.exe" train_histogene_omiclip.py \
    --patient HYZ15040 --lr 5e-05 --num_epochs 3
```

---

## 7. 新病例兼容

与 Virchow2 完全相同的模式：
1. 数据放 `data_new_3ST/patch_noov_spilt/{NEW}_noov_split/`
2. 标签放 `data_new_3ST/ssGSEA_zscore/{NEW}_ssGSEA_zscore.csv`
3. 在 PATIENT_PATHS 和 PATIENT_CONFIG 中添加
4. 运行提取→训练

---

## 8. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| Vision encoder 不可单独访问 | 中 | Fallback 到策略 B（端到端基因预测），或从 checkpoint dict 重建 vision encoder |
| 仅输出 1D CLS 向量（无空间 token） | 中 | 可接受，简化模型仍能对比（CLS-pooled 也是一种有效表征） |
| Python 3.9 与 CUDA 11.8 兼容性 | 低 | PyTorch 2.0.x 支持 Python 3.9 + CUDA 11.8 |
| 7.14 GB 权重下载失败 | 中 | 使用 `wget -c` 断点续传或 HF 国内镜像 |
| OmiCLIP embedding 效果不如 UNI | 高 | 正常学术对比结果，记录即可 |

---

## 9. 预期效果

OmiCLIP 直接桥接 H&E 与空间转录组，理论上对 ST 预测任务更适配。但由于：
- CLIP-style 训练目标 ≠ 通路评分回归
- 泛化到食管癌数据的 domain shift
- 可能仅输出 CLS 向量（缺失空间 tokens 信息）

暂不做具体 PCC 提升预估。实际结果将成为 UNI vs OmiCLIP 的学术对比数据。
