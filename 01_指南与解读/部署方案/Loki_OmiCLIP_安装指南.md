# Loki (OmiCLIP) 安装与部署指南

> **模型全称**：OmiCLIP — A visual–omics foundation model to bridge histopathology with spatial transcriptomics  
> **平台名称**：Loki  
> **论文**：Nature Methods (2025) | DOI: [10.1038/s41592-025-02707-1](https://www.nature.com/articles/s41592-025-02707-1)  
> **作者**：Weiqing Chen, Pengzhi Zhang 等 (Guangyu Wang Lab)  
> **许可**：BSD-3-Clause  
> **创建日期**：2026-05-16  

---

## 1. 资源清单

| 资源 | 地址 | 说明 |
|------|------|------|
| 📂 **源代码** | [GitHub: GuangyuWangLab2021/Loki](https://github.com/GuangyuWangLab2021/Loki) | ⭐139 · 🍴13 · Python |
| 🤗 **预训练权重** | [HuggingFace: WangGuangyuLab/Loki](https://huggingface.co/WangGuangyuLab/Loki) | `checkpoint.pt` ~7.14 GB |
| 🗄️ **STbank 数据库** | [Google Drive](https://drive.google.com/drive/folders/1J15cO-pXTwkTjRAR-v-_nQkqXNfcCNn3) | 训练/评估用空间转录组数据 |
| 📖 **文档网站** | [https://guangyuwanglab2021.github.io/Loki/](https://guangyuwanglab2021.github.io/Loki/) | 使用手册 + Notebooks |
| 📄 **论文全文** | [Nature Methods](https://www.nature.com/articles/s41592-025-02707-1) | 正式发表版本 |
| 📝 **预印本** | [Research Square](https://doi.org/10.21203/rs.3.rs-5183775/v1) | 免费访问 |

---

## 2. 环境要求

| 项目 | 要求 |
|------|------|
| **操作系统** | Linux / macOS / Windows (推荐 Linux) |
| **Python** | 3.9（⚠️ 必须，非 3.10+） |
| **Conda** | Miniconda 或 Anaconda |
| **GPU** | NVIDIA GPU (CUDA)，推理约需 8GB 显存，训练需更多 |
| **磁盘空间** | 至少 30 GB（含权重 7.14 GB + STbank 数据库） |
| **内存** | 建议 ≥ 32 GB |

---

## 3. 安装步骤

### 3.1 克隆仓库

```bash
git clone https://github.com/GuangyuWangLab2021/Loki.git
cd Loki
```

### 3.2 创建 Conda 环境

```bash
conda create -n loki_env python=3.9 -y
conda activate loki_env
```

### 3.3 安装 Loki

```bash
cd src
pip install .
```

> 官方说安装约需 5 分钟（MacBook Pro 基准）。

### 3.4 安装 PyTorch（如未自动安装）

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3.5 验证安装

```bash
python -c "import loki; import loki.preprocess; import loki.utils; import loki.plot; import loki.align; import loki.annotate; import loki.decompose; import loki.retrieve; import loki.predex; print('Loki import success!')"
```

---

## 4. 下载资源

### 4.1 预训练权重 (~7.14 GB)

```bash
# 方法一：wget（推荐，支持断点续传）
wget -c https://huggingface.co/WangGuangyuLab/Loki/resolve/main/checkpoint.pt -O checkpoint.pt

# 方法二：curl
curl -L -o checkpoint.pt https://huggingface.co/WangGuangyuLab/Loki/resolve/main/checkpoint.pt

# 方法三：huggingface_hub Python API
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('WangGuangyuLab/Loki', 'checkpoint.pt', local_dir='./pretrained')"
```

> 权重文件放在项目根目录或 `--ckpt_path` 指定的路径即可。

### 4.2 STbank 数据库

从 [Google Drive 链接](https://drive.google.com/drive/folders/1J15cO-pXTwkTjRAR-v-_nQkqXNfcCNn3) 下载以下文件：

```bash
# 使用 gdown 下载（推荐）
pip install gdown

# 下载 text.csv（基因句子 + 配对图像块）
gdown --folder "https://drive.google.com/drive/folders/1J15cO-pXTwkTjRAR-v-_nQkqXNfcCNn3"

# 注意：image.tar.gz 较大，可能需要分步下载
```

STbank 包含的文件：
- `links_to_raw_data.xlsx` — 原始数据论文来源、DOI 和下载链接
- `text.csv` — 基因句子与配对图像块
- `image.tar.gz` — 图像块压缩包

---

## 5. 核心功能模块

Loki 平台的五大功能：

| 模块 | 函数入口 | 功能 |
|------|----------|------|
| **组织对齐** | `loki.align` | 用 ST 或 H&E 图像对齐组织切片 |
| **细胞分解** | `loki.decompose` | 以 scRNA-seq 为参考，分解 ST/H&E 的细胞类型 |
| **组织注释** | `loki.annotate` | 基于 bulk RNA-seq 或 marker 基因注释组织区域 |
| **基因预测** | `loki.predex` | 从 H&E 图像预测 ST 基因表达 |
| **图像-转录组检索** | `loki.retrieve` | 组织学图像 ↔ 转录组相互检索 |

---

## 6. 快速测试

```python
import loki

# ---- 示例：从 H&E 预测基因表达 ----
import loki.predex
from loki.predex import OmiCLIP_Predictor

# 加载预训练模型
predictor = OmiCLIP_Predictor(ckpt_path="./checkpoint.pt")

# 预测（具体参数见官方文档）
# result = predictor.predict(he_image_path, gene_list)
```

---

## 7. 常见问题

### Q1: ImportError / 模块缺失
```bash
pip install -r src/requirements.txt
```

### Q2: CUDA out of memory
- 减小 batch size 或图像分辨率
- 使用 CPU 模式推理（速度较慢）

### Q3: checkpoint.pt 下载慢
- 使用 `wget -c` 断点续传
- 或使用 [HF Mirror](https://hf-mirror.com)（国内加速）

### Q4: Conda 环境已存在但安装失败
```bash
conda remove -n loki_env --all
conda create -n loki_env python=3.9 -y
```

---

## 8. 与 PFMval 集成建议

该模型可作为 PFMval 基准测试的重要 Baseline，理由：

1. **全开源** — 代码 + 权重 + 数据三要素齐全
2. **多模态** — 直接桥接 H&E 图像与空间转录组，契合 PFMval 评估方向
3. **功能丰富** — 提供 5 种核心分析模式，可多维度对比
4. **Nature Methods** — 权威性强，适合作为 SOTA Baseline

---

## 9. 参考链接汇总

| 用途 | 链接 |
|------|------|
| GitHub 仓库 | https://github.com/GuangyuWangLab2021/Loki |
| HuggingFace 权重 | https://huggingface.co/WangGuangyuLab/Loki |
| 权重直链下载 | https://huggingface.co/WangGuangyuLab/Loki/resolve/main/checkpoint.pt |
| STbank 数据库 | https://drive.google.com/drive/folders/1J15cO-pXTwkTjRAR-v-_nQkqXNfcCNn3 |
| 官方文档 | https://guangyuwanglab2021.github.io/Loki/ |
| 论文 Nature Methods | https://www.nature.com/articles/s41592-025-02707-1 |
| 预印本 | https://doi.org/10.21203/rs.3.rs-5183775/v1 |
