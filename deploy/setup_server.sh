#!/bin/bash
# ============================================================
# setup_server.sh — 服务器首次环境安装
# ============================================================
# 用法: 将此脚本上传到服务器，然后执行:
#   bash setup_server.sh
# 或通过 SSH 一键执行:
#   ssh user@host 'bash -s' < deploy/setup_server.sh
# ============================================================

set -e

echo "=============================================="
echo "  PFMval 服务器环境安装"
echo "=============================================="

# ---- 用户配置（按需修改）----
# 代理设置（服务器需代理才能上网时填写，留空则不设置）
PROXY_HTTP="${PROXY_HTTP:-}"
PROXY_HTTPS="${PROXY_HTTPS:-}"
# HuggingFace 镜像（国内服务器可用 https://hf-mirror.com）
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# Conda 安装路径
CONDA_DIR="${HOME}/miniconda3"
# 项目在服务器上的路径
PROJECT_DIR="${HOME}/PFMval_new"

echo ""
echo ">>> 配置: PROXY_HTTP=${PROXY_HTTP:-未设置}"
echo "         HF_ENDPOINT=${HF_ENDPOINT:-默认}"
echo "         CONDA_DIR=${CONDA_DIR}"
echo ""

# ---- 1. 基础检查 ----
echo "=============================================="
echo "  1/7: 检查系统基础环境"
echo "=============================================="

echo "OS: $(uname -a)"
echo "CPU: $(nproc --all 2>/dev/null || echo 'unknown') cores"
echo "RAM: $(free -h 2>/dev/null | grep Mem | awk '{print $2}' || echo 'unknown')"
echo "Disk: $(df -h ${HOME} 2>/dev/null | tail -1 | awk '{print $4}' || echo 'unknown') available on HOME"

# ---- 2. 配置代理 ----
echo ""
echo "=============================================="
echo "  2/7: 配置代理"
echo "=============================================="

if [ -n "$PROXY_HTTP" ]; then
    echo ">>> 设置 HTTP/HTTPS 代理..."

    # Shell 环境变量
    export http_proxy="${PROXY_HTTP}"
    export https_proxy="${PROXY_HTTPS:-${PROXY_HTTP}}"
    export HTTP_PROXY="${PROXY_HTTP}"
    export HTTPS_PROXY="${PROXY_HTTPS:-${PROXY_HTTP}}"

    # 写入 bashrc 持久化
    grep -q "http_proxy" ~/.bashrc 2>/dev/null || {
        echo "" >> ~/.bashrc
        echo "# PFMval 代理配置" >> ~/.bashrc
        echo "export http_proxy=${PROXY_HTTP}" >> ~/.bashrc
        echo "export https_proxy=${PROXY_HTTPS:-${PROXY_HTTP}}" >> ~/.bashrc
        echo "export HTTP_PROXY=${PROXY_HTTP}" >> ~/.bashrc
        echo "export HTTPS_PROXY=${PROXY_HTTPS:-${PROXY_HTTP}}" >> ~/.bashrc
    }

    # Git 代理
    git config --global http.proxy "${PROXY_HTTP}" 2>/dev/null || true
    git config --global https.proxy "${PROXY_HTTPS:-${PROXY_HTTP}}" 2>/dev/null || true

    echo ">>> 代理配置完成"
else
    echo ">>> 未配置代理（跳过）"
fi

# ---- 3. 安装 Miniconda ----
echo ""
echo "=============================================="
echo "  3/7: 安装 Miniconda"
echo "=============================================="

if command -v conda &>/dev/null; then
    echo ">>> conda 已安装: $(conda --version)"
else
    if [ -d "${CONDA_DIR}" ]; then
        echo ">>> 发现已有 Miniconda 目录，初始化..."
    else
        echo ">>> 下载 Miniconda..."
        INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
        if [ -n "$PROXY_HTTP" ]; then
            wget -e use_proxy=yes -e http_proxy="${PROXY_HTTP}" \
                "https://repo.anaconda.com/miniconda/${INSTALLER}" -O "/tmp/${INSTALLER}"
        else
            wget "https://repo.anaconda.com/miniconda/${INSTALLER}" -O "/tmp/${INSTALLER}"
        fi
        bash "/tmp/${INSTALLER}" -b -p "${CONDA_DIR}"
        rm -f "/tmp/${INSTALLER}"
        echo ">>> Miniconda 安装完成"
    fi

    # 初始化 conda
    "${CONDA_DIR}/bin/conda" init bash
    source "${CONDA_DIR}/bin/activate"
    echo ">>> conda 初始化完成"
fi

# ---- 4. 创建 conda 环境 ----
echo ""
echo "=============================================="
echo "  4/7: 创建 Python 训练环境"
echo "=============================================="

# 确保 conda 可用
if ! command -v conda &>/dev/null; then
    source "${CONDA_DIR}/bin/activate" 2>/dev/null || true
fi

# 配置 conda 代理
if [ -n "$PROXY_HTTP" ]; then
    conda config --set proxy_servers.http "${PROXY_HTTP}" 2>/dev/null || true
    conda config --set proxy_servers.https "${PROXY_HTTPS:-${PROXY_HTTP}}" 2>/dev/null || true
fi

# HisToGene 环境
if conda env list | grep -q "pfmval_histogene"; then
    echo ">>> pfmval_histogene 环境已存在"
else
    echo ">>> 创建 pfmval_histogene 环境（需要5-15分钟）..."
    if [ -f "${PROJECT_DIR}/env_histogene.yml" ]; then
        conda env create -f "${PROJECT_DIR}/env_histogene.yml" 2>&1 || {
            echo "!!! 从 YML 创建失败，尝试分步安装..."
            conda create -n pfmval_histogene python=3.10 -y
            conda activate pfmval_histogene
            conda install -y numpy pandas scipy scikit-learn matplotlib pyyaml tqdm pillow opencv einops
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
            pip install transformers timm albumentations huggingface_hub open_clip_torch
        }
    else
        echo "!!! ${PROJECT_DIR}/env_histogene.yml 不存在，请先推送代码"
        echo ">>> 跳过 HisToGene 环境创建"
    fi
fi

# EGN-v2 环境
if conda env list | grep -q "pfmval_egnv2"; then
    echo ">>> pfmval_egnv2 环境已存在"
else
    echo ">>> 创建 pfmval_egnv2 环境..."
    if [ -f "${PROJECT_DIR}/env_egnv2.yml" ]; then
        conda env create -f "${PROJECT_DIR}/env_egnv2.yml" 2>&1 || true
    fi
fi

# ---- 5. 配置 HuggingFace ----
echo ""
echo "=============================================="
echo "  5/7: 配置 HuggingFace"
echo "=============================================="

if [ -n "$HF_ENDPOINT" ]; then
    export HF_ENDPOINT="${HF_ENDPOINT}"
    grep -q "HF_ENDPOINT" ~/.bashrc 2>/dev/null || {
        echo "export HF_ENDPOINT=${HF_ENDPOINT}" >> ~/.bashrc
    }
    echo ">>> HF_ENDPOINT=${HF_ENDPOINT}"
fi

# 检查 HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "!!! 注意: HF_TOKEN 未设置"
    echo "    如需下载受限制模型，请设置: export HF_TOKEN='hf_xxx'"
    echo "    并将其写入 ~/.bashrc"
fi

# ---- 6. 安装 tmux ----
echo ""
echo "=============================================="
echo "  6/7: 安装 tmux（后台训练必需）"
echo "=============================================="

if command -v tmux &>/dev/null; then
    echo ">>> tmux 已安装: $(tmux -V)"
else
    echo ">>> 尝试安装 tmux..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -y && sudo apt-get install -y tmux
    elif command -v yum &>/dev/null; then
        sudo yum install -y tmux
    elif command -v conda &>/dev/null; then
        conda install -y -c conda-forge tmux
    else
        echo "!!! 无法自动安装 tmux，请手动安装"
        echo "    Ubuntu/Debian: sudo apt-get install tmux"
        echo "    CentOS/RHEL:   sudo yum install tmux"
    fi
fi

# ---- 7. 验证 ----
echo ""
echo "=============================================="
echo "  7/7: 环境验证"
echo "=============================================="

echo "Python:"
source "${CONDA_DIR}/bin/activate" pfmval_histogene 2>/dev/null && python --version || echo "  (待验证)"

echo "PyTorch + CUDA:"
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "  (待验证)"

echo "GPU:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi 不可用"

echo "tmux: $(tmux -V 2>/dev/null || echo '未安装')"
echo "Disk: $(df -h ${HOME} | tail -1 | awk '{print "used "$3" / "$2" ("$5" full)"}')"

echo ""
echo "=============================================="
echo "  安装完毕！"
echo "=============================================="
echo ""
echo "  后续步骤:"
echo "  1. 修改 ${PROJECT_DIR}/config.yaml 指向服务器数据路径"
echo "  2. python config_utils.py     # 验证路径配置"
echo "  3. 传输/下载模型权重和特征缓存到服务器"
echo "  4. python train_xxx.py --epochs 5   # 冒烟测试"
echo ""
