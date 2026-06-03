#!/bin/bash
# ============================================================
# PFMval 服务器部署 — 集中配置文件
# ============================================================
# 目标服务器: Windows 10 Pro, RTX 4080 (16GB), 1TB RAM
# 用法：修改下方变量后，source 此文件，其他脚本自动读取。
# 这是你唯一需要编辑的文件。
# ============================================================

# ---- 服务器连接 ----
SERVER_HOST="<填写IP或主机名，如 192.168.1.100>"
SERVER_USER="<填写Windows用户名>"
SERVER_PORT="22"

# ---- 服务器路径（Windows 格式，正斜杠兼容 SSH）----
# 代码在服务器上的位置（建议放在 SSD 上）
SERVER_PROJECT_DIR="D:/AIPatho/qzs"

# 服务器上已有病理图像数据路径（留空则后续填写）
SERVER_DATA_BASE="D:/data/HE_patches"

# 服务器上标签文件路径（留空则后续填写）
SERVER_SSGSEA_BASE="D:/data/ssGSEA_scores"

# 服务器上缓存/模型/输出目录
SERVER_CACHE_DIR="D:/data/caches"
SERVER_MODEL_DIR="D:/data/models"
SERVER_OUTPUT_DIR="D:/data/outputs"

# ---- 代理配置（服务器需代理才能上网时填写）----
PROXY_HTTP="http://proxy.example.com:8080"
PROXY_HTTPS="http://proxy.example.com:8080"
# HuggingFace 镜像（国内可用 https://hf-mirror.com）
HF_ENDPOINT=""

# ---- HTTP 命令服务配置 ----
# 服务器端运行: python deploy/cmd_server.py --port $CMD_PORT --token $CMD_TOKEN
# 客户端连接: python deploy/cmd_client.py --host $SERVER_HOST --port $CMD_PORT --token $CMD_TOKEN "command"
CMD_PORT="8080"                        # 服务器监听端口（需在路由器做端口映射）
CMD_TOKEN="${PFMVAL_SERVER_PASSWORD}"  # 鉴权 Token（默认复用服务器密码）

# ---- SSH 配置 ----
SSH_KEY="$HOME/.ssh/pfmval_server"
SSH_CMD="ssh -i ${SSH_KEY} -p ${SERVER_PORT} -o StrictHostKeyChecking=no"
SCP_CMD="scp -i ${SSH_KEY} -P ${SERVER_PORT} -o StrictHostKeyChecking=no"

# ---- scp 排除规则 ----
# Windows 使用 scp 而非 rsync，排除项通过 --exclude 实现（需 rsync）
# 如服务器有 Git Bash + rsync，取消下行注释启用 rsync 模式：
# RSYNC_CMD="rsync -avz --progress -e 'ssh -i ${SSH_KEY} -p ${SERVER_PORT} -o StrictHostKeyChecking=no'"
#
# 默认使用 Git 方式同步（推荐，避免传输大文件）：
#   本地: git push server main
#   服务器: git pull

RSYNC_EXCLUDES=(
    "data_new_3ST/"
    "data/"
    "HYZ15040_old/"
    "*.tiff"
    "*.tif"
    "*.zip"
    "uni2h_cache*/"
    "uni2h_cache_tokens*/"
    "omiclip_cache*/"
    "virchow2_cache*/"
    "hf_cache/"
    "**/cache/"
    "*.pth"
    "checkpoints/"
    "pretrained_omiclip/"
    "__pycache__/"
    "*.pyc"
    "*.pyo"
    ".venv/"
    "venv/"
    "env/"
    ".vscode/"
    ".idea/"
    "Thumbs.db"
    ".DS_Store"
    "desktop.ini"
    "**/results_vis/"
    "tv_sweep_results/"
    "tv_3fold_cv_results/"
    "tv_loss_search_results/"
    "ensemble_results/"
    "phase3_search_results/"
    "report_figures/"
    "attnpool_analysis/"
    "*.log"
    "PAUSE_TRAINING"
    "training_status_*.txt"
    "*.tmp"
    "*.bak"
    "temp_*.py"
    "inspect_*.py"
    ".git/"
    "virchow2_repo/"
    "loki_src/"
    "openmidnight/"
    "HisToGene_UNI_Tokens_打包/"
    "docs/"
    "Ai病理项目文献汇总/"
)

build_excludes() {
    local excludes=""
    for pattern in "${RSYNC_EXCLUDES[@]}"; do
        excludes="${excludes} --exclude='${pattern}'"
    done
    echo "$excludes"
}
