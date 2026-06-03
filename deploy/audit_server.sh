#!/bin/bash
# ============================================================
# audit_server.sh — 服务器环境审计
# ============================================================
# 用法: 将此脚本上传到服务器，然后执行:
#   bash audit_server.sh
# 或远程执行:
#   ssh user@host 'bash -s' < deploy/audit_server.sh
#
# 输出完整的服务器环境报告，用于判断是否满足训练需求。
# ============================================================

echo "=============================================="
echo "  PFMval 服务器环境审计报告"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

# ---- 1. 系统信息 ----
echo ""
echo "--- 系统信息 ---"
echo "Hostname: $(hostname)"
echo "OS: $(cat /etc/os-release 2>/dev/null | head -3 || uname -a)"
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"

# ---- 2. CPU 和内存 ----
echo ""
echo "--- CPU ---"
echo "型号: $(grep 'model name' /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | xargs)"
echo "核心数: $(nproc --all 2>/dev/null || echo 'unknown')"
echo "物理CPU: $(grep 'physical id' /proc/cpuinfo 2>/dev/null | sort -u | wc -l)"

echo ""
echo "--- 内存 ---"
free -h 2>/dev/null || echo "无法获取内存信息"

# ---- 3. 磁盘 ----
echo ""
echo "--- 磁盘空间 ---"
df -h / /home /data /tmp 2>/dev/null | grep -v "tmpfs\|snap" || df -h

# ---- 4. GPU ----
echo ""
echo "--- GPU ---"
if command -v nvidia-smi &>/dev/null; then
    echo "驱动版本: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    echo "CUDA 版本: $(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | head -1)"
    echo ""
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu --format=csv 2>/dev/null
    echo ""
    echo "GPU 进程:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || echo "  (无运行中进程)"
else
    echo "!!! nvidia-smi 不可用 — 服务器可能无 GPU 或驱动未安装"
fi

# ---- 5. CUDA Libraries ----
echo ""
echo "--- CUDA 库 ---"
for ver in 11.8 12.1 12.4 12.6; do
    if [ -d "/usr/local/cuda-${ver}" ]; then
        echo "/usr/local/cuda-${ver}: 存在"
    fi
done
if [ -z "$(ls -d /usr/local/cuda-* 2>/dev/null)" ]; then
    echo "未找到 /usr/local/cuda-* 目录"
fi
echo "nvcc: $(nvcc --version 2>/dev/null | grep 'release' || echo '未安装或不在 PATH 中')"

# ---- 6. Python 环境 ----
echo ""
echo "--- Python ---"
for py in python python3 python3.10 python3.11 python3.12; do
    if command -v $py &>/dev/null; then
        echo "$py: $($py --version 2>&1) [$(which $py)]"
    fi
done

echo ""
echo "--- Conda ---"
if command -v conda &>/dev/null; then
    echo "conda: $(conda --version 2>&1)"
    echo ""
    echo "已安装环境:"
    conda env list 2>/dev/null
else
    echo "conda 未安装"
fi

# ---- 7. 网络 ----
echo ""
echo "--- 网络 ---"
echo "外网连通性:"
for host in google.com baidu.com pypi.org huggingface.co; do
    if ping -c 1 -W 3 $host &>/dev/null; then
        echo "  $host: 可连通"
    else
        echo "  $host: 不可达"
    fi
done

echo ""
echo "代理设置:"
echo "  http_proxy=${http_proxy:-未设置}"
echo "  https_proxy=${https_proxy:-未设置}"

# ---- 8. 基础工具 ----
echo ""
echo "--- 工具链 ---"
for tool in git wget curl rsync tmux screen gcc g++ make cmake; do
    if command -v $tool &>/dev/null; then
        echo "  $tool: $(command -v $tool)"
    else
        echo "  $tool: 未安装"
    fi
done

# ---- 9. 权限 ----
echo ""
echo "--- 权限 ---"
echo "当前用户: $(whoami) [UID=$(id -u), GID=$(id -g)]"
echo "用户组: $(groups)"
echo "sudo 权限: $(sudo -n true 2>/dev/null && echo '有' || echo '无（或需密码）')"
echo "HOME: ${HOME} (可用: $(df -h ${HOME} | tail -1 | awk '{print $4}'))"

# ---- 10. 已运行进程 ----
echo ""
echo "--- 可能冲突的 GPU 进程 ---"
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || echo "  无法查询"

# ---- 总结 ----
echo ""
echo "=============================================="
echo "  审计完毕"
echo "=============================================="
