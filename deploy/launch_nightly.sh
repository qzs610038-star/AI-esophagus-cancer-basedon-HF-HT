#!/bin/bash
# ============================================================
# launch_nightly.sh — 推送代码到服务器 + 启动夜间实验
# ============================================================
# 用法:
#   ./deploy/launch_nightly.sh           # 推送+后台启动
#   ./deploy/launch_nightly.sh --status  # 查看运行状态
#   ./deploy/launch_nightly.sh --pull    # 拉取结果
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ══════════════════════════════════════════════
# 服务器连接参数（来自 CLAUDE.md + ~/.ssh/config）
# ══════════════════════════════════════════════
SERVER_HOST="117.68.10.96"
SERVER_USER="AIPatho1"
SERVER_PORT="22"
SSH_KEY="$HOME/.ssh/pfmval_server"

# 服务器路径
SERVER_PROJECT_DIR="D:/AIPatho/qzs/pfmval_deploy_git"
SERVER_PYTHON="C:/Users/AIPatho1/pfmval_env/Scripts/python.exe"

# SSH 基础命令
SSH_CMD="ssh -i ${SSH_KEY} -p ${SERVER_PORT} -o StrictHostKeyChecking=no"
SCP_CMD="scp -i ${SSH_KEY} -P ${SERVER_PORT} -o StrictHostKeyChecking=no"

MODE="${1:-run}"

case "$MODE" in
    --status|-s)
        echo ">>> 检查服务器夜间实验状态..."
        echo ""
        ${SSH_CMD} ${SERVER_USER}@${SERVER_HOST} \
            "powershell -Command \"Get-ChildItem '${SERVER_PROJECT_DIR}/logs/nightly/' | Sort-Object LastWriteTime -Descending | Select-Object -First 10 Name, Length, LastWriteTime | Format-Table -AutoSize\""
        echo ""
        echo ">>> 最新日志尾部:"
        LATEST_LOG=$(${SSH_CMD} ${SERVER_USER}@${SERVER_HOST} \
            "powershell -Command \"Get-ChildItem '${SERVER_PROJECT_DIR}/logs/nightly/batch_*.log' | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName\"" 2>/dev/null)
        if [ -n "$LATEST_LOG" ]; then
            LATEST_LOG=$(echo "$LATEST_LOG" | tr -d '\r')
            ${SSH_CMD} ${SERVER_USER}@${SERVER_HOST} \
                "powershell -Command \"Get-Content '${LATEST_LOG}' -Tail 30\""
        else
            echo "  (无日志文件)"
        fi
        echo ""
        echo ">>> 进程状态:"
        ${SSH_CMD} ${SERVER_USER}@${SERVER_HOST} \
            "powershell -Command \"Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, StartTime, @{N='MemMB';E={[math]::Round(\$_.WorkingSet64/1MB,0)}} | Format-Table -AutoSize\""
        ;;

    --pull|-p)
        echo ">>> 从服务器拉取训练结果..."
        echo ""
        echo "--- 拉取 checkpoints/online_cls ---"
        ${SCP_CMD} -r ${SERVER_USER}@${SERVER_HOST}:"${SERVER_PROJECT_DIR}/checkpoints/online_cls/" \
            "${PROJECT_DIR}/checkpoints/online_cls/" 2>/dev/null
        echo "--- 拉取 logs/nightly ---"
        ${SCP_CMD} -r ${SERVER_USER}@${SERVER_HOST}:"${SERVER_PROJECT_DIR}/logs/nightly/" \
            "${PROJECT_DIR}/logs/nightly/" 2>/dev/null
        echo ""
        echo ">>> 拉取完成。请运行以下命令分析结果:"
        echo "    ls checkpoints/online_cls/"
        echo "    cat logs/nightly/summary_*.csv | tail -20"
        ;;

    --help|-h)
        echo "用法: ./deploy/launch_nightly.sh [--status|--pull|--help]"
        echo ""
        echo "  (无参数)  推送代码 + 在服务器后台启动所有实验"
        echo "  --status  查看服务器运行状态（日志尾部 + 进程）"
        echo "  --pull    从服务器拉取训练结果（checkpoints + logs）"
        ;;

    *)
        # 默认：推送 + 启动
        echo "=============================================="
        echo "  PFMval 夜间 LoRA 验证实验 — 启动"
        echo "=============================================="
        echo ""
        echo "服务器: ${SERVER_USER}@${SERVER_HOST}"
        echo "项目路径: ${SERVER_PROJECT_DIR}"
        echo "实验脚本: deploy/run_nightly_experiments.ps1"
        echo ""

        # Step 1: 推送代码到 GitHub
        echo ">>> Step 1/3: 推送代码到 GitHub..."
        cd "$PROJECT_DIR"
        git push origin main
        if [ $? -ne 0 ]; then
            echo "!!! GitHub 推送失败，请先处理"
            exit 1
        fi
        echo ""

        # Step 2: 服务器拉取代码
        echo ">>> Step 2/3: 服务器同步代码..."
        ${SSH_CMD} ${SERVER_USER}@${SERVER_HOST} \
            "cd ${SERVER_PROJECT_DIR} && git pull origin main 2>&1"
        echo ""

        # Step 3: 启动 PowerShell 实验脚本（后台运行）
        echo ">>> Step 3/3: 服务器后台启动实验..."
        NOW=$(date +%Y%m%d_%H%M%S)
        REMOTE_LOG="${SERVER_PROJECT_DIR}/logs/nightly/launch_${NOW}.log"

        # 使用 PowerShell Start-Process 后台执行
        # -NoNewWindow: 不弹窗
        # -Wait:$false: 不等待（后台）
        ${SSH_CMD} ${SERVER_USER}@${SERVER_HOST} \
            "powershell -Command \"
                Set-Location '${SERVER_PROJECT_DIR}';
                New-Item -ItemType Directory -Force -Path 'logs/nightly' | Out-Null;
                Start-Process powershell -ArgumentList '
                    -NoProfile -ExecutionPolicy Bypass -File deploy/run_nightly_experiments.ps1
                ' -NoNewWindow -Wait:\$false;
                Write-Host '>>> 实验已在服务器后台启动';
                Write-Host '>>> 查看进度: ssh ${SERVER_USER}@${SERVER_HOST} powershell -Command \\\"Get-Content ${REMOTE_LOG} -Tail 20\\\"';
            \""

        echo ""
        echo "=============================================="
        echo "  实验已提交到服务器后台"
        echo "=============================================="
        echo ""
        echo "查看状态:"
        echo "  ./deploy/launch_nightly.sh --status"
        echo ""
        echo "拉取结果:"
        echo "  ./deploy/launch_nightly.sh --pull"
        echo ""
        echo "预计耗时: 4-6 小时（9 个实验）"
        echo "=============================================="
        ;;
esac
