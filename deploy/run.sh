#!/bin/bash
# ============================================================
# run.sh — 推送 + 在 Windows 服务器上执行训练命令
# ============================================================
# 用法:
#   ./deploy/run.sh "python train_xxx.py --epochs 100"
#   ./deploy/run.sh --no-push "python train_xxx.py"
#   ./deploy/run.sh --bg "python train_xxx.py --epochs 200"   # 后台运行
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "${SCRIPT_DIR}/config.sh"

SKIP_PUSH=false
RUN_BG=false

TRAINING_CMD=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-push)
            SKIP_PUSH=true
            shift
            ;;
        --bg|--background)
            RUN_BG=true
            shift
            ;;
        --help|-h)
            echo "用法: ./deploy/run.sh [选项] '<训练命令>'"
            echo ""
            echo "选项:"
            echo "  --no-push       跳过代码推送"
            echo "  --bg            后台运行（使用 PowerShell Start-Process）"
            echo ""
            echo "示例:"
            echo "  ./deploy/run.sh 'python train_histogene_uni_tokens_augmix.py --epochs 100'"
            echo "  ./deploy/run.sh --bg 'python run_tv_sweep.py'"
            exit 0
            ;;
        *)
            TRAINING_CMD="$*"
            break
            ;;
    esac
done

if [ -z "$TRAINING_CMD" ]; then
    echo "用法: ./deploy/run.sh [--no-push] [--bg] '<训练命令>'"
    exit 1
fi

# Step 1: 推送代码
if [ "$SKIP_PUSH" = false ]; then
    echo "=============================================="
    echo "  Step 1/3: 推送代码到服务器"
    echo "=============================================="
    bash "${SCRIPT_DIR}/push.sh"
    echo ""

    # 服务器端 git pull（如果是 git 模式）
    echo ">>> 服务器端同步代码..."
    ${SSH_CMD} ${SERVER_USER}@${SERVER_HOST} \
        "cd ${SERVER_PROJECT_DIR} && git pull 2>/dev/null || echo '  (非 git 模式，跳过 pull)'"
    echo ""
fi

# Step 2: 远程执行
echo "=============================================="
echo "  Step 2/3: 服务器上执行训练"
echo "=============================================="
echo "命令: ${TRAINING_CMD}"
echo "服务器: ${SERVER_HOST}"
echo ""

# Windows 训练需要 PYTHONIOENCODING=utf-8（处理中文输出）
REMOTE_CMD="cd ${SERVER_PROJECT_DIR} && set PYTHONIOENCODING=utf-8 && ${TRAINING_CMD}"

if [ "$RUN_BG" = true ]; then
    # 使用 PowerShell Start-Process 后台运行
    # 输出重定向到日志文件
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOGFILE="${SERVER_PROJECT_DIR}/training_log_${TIMESTAMP}.txt"
    echo ">>> 后台运行，日志: ${LOGFILE}"
    ${SSH_CMD} ${SERVER_USER}@${SERVER_HOST} \
        "powershell -Command \"Start-Process cmd -ArgumentList '/c cd /d ${SERVER_PROJECT_DIR} && set PYTHONIOENCODING=utf-8 && ${TRAINING_CMD} > ${LOGFILE} 2>&1' -NoNewWindow -Wait:\$false\""
    echo ">>> 训练已在服务器后台启动"
    echo ">>> 查看日志: ssh ${SERVER_USER}@${SERVER_HOST} 'type ${LOGFILE}'"
else
    # 前台运行（SSH 保持连接，Ctrl+C 可中断）
    ${SSH_CMD} -t ${SERVER_USER}@${SERVER_HOST} \
        "cmd /c \"cd /d ${SERVER_PROJECT_DIR} && set PYTHONIOENCODING=utf-8 && ${TRAINING_CMD}\""
fi

echo ""

# Step 3: 询问是否拉取结果
echo "=============================================="
echo "  Step 3/3: 拉取训练结果"
echo "=============================================="
read -p "是否现在拉取训练结果？[y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    bash "${SCRIPT_DIR}/pull.sh"
fi

echo ""
echo ">>> 完成。"
