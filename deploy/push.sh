#!/bin/bash
# ============================================================
# push.sh — 推送代码到 Windows 服务器（Git 方式）
# ============================================================
# 用法: ./deploy/push.sh              # git push 到服务器
#       ./deploy/push.sh --scp        # scp 方式（全量复制）
#       ./deploy/push.sh --status     # 查看本地未推送的 commit
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "${SCRIPT_DIR}/config.sh"

MODE="${1:-git}"

cd "$PROJECT_DIR"

case "$MODE" in
    --status|-s)
        echo ">>> 本地相对于服务器的提交状态："
        echo ""
        git log --oneline server/main..HEAD 2>/dev/null || {
            echo "无法比较。请先设置 server remote："
            echo "  git remote add server ssh://${SERVER_USER}@${SERVER_HOST}:${SERVER_PORT}/$(cygpath -w "${SERVER_PROJECT_DIR}.git" 2>/dev/null || echo "${SERVER_PROJECT_DIR}.git")"
        }
        ;;

    --scp)
        echo ">>> SCP 方式推送代码到 ${SERVER_USER}@${SERVER_HOST}:${SERVER_PROJECT_DIR}/"
        echo ">>> 注意: SCP 会全量复制，不排除大文件，建议用于首次部署"
        echo ""

        # 使用 tar + ssh 管道方式传输（支持排除）
        EXCLUDES=$(build_excludes)
        eval tar czf - ${EXCLUDES} --exclude='.git' . | \
            ${SSH_CMD} ${SERVER_USER}@${SERVER_HOST} \
            "powershell -Command \"cd '${SERVER_PROJECT_DIR}'; tar xzf - 2>&1\""

        echo ""
        echo ">>> SCP 推送完成"
        ;;

    --help|-h)
        echo "用法: ./deploy/push.sh [git|--scp|--status]"
        echo ""
        echo "  (无参数)    Git push 到服务器（推荐，增量同步）"
        echo "  --scp       全量 tar+ssh 传输（用于首次部署）"
        echo "  --status    查看本地与服务器的差异"
        ;;

    *)
        # 默认：Git 方式推送
        echo ">>> Git push 到服务器..."
        echo ""

        # 检查 remote 是否已配置
        if ! git remote | grep -q "^server$"; then
            echo "!!! 未配置 server remote，请执行："
            echo ""
            echo "    git remote add server ssh://${SERVER_USER}@${SERVER_HOST}:${SERVER_PORT}/$(cygpath -w "${SERVER_PROJECT_DIR}.git" 2>/dev/null || echo "${SERVER_PROJECT_DIR}.git")"
            echo ""
            echo "    或使用 SCP 方式: ./deploy/push.sh --scp"
            exit 1
        fi

        git push server main
        echo ""
        echo ">>> 推送完成。服务器上执行: cd ${SERVER_PROJECT_DIR} && git pull"
        ;;
esac
