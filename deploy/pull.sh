#!/bin/bash
# ============================================================
# pull.sh — 从 Windows 服务器拉取训练结果回本地
# ============================================================
# 用法: ./deploy/pull.sh                    # 拉取全部结果
#       ./deploy/pull.sh checkpoints        # 只拉取 checkpoints
#       ./deploy/pull.sh results_vis        # 只拉取 results_vis
#       ./deploy/pull.sh --dry-run          # 预览
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "${SCRIPT_DIR}/config.sh"

TARGET="${1:-all}"
DRY_RUN=""

if [ "$1" = "--dry-run" ] || [ "$1" = "-n" ]; then
    echo ">>> 预览模式"
    TARGET="all"
    DRY_RUN="true"
fi

cd "$PROJECT_DIR"

pull_path() {
    local remote_path="$1"
    local local_path="$2"
    local desc="$3"

    if [ "$DRY_RUN" = "true" ]; then
        echo "  [预览] ${SERVER_USER}@${SERVER_HOST}:${remote_path} → ${local_path}"
        return
    fi

    echo ">>> 拉取: ${desc}"
    ${SCP_CMD} -r "${SERVER_USER}@${SERVER_HOST}:${remote_path}" "${local_path}" 2>&1 || {
        echo "     (路径可能不存在，跳过)"
    }
}

case "$TARGET" in
    all)
        pull_path "${SERVER_PROJECT_DIR}/histogene/checkpoints/results_vis/" "./histogene/checkpoints/" "results_vis"
        pull_path "${SERVER_PROJECT_DIR}/checkpoints/" "./" "checkpoints"
        pull_path "${SERVER_PROJECT_DIR}/tv_sweep_results/" "./" "tv_sweep_results"
        pull_path "${SERVER_PROJECT_DIR}/tv_3fold_cv_results/" "./" "tv_3fold_cv_results"
        pull_path "${SERVER_PROJECT_DIR}/ensemble_results/" "./" "ensemble_results"
        pull_path "${SERVER_PROJECT_DIR}/phase3_search_results/" "./" "phase3_search_results"
        pull_path "${SERVER_PROJECT_DIR}/*.csv" "./" "CSV 结果"
        ;;
    checkpoints)
        pull_path "${SERVER_PROJECT_DIR}/checkpoints/" "./" "checkpoints"
        ;;
    results_vis)
        pull_path "${SERVER_PROJECT_DIR}/histogene/checkpoints/results_vis/" "./histogene/checkpoints/" "results_vis"
        ;;
    csv)
        pull_path "${SERVER_PROJECT_DIR}/*.csv" "./" "CSV"
        ;;
    *)
        pull_path "${SERVER_PROJECT_DIR}/${TARGET}" "./" "${TARGET}"
        ;;
esac

echo ""
if [ "$DRY_RUN" != "true" ]; then
    echo ">>> 结果拉取完成。"
fi
