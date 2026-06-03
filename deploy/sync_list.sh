#!/bin/bash
# ============================================================
# PFMval 手动同步 — 文件清单生成器
# ============================================================
# 用法:
#   ./deploy/sync_list.sh              # 显示自上次同步点以来的变更
#   ./deploy/sync_list.sh --mark       # 标记当前为已同步（创建 git tag）
#   ./deploy/sync_list.sh --all        # 列出所有需关注的 Python 文件
#   ./deploy/sync_list.sh --files a.py b.py  # 手动指定文件
#
# 每次手动同步到服务器后，运行 --mark 更新同步基准点。
# ============================================================

SYNC_TAG="last-sync"

# ---- 排除：不需要同步的文件/目录 ----
EXCLUDES=(
    "deploy/"
    "docs/"
    ".claude/"
    ".qoder/"
    "__pycache__/"
    "*.pyc"
    "*.log"
    "temp_*.py"
    "PAUSE_TRAINING"
    "training_status_*.txt"
)

build_grep_exclude() {
    local pattern=""
    for e in "${EXCLUDES[@]}"; do
        pattern="${pattern} -e '${e}'"
    done
    echo "$pattern"
}

show_files() {
    local files="$1"
    if [ -z "$files" ]; then
        echo "没有需要同步的文件。"
        return
    fi

    echo "========================================"
    echo " 需要同步的文件清单 (本地 -> 服务器)"
    echo "========================================"
    echo ""
    local count=0
    while IFS= read -r f; do
        [ -z "$f" ] && continue
        count=$((count + 1))
        # 标注文件类型
        case "$f" in
            train_*.py)      tag="[训练脚本]" ;;
            model_*.py)      tag="[模型定义]" ;;
            dataset_*.py)    tag="[数据集]" ;;
            config*.py)      tag="[配置工具]" ;;
            config.yaml)     tag="[配置文件] ⚠️ 服务器路径不同，请手动合并" ;;
            augment*.py)     tag="[数据增强]" ;;
            evaluate*.py)    tag="[评估]" ;;
            ensemble*.py)    tag="[集成]" ;;
            *.py)            tag="[脚本]" ;;
            *)               tag="" ;;
        esac
        printf "  %3d. %s %s\n" "$count" "$f" "$tag"
    done <<< "$files"

    echo ""
    echo "共 $count 个文件。"
    echo ""
    echo "同步方法:"
    echo "  1. 将上述文件复制到 U 盘/共享文件夹"
    echo "  2. 在服务器上覆盖到对应目录: ${SERVER_PROJECT_DIR:-<项目根目录>}"
    echo "  3. 运行 ./deploy/sync_list.sh --mark 标记同步完成"
}

case "${1:-}" in
    --mark)
        git tag -f "$SYNC_TAG" 2>/dev/null
        echo "[OK] 同步基准点已更新 (git tag: $SYNC_TAG)"
        echo "上次同步: $(git log -1 --format='%ci' "$SYNC_TAG")"
        ;;

    --all)
        # 列出所有 Python 源文件（排除 deploy/ 和受保护目录）
        FILES=$(git ls-files "*.py" "*.yaml" "*.yml" "*.sh" \
            | grep -v $(build_grep_exclude) \
            | grep -v "^histogene/" \
            | grep -v "^egnv1/" \
            | grep -v "^egnv2/" \
            | sort)
        show_files "$FILES"
        ;;

    --files)
        shift
        if [ $# -eq 0 ]; then
            echo "用法: $0 --files <file1> <file2> ..."
            exit 1
        fi
        FILES=""
        for f in "$@"; do
            if [ -f "$f" ]; then
                FILES="${FILES}${f}\n"
            else
                echo "[WARN] 文件不存在: $f"
            fi
        done
        show_files "$(echo -e "$FILES")"
        ;;

    *)
        # 默认：显示自上次同步以来的变更
        if git rev-parse "$SYNC_TAG" >/dev/null 2>&1; then
            echo "# 自 $(git log -1 --format='%ci' "$SYNC_TAG") 以来的变更"
            FILES=$(git diff --name-only "$SYNC_TAG" -- "*.py" "*.yaml" "*.yml" \
                | grep -v $(build_grep_exclude) \
                | grep -v "^histogene/" \
                | grep -v "^egnv1/" \
                | grep -v "^egnv2/")
        else
            echo "# 未找到同步基准点 (tag: $SYNC_TAG)，显示最近一次 commit 的变更"
            FILES=$(git diff --name-only HEAD~1 -- "*.py" "*.yaml" "*.yml" \
                | grep -v $(build_grep_exclude) \
                | grep -v "^histogene/" \
                | grep -v "^egnv1/" \
                | grep -v "^egnv2/")
        fi
        show_files "$FILES"
        ;;
esac
