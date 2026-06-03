#!/bin/bash
# ============================================================
# PFMval 部署包打包脚本
# ============================================================
# 生成包含最佳模型 + 代码 + 配置的部署压缩包。
# 输出: pfmval_deploy_YYYYMMDD_HHMMSS.zip
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ZIP_NAME="pfmval_deploy_${TIMESTAMP}.zip"
TMPDIR="$PROJECT_ROOT/deploy/.pack_tmp"

echo "=== PFMval 部署包打包 ==="
echo "输出文件: $PROJECT_ROOT/$ZIP_NAME"
echo ""

# 清理旧的临时目录
rm -rf "$TMPDIR"
mkdir -p "$TMPDIR/pfmval_deploy"

cd "$PROJECT_ROOT"

# ---- 要打包的文件清单 ----
FILES=(
    # 训练脚本（最佳模型）
    "train_histogene_uni_tokens_augmix.py"

    # 模型架构
    "model_uni_tokens.py"

    # 数据集
    "dataset_uni_tokens_augmix.py"

    # 正则化
    "spatial_tv_loss.py"

    # 通知工具
    "notify_utils.py"

    # 路径配置
    "config_utils.py"

    # 配置文件
    "config.yaml"

    # histogene 受保护文件（不可修改，原样复制）
    "histogene/utils.py"

    # 部署文档
    "deploy/SYNC_GUIDE.md"
)

echo "复制代码文件..."
for f in "${FILES[@]}"; do
    if [ ! -f "$PROJECT_ROOT/$f" ]; then
        echo "  [WARN] 文件不存在，跳过: $f"
        continue
    fi
    dest_dir="$TMPDIR/pfmval_deploy/$(dirname "$f")"
    mkdir -p "$dest_dir"
    cp "$PROJECT_ROOT/$f" "$dest_dir/"
    echo "  [OK] $f"
done

# ---- 最佳模型权重 ----
echo ""
echo "复制最佳模型权重..."

BEST_CKPT="histogene/checkpoints/HisToGene_UNI_Tokens_AugMix/TV_Sweep_tv_l2_w0.01/best_histogene_uni_tokens_augmix.pth"
BEST_DESC="TV L2 w=0.01, Fold1 PCC=0.4170, 3-fold mean=0.3943"

# 也复制 3 折 CV 的 Fold2/Fold3 权重（如果存在）
CKPT_FOLD2="histogene/checkpoints/HisToGene_UNI_Tokens_AugMix/TV_3Fold_Fold2/best_histogene_uni_tokens_augmix.pth"
CKPT_FOLD3="histogene/checkpoints/HisToGene_UNI_Tokens_AugMix/TV_3Fold_Fold3/best_histogene_uni_tokens_augmix.pth"

if [ -f "$PROJECT_ROOT/$BEST_CKPT" ]; then
    ckpt_dir="$TMPDIR/pfmval_deploy/checkpoints/TV_Sweep_tv_l2_w0.01"
    mkdir -p "$ckpt_dir"
    cp "$PROJECT_ROOT/$BEST_CKPT" "$ckpt_dir/"
    echo "  [OK] TV_Sweep_tv_l2_w0.01 (Fold1 最佳, PCC=0.4170, ~112MB)"
else
    echo "  [WARN] 未找到最佳权重: $BEST_CKPT"
fi

for ckpt in "$CKPT_FOLD2" "$CKPT_FOLD3"; do
    if [ -f "$PROJECT_ROOT/$ckpt" ]; then
        fold_name=$(basename "$(dirname "$ckpt")")
        ckpt_dir="$TMPDIR/pfmval_deploy/checkpoints/$fold_name"
        mkdir -p "$ckpt_dir"
        cp "$PROJECT_ROOT/$ckpt" "$ckpt_dir/"
        echo "  [OK] $fold_name"
    fi
done

# ---- 如果权重未打包进去，检查历史最佳跨患者模型 ----
CROSS_PT_CKPT="histogene/checkpoints/HisToGene_UNI_Tokens_AugMix/CrossPatient_JFX_LMZ_to_HYZ_UNI_tokens_AugMix/best_histogene_uni_tokens_augmix.pth"
if [ -f "$PROJECT_ROOT/$CROSS_PT_CKPT" ] && [ ! -f "$TMPDIR/pfmval_deploy/checkpoints/CrossPatient_JFX_LMZ_to_HYZ/best.pth" ]; then
    ckpt_dir="$TMPDIR/pfmval_deploy/checkpoints/CrossPatient_Fold1"
    mkdir -p "$ckpt_dir"
    cp "$PROJECT_ROOT/$CROSS_PT_CKPT" "$ckpt_dir/best_histogene_uni_tokens_augmix.pth"
    echo "  [OK] CrossPatient_Fold1 (PCC=0.4212 TV model)"
fi

# ---- 生成文件清单 ----
echo ""
echo "生成文件清单..."
MANIFEST="$TMPDIR/pfmval_deploy/MANIFEST.txt"
cat > "$MANIFEST" << EOF
PFMval 部署包 — 文件清单
==========================
打包时间: $(date '+%Y-%m-%d %H:%M:%S')
Git 版本: $(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

最佳模型: HisToGene-UNI-Tokens + AugMix + TV Loss (L2 w=0.01)
性能指标: Fold1 PCC=0.4170, 3-fold CV Mean=0.3943
架构参数: LightweightTokenEncoder(hidden=512, layers=2, heads=8) + CoordEmbed(n_pos=128) + MLP(2048, dropout=0.5)
训练超参: lr=3e-5, batch_size=64, HuberLoss, tv_weight=0.01, tv_mode=l2

包含文件:
EOF

find "$TMPDIR/pfmval_deploy" -type f ! -name "MANIFEST.txt" | sort | while read -r f; do
    rel="${f#$TMPDIR/pfmval_deploy/}"
    size=$(stat --format=%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo "?")
    if [ "$size" != "?" ] && [ "$size" -gt 1048576 ]; then
        sz_mb=$(awk "BEGIN {printf \"%.1f MB\", $size/1048576}")
        echo "  $rel ($sz_mb)"
    else
        echo "  $rel"
    fi
done >> "$MANIFEST"

# ---- 打包 ----
echo ""
echo "创建压缩包..."
cd "$TMPDIR"
if command -v zip &>/dev/null; then
    zip -r "$PROJECT_ROOT/$ZIP_NAME" pfmval_deploy/ -x "*.pyc" "__pycache__/*"
    echo ""
    echo "========================================"
    echo " 打包完成!"
    echo " 文件: $ZIP_NAME"
    echo " 大小: $(du -h "$PROJECT_ROOT/$ZIP_NAME" 2>/dev/null || stat --format=%s "$PROJECT_ROOT/$ZIP_NAME" 2>/dev/null || echo "?")"
    echo "========================================"
else
    # 回退到 tar.gz
    tar -czf "$PROJECT_ROOT/${ZIP_NAME%.zip}.tar.gz" pfmval_deploy/ --exclude="*.pyc" --exclude="__pycache__"
    echo ""
    echo "========================================"
    echo " 打包完成! (tar.gz 格式，无 zip 命令)"
    echo " 文件: ${ZIP_NAME%.zip}.tar.gz"
    echo "========================================"
fi

# 清理临时目录
rm -rf "$TMPDIR"

echo ""
echo "下一步:"
echo "  1. 将 $ZIP_NAME 传输到服务器"
echo "  2. 服务器上解压: unzip $ZIP_NAME -d D:/AIPatho/qzs/"
echo "  3. 参照 pfmval_deploy/SETUP_服务器配置指南.txt 配置环境"
